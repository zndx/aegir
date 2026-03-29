"""Aegir training script for CTA/CPA column annotation tasks.

Usage:
    # Smoke test with synthetic data (no dataset needed)
    uv run python train.py --smoke-test

    # Train on SOTAB CTA
    uv run python train.py --task sotab --data-dir /path/to/sotab

    # Multi-GPU (DDP)
    uv run torchrun --nproc_per_node=6 train.py --task sotab --data-dir /path/to/sotab
"""

import argparse
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

from aegir.models.config import AegirConfig, SSMConfig, AttnConfig, RWKVConfig
from aegir.models.heads import AegirForColumnAnnotation, ColumnAnnotationOutput
from aegir.utils.train import load_balancing_loss, group_params, f1_score_multilabel
from aegir.data.table_dataset import TASK_NUM_CLASSES, CPA_TASKS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SyntheticTableDataset(Dataset):
    """Synthetic dataset for smoke testing — no real data needed."""

    def __init__(self, num_samples=256, seq_len=64, vocab_size=65536, num_classes=91):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "role_ids": torch.cat([
                torch.zeros(self.seq_len // 2, dtype=torch.long),
                torch.ones(self.seq_len // 2, dtype=torch.long),
            ]),
            "cls_index": torch.tensor(0, dtype=torch.long),
            "label": torch.tensor(idx % self.num_classes, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate batch with padding to max length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)

    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    role_ids = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    cls_indexes = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        role_ids[i, :L] = b["role_ids"]
        mask[i, :L] = True
        cls_indexes[i] = b["cls_index"]
        labels[i] = b["label"]

    return {
        "input_ids": input_ids,
        "role_ids": role_ids,
        "mask": mask,
        "cls_indexes": cls_indexes,
        "labels": labels,
    }


def make_model(args) -> AegirForColumnAnnotation:
    """Create model from args."""
    num_classes = TASK_NUM_CLASSES.get(args.task, args.num_classes)

    if args.model_size == "tiny":
        config = AegirConfig(
            arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
            d_model=[128, 192, 192],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[8, 12, 12], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=num_classes,
            task_type="cpa" if args.task in CPA_TASKS else "cta",
        )
    elif args.model_size == "small":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
            d_model=[256, 384, 384],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(),
            attn_cfg=AttnConfig(num_heads=[4, 6, 6], rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=num_classes,
            task_type="cpa" if args.task in CPA_TASKS else "cta",
        )
    elif args.model_size == "base":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w12"], "w4"], "w4"],
            d_model=[768, 1024, 1024],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(),
            attn_cfg=AttnConfig(num_heads=[12, 16, 16], rotary_emb_dim=[32, 48, 64], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=num_classes,
            task_type="cpa" if args.task in CPA_TASKS else "cta",
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    return AegirForColumnAnnotation(config)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, args):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_lb_loss = 0.0
    all_preds = []
    all_labels = []
    is_cpa = args.task in CPA_TASKS
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        cls_indexes = batch["cls_indexes"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            output: ColumnAnnotationOutput = model(
                input_ids, role_ids=role_ids, cls_indexes=cls_indexes, mask=mask,
            )

            task_loss = loss_fn(output.logits, labels)

            lb_loss = torch.tensor(0.0, device=device)
            if output.bpred_output:
                for bpred in output.bpred_output:
                    lb_loss = lb_loss + load_balancing_loss(bpred, N=args.downsample_factor)
                lb_loss = lb_loss / len(output.bpred_output)

            loss = task_loss + args.lambda_lb * lb_loss

        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_lb_loss += lb_loss.item()
        num_batches += 1

        if is_cpa:
            preds = (output.logits.detach() >= 0.0).int().cpu().numpy()
        else:
            preds = output.logits.detach().argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    avg_task_loss = total_task_loss / max(num_batches, 1)
    avg_lb_loss = total_lb_loss / max(num_batches, 1)

    micro_f1, macro_f1, _ = f1_score_multilabel(all_labels, all_preds, num_classes=model.config.num_labels)

    return {
        "loss": avg_loss,
        "task_loss": avg_task_loss,
        "lb_loss": avg_lb_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, args):
    """Run evaluation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    is_cpa = args.task in CPA_TASKS
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        cls_indexes = batch["cls_indexes"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            output = model(
                input_ids, role_ids=role_ids, cls_indexes=cls_indexes, mask=mask,
            )
            loss = loss_fn(output.logits, labels)

        total_loss += loss.item()
        num_batches += 1

        if is_cpa:
            preds = (output.logits >= 0.0).int().cpu().numpy()
        else:
            preds = output.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    micro_f1, macro_f1, _ = f1_score_multilabel(all_labels, all_preds, num_classes=model.config.num_labels)

    return {
        "loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Aegir CTA/CPA Training")
    # Data
    parser.add_argument("--task", type=str, default="sotab", choices=list(TASK_NUM_CLASSES.keys()))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-context-cols", type=int, default=8)

    # Model
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--num-classes", type=int, default=91)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-lb", type=float, default=0.1, help="Load balancing loss weight")
    parser.add_argument("--downsample-factor", type=float, default=2.0)

    # System
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=4649)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic data for smoke testing")
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    if args.no_amp:
        args.amp = False

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    set_seed(args.seed)

    if is_main:
        print(f"=== Aegir Training ===")
        print(f"Task: {args.task}")
        print(f"Model size: {args.model_size}")
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed}")
        print(f"AMP: {args.amp}")
        print()

    # Model
    model = make_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Parameters: {num_params:,}")

    model = model.to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Dataset
    if args.smoke_test or args.data_dir is None:
        num_classes = TASK_NUM_CLASSES.get(args.task, args.num_classes)
        train_dataset = SyntheticTableDataset(
            num_samples=256, seq_len=args.max_length,
            vocab_size=args.vocab_size, num_classes=num_classes,
        )
        val_dataset = SyntheticTableDataset(
            num_samples=64, seq_len=args.max_length,
            vocab_size=args.vocab_size, num_classes=num_classes,
        )
        if is_main:
            print(f"Using synthetic data: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        raise NotImplementedError(
            "Real dataset loading not yet implemented. "
            "Use --smoke-test for testing with synthetic data."
        )

    # DataLoaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # Optimizer & scheduler
    param_groups = group_params(raw_model)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Loss function
    is_cpa = args.task in CPA_TASKS
    if is_cpa:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if is_main:
        print(f"Loss: {'BCEWithLogitsLoss' if is_cpa else 'CrossEntropyLoss'}")
        print(f"Training steps: {num_training_steps} ({num_warmup_steps} warmup)")
        print()

    # Training loop
    best_val_f1 = -1.0
    best_model_state = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, args)
        train_time = time.time() - t0

        val_metrics = evaluate(raw_model, val_loader, loss_fn, device, args)

        if is_main:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"train loss={train_metrics['loss']:.4f} "
                f"(task={train_metrics['task_loss']:.4f} lb={train_metrics['lb_loss']:.4f}) "
                f"F1={train_metrics['micro_f1']:.3f}/{train_metrics['macro_f1']:.3f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"F1={val_metrics['micro_f1']:.3f}/{val_metrics['macro_f1']:.3f} | "
                f"{train_time:.1f}s"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_model_state = deepcopy(raw_model.state_dict())
                torch.save(best_model_state, output_dir / "best_model.pt")
                print(f"  -> New best val macro F1: {best_val_f1:.4f}")

    if is_main:
        print(f"\nTraining complete. Best val macro F1: {best_val_f1:.4f}")
        if best_model_state is not None:
            torch.save(best_model_state, output_dir / "best_model.pt")
            print(f"Best model saved to {output_dir / 'best_model.pt'}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
