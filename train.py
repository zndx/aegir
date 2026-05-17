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
from aegir.utils.runs import RunArtifacts
from aegir.utils.train import (
    load_balancing_loss,
    group_params,
    f1_score_multilabel,
    boundary_diagnostics,
    format_boundary_diagnostics,
)
from aegir.data.table_dataset import (
    TASK_NUM_CLASSES,
    CPA_TASKS,
    GitTablesSignalsDataset,
    GitTablesDbpediaDataset,
    GitTablesSchemaorgDataset,
    SotabCTADataset,
    SotabCPADataset,
    SotabCTADbpediaDataset,
    SotabCPADbpediaDataset,
)
from aegir.data.tokenizer import ByteTokenizer


# Task id → Dataset class. train.py uses this to dispatch real-data
# loading when the user passes ``--task <id>`` without ``--smoke-test``.
# ``just benchmarks`` iterates over this table (minus smoke aliases) so
# registering a new benchmark here is the only wiring required.
TASK_DATASET_CLASSES = {
    "gt-signals-dbpedia":   GitTablesSignalsDataset,
    "sotab":                SotabCTADataset,
    "sotab-re":             SotabCPADataset,
    "sotab-dbp":            SotabCTADbpediaDataset,
    "sotab-dbp-re":         SotabCPADbpediaDataset,
    "gittables-dbpedia":    GitTablesDbpediaDataset,
    "gittables-schemaorg":  GitTablesSchemaorgDataset,
}


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

    def __init__(self, num_samples=256, seq_len=64, vocab_size=65536, num_classes=82):
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


def _make_datasets(args, is_main: bool):
    """Build (train, val) datasets. Smoke-test uses synthetic; otherwise
    dispatches via TASK_DATASET_CLASSES and requires the real data on disk.

    Keeps dataset construction out of main() so num_classes can be
    reconciled against the real label vocab before the model is built.
    """
    if args.smoke_test:
        num_classes = args.num_classes or TASK_NUM_CLASSES.get(args.task, 82)
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
        return train_dataset, val_dataset

    if args.task not in TASK_DATASET_CLASSES:
        raise NotImplementedError(
            f"Real dataset loading for task={args.task!r} is not yet implemented. "
            f"Supported: {sorted(TASK_DATASET_CLASSES)}. "
            f"Pass --smoke-test to use synthetic data instead."
        )
    dataset_cls = TASK_DATASET_CLASSES[args.task]
    tokenizer = ByteTokenizer()
    train_dataset = dataset_cls(
        data_dir=args.data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_context_cols=args.max_context_cols,
    )
    val_dataset = dataset_cls(
        data_dir=args.data_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_context_cols=args.max_context_cols,
    )
    if is_main:
        print(f"Loaded {args.task}: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset


def _observed_num_classes(*datasets) -> int | None:
    """Return ``max(label)+1`` across the provided datasets, or None if empty.

    Treats each dataset sample's ``label`` field as either an int or a
    sequence (multi-label CPA). None/empty datasets → None.
    """
    top = -1
    for ds in datasets:
        if ds is None or len(ds) == 0:
            continue
        for i in range(len(ds)):
            lbl = ds[i]["label"]
            if hasattr(lbl, "max"):
                v = int(lbl.max())
            else:
                v = int(lbl)
            if v > top:
                top = v
    return top + 1 if top >= 0 else None


def make_model(args) -> AegirForColumnAnnotation:
    """Create model from args.

    Respects ``args.num_classes`` when explicitly set (e.g. train.py has
    already probed the loaded dataset and knows the *real* vocab size);
    otherwise falls back to the pinned ``TASK_NUM_CLASSES`` entry. This
    matters for datasets whose label vocab is discovered at load time
    (GitTables 1M has ~1100+ DBpedia IRIs, not the 835 figure the paper
    headlines — properties + classes combined).
    """
    num_classes = args.num_classes or TASK_NUM_CLASSES.get(args.task)
    if num_classes is None:
        raise ValueError(
            f"num_classes undetermined for task={args.task!r}. "
            f"Set --num-classes explicitly or register in TASK_NUM_CLASSES."
        )

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


def _accumulate_boundary(accum: dict[str, float], diag: dict[str, float]) -> None:
    for k, v in diag.items():
        accum[k] = accum.get(k, 0.0) + v


def _finalize_boundary(accum: dict[str, float], num_batches: int) -> dict[str, float]:
    if num_batches == 0:
        return {}
    return {k: v / num_batches for k, v in accum.items()}


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
    boundary_accum: dict[str, float] = {}

    max_steps = getattr(args, "max_train_steps", 0) or 0
    for step_idx, batch in enumerate(loader):
        if max_steps and step_idx >= max_steps:
            break
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

        _accumulate_boundary(
            boundary_accum,
            boundary_diagnostics(output.bpred_output, target_N=args.downsample_factor),
        )

        if is_cpa:
            preds = (output.logits.detach() >= 0.0).int().cpu().numpy()
        else:
            preds = output.logits.detach().argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    avg_task_loss = total_task_loss / max(num_batches, 1)
    avg_lb_loss = total_lb_loss / max(num_batches, 1)

    # DDP-wrapped models hide .config behind .module; unwrap defensively.
    num_classes = getattr(model, "module", model).config.num_labels
    micro_f1, macro_f1, _ = f1_score_multilabel(all_labels, all_preds, num_classes=num_classes)

    return {
        "loss": avg_loss,
        "task_loss": avg_task_loss,
        "lb_loss": avg_lb_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "boundary": _finalize_boundary(boundary_accum, num_batches),
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
    boundary_accum: dict[str, float] = {}

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

        _accumulate_boundary(
            boundary_accum,
            boundary_diagnostics(output.bpred_output, target_N=args.downsample_factor),
        )

        if is_cpa:
            preds = (output.logits >= 0.0).int().cpu().numpy()
        else:
            preds = output.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    num_classes = getattr(model, "module", model).config.num_labels
    micro_f1, macro_f1, _ = f1_score_multilabel(all_labels, all_preds, num_classes=num_classes)

    return {
        "loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "boundary": _finalize_boundary(boundary_accum, num_batches),
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
    # Fast-path flags for benchmark smoke runs. Non-zero caps truncate
    # the respective loader so a single tiny-model pass through every
    # registered benchmark stays under 5-10 min. Defaults 0 = unlimited
    # (production behaviour).
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Truncate train set to N samples (0 = full set).")
    parser.add_argument("--max-val-samples", type=int, default=0,
                        help="Truncate val set to N samples (0 = full set).")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Cap optimizer steps per epoch (0 = full train loader).")

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

    # Run artifact writer (only on rank 0 — avoid duplicate sidecars under DDP).
    run = None
    if is_main:
        run = RunArtifacts.start(args, runs_root=Path(args.output_dir) / "runs")
        print(f"Run ID: {run.run_id}")

    # ── Dataset first, so num_classes can be discovered from the actual
    # label vocab before we build a model of the wrong output shape. ──
    train_dataset, val_dataset = _make_datasets(args, is_main)

    # Optional subsampling for fast benchmark smoke runs.
    if args.max_train_samples and len(train_dataset) > args.max_train_samples:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(args.max_train_samples)))
        if is_main:
            print(f"Subsampled train to first {args.max_train_samples} samples")
    if args.max_val_samples and len(val_dataset) > args.max_val_samples:
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, list(range(args.max_val_samples)))
        if is_main:
            print(f"Subsampled val to first {args.max_val_samples} samples")

    # Reconcile num_classes against the dataset's actual label vocab
    # (matters for GitTables where the paper's 835 figure differs from
    # the live data). Prefer max(label)+1 over len(vocab) because sparse
    # label indices also work.
    actual_num_classes = _observed_num_classes(train_dataset, val_dataset)
    task_default = TASK_NUM_CLASSES.get(args.task)
    if actual_num_classes is not None:
        args.num_classes = max(actual_num_classes, task_default or 0)
        if is_main and task_default and actual_num_classes > task_default:
            print(f"Note: observed {actual_num_classes} label classes in dataset, "
                  f"exceeds TASK_NUM_CLASSES[{args.task!r}]={task_default}. "
                  f"Using {args.num_classes}.")
    else:
        args.num_classes = task_default

    # Model
    model = make_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Parameters: {num_params:,}")
        if run is not None:
            run.set_num_params(num_params)

    model = model.to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Dataset already built above; the DataLoaders come from it.
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
            train_chunk_line = format_boundary_diagnostics(train_metrics.get("boundary", {}))
            if train_chunk_line:
                print(f"  train {train_chunk_line}")
            val_chunk_line = format_boundary_diagnostics(val_metrics.get("boundary", {}))
            if val_chunk_line:
                print(f"  val   {val_chunk_line}")

            if run is not None:
                run.add_epoch_metrics({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_task_loss": train_metrics.get("task_loss"),
                    "train_lb_loss": train_metrics.get("lb_loss"),
                    "val_loss": val_metrics["loss"],
                    "micro_f1": val_metrics["micro_f1"],
                    "macro_f1": val_metrics["macro_f1"],
                    "boundary": val_metrics.get("boundary", {}),
                    "wall_seconds": train_time,
                })

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
        if run is not None:
            run.finalize()
            print(f"Run artifacts: {run.run_dir}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
