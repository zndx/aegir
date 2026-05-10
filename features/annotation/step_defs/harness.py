"""Test harness for GitTables CTA scenarios.

Encapsulates model construction, dataset loading, and training loop access so
step definitions stay thin. ``build_tiny_model`` mirrors the ``tiny`` config
in ``train.py::make_model`` but with 120 classes for the gt-signals task and
adjustments to keep sequence lengths / head counts sane at small d_model.

Rule of thumb: any logic that belongs in a production training script goes in
``train.py`` or ``src/aegir/…``. This file is for orchestrating *behave
scenarios* over that code — glue, not reimplementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from aegir.data.table_dataset import GitTablesSignalsDataset, TASK_NUM_CLASSES
from aegir.data.tokenizer import ByteTokenizer
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForColumnAnnotation


def _tiny_args(task: str = "gt-signals-dbpedia") -> argparse.Namespace:
    """Args namespace compatible with train.py's make_model + training helpers."""
    num_classes = TASK_NUM_CLASSES[task]
    return argparse.Namespace(
        task=task,
        model_size="tiny",
        vocab_size=ByteTokenizer.vocab_size,
        num_classes=num_classes,
        batch_size=4,
        lr=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lambda_lb=0.1,
        downsample_factor=2.0,
        amp=False,  # BDD scenarios run without AMP for determinism
        seed=4649,
        num_workers=0,
        output_dir="outputs",
        smoke_test=False,
        log_interval=10,
        max_length=128,
        max_context_cols=4,
    )


def build_tiny_model(task: str = "gt-signals-dbpedia", device: torch.device | None = None) -> AegirForColumnAnnotation:
    """Construct the ``tiny`` Aegir config adapted for the gt-signals task.

    Mirrors ``train.py::make_model(args)`` with ``--model-size tiny`` but:
      - swaps ``num_labels`` to match TASK_NUM_CLASSES[task]
      - uses the byte tokenizer's vocab size
    """
    cfg = AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192],
        d_intermediate=[0, 0, 0],
        vocab_size=ByteTokenizer.vocab_size,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[8, 12, 12], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=TASK_NUM_CLASSES[task],
        task_type="cta",
    )
    model = AegirForColumnAnnotation(cfg)
    if device is not None:
        model = model.to(device)
    return model


@dataclass
class TrainingHarness:
    """Bundle model + dataset + optimizer for a scenario.

    Attributes:
        model: AegirForColumnAnnotation on the specified device.
        train_loader: DataLoader yielding tiny batches from the tiny fixture.
        val_loader: DataLoader for the val split.
        optimizer: AdamW over grouped params.
        scheduler: Cosine LR with warmup (identity if warmup_steps==0).
        loss_fn: CrossEntropyLoss (CTA single-label).
        args: Namespace of training knobs consumed by train_epoch.
        device: torch device.
        loss_history: per-step training loss, populated by train_n_steps.
    """

    model: AegirForColumnAnnotation
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: torch.nn.Module
    args: argparse.Namespace
    device: torch.device
    loss_history: list[float] = field(default_factory=list)


def _make_loaders(
    data_dir: Path | None,
    tokenizer: ByteTokenizer,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    from train import collate_fn  # reuse the project's collate; no reimplementation

    train_ds = GitTablesSignalsDataset(
        data_dir=data_dir, split="train",
        tokenizer=tokenizer, max_length=args.max_length,
        max_context_cols=args.max_context_cols,
    )
    val_ds = GitTablesSignalsDataset(
        data_dir=data_dir, split="val",
        tokenizer=tokenizer, max_length=args.max_length,
        max_context_cols=args.max_context_cols,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False,
    )
    return train_loader, val_loader


def make_harness(
    data_dir: Path | None,
    device: torch.device,
    *,
    batch_size: int = 4,
    max_context_cols: int = 4,
    lr: float = 5e-4,
    max_length: int = 128,
) -> TrainingHarness:
    """Build a scenario-ready harness. Small defaults keep tier-1 fast.

    ``data_dir=None`` uses the default gt-signals-dbpedia path (full benchmark);
    pass the tiny-fixture path to run against the 16-table subset.
    """
    from aegir.utils.train import group_params

    args = _tiny_args()
    args.batch_size = batch_size
    args.max_context_cols = max_context_cols
    args.lr = lr
    args.max_length = max_length
    tokenizer = ByteTokenizer()
    train_loader, val_loader = _make_loaders(data_dir, tokenizer, args)
    model = build_tiny_model(task=args.task, device=device)

    optimizer = torch.optim.AdamW(group_params(model), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    loss_fn = torch.nn.CrossEntropyLoss()

    return TrainingHarness(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        args=args,
        device=device,
    )


def train_n_steps(h: TrainingHarness, n: int) -> dict:
    """Run exactly ``n`` optimizer steps, cycling the train loader as needed.

    Populates ``h.loss_history`` with the per-step task loss. Returns a summary
    dict with first/last mean losses plus mean boundary diagnostics over the run.
    """
    from aegir.utils.train import load_balancing_loss, boundary_diagnostics

    h.model.train()
    it = iter(h.train_loader)
    boundary_accum: dict[str, float] = {}
    step = 0
    while step < n:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(h.train_loader)
            batch = next(it)

        batch = {k: v.to(h.device) for k, v in batch.items()}
        output = h.model(
            batch["input_ids"],
            role_ids=batch["role_ids"],
            cls_indexes=batch["cls_indexes"],
            mask=batch["mask"],
        )
        task_loss = h.loss_fn(output.logits, batch["labels"])
        lb_loss = torch.tensor(0.0, device=h.device)
        if output.bpred_output:
            for bp in output.bpred_output:
                lb_loss = lb_loss + load_balancing_loss(bp, N=h.args.downsample_factor)
            lb_loss = lb_loss / len(output.bpred_output)
        loss = task_loss + h.args.lambda_lb * lb_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.model.parameters(), h.args.max_grad_norm)
        h.optimizer.step()
        h.scheduler.step()
        h.optimizer.zero_grad(set_to_none=True)

        h.loss_history.append(task_loss.item())
        for k, v in boundary_diagnostics(output.bpred_output, target_N=h.args.downsample_factor).items():
            boundary_accum[k] = boundary_accum.get(k, 0.0) + v
        step += 1

    if n > 0 and boundary_accum:
        boundary_accum = {k: v / n for k, v in boundary_accum.items()}

    k = min(10, len(h.loss_history) // 2) if len(h.loss_history) >= 2 else len(h.loss_history)
    first = h.loss_history[:k]
    last = h.loss_history[-k:] if k > 0 else h.loss_history
    return {
        "first_mean": sum(first) / max(len(first), 1),
        "last_mean": sum(last) / max(len(last), 1),
        "num_steps": len(h.loss_history),
        "boundary": boundary_accum,
    }


def evaluate_micro_macro(h: TrainingHarness) -> dict[str, float]:
    """Run the val loader and return micro/macro F1."""
    from aegir.utils.train import f1_score_multilabel

    h.model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in h.val_loader:
            batch = {k: v.to(h.device) for k, v in batch.items()}
            output = h.model(
                batch["input_ids"],
                role_ids=batch["role_ids"],
                cls_indexes=batch["cls_indexes"],
                mask=batch["mask"],
            )
            preds = output.logits.argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    micro, macro, _ = f1_score_multilabel(all_labels, all_preds, num_classes=h.model.config.num_labels)
    return SimpleNamespace(micro_f1=micro, macro_f1=macro).__dict__
