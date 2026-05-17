#!/usr/bin/env python
"""Evaluate an Aegir SOTAB-CTA checkpoint on all SOTAB-v2 test splits.

The standard SOTAB-v2 benchmark ships five test CSVs of growing
difficulty:

  - ``sotab_v2_cta_test_set.csv``                         — main test
  - ``sotab_v2_cta_random_test_set.csv``                  — random subset
  - ``sotab_v2_cta_corner_cases_test_set.csv``            — challenging
  - ``sotab_v2_cta_missing_values_test_set.csv``          — missing values
  - ``sotab_v2_cta_format_heterogeneity_test_set.csv``    — varied formats

We point the dataset class at each one in turn via a small monkey-patch
of ``_gt_csv_path`` (the class wasn't built with robustness-split
support; here we add it externally without touching the dataset code).

Output: per-split micro/macro F1 + accuracy + sample count, written to
``<output>/sotab_test_metrics.json``.

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/aegir_eval_sotab.py \\
        --checkpoint /raid/checkpoints/aegir-sotab-cta-v0.1/<run-id>/final.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("aegir-eval-sotab")


# The five SOTAB-v2 CTA test splits, in the order we report them.
SOTAB_CTA_TEST_SPLITS = [
    ("test",                  "sotab_v2_cta_test_set.csv"),
    ("random",                "sotab_v2_cta_random_test_set.csv"),
    ("corner_cases",          "sotab_v2_cta_corner_cases_test_set.csv"),
    ("missing_values",        "sotab_v2_cta_missing_values_test_set.csv"),
    ("format_heterogeneity",  "sotab_v2_cta_format_heterogeneity_test_set.csv"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pt (raw state_dict).")
    p.add_argument("--data-dir", default="/raid/datasets/sotab")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Defaults to "
                        "<checkpoint-dir>/sotab_test_metrics.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-context-cols", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate at most this many samples per split. "
                        "0 = all. Useful for sanity checks.")
    # Model rebuild args — train.py saves only the raw state_dict
    # (`best_model_state = deepcopy(raw_model.state_dict())`), so we
    # re-derive the architecture from the user's training flags rather
    # than reading them from the checkpoint.
    p.add_argument("--model-size", default="small",
                   choices=("tiny", "small", "base"))
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--num-classes", type=int, default=82,
                   help="Must match the head dim of the trained model.")
    p.add_argument("--downsample-factor", type=float, default=2.0)
    return p.parse_args()


def _patch_gt_csv_path(dataset_cls, csv_filename: str) -> None:
    """Monkey-patch ``_gt_csv_path`` on a SOTAB dataset class so it
    returns the requested CSV filename verbatim, regardless of split
    name. Lets us reuse the existing dataset for the 5 test variants.
    """
    def _gt_csv_path(self, split):  # noqa: ARG001 — split ignored
        return Path(self.data_dir) / csv_filename
    dataset_cls._gt_csv_path = _gt_csv_path  # type: ignore[method-assign]


def _evaluate_one_split(model, dataset, device, batch_size, num_workers,
                        loss_fn) -> dict:
    """Run evaluation on a single dataset; mirrors train.py's
    ``evaluate()`` but standalone and metric-only."""
    import torch
    from torch.utils.data import DataLoader

    from aegir.utils.train import f1_score_multilabel

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            cls_indexes = batch["cls_indexes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids, role_ids=role_ids,
                    cls_indexes=cls_indexes, mask=mask,
                )
                loss = loss_fn(output.logits, labels)

            total_loss += loss.item()
            num_batches += 1
            num_samples += labels.size(0)

            all_preds.append(output.logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

    import numpy as np
    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    num_classes = int(model.config.num_labels)
    micro_f1, macro_f1, _ = f1_score_multilabel(
        all_labels_np.tolist(), all_preds_np.tolist(),
        num_classes=num_classes,
    )
    accuracy = float((all_preds_np == all_labels_np).mean())

    return {
        "loss": total_loss / max(num_batches, 1),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "n_samples": int(num_samples),
        "n_batches": int(num_batches),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    import torch
    import torch.nn as nn

    from aegir.data.table_dataset import SotabCTADataset
    from aegir.data.tokenizer import ByteTokenizer
    from aegir.models.heads import AegirForColumnAnnotation
    from aegir.models.config import AegirConfig

    print("=" * 70)
    print("Aegir SOTAB-CTA held-out evaluation")
    print("=" * 70)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"[1/4] loading checkpoint: {ckpt_path}")
    # train.py saves a raw state_dict, so we rebuild the architecture
    # from CLI flags. Match what make_model() in train.py does.
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # Strip any DDP "module." prefix
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    print(f"[2/4] rebuilding {args.model_size} arch + loading state…")
    from aegir.models.config import AttnConfig, RWKVConfig, SSMConfig
    if args.model_size == "tiny":
        config = AegirConfig(
            arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
            d_model=[128, 192, 192],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[2, 3, 3],
                                rotary_emb_dim=[8, 12, 12], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=args.num_classes,
            task_type="cta",
        )
    elif args.model_size == "small":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
            d_model=[256, 384, 384],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[4, 6, 6],
                                rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=args.num_classes,
            task_type="cta",
        )
    else:  # base
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w12"], "w4"], "w4"],
            d_model=[768, 1024, 1024],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=128, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[12, 16, 16],
                                rotary_emb_dim=[48, 64, 64], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=args.num_classes,
            task_type="cta",
        )
    model = AegirForColumnAnnotation(config).to(args.device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning("unexpected keys when loading state_dict: %r",
                       unexpected[:5])
    if missing:
        logger.warning("missing keys when loading state_dict: %r",
                       missing[:5])

    tokenizer = ByteTokenizer()
    loss_fn = nn.CrossEntropyLoss()

    print(f"[3/4] evaluating on {len(SOTAB_CTA_TEST_SPLITS)} test splits")
    results: dict[str, dict] = {}
    t_total = time.time()

    for split_name, csv_filename in SOTAB_CTA_TEST_SPLITS:
        t0 = time.time()
        csv_path = Path(args.data_dir) / csv_filename
        if not csv_path.exists():
            print(f"  skip {split_name:<22s}  (CSV missing: {csv_path})")
            continue

        # Patch the dataset class to use this CSV.
        _patch_gt_csv_path(SotabCTADataset, csv_filename)
        try:
            dataset = SotabCTADataset(
                data_dir=args.data_dir, task="sotab", split="test",
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_context_cols=args.max_context_cols,
            )
        except Exception as exc:
            print(f"  failed on {split_name}: {exc}")
            results[split_name] = {"error": str(exc)}
            continue

        if args.limit > 0 and args.limit < len(dataset):
            from torch.utils.data import Subset
            dataset = Subset(dataset, list(range(args.limit)))

        n = len(dataset)
        print(f"  → {split_name:<22s}  n={n:6d}  evaluating…", flush=True)
        metrics = _evaluate_one_split(
            model, dataset, args.device,
            args.batch_size, args.num_workers, loss_fn,
        )
        elapsed = time.time() - t0
        metrics["elapsed_sec"] = elapsed
        results[split_name] = metrics
        print(f"    micro_f1={metrics['micro_f1']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"({elapsed:.1f}s)")

    print()
    print("=" * 70)
    print(f"=== eval complete — {time.time()-t_total:.1f}s total ===")
    out_path = (
        Path(args.output) if args.output
        else ckpt_path.parent / "sotab_test_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "splits": results,
        "num_labels": config.num_labels,
    }, indent=2))
    print(f"results: {out_path}")
    print()
    print("summary:")
    print(f"{'split':<22s}  {'n':>6s}  {'micro_f1':>9s}  "
          f"{'macro_f1':>9s}  {'accuracy':>9s}")
    for split_name, _ in SOTAB_CTA_TEST_SPLITS:
        if split_name not in results or "error" in results[split_name]:
            continue
        r = results[split_name]
        print(f"  {split_name:<20s}  {r['n_samples']:>6d}  "
              f"{r['micro_f1']:>9.4f}  {r['macro_f1']:>9.4f}  "
              f"{r['accuracy']:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
