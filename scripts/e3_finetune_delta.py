#!/usr/bin/env python
"""E3 (redefined 2026-06-15) — load-bearing claim as a FINE-TUNING DELTA.

Frozen-probe E3 was flat (E2(b) = byte-value-overlap, arm-invariant). New test:
fine-tune each ablation arm's byte-pretrained backbone end-to-end on a relational
task (SOTAB-DBpedia CPA = column-pair relation prediction), and ask whether the
DDL-injected `full` arm fine-tunes to higher macro-F1 than `no_schema` —
especially at low data (sample-efficiency, where a better structural prior helps
most). Same hyperparams/seeds across arms; `no_ontology` reported for sanity.

Decision: full − no_schema > 0 with 95% CI excluding 0 at ≥1 train size.
Drives train.py (with --pretrained/--max-train-samples/--metrics-out) as subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARMS = {
    "full": "/raid/checkpoints/aegir-artifacts/ablation_v1_ckpts/full/runs/20260605T032856Z_952013c6ca_tiny_pretrain/best_model.pt",
    "no_schema": "/raid/checkpoints/aegir-artifacts/ablation_v1_ckpts/no_schema/runs/20260605T033826Z_952013c6ca_tiny_pretrain/best_model.pt",
    "no_ontology": "/raid/checkpoints/aegir-artifacts/ablation_v1_ckpts/no_ontology/runs/20260605T033436Z_952013c6ca_tiny_pretrain/best_model.pt",
}


def run_one(arm, ckpt, size, seed, args, outdir) -> float | None:
    tag = f"{arm}_n{size}_s{seed}"
    mout = outdir / f"{tag}.json"
    if mout.exists():
        return float(json.loads(mout.read_text())["best_val_macro_f1"])
    cmd = [
        "uv", "run", "--no-sync", "python", str(REPO / "train.py"),
        "--task", args.task, "--data-dir", args.data_dir, "--single-label-ce",
        "--pretrained", ckpt, "--model-size", "tiny", "--vocab-size", "65536",
        "--max-train-samples", str(size), "--max-val-samples", str(args.max_val),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--seed", str(seed),
        "--metrics-out", str(mout), "--output-dir", str(outdir / tag),
    ]
    env = {**os.environ, "AEGIR_SOTAB_MAX_TABLES": str(args.max_tables)}
    print(f"  RUN {tag} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not mout.exists():
        print(f"    FAILED {tag} (rc={r.returncode}): {r.stderr[-300:]}")
        return None
    return float(json.loads(mout.read_text())["best_val_macro_f1"])


def ci_delta(a: list[float], b: list[float], rng, n=5000):
    """Bootstrap CI of mean(a) - mean(b) over independent seed samples."""
    aa, bb = np.array(a), np.array(b)
    d = np.array([rng.choice(aa, len(aa)).mean() - rng.choice(bb, len(bb)).mean() for _ in range(n)])
    return float(aa.mean() - bb.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="sotab-dbp-re")
    ap.add_argument("--data-dir", default="/raid/datasets/sotab")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 0])  # 0 = full train
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--arms", nargs="+", default=["full", "no_schema", "no_ontology"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--max-tables", type=int, default=1500,
                    help="cap SOTAB tables loaded per run (deterministic subset; same across arms)")
    ap.add_argument("--outdir", default="/raid/checkpoints/aegir-artifacts/evidence/e3")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    results: dict = {}  # (arm,size) -> [f1 per seed]
    for size in args.sizes:
        for arm in args.arms:
            f1s = []
            for seed in args.seeds:
                v = run_one(arm, ARMS[arm], size, seed, args, outdir)
                if v is not None:
                    f1s.append(v)
            results[f"{arm}|{size}"] = f1s
            print(f"[{arm} n={size or 'full'}] macro-F1: "
                  f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (n={len(f1s)})")

    print("\n=== E3 fine-tuning delta (full − no_schema) ===")
    summary = {"task": args.task, "sizes": args.sizes, "seeds": args.seeds, "by_arm_size": results, "deltas": {}}
    any_clean = False
    for size in args.sizes:
        full = results.get(f"full|{size}", [])
        nos = results.get(f"no_schema|{size}", [])
        noo = results.get(f"no_ontology|{size}", [])
        if not full or not nos:
            continue
        d, lo, hi = ci_delta(full, nos, rng)
        clean = lo > 0
        any_clean = any_clean or clean
        sane = (f"  [no_ontology {np.mean(noo):.4f}]" if noo else "")
        print(f"  n={size or 'full':<5} full {np.mean(full):.4f} − no_schema {np.mean(nos):.4f} "
              f"= Δ{d:+.4f} [{lo:+.4f},{hi:+.4f}]  {'CLEAN' if clean else 'spans-0'}{sane}")
        summary["deltas"][str(size)] = {"full": np.mean(full).item(), "no_schema": np.mean(nos).item(),
                                        "no_ontology": (np.mean(noo).item() if noo else None),
                                        "delta": d, "ci": [lo, hi], "ci_clean": bool(clean)}
    summary["G_rel_supported"] = bool(any_clean)
    (outdir / "e3_delta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nG-rel (full > no_schema CI-clean at ≥1 size): {any_clean}  → {outdir}/e3_delta_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
