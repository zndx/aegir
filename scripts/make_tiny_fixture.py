#!/usr/bin/env python3
"""Materialize a 16-table tiny fixture for BDD tier-1 scenarios.

Reads the cldr/signals GitTables drop and emits a column-subset parquet +
filtered ground-truth JSON under ``features/fixtures/gittables_tiny/``.
Idempotent: if outputs already exist, skips unless ``--force`` is passed.

Rationale: the full drop is 250 KB so copying it wholesale would be fine,
but a 16-table subset keeps tier-1 scenarios fast (<60 s each) and exercises
the exact same code path as the full run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SOURCE_DIR = Path("~/local/src/cldr/signals/build/datasets/gittables").expanduser()
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "features" / "fixtures" / "gittables_tiny"
NUM_TABLES = 16
SEED = 4649


def _hash_order(table_id: str, seed: int) -> int:
    """Deterministic ordering — pick the 16 lowest-hash tables."""
    return int(hashlib.blake2b(f"{seed}:{table_id}".encode(), digest_size=8).hexdigest(), 16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--num-tables", type=int, default=NUM_TABLES)
    parser.add_argument("--force", action="store_true", help="Re-build even if outputs exist")
    args = parser.parse_args(argv)

    src_parquet = args.source / "gittables_columns.parquet"
    src_gt = args.source / "gittables_gt.json"
    if not src_parquet.exists() or not src_gt.exists():
        print(f"ERROR: source not found at {args.source}", file=sys.stderr)
        return 2

    dst_parquet = args.dest / "gittables_columns.parquet"
    dst_gt = args.dest / "gittables_gt.json"
    if dst_parquet.exists() and dst_gt.exists() and not args.force:
        print(f"fixture already present at {args.dest} (use --force to rebuild)")
        return 0

    import pandas as pd

    args.dest.mkdir(parents=True, exist_ok=True)

    with open(src_gt) as f:
        gt = json.load(f)
    mappings: dict[str, str] = gt["mappings"]
    df = pd.read_parquet(src_parquet)

    # Restrict to tables that have at least one labeled column (most do).
    tables_with_labels = {key.split(".", 1)[0] for key in mappings}
    candidate_tables = sorted(
        set(df["source_table"].unique()) & tables_with_labels,
        key=lambda t: _hash_order(str(t), SEED),
    )
    picked = candidate_tables[: args.num_tables]
    picked_set = set(picked)

    df_tiny = df[df["source_table"].isin(list(picked_set))].reset_index(drop=True)
    gt_tiny = {k: v for k, v in mappings.items() if k.split(".", 1)[0] in picked_set}

    df_tiny.to_parquet(dst_parquet, index=False)
    with open(dst_gt, "w") as f:
        json.dump({"mappings": gt_tiny}, f, indent=2)

    labels = sorted(set(gt_tiny.values()))
    print(f"wrote {dst_parquet} ({len(df_tiny)} rows, {len(picked_set)} tables)")
    print(f"wrote {dst_gt} ({len(gt_tiny)} labeled cols, {len(labels)} distinct labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
