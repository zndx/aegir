#!/usr/bin/env python
"""Split a parquet of pretraining byte sequences into train / val.

Reads a single parquet (output of ``tapas_proto_to_aegir.py``), shuffles
deterministically by seed, and writes two parquets with the same schema.

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/split_pretrain_parquet.py \\
        --input /raid/datasets/tapas/aegir_bytes_v0.parquet \\
        --train-out /raid/datasets/tapas/aegir_bytes_v0_train.parquet \\
        --val-out /raid/datasets/tapas/aegir_bytes_v0_val.parquet \\
        --val-ratio 0.05 \\
        --seed 4649
"""

from __future__ import annotations

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--train-out", required=True)
    p.add_argument("--val-out", required=True)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=4649)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"reading {args.input}")
    table = pq.read_table(args.input)
    n = table.num_rows
    print(f"  {n:,} rows")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    print(f"  split: train={n_train:,} val={n_val:,}")

    train_tbl = table.take(pa.array(train_idx))
    val_tbl = table.take(pa.array(val_idx))

    pq.write_table(train_tbl, args.train_out, compression="zstd")
    pq.write_table(val_tbl, args.val_out, compression="zstd")
    print(f"wrote {args.train_out}")
    print(f"wrote {args.val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
