#!/usr/bin/env python
"""Re-emit a parquet with smaller row groups for rank-shardable reads.

Phase 0's split helper used pyarrow defaults that coalesced 2M rows
into just 3 row groups, which can't be split 6 ways for DDP without
falling back to in-memory slicing. This script streams the source
parquet through with a smaller batch size so each batch becomes one
row group.

Stream-reads in batches (no full-table load), so memory stays bounded.

Usage::

    just setup-libs  # if you haven't yet
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/reshard_pretrain_parquet.py \\
        --input  /raid/datasets/tapas/aegir_bytes_v0_train.parquet \\
        --output /raid/datasets/tapas/aegir_bytes_v0_train_shardable.parquet \\
        --rows-per-rg 32000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows-per-rg", type=int, default=32000,
                   help="Rows per row group. 32k × 6-way DDP = ~33 row "
                        "groups for a 2M-row train parquet; comfortable.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    if dst.exists():
        print(f"NOTE: overwriting existing {dst}")

    pf = pq.ParquetFile(str(src), memory_map=True)
    schema = pf.schema_arrow
    md = pf.metadata
    print(f"input:  {src}  rows={md.num_rows:,}  row_groups={md.num_row_groups}")
    print(f"output: {dst}  rows-per-rg={args.rows_per_rg:,}")

    writer = pq.ParquetWriter(str(dst), schema, compression="zstd")
    t0 = time.time()
    n_written = 0
    n_rg_out = 0
    for batch in pf.iter_batches(batch_size=args.rows_per_rg):
        writer.write_batch(batch)
        n_written += batch.num_rows
        n_rg_out += 1
        if n_rg_out % 10 == 0:
            elapsed = time.time() - t0
            rate = n_written / max(elapsed, 1e-9)
            print(f"  wrote {n_written:>10,} rows / {md.num_rows:,}  "
                  f"({n_rg_out} row groups)  rate={rate:.0f}/s")
    writer.close()
    elapsed = time.time() - t0
    print(f"\ndone: {n_written:,} rows in {n_rg_out} row groups ({elapsed:.1f}s)")

    # Verify
    md_out = pq.read_metadata(str(dst))
    print(f"verify: rows={md_out.num_rows:,}  row_groups={md_out.num_row_groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
