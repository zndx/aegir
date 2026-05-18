#!/usr/bin/env python
"""Convert generated chapter parquet → byte-level training parquet.

The generated chapter parquet from ``scripts/generate_chapter.py`` has rows
of (chapter_id, response_text, ablation, family, model, ...) with markdown
response_text 4-8K chars each. The training script ``train_pretrain.py``
consumes a parquet of (table_id, byte_ids: list<uint32>, length: uint32)
rows that the ``ParquetByteDataset`` reads row-group-sharded across DDP
ranks.

This script bridges the two. For each chapter, encode response_text with
the byte tokenizer, optionally split into fixed chunks (so we get more
training rows and bound per-sample memory), and emit one parquet row per
chunk.

We also split into train/val deterministically (90/10 by chapter_id hash)
so the val set is disjoint from train within an arm. The output goes to
two parquets so train_pretrain.py can ingest them via --train-parquet and
--val-parquet flags.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/chapters_to_byte_parquet.py \\
        --chapters-parquet /raid/.../ablation_v0/<arm>/chapters.parquet \\
        --out-dir /raid/.../ablation_v0_bytes/<arm>/ \\
        --chunk-size 2048 \\
        --val-frac 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.data.tokenizer import ByteTokenizer  # noqa: E402

logger = logging.getLogger("chapters→bytes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-parquet", required=True,
                   help="Source chapters.parquet from generate_chapter.py")
    p.add_argument("--out-dir", required=True,
                   help="Output directory (writes train.parquet + val.parquet)")
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="Max bytes per output row. Long chapters split into N chunks.")
    p.add_argument("--min-chunk-size", type=int, default=128,
                   help="Drop tail chunks shorter than this (avoid degenerate samples)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Hash-deterministic train/val split fraction")
    p.add_argument("--rg-size", type=int, default=64,
                   help="Rows per row-group (controls DDP shard granularity)")
    return p.parse_args()


def is_val(chapter_id: str, val_frac: float) -> bool:
    """Deterministic hash split — same chapter always lands in same split."""
    h = int(hashlib.sha256(chapter_id.encode()).hexdigest()[:12], 16)
    return (h % 10_000) / 10_000.0 < val_frac


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = pq.read_table(args.chapters_parquet).to_pandas()
    logger.info(f"input: {len(chapters)} chapters from {args.chapters_parquet}")

    tok = ByteTokenizer()
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    total_bytes_in = 0
    total_bytes_out = 0
    for _, ch in chapters.iterrows():
        chapter_id = str(ch["chapter_id"])
        text = ch.get("response_text") or ""
        if not text:
            continue
        ids = tok.encode(text)
        total_bytes_in += len(text)
        total_bytes_out += len(ids)
        target = val_rows if is_val(chapter_id, args.val_frac) else train_rows

        # Split into chunks of chunk_size; the last chunk is kept if ≥ min_chunk_size
        for i in range(0, len(ids), args.chunk_size):
            chunk = ids[i:i + args.chunk_size]
            if len(chunk) < args.min_chunk_size and i > 0:
                continue
            target.append({
                "table_id": f"{chapter_id}_{i // args.chunk_size:03d}",
                "byte_ids": chunk,
                "length": len(chunk),
            })

    logger.info(f"train: {len(train_rows)} rows, val: {len(val_rows)} rows")
    logger.info(f"total bytes in: {total_bytes_in:,}, encoded ids out: {total_bytes_out:,}")

    schema = pa.schema([
        ("table_id", pa.string()),
        ("byte_ids", pa.list_(pa.uint32())),
        ("length", pa.uint32()),
    ])

    for name, rows in [("train", train_rows), ("val", val_rows)]:
        if not rows:
            logger.warning(f"split {name!r} is empty — skipping")
            continue
        path = out_dir / f"{name}.parquet"
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table, path,
            compression="zstd",
            row_group_size=args.rg_size,
        )
        pf = pq.ParquetFile(str(path))
        logger.info(f"wrote {path} — rows={pf.metadata.num_rows} "
                    f"row_groups={pf.num_row_groups}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
