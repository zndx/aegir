#!/usr/bin/env python
"""TAPAS interaction.proto → Aegir byte-sequence parquet converter.

Reads ``interactions.txtpb.gz`` (proto text format, one record per line)
from TAPAS's Wikipedia table-text dump and emits a parquet file with one
row per accepted table, containing a byte-level serialization suitable
for Aegir pretraining.

Serialization layout (with explicit cell-boundary sentinel)::

    [BOS] <surrounding text> [SEP]
      [BOUNDARY] col1 [BOUNDARY] col2 ... [BOUNDARY] colN
      [BOUNDARY] row1cell1 [BOUNDARY] row1cell2 ... [BOUNDARY] row1cellN
      [BOUNDARY] row2cell1 ...
    [EOS]

Where ``[BOS]`` / ``[SEP]`` / ``[BOUNDARY]`` / ``[EOS]`` are the special
IDs defined in ``aegir.data.tokenizer``.

The surrounding text concatenates the TITLE / DESCRIPTION /
SEGMENT_TITLE / SEGMENT_TEXT ``questions`` entries that TAPAS uses to
encode the page context around each table.

Filters:
- Drop tables with fewer than 2 columns or 2 data rows.
- Drop sequences shorter than ``--min-length`` (after serialization).

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/tapas_proto_to_aegir.py \\
        --input /raid/datasets/tapas/interactions.txtpb.gz \\
        --output /raid/datasets/tapas/aegir_bytes_v0.parquet \\
        --max-length 1024 \\
        --batch-size 10000
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "protos"))

from aegir.data.tokenizer import (  # noqa: E402
    BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
    CELL_BOUNDARY_TOKEN_ID, _BYTE_OFFSET,
)
import interaction_pb2 as ip  # type: ignore[import-not-found]  # noqa: E402
from google.protobuf import text_format  # noqa: E402

logger = logging.getLogger("tapas-convert")

CONTEXT_QUESTION_IDS = ("TITLE", "DESCRIPTION", "SEGMENT_TITLE", "SEGMENT_TEXT")


def _encode_bytes(text: str) -> list[int]:
    """UTF-8 encode + apply tokenizer's _BYTE_OFFSET."""
    return [b + _BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]


def serialize_for_pretrain(interaction: ip.Interaction, max_length: int) -> list[int]:
    """Serialize one Interaction into a byte-id sequence.

    Layout::
        [BOS] context_text [SEP]
        [BOUNDARY] col1 [BOUNDARY] col2 ...
        [BOUNDARY] r1c1 [BOUNDARY] r1c2 ...
        ...
        [EOS]

    Truncates at ``max_length`` if needed (always preserving [EOS]).
    Returns the id sequence; caller filters/drops if too short.
    """
    table = interaction.table
    if len(table.columns) < 2 or len(table.rows) < 2:
        return []

    # Surrounding text from questions.
    context_parts: list[str] = []
    for q in interaction.questions:
        if q.id in CONTEXT_QUESTION_IDS and q.original_text:
            context_parts.append(q.original_text)
    context_text = " | ".join(context_parts)

    ids: list[int] = [BOS_TOKEN_ID]
    ids.extend(_encode_bytes(context_text))
    ids.append(SEP_TOKEN_ID)

    # Header row.
    for col in table.columns:
        ids.append(CELL_BOUNDARY_TOKEN_ID)
        ids.extend(_encode_bytes(col.text))
        if len(ids) >= max_length - 1:
            break

    # Data rows.
    if len(ids) < max_length - 1:
        for row in table.rows:
            for cell in row.cells:
                ids.append(CELL_BOUNDARY_TOKEN_ID)
                ids.extend(_encode_bytes(cell.text))
                if len(ids) >= max_length - 1:
                    break
            if len(ids) >= max_length - 1:
                break

    ids = ids[:max_length - 1]
    ids.append(EOS_TOKEN_ID)
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Path to interactions.txtpb.gz")
    p.add_argument("--output", required=True,
                   help="Path to output parquet")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Max byte-id sequence length (default 1024)")
    p.add_argument("--min-length", type=int, default=64,
                   help="Drop sequences shorter than this (default 64)")
    p.add_argument("--batch-size", type=int, default=10000,
                   help="Records per parquet row-group")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, stop after this many input records (debug)")
    p.add_argument("--progress-interval", type=int, default=50000)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    schema = pa.schema([
        pa.field("table_id", pa.string()),
        pa.field("byte_ids", pa.list_(pa.uint32())),
        pa.field("length", pa.uint32()),
    ])

    writer = pq.ParquetWriter(args.output, schema, compression="zstd")

    n_seen = 0
    n_kept = 0
    n_dropped_small_table = 0
    n_dropped_short_seq = 0
    batch_table_ids: list[str] = []
    batch_byte_ids: list[list[int]] = []
    batch_lengths: list[int] = []
    t0 = time.time()

    def _flush():
        if not batch_table_ids:
            return
        rb = pa.record_batch(
            [batch_table_ids, batch_byte_ids, batch_lengths],
            schema=schema,
        )
        writer.write_batch(rb)
        batch_table_ids.clear()
        batch_byte_ids.clear()
        batch_lengths.clear()

    interaction = ip.Interaction()
    with gzip.open(args.input, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_seen += 1
            line = line.strip()
            if not line:
                continue

            interaction.Clear()
            try:
                text_format.Parse(line, interaction)
            except text_format.ParseError:
                continue

            ids = serialize_for_pretrain(interaction, args.max_length)
            if not ids:
                n_dropped_small_table += 1
            elif len(ids) < args.min_length:
                n_dropped_short_seq += 1
            else:
                batch_table_ids.append(interaction.table.table_id or interaction.id)
                batch_byte_ids.append(ids)
                batch_lengths.append(len(ids))
                n_kept += 1
                if len(batch_table_ids) >= args.batch_size:
                    _flush()

            if n_seen % args.progress_interval == 0:
                elapsed = time.time() - t0
                rate = n_seen / max(elapsed, 1e-9)
                logger.info(
                    f"seen={n_seen:,} kept={n_kept:,} "
                    f"small={n_dropped_small_table:,} "
                    f"short={n_dropped_short_seq:,} "
                    f"rate={rate:,.0f}/s"
                )

            if args.limit and n_seen >= args.limit:
                break

    _flush()
    writer.close()
    elapsed = time.time() - t0
    print(f"\ndone: seen={n_seen:,} kept={n_kept:,} "
          f"(small={n_dropped_small_table:,}, "
          f"short={n_dropped_short_seq:,}) in {elapsed:.1f}s")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
