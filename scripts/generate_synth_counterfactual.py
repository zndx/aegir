#!/usr/bin/env python
"""Generate the Phase 0.5 counterfactual intermediate-pretrain corpus.

Reads ``interactions.txtpb.gz`` (TAPAS proto text format) and for each
table with surrounding text (TITLE / DESCRIPTION / SEGMENT_TITLE /
SEGMENT_TEXT questions), runs
``aegir.data.synth_counterfactual.generate_counterfactuals`` to emit
(table_bytes, text_bytes, label) parquet rows.

Output parquet schema::

    table_id:       string
    table_bytes:    list[uint32]    — serialized table with cell-boundary sentinels
    text_bytes:     list[uint32]    — surrounding text (original or perturbed)
    label:          bool            — True = corrupted, False = original
    swap_info:      string          — "" for label=False; "X→Y" for label=True

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/generate_synth_counterfactual.py \\
        --input /raid/datasets/tapas/interactions.txtpb.gz \\
        --output /raid/datasets/tapas/aegir_synth_counterfactual_v0.parquet \\
        --pairs-per-table 2 --seed 4649 \\
        --max-table-bytes 768 --max-text-bytes 256
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "protos"))

from aegir.data.tokenizer import (  # noqa: E402
    CELL_BOUNDARY_TOKEN_ID, _BYTE_OFFSET,
)
from aegir.data.synth_counterfactual import generate_counterfactuals  # noqa: E402
import interaction_pb2 as ip  # type: ignore[import-not-found]  # noqa: E402
from google.protobuf import text_format  # noqa: E402

logger = logging.getLogger("synth-cf")

CONTEXT_QUESTION_IDS = ("TITLE", "DESCRIPTION", "SEGMENT_TITLE", "SEGMENT_TEXT")


def _encode_bytes(text: str) -> list[int]:
    return [b + _BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]


def serialize_table_for_cf(interaction: ip.Interaction, max_bytes: int) -> list[int]:
    """Serialize just the table cells with cell-boundary sentinels."""
    table = interaction.table
    ids: list[int] = []
    for col in table.columns:
        ids.append(CELL_BOUNDARY_TOKEN_ID)
        ids.extend(_encode_bytes(col.text))
        if len(ids) >= max_bytes:
            return ids[:max_bytes]
    for row in table.rows:
        for cell in row.cells:
            ids.append(CELL_BOUNDARY_TOKEN_ID)
            ids.extend(_encode_bytes(cell.text))
            if len(ids) >= max_bytes:
                return ids[:max_bytes]
    return ids[:max_bytes]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs-per-table", type=int, default=2,
                   help="Max (original, perturbed) pairs per table")
    p.add_argument("--max-table-bytes", type=int, default=768)
    p.add_argument("--max-text-bytes", type=int, default=256)
    p.add_argument("--seed", type=int, default=4649)
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N input records (debug)")
    p.add_argument("--progress-interval", type=int, default=50000)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    schema = pa.schema([
        pa.field("table_id", pa.string()),
        pa.field("table_bytes", pa.list_(pa.uint32())),
        pa.field("text_bytes", pa.list_(pa.uint32())),
        pa.field("label", pa.bool_()),
        pa.field("swap_info", pa.string()),
    ])
    writer = pq.ParquetWriter(args.output, schema, compression="zstd")

    batch_size = 5000
    buf_tid: list[str] = []
    buf_tb: list[list[int]] = []
    buf_txt: list[list[int]] = []
    buf_lbl: list[bool] = []
    buf_swap: list[str] = []

    def _flush():
        if not buf_tid:
            return
        rb = pa.record_batch(
            [buf_tid, buf_tb, buf_txt, buf_lbl, buf_swap], schema=schema,
        )
        writer.write_batch(rb)
        buf_tid.clear()
        buf_tb.clear()
        buf_txt.clear()
        buf_lbl.clear()
        buf_swap.clear()

    interaction = ip.Interaction()
    rng = random.Random(args.seed)
    n_seen = 0
    n_tables_used = 0
    n_examples = 0
    n_true = 0
    t0 = time.time()

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
            table = interaction.table
            if len(table.columns) < 2 or len(table.rows) < 2:
                continue

            text_parts = [q.original_text for q in interaction.questions
                          if q.id in CONTEXT_QUESTION_IDS and q.original_text]
            if not text_parts:
                continue
            surrounding = " ".join(text_parts)

            col_names = [c.text for c in table.columns]
            rows = [[cell.text for cell in r.cells] for r in table.rows]

            exs = generate_counterfactuals(
                col_names, rows, surrounding,
                rng=rng, max_pairs=args.pairs_per_table,
            )
            if not exs:
                continue

            tbytes = serialize_table_for_cf(interaction, args.max_table_bytes)
            tid = table.table_id or interaction.id

            for ex in exs:
                text_bytes = _encode_bytes(ex.text)[:args.max_text_bytes]
                buf_tid.append(tid)
                buf_tb.append(tbytes)
                buf_txt.append(text_bytes)
                buf_lbl.append(ex.label)
                if ex.swap_info is None:
                    buf_swap.append("")
                else:
                    buf_swap.append(f"{ex.swap_info[0]} -> {ex.swap_info[1]}")
                n_examples += 1
                if ex.label:
                    n_true += 1

            n_tables_used += 1
            if len(buf_tid) >= batch_size:
                _flush()

            if n_seen % args.progress_interval == 0:
                elapsed = time.time() - t0
                rate = n_seen / max(elapsed, 1e-9)
                logger.info(
                    f"seen={n_seen:,} tables_used={n_tables_used:,} "
                    f"examples={n_examples:,} (T={n_true:,} F={n_examples - n_true:,}) "
                    f"rate={rate:.0f}/s"
                )
            if args.limit and n_seen >= args.limit:
                break

    _flush()
    writer.close()
    elapsed = time.time() - t0
    print(
        f"\ndone: seen={n_seen:,} tables_used={n_tables_used:,} "
        f"examples={n_examples:,} "
        f"(T={n_true:,} F={n_examples - n_true:,}) "
        f"in {elapsed:.1f}s"
    )
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
