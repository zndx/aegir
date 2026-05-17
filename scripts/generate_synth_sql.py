#!/usr/bin/env python
"""Generate the Phase 0.5 synthetic-SQL intermediate-pretrain corpus.

Reads ``interactions.txtpb.gz`` (TAPAS proto text format), filters tables
suitable for SQL-grounded statement generation (>=2 cols, >=2 rows, at
least one numeric column), and for each table emits up to ``--per-table``
synthetic statements via ``aegir.data.synth_table_qa.generate_examples``.

Output parquet schema::

    table_id:       string
    table_bytes:    list[uint32]    — serialized table (no statement),
                                       with cell-boundary sentinels
    statement_bytes:list[uint32]    — natural-language statement bytes
    label:          bool            — truth value
    pattern:        uint8           — for diagnostics (0=A, 1=B, 2=C)

The trainer chooses how to combine table+statement at load time
(prefix/suffix concat for LM, or two-segment input for classification).

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/generate_synth_sql.py \\
        --input /raid/datasets/tapas/interactions.txtpb.gz \\
        --output /raid/datasets/tapas/aegir_synth_sql_v0.parquet \\
        --per-table 4 --seed 4649 \\
        --max-table-bytes 768 --max-statement-bytes 192
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
    EOS_TOKEN_ID, SEP_TOKEN_ID,
    CELL_BOUNDARY_TOKEN_ID, _BYTE_OFFSET,
)
from aegir.data.synth_table_qa import generate_examples  # noqa: E402
import interaction_pb2 as ip  # type: ignore[import-not-found]  # noqa: E402
from google.protobuf import text_format  # noqa: E402

logger = logging.getLogger("synth-sql")


def _encode_bytes(text: str) -> list[int]:
    return [b + _BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]


def serialize_table_for_synth(interaction: ip.Interaction, max_bytes: int) -> list[int]:
    """Serialize just the table (no surrounding text, no BOS/EOS).

    The statement gets concatenated downstream; this function only emits
    the structural cells. Caller wraps in BOS/SEP/EOS as needed.
    """
    table = interaction.table
    ids: list[int] = []
    # Header row.
    for col in table.columns:
        ids.append(CELL_BOUNDARY_TOKEN_ID)
        ids.extend(_encode_bytes(col.text))
        if len(ids) >= max_bytes:
            return ids[:max_bytes]
    # Data rows.
    for row in table.rows:
        for cell in row.cells:
            ids.append(CELL_BOUNDARY_TOKEN_ID)
            ids.extend(_encode_bytes(cell.text))
            if len(ids) >= max_bytes:
                return ids[:max_bytes]
    return ids[:max_bytes]


def encode_statement(text: str, max_bytes: int) -> list[int]:
    """Encode a natural-language statement with BOS prefix + EOS suffix."""
    body = _encode_bytes(text)[: max_bytes - 2]
    return [SEP_TOKEN_ID] + body + [EOS_TOKEN_ID]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-table", type=int, default=4,
                   help="Max synth statements per table")
    p.add_argument("--max-table-bytes", type=int, default=768)
    p.add_argument("--max-statement-bytes", type=int, default=192)
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
        pa.field("statement_bytes", pa.list_(pa.uint32())),
        pa.field("label", pa.bool_()),
        pa.field("pattern_text", pa.string()),
    ])
    writer = pq.ParquetWriter(args.output, schema, compression="zstd")

    batch_size = 5000
    buf_table_id: list[str] = []
    buf_table_bytes: list[list[int]] = []
    buf_stmt_bytes: list[list[int]] = []
    buf_label: list[bool] = []
    buf_pattern: list[str] = []

    def _flush():
        if not buf_table_id:
            return
        rb = pa.record_batch(
            [buf_table_id, buf_table_bytes, buf_stmt_bytes, buf_label, buf_pattern],
            schema=schema,
        )
        writer.write_batch(rb)
        buf_table_id.clear()
        buf_table_bytes.clear()
        buf_stmt_bytes.clear()
        buf_label.clear()
        buf_pattern.clear()

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

            # Extract structured columns/rows for the generator.
            col_names = [c.text for c in table.columns]
            rows = [[cell.text for cell in r.cells] for r in table.rows]

            examples = generate_examples(
                col_names, rows,
                n_max=args.per_table, rng=rng, label_balance=True,
            )
            if not examples:
                continue

            table_bytes = serialize_table_for_synth(interaction, args.max_table_bytes)
            tid = table.table_id or interaction.id

            for ex in examples:
                stmt_bytes = encode_statement(ex.statement, args.max_statement_bytes)
                buf_table_id.append(tid)
                buf_table_bytes.append(table_bytes)
                buf_stmt_bytes.append(stmt_bytes)
                buf_label.append(ex.label)
                buf_pattern.append(
                    # Cheap pattern tagging via heuristic; not load-bearing.
                    "C" if " greater than the " in ex.statement
                          or " less than the " in ex.statement
                          or " equal to the " in ex.statement
                    else ("A" if ex.statement.startswith("the ") else "B")
                )
                n_examples += 1
                if ex.label:
                    n_true += 1

            n_tables_used += 1
            if len(buf_table_id) >= batch_size:
                _flush()

            if n_seen % args.progress_interval == 0:
                elapsed = time.time() - t0
                rate = n_seen / max(elapsed, 1e-9)
                tn_rate = n_examples / max(n_tables_used, 1)
                logger.info(
                    f"seen={n_seen:,} tables_used={n_tables_used:,} "
                    f"examples={n_examples:,} (T/F={n_true:,}/{n_examples - n_true:,}) "
                    f"examples_per_table={tn_rate:.2f} input_rate={rate:.0f}/s"
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
