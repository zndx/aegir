#!/usr/bin/env python3
"""Corpus item 6: SQaLe text-to-SQL dataset.

Pulls trl-lab/SQaLe-text-to-SQL-dataset (MIT, ~3.6 GB, 517,676 rows).
Each row is a (schema_ddl, question, query) triple where the SQL has
been execution-validated against the schema.

Renders rows to text shards using a CTA-friendly format:

    <schema_ddl>
    -- Q: <question>
    <query>
    \x03

5% held out (deterministic by seed) as `sqale_eval.txt`. Output goes to
/raid/datasets/aegir-corpus-v1/sqale/.

Usage:
    uv run --no-sync python scripts/download_corpus_item6_sqale.py
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

log = logging.getLogger("corpus_item6")

REPO = "trl-lab/SQaLe-text-to-SQL-dataset"
DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1/sqale")
DOC_DELIM = b"\x03"
EVAL_FRACTION = 0.05


def _format_row(schema, question, query) -> bytes:
    if not (schema and question and query):
        return b""
    text = f"{schema}\n-- Q: {question}\n{query}\n"
    return text.encode("utf-8", errors="replace")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--seed", type=int, default=4649)
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    train_path = args.dest / "sqale_train.txt"
    eval_path = args.dest / "sqale_eval.txt"

    if train_path.exists() and train_path.stat().st_size > 0:
        log.info("SKIP (exists): %s (%.2f MB)",
                 train_path.name, train_path.stat().st_size / 2**20)
        return 0

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        log.error("No parquet shards found in %s", REPO)
        return 1
    log.info("%s: %d shard(s)", REPO, len(shards))

    rng = random.Random(args.seed)
    n_train = b_train = n_eval = b_eval = 0

    with open(train_path, "wb") as ftrain, open(eval_path, "wb") as feval:
        for shard_rel in shards:
            log.info("  fetching %s ...", shard_rel)
            local = hf_hub_download(REPO, shard_rel, repo_type="dataset")
            pf = pq.ParquetFile(local)
            cols = pf.schema_arrow.names
            log.info("    columns=%s rows=%d", cols, pf.metadata.num_rows)

            schema_col = "schema" if "schema" in cols else cols[0]
            q_col = "question" if "question" in cols else None
            sql_col = "query" if "query" in cols else (
                "sql" if "sql" in cols else None
            )
            if q_col is None or sql_col is None:
                log.warning("    missing question/query columns, skipping shard")
                continue

            for batch in pf.iter_batches(batch_size=4096,
                                          columns=[schema_col, q_col, sql_col]):
                d = batch.to_pydict()
                for s, q, sql in zip(d[schema_col], d[q_col], d[sql_col]):
                    payload = _format_row(s, q, sql)
                    if not payload:
                        continue
                    if rng.random() < EVAL_FRACTION:
                        feval.write(payload)
                        feval.write(DOC_DELIM)
                        n_eval += 1
                        b_eval += len(payload) + 1
                    else:
                        ftrain.write(payload)
                        ftrain.write(DOC_DELIM)
                        n_train += 1
                        b_train += len(payload) + 1
            log.info("    cumulative train=%d eval=%d", n_train, n_eval)

    log.info("=" * 60)
    log.info("train: %d docs, %.2f MB → %s", n_train, b_train / 2**20, train_path.name)
    log.info("eval:  %d docs, %.2f MB → %s", n_eval, b_eval / 2**20, eval_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
