#!/usr/bin/env python3
"""Corpus item 8: SchemaPile DDL renderer.

Pivot from OpenCoder (which only publishes Stack v2 metadata, not raw
content) to SchemaPile — the actual source SQaLe was built from.
22,989 real-world SQL schemas scraped from public GitHub repos with
permissive licenses, parsed into structured TABLES with column types,
constraints, sample values, and foreign-key relationships.

Each row is rendered as:

    -- {URL}
    -- License: {LICENSE}

    CREATE TABLE {table_name} (
        {col_name} {col_type} [PRIMARY KEY] [NOT NULL] [UNIQUE] [DEFAULT ...]
        [CHECK (...)],
        ...
    );

    -- Sample values:
    -- {col_name}: val1, val2, val3
    -- ...

    [next table ...]
    \\x03

Gives the model exposure to real DDL syntax variation, sample value
distributions, and FK structure — complementary to SQaLe's
LLM-extended schemas and our v1 NL+SQL pair scrapes.

Usage:
    uv run --no-sync python scripts/download_corpus_item8_opencoder_sql.py
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

log = logging.getLogger("corpus_item8")

REPO = "trl-lab/schemapile"
DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1/schemapile")
DOC_DELIM = b"\x03"
EVAL_FRACTION = 0.05
MAX_VALUES_PER_COL = 5  # cap sample-value listing


def _render_column(c: dict) -> str:
    name = c.get("NAME", "")
    typ = c.get("TYPE", "")
    parts = [f"  {name} {typ}"]
    if c.get("IS_PRIMARY"):
        parts.append("PRIMARY KEY")
    if not c.get("NULLABLE", True):
        parts.append("NOT NULL")
    if c.get("UNIQUE"):
        parts.append("UNIQUE")
    default = c.get("DEFAULT")
    if default is not None:
        parts.append(f"DEFAULT {default}")
    checks = c.get("CHECKS") or []
    for chk in checks:
        if chk:
            parts.append(f"CHECK ({chk})")
    return " ".join(parts)


def _render_table(t: dict) -> str:
    name = t.get("TABLE_NAME", "unknown")
    cols = t.get("COLUMNS") or []
    if not cols:
        return ""
    col_lines = [_render_column(c) for c in cols if c.get("NAME")]
    body = ",\n".join(col_lines)
    out = [f"CREATE TABLE {name} (\n{body}\n);"]
    # Append sample values per column
    sample_lines = []
    for c in cols:
        cname = c.get("NAME")
        vals = c.get("VALUES") or []
        if not (cname and vals):
            continue
        cap = vals[:MAX_VALUES_PER_COL]
        sample_lines.append(f"-- {cname}: " + ", ".join(str(v) for v in cap))
    if sample_lines:
        out.append("-- Sample values:")
        out.extend(sample_lines)
    return "\n".join(out)


def _render_row(row: dict) -> bytes:
    url = row.get("URL") or ""
    license_ = row.get("LICENSE") or "unknown"
    tables = row.get("TABLES") or []
    if not tables:
        return b""
    parts = [f"-- {url.strip()}", f"-- License: {license_}", ""]
    rendered_any = False
    for t in tables:
        rendered = _render_table(t)
        if rendered:
            parts.append(rendered)
            parts.append("")
            rendered_any = True
    if not rendered_any:
        return b""
    return ("\n".join(parts)).encode("utf-8", errors="replace")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--seed", type=int, default=4649)
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    train_path = args.dest / "schemapile_train.txt"
    eval_path = args.dest / "schemapile_eval.txt"

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
        log.error("No parquet shards in %s", REPO)
        return 1
    log.info("%s: %d shard(s)", REPO, len(shards))

    rng = random.Random(args.seed)
    n_train = b_train = n_eval = b_eval = 0

    with open(train_path, "wb") as ftrain, open(eval_path, "wb") as feval:
        for shard_rel in shards:
            log.info("  fetching %s ...", shard_rel)
            local = hf_hub_download(REPO, shard_rel, repo_type="dataset")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(
                batch_size=512,
                columns=["URL", "LICENSE", "TABLES"],
            ):
                d = batch.to_pydict()
                for url, lic, tables in zip(d["URL"], d["LICENSE"], d["TABLES"]):
                    payload = _render_row({"URL": url, "LICENSE": lic, "TABLES": tables})
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
            log.info("    cumulative train=%d (%.1f MB) eval=%d (%.1f MB)",
                     n_train, b_train / 2**20, n_eval, b_eval / 2**20)

    log.info("=" * 60)
    log.info("train: %d schemas, %.2f MB → %s",
             n_train, b_train / 2**20, train_path.name)
    log.info("eval:  %d schemas, %.2f MB → %s",
             n_eval, b_eval / 2**20, eval_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
