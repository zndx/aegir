#!/usr/bin/env python3
"""Corpus item 2: Stack v2 dedup — SQL dialect slices.

Fetches the pure-SQL language slices from ``bigcode/the-stack-v2-dedup``:
SQL, PLpgSQL (Postgres), TSQL (SQL Server), PLSQL (Oracle), SQLPL (DB2),
HiveQL. Each is a single parquet shard.

Writes UTF-8 text shards to /raid/datasets/aegir-corpus-v1/code/sql/
with \\x03 as document separator (same convention as the FineWeb
downloader).

Idempotent: skips shards already on disk. Each output is ~0.1-1 GB;
total domain-specific SQL corpus ~1-3 GB uncompressed.

Usage:
    uv run --no-sync python scripts/download_corpus_item2.py
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

log = logging.getLogger("corpus_item2")

SQL_DIALECTS = ["SQL", "PLpgSQL", "TSQL", "PLSQL", "SQLPL", "HiveQL"]
REPO = "bigcode/the-stack-v2-dedup"
DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1/code/sql")
DOC_DELIM = b"\x03"

# Non-gated fallback datasets: when Stack v2 is inaccessible, pull
# SQL text from these sources. Each is a single parquet/json shard
# with a 'sql' column and usually extra question/context. We take
# only the SQL text for pretraining coverage.
FALLBACK_SQL_REPOS = [
    ("NumbersStation/NSText2SQL", ["sql"]),  # 290k SQL statements
    ("gretelai/synthetic_text_to_sql", ["sql"]),  # synthetic pairs
    ("Clinton/Text-to-sql-v1", ["response"]),  # many SQL responses
    ("knowrohit07/know_sql", ["Answer"]),  # LLM SQL Q&A
    ("kaxap/pg-wikiSQL-sql-instructions-80k", ["sql"]),  # Postgres wikiSQL
    ("lamini/spider_text_to_sql", ["output"]),  # spider-style
]


def _fetch_fallback(repo: str, sql_cols: list[str], out_path: Path) -> tuple[int, int]:
    """Pull SQL text from a non-gated alternate dataset."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    import json

    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("  SKIP (exists): %s (%.2f MB)", out_path.name,
                 out_path.stat().st_size / 2**20)
        return 0, out_path.stat().st_size

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    shards = [f for f in files if f.endswith(".parquet")]
    is_json = False
    if not shards:
        shards = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
        is_json = True
    if not shards:
        log.warning("  %s: no shards", repo)
        return 0, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_bytes = 0
    with open(out_path, "wb") as out:
        for shard_rel in shards:
            log.info("    fetching %s / %s ...", repo, shard_rel)
            try:
                local = hf_hub_download(repo, shard_rel, repo_type="dataset")
            except Exception as e:
                log.warning("    %s: %s", shard_rel, e)
                continue
            if is_json:
                # Try JSON array first, then JSONL
                try:
                    data = json.loads(Path(local).read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        data = [data]
                    rows = data
                except json.JSONDecodeError:
                    rows = []
                    with open(local, encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                table = pq.read_table(local)
                # Find a SQL-looking column. Use the provided sql_cols or
                # probe common names.
                available = table.column_names
                pick = next((c for c in sql_cols if c in available), None)
                if pick is None:
                    for c in ("sql", "query", "response", "output", "Answer", "answer"):
                        if c in available:
                            pick = c
                            break
                if pick is None:
                    log.warning("    %s: no sql column in %s", repo, available)
                    continue
                rows = [{"sql": str(v)} for v in table.column(pick).to_pylist() if v]

            for row in rows:
                # Take first string value that looks like SQL
                val = None
                for key in ("sql", "query", "response", "output", "Answer", "answer"):
                    if key in row and row[key]:
                        val = str(row[key])
                        break
                if val is None:
                    # Some JSONL may have {sql: ...} or be a plain string
                    continue
                b = val.encode("utf-8", errors="replace")
                if not b.strip():
                    continue
                out.write(b)
                out.write(DOC_DELIM)
                n_docs += 1
                n_bytes += len(b) + 1
    log.info("  %s: %d docs, %.2f MB", repo, n_docs, n_bytes / 2**20)
    return n_docs, n_bytes


def _fetch_dialect(dialect: str, dest: Path) -> tuple[int, int]:
    """Stream one dialect's parquet(s) to a single .txt shard.

    Returns (n_docs, n_bytes).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out_path = dest / f"{dialect.lower()}.txt"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("  SKIP (exists): %s (%.2f MB)",
                 out_path.name, out_path.stat().st_size / 2**20)
        return 0, out_path.stat().st_size

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    shards = [f for f in files if f.startswith(f"data/{dialect}/") and f.endswith(".parquet")]
    if not shards:
        log.warning("  %s: no shards found", dialect)
        return 0, 0
    log.info("  %s: %d shard(s)", dialect, len(shards))

    dest.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_bytes = 0
    with open(out_path, "wb") as out:
        for shard_rel in shards:
            log.info("    fetching %s ...", shard_rel)
            t0 = time.time()
            local = hf_hub_download(REPO, shard_rel, repo_type="dataset")
            log.info("    read %.2f MB in %.1fs", Path(local).stat().st_size / 2**20,
                     time.time() - t0)
            # Stack v2 parquet schema typically has 'content' column
            # with the source text. Read lazily to control memory.
            t0 = time.time()
            table = pq.read_table(local, columns=["content"])
            col = table.column("content")
            for chunk in col.chunks:
                for val in chunk.to_pylist():
                    if val is None:
                        continue
                    b = str(val).encode("utf-8", errors="replace")
                    if not b:
                        continue
                    out.write(b)
                    out.write(DOC_DELIM)
                    n_docs += 1
                    n_bytes += len(b) + 1
            log.info("    wrote %d docs in %.1fs", n_docs, time.time() - t0)
    log.info("  %s: total %d docs, %.2f MB", dialect, n_docs, n_bytes / 2**20)
    return n_docs, n_bytes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--dialects", nargs="*", default=SQL_DIALECTS)
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    log.info("Dest: %s", args.dest)
    log.info("Dialects: %s", args.dialects)

    total_docs = 0
    total_bytes = 0
    any_dialect_worked = False
    for d in args.dialects:
        try:
            n, b = _fetch_dialect(d, args.dest)
            if n > 0 or b > 0:
                any_dialect_worked = True
            total_docs += n
            total_bytes += b
        except Exception as e:
            log.warning("  %s failed: %s", d, e)
            if "Gated" in str(type(e).__name__) or "403" in str(e):
                log.warning("  Stack v2 is gated — falling back to non-gated sources")
                break

    if not any_dialect_worked:
        log.info("=" * 60)
        log.info("Stack v2 dedup gated; using non-gated SQL sources instead")
        log.info("=" * 60)
        fallback_dir = args.dest
        for repo, sql_cols in FALLBACK_SQL_REPOS:
            name = repo.split("/")[-1].replace("-", "_")
            try:
                n, b = _fetch_fallback(repo, sql_cols, fallback_dir / f"{name}.txt")
                total_docs += n
                total_bytes += b
            except Exception as e:
                log.warning("  %s failed: %s", repo, e)

    log.info("=" * 60)
    log.info("Done. %d docs, %.2f GB in %s.",
             total_docs, total_bytes / 2**30, args.dest)
    log.info("Files on disk:")
    for f in sorted(args.dest.glob("*.txt")):
        log.info("  %-20s  %.2f MB", f.name, f.stat().st_size / 2**20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
