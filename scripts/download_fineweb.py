#!/usr/bin/env python3
"""Stream-download FineWeb-Edu shards to /raid/datasets/fineweb-edu/.

Fetches parquet shards from HuggingFaceFW/fineweb-edu (the sample-10BT
subset by default) via ``huggingface_hub``, reads the ``text`` column
from each, and writes a flat UTF-8 text file per shard with documents
separated by a FineWeb-Edu document delimiter byte (``\\x03``).

Byte-level training on raw text is our target — these .txt files can be
read by a simple IterableDataset without the datasets library.

Idempotent: skips shards already on disk. Each output file is ~1 GB of
text. Default budget is 8 shards ≈ 8 GB.

Usage:
    uv run --no-sync python scripts/download_fineweb.py
    uv run --no-sync python scripts/download_fineweb.py --num-shards 5 --subset sample/10BT
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("download_fineweb")

REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_DEST = Path("/raid/datasets/fineweb-edu")
DOC_DELIM = b"\x03"  # byte separating documents in the output stream


def _list_parquet_files(repo_id: str, subset: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    pq = [f for f in all_files if f.startswith(f"{subset}/") and f.endswith(".parquet")]
    return sorted(pq)


def _fetch_and_convert(repo_id: str, parquet_rel: str, dest: Path) -> tuple[int, int]:
    """Download one parquet, stream its rows as UTF-8 text into ``dest``.

    Returns (num_docs, num_bytes_written).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    log.info("  fetching %s ...", parquet_rel)
    t0 = time.time()
    local = hf_hub_download(repo_id, parquet_rel, repo_type="dataset")
    log.info("  downloaded in %.1fs -> %s", time.time() - t0, local)

    log.info("  reading + writing text to %s ...", dest)
    t0 = time.time()
    table = pq.read_table(local, columns=["text"])
    col = table.column("text")
    n_docs = 0
    n_bytes = 0
    # Iterate chunks of the ChunkedArray; each chunk is a pa.Array which
    # we materialize to Python str list (RAM-efficient per chunk).
    with open(dest, "wb") as out:
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
    log.info("  wrote %d docs, %.2f MB in %.1fs", n_docs, n_bytes / 2**20, time.time() - t0)
    return n_docs, n_bytes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--subset", default="sample/10BT", help="HF path prefix")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip the first N shards (for resuming partial runs)")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    log.info("Dest: %s", args.dest)
    log.info("Listing parquet files under %s/%s ...", REPO_ID, args.subset)
    all_shards = _list_parquet_files(REPO_ID, args.subset)
    log.info("  found %d total shards", len(all_shards))
    if not all_shards:
        log.error("No shards matched subset %s", args.subset)
        return 1

    picked = all_shards[args.skip : args.skip + args.num_shards]
    log.info("Fetching %d shards (skip=%d)", len(picked), args.skip)

    total_docs = 0
    total_bytes = 0
    for i, rel in enumerate(picked, 1):
        # Save each shard with a readable name derived from the parquet path
        name = rel.replace("/", "_").replace(".parquet", ".txt")
        dest_file = args.dest / name
        if dest_file.exists() and dest_file.stat().st_size > 0:
            log.info("[%d/%d] SKIP (exists): %s (%.1f MB)",
                     i, len(picked), name, dest_file.stat().st_size / 2**20)
            total_bytes += dest_file.stat().st_size
            continue
        log.info("[%d/%d] %s", i, len(picked), rel)
        n_docs, n_bytes = _fetch_and_convert(REPO_ID, rel, dest_file)
        total_docs += n_docs
        total_bytes += n_bytes

    log.info("Done. %d docs, %.2f GB across %d shards.",
             total_docs, total_bytes / 2**30, len(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
