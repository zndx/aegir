#!/usr/bin/env python3
"""Corpus survey item 1: pull the small high-signal pieces.

Per docs/notes/2026-04-20/162800_corpus_survey_metadata_domain.md,
these are the cheap wins — small enough to fetch in minutes, concentrated
enough in our domain to justify inclusion by themselves.

Fetches:
  annotations/
    - spider.jsonl          (Spider text-to-SQL, ~10 k examples)
    - sql-create-context.jsonl  (schema + question + SQL, ~78 k)
  ontology/
    - schemaorg.jsonld       (all of schema.org, ~3 MB)
    - dbpedia.owl            (DBpedia ontology, ~2 MB)
    - bfo.owl                (BFO 2020 upper ontology)
    - cco.owl                (Common Core Ontology mid-level)

Idempotent: skips anything already present on disk.

Usage:
    uv run --no-sync python scripts/download_corpus_item1.py
    uv run --no-sync python scripts/download_corpus_item1.py --dest /raid/datasets/aegir-corpus-v1
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("corpus_item1")


DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1")

# Direct-download ontology URLs (stable, no auth).
# Verified 2026-04-20; see docs/notes/2026-04-20/162800 survey.
ONTOLOGY_URLS = {
    "schemaorg.jsonld":  "https://schema.org/version/latest/schemaorg-current-https.jsonld",
    # DBpedia ontology: served from the mappings server; the dbpedia.org
    # path returns HTML since the ontology file was moved.
    "dbpedia.owl":       "https://mappings.dbpedia.org/server/ontology/dbpedia.owl",
    # BFO 2020 (formally ISO/IEC 21838-2): bfo-core.owl under the
    # versioned standard path.
    "bfo.owl":           "https://raw.githubusercontent.com/BFO-ontology/BFO-2020/master/21838-2/owl/bfo-core.owl",
    # CCO: the merged all-core file — ~200 classes in one shot.
    "cco.ttl":           "https://raw.githubusercontent.com/CommonCoreOntology/CommonCoreOntologies/master/src/cco-merged/CommonCoreOntologiesMerged.ttl",
}


def fetch_url(url: str, dest: Path) -> int:
    """Download url to dest. Returns bytes written."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("  SKIP (exists): %s (%.2f MB)", dest.name, dest.stat().st_size / 2**20)
        return dest.stat().st_size
    log.info("  fetching %s ...", url)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aegir-corpus/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as e:
        log.error("  FAILED: %s (%s)", url, e)
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    log.info("  wrote %s (%.2f MB) in %.1fs", dest.name, len(data) / 2**20, time.time() - t0)
    return len(data)


def fetch_hf_dataset_as_jsonl(repo: str, out_path: Path, columns: list[str] | None = None,
                               max_rows: int | None = None) -> int:
    """Fetch a HuggingFace dataset and emit as JSONL with selected columns.

    Uses huggingface_hub to find and download the parquet shards directly,
    reads with pyarrow, writes JSONL. Keeps things minimal without needing
    the heavyweight datasets library.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("  SKIP (exists): %s (%.2f MB)", out_path.name, out_path.stat().st_size / 2**20)
        return out_path.stat().st_size

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    shards = [f for f in files if f.endswith(".parquet")]
    if not shards:
        # Some datasets ship JSON or CSV
        shards = [f for f in files if f.endswith((".json", ".jsonl", ".csv"))]
    if not shards:
        log.error("  no parquet/json shards in %s", repo)
        return 0
    log.info("  %s: %d shard files", repo, len(shards))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for shard_rel in shards:
            if shard_rel.endswith(".parquet"):
                log.info("  downloading %s ...", shard_rel)
                local = hf_hub_download(repo, shard_rel, repo_type="dataset")
                t = pq.read_table(local, columns=columns)
                for batch in t.to_batches():
                    for row in batch.to_pylist():
                        line = json.dumps(row, ensure_ascii=False)
                        out.write(line)
                        out.write("\n")
                        total_rows += 1
                        total_bytes += len(line) + 1
                        if max_rows and total_rows >= max_rows:
                            log.info("  reached max_rows=%d, stopping", max_rows)
                            return total_bytes
            elif shard_rel.endswith(".json"):
                # JSON array → one row per element, emitted as JSONL
                log.info("  downloading %s (JSON) ...", shard_rel)
                local = hf_hub_download(repo, shard_rel, repo_type="dataset")
                with open(local, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    log.error("  unexpected JSON shape (not array)")
                    continue
                for row in data:
                    line = json.dumps(row, ensure_ascii=False)
                    out.write(line)
                    out.write("\n")
                    total_rows += 1
                    total_bytes += len(line) + 1
                    if max_rows and total_rows >= max_rows:
                        return total_bytes
            elif shard_rel.endswith(".jsonl"):
                log.info("  downloading %s (JSONL) ...", shard_rel)
                local = hf_hub_download(repo, shard_rel, repo_type="dataset")
                with open(local, encoding="utf-8") as fh:
                    for line in fh:
                        out.write(line)
                        if not line.endswith("\n"):
                            out.write("\n")
                        total_rows += 1
                        total_bytes += len(line)
                        if max_rows and total_rows >= max_rows:
                            return total_bytes
            else:
                log.warning("  unknown format %s, skipping", shard_rel)
    log.info("  %s: wrote %d rows, %.2f MB", out_path.name, total_rows, total_bytes / 2**20)
    return total_bytes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    total = 0

    # --- ontology raw sources ---
    log.info("=" * 60)
    log.info("Ontology raw sources")
    log.info("=" * 60)
    for name, url in ONTOLOGY_URLS.items():
        total += fetch_url(url, args.dest / "ontology" / name)

    # --- HF SQL/NL annotation pairs ---
    log.info("=" * 60)
    log.info("HF annotation pairs")
    log.info("=" * 60)
    total += fetch_hf_dataset_as_jsonl(
        "xlangai/spider",
        args.dest / "annotations" / "spider.jsonl",
    )
    total += fetch_hf_dataset_as_jsonl(
        "b-mc2/sql-create-context",
        args.dest / "annotations" / "sql-create-context.jsonl",
    )

    log.info("=" * 60)
    log.info("Done. Total bytes written this run: %.2f MB", total / 2**20)
    # Summary of what's on disk
    for sub in ("ontology", "annotations"):
        d = args.dest / sub
        if d.exists():
            log.info("%s/:", sub)
            for f in sorted(d.iterdir()):
                log.info("  %-30s  %.2f MB", f.name, f.stat().st_size / 2**20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
