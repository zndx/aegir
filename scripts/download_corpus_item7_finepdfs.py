#!/usr/bin/env python3
"""Corpus item 7: FinePDFs lab/clinical/regulatory subset.

Streams shards of HuggingFaceFW/finepdfs (eng_Latn config), keeps
documents whose text matches our lab/clinical/regulatory keyword regex.
Stops when --max-bytes have been collected. Final --eval-bytes worth
of collected documents are written to a separate eval shard.

Provides a formal-prose track absent from FineWeb-Edu — SOPs, methods
sections, regulatory documents, in-vitro/in-vivo lab write-ups.
Vocabulary grounding for v3's LIMS-oriented synthetic-table generation.

Usage:
    uv run --no-sync python scripts/download_corpus_item7_finepdfs.py
    uv run --no-sync python scripts/download_corpus_item7_finepdfs.py --max-bytes 5_000_000_000
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

log = logging.getLogger("corpus_item7")

REPO = "HuggingFaceFW/finepdfs"
SHARD_PREFIX = "data/eng_Latn/"
DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1/finepdfs-lab")
DOC_DELIM = b"\x03"

LAB_VOCAB = re.compile(
    r"\b("
    r"HPLC|qPCR|PCR|ICP-MS|GC-MS|LC-MS|NMR|ELISA|FACS|RNA-seq|"
    r"assay|specimen|analyte|reagent|titration|chromatograph\w*|"
    r"spectrometr\w*|spectroscop\w*|nucleotide|oligonucleotide|"
    r"enzyme|substrate|catalyst|in vitro|in vivo|ex vivo|"
    r"cell line|cell culture|microbiome|metabolite|biomarker|"
    r"GLP|GMP|GCP|SOP|CFR|FDA|EMA|"
    r"mg/mL|µg/mL|ng/mL|µmol|nmol|pmol|"
    r"protocol\s+for|methodology|method\s+section"
    r")\b",
    re.IGNORECASE,
)

# Cheap pre-filter: any of these substrings present (case-insensitive)
# before we pay for the full regex.
PRE_HINTS = ("method", "assay", "specimen", "in vitro", "hplc", "gmp",
             "fda", "protocol", "qpcr", "elisa")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--max-bytes", type=int, default=20_000_000_000)
    ap.add_argument("--eval-bytes", type=int, default=512_000_000)
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Minimum keyword matches to accept a document.")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    train_path = args.dest / "finepdfs_lab_train.txt"
    eval_path = args.dest / "finepdfs_lab_eval.txt"

    if train_path.exists() and train_path.stat().st_size > 0:
        log.info("SKIP (exists): %s (%.2f MB)",
                 train_path.name, train_path.stat().st_size / 2**20)
        return 0

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    log.info("Listing %s shards under %s ...", REPO, SHARD_PREFIX)
    files = api.list_repo_files(REPO, repo_type="dataset")
    shards = sorted(f for f in files
                    if f.startswith(SHARD_PREFIX) and f.endswith(".parquet"))
    if not shards:
        log.error("No parquet shards found under %s", SHARD_PREFIX)
        return 1
    log.info("  found %d shards", len(shards))

    n_seen = 0
    n_kept = 0
    b_collected = 0
    train_threshold = args.max_bytes - args.eval_bytes

    with open(train_path, "wb") as ftrain, open(eval_path, "wb") as feval:
        for shard_rel in shards:
            if b_collected >= args.max_bytes:
                break
            log.info("  fetching %s ...", shard_rel)
            try:
                local = hf_hub_download(REPO, shard_rel, repo_type="dataset")
            except Exception as e:
                log.warning("    %s: %s", shard_rel, e)
                continue
            pf = pq.ParquetFile(local)
            cols = pf.schema_arrow.names
            text_col = "text" if "text" in cols else None
            if text_col is None:
                log.warning("    no 'text' column in %s", cols)
                continue

            for batch in pf.iter_batches(batch_size=1024, columns=[text_col]):
                for text in batch.column(text_col).to_pylist():
                    n_seen += 1
                    if not text:
                        continue
                    low = text.lower()
                    if not any(h in low for h in PRE_HINTS):
                        continue
                    if len(LAB_VOCAB.findall(text)) < args.min_hits:
                        continue
                    payload = text.encode("utf-8", errors="replace")
                    if not payload:
                        continue
                    target = ftrain if b_collected < train_threshold else feval
                    target.write(payload)
                    target.write(DOC_DELIM)
                    n_kept += 1
                    b_collected += len(payload) + 1
                    if b_collected >= args.max_bytes:
                        break
                if b_collected >= args.max_bytes:
                    break
                if n_seen % 5000 == 0 and n_seen > 0:
                    log.info("    seen=%d kept=%d (%.1f%%) collected=%.2f GB",
                             n_seen, n_kept, 100 * n_kept / max(n_seen, 1),
                             b_collected / 2**30)
            log.info("  shard done: seen=%d kept=%d collected=%.2f GB",
                     n_seen, n_kept, b_collected / 2**30)
            # Don't keep parquet shards on disk after consuming
            try:
                Path(local).unlink()
            except OSError:
                pass

    log.info("=" * 60)
    log.info("Final: seen=%d kept=%d collected=%.2f GB", n_seen, n_kept,
             b_collected / 2**30)
    log.info("train: %.2f MB → %s",
             train_path.stat().st_size / 2**20, train_path.name)
    log.info("eval:  %.2f MB → %s",
             eval_path.stat().st_size / 2**20, eval_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
