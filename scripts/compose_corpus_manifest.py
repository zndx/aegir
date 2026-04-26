#!/usr/bin/env python3
"""Compose MIXTURE.json + MIXTURE_eval.json for v2 corpus mix.

Two manifests:
- MIXTURE.json — training slices with normalised weights
- MIXTURE_eval.json — held-out slices for stratified per-slice bits/byte
  evaluation (no weights; eval loop iterates them sequentially)

v2 mix (weights sum to 1.0 after normalisation; before normalisation
the listed weights sum to 0.95 — the gap is the held-out budget):

    prose
      fineweb-edu        0.35   backbone
      finepdfs-lab       0.10   formal scientific prose (item 7)
    code
      schemapile         0.10   real-world DDL (item 8)
      sql                0.05   v1 NL+SQL pair scrapes (kept)
      docs               0.05   Apache project docs (item 3)
    ontology
      raw                0.03   OWL/JSONLD bytes
      prose              0.10   class definitions
    synthetic
      tables             0.07   Schema.org templated tables (item 5)
    sqale                0.10   execution-validated NL+DDL+SQL (item 6)

Usage:
    uv run --no-sync python scripts/compose_corpus_manifest.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("compose_corpus")

DEFAULT_ROOT = Path("/raid/datasets/aegir-corpus-v1")


# Per-slice target weight (relative — will be normalised). Empty slices
# drop out; the remaining weights renormalise automatically.
SLICES = [
    # (slice_name, glob relative to root, weight, description)
    ("prose.fineweb-edu",     "../fineweb-edu/*.txt",                       0.35, "FineWeb-Edu text shards (existing)"),
    ("prose.finepdfs-lab",    "finepdfs-lab/finepdfs_lab_train.txt",        0.10, "FinePDFs lab/clinical/regulatory subset (item 7)"),
    ("code.schemapile",       "schemapile/schemapile_train.txt",            0.10, "SchemaPile real-world DDL (item 8)"),
    ("code.sql",              "code/sql/*.txt",                             0.05, "v1 SQL NL/pair scrapes (kept)"),
    ("code.docs",             "code/docs/*.txt",                            0.05, "Apache data-systems project docs"),
    ("ontology.raw",          "ontology/*",                                 0.03, "OWL/JSONLD raw"),
    ("ontology.prose",        "ontology-prose/*.txt",                       0.10, "Ontology class definitions as prose"),
    ("synthetic.tables",      "synthetic/schemaorg-tables.txt",             0.07, "Templated Schema.org tables (item 5)"),
    ("sqale.nl-sql",          "sqale/sqale_train.txt",                      0.10, "SQaLe execution-validated NL+DDL+SQL (item 6)"),
]


# Held-out slices for stratified eval (no weights; eval loop reads
# sequentially from each).
EVAL_SLICES = [
    ("eval.fineweb-held",      "../fineweb-edu-eval/*.txt",                 "Held-out FineWeb-Edu shard"),
    ("eval.finepdfs-lab-held", "finepdfs-lab/finepdfs_lab_eval.txt",        "Held-out FinePDFs lab subset"),
    ("eval.schemapile-held",   "schemapile/schemapile_eval.txt",            "Held-out SchemaPile rows"),
    ("eval.sqale-held",        "sqale/sqale_eval.txt",                      "Held-out SQaLe rows (5%)"),
    ("eval.spider",            "annotations/*.jsonl",                       "Spider + sql-create-context (held-out, was train in v1)"),
]


def _discover(root: Path, glob: str) -> list[Path]:
    """Return resolved file paths matching the glob."""
    results: list[Path] = []
    for p in root.glob(glob):
        if p.is_file() and p.stat().st_size > 0:
            results.append(p.resolve())
    return sorted(results)


def _build_train_manifest(root: Path) -> dict:
    manifest = {"version": "v2", "root": str(root), "slices": []}
    total_bytes = 0
    total_weight = 0.0
    for name, glob_rel, weight, desc in SLICES:
        files = _discover(root, glob_rel)
        if not files:
            log.warning("  TRAIN slice %s: no files matching %s — SKIP",
                        name, glob_rel)
            continue
        size = sum(f.stat().st_size for f in files)
        manifest["slices"].append({
            "name": name,
            "weight": weight,
            "description": desc,
            "glob": glob_rel,
            "n_files": len(files),
            "bytes": size,
            "files": [str(f) for f in files],
        })
        total_bytes += size
        total_weight += weight
        log.info("  TRAIN %-24s  %d files  %8.2f MB  w=%.2f", name,
                 len(files), size / 2**20, weight)
    if total_weight > 0:
        for s in manifest["slices"]:
            s["weight_normalised"] = s["weight"] / total_weight
    manifest["total_bytes"] = total_bytes
    manifest["total_weight_raw"] = total_weight
    return manifest


def _build_eval_manifest(root: Path) -> dict:
    manifest = {"version": "v2", "root": str(root), "slices": []}
    total_bytes = 0
    for name, glob_rel, desc in EVAL_SLICES:
        files = _discover(root, glob_rel)
        if not files:
            log.warning("  EVAL slice %s: no files matching %s — SKIP",
                        name, glob_rel)
            continue
        size = sum(f.stat().st_size for f in files)
        manifest["slices"].append({
            "name": name,
            "description": desc,
            "glob": glob_rel,
            "n_files": len(files),
            "bytes": size,
            "files": [str(f) for f in files],
        })
        total_bytes += size
        log.info("  EVAL  %-24s  %d files  %8.2f MB", name, len(files),
                 size / 2**20)
    manifest["total_bytes"] = total_bytes
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_ROOT / "MIXTURE.json")
    ap.add_argument("--out-eval", type=Path,
                    default=DEFAULT_ROOT / "MIXTURE_eval.json")
    args = ap.parse_args()

    train = _build_train_manifest(args.root)
    log.info("=" * 60)
    log.info("TRAIN: %d slices, %.2f GB",
             len(train["slices"]), train["total_bytes"] / 2**30)

    log.info("=" * 60)
    eval_ = _build_eval_manifest(args.root)
    log.info("=" * 60)
    log.info("EVAL : %d slices, %.2f MB",
             len(eval_["slices"]), eval_["total_bytes"] / 2**20)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(train, indent=2))
    args.out_eval.write_text(json.dumps(eval_, indent=2))
    log.info("Train manifest: %s", args.out)
    log.info("Eval manifest:  %s", args.out_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
