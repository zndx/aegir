#!/usr/bin/env python3
"""Compose MIXTURE.json: enumerate corpus slices with sampling weights.

Produces a manifest that a streaming dataloader can consume to pick
one slice per micro-batch, weighted by the target mix proportion from
the survey note (docs/notes/2026-04-20/162800_corpus_survey_metadata_domain.md).

The manifest is informational — the streaming sampler in the training
script reads it and uses the weights to bias source selection. Actual
raw data files are not duplicated.

Default mix (what this script proposes) — weights sum to 1.0:

    prose           0.55    general language competence
      fineweb-edu   0.55    (we have 26 GB on disk)
    domain-code     0.15    SQL dialects + annotation pairs
      sql-text      0.10    from item 2 (SQL text only)
      docs          0.05    from item 3 (Apache project docs)
    ontology        0.15    schema/class vocabulary
      raw           0.03    OWLs (for byte-level pattern exposure)
      prose         0.12    generated descriptions
    synthetic       0.10    ontology-grounded tables with known GT
    annotations     0.05    Spider + sql-create-context

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
    ("prose.fineweb-edu",     "../fineweb-edu/*.txt",                      0.55, "FineWeb-Edu text shards (existing)"),
    ("code.sql",              "code/sql/*.txt",                             0.10, "SQL statements from item 2"),
    ("code.docs",             "code/docs/*.txt",                            0.05, "Apache data-systems project docs"),
    ("ontology.raw",          "ontology/*",                                 0.03, "OWL/JSONLD raw"),
    ("ontology.prose",        "ontology-prose/*.txt",                       0.12, "Ontology class definitions as prose"),
    ("synthetic.tables",      "synthetic/schemaorg-tables.txt",             0.10, "Templated Schema.org tables"),
    ("annotations",           "annotations/*.jsonl",                        0.05, "Spider + sql-create-context"),
]


def _discover(root: Path, glob: str) -> list[Path]:
    """Return resolved file paths matching the glob."""
    results: list[Path] = []
    for p in root.glob(glob):
        if p.is_file() and p.stat().st_size > 0:
            results.append(p.resolve())
    return sorted(results)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--out", type=Path,
        default=DEFAULT_ROOT / "MIXTURE.json",
    )
    args = ap.parse_args()

    manifest = {
        "version": "v1",
        "root": str(args.root),
        "slices": [],
    }

    total_bytes = 0
    total_weight = 0.0
    for name, glob_rel, weight, desc in SLICES:
        files = _discover(args.root, glob_rel)
        if not files:
            log.warning("  slice %s: no files matching %s — SKIP", name, glob_rel)
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
        log.info("  %-22s  %d files  %8.2f MB  w=%.2f", name, len(files),
                 size / 2**20, weight)

    # Normalise weights
    if total_weight > 0:
        for s in manifest["slices"]:
            s["weight_normalised"] = s["weight"] / total_weight

    manifest["total_bytes"] = total_bytes
    manifest["total_weight_raw"] = total_weight

    log.info("=" * 60)
    log.info("Total: %d slices, %.2f GB", len(manifest["slices"]), total_bytes / 2**30)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
