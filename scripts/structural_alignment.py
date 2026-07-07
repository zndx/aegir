#!/usr/bin/env python
"""Structural alignment proxy: do generated tables produce columns
whose names align with downstream-task label vocabularies?

The ontology→corpus→downstream chain is hard to evaluate at small
scale. This script computes a CHEAP proxy that doesn't require any
model training: for each ablation arm, parse all generated chapters'
markdown tables, extract column headers, and measure their overlap
with downstream task label vocabularies.

For SOTAB-CTA (Schema.org Column Type Annotation, 82 labels), this
asks: would the corpus naturally produce training rows useful for
predicting the 82 schema.org column types?

For ontology-IRI vocabulary, it asks: do the column names actually
reference ontology slot types?

The intuition: even if no-ontology chapters have better topic
correspondence to FinePDFs, they may produce tables with arbitrary
column names that don't align with the downstream task we care
about. The ontology-grounded chapters may "drift" in topic space
but produce tables that are structurally closer to what SOTAB-CTA
or a similar downstream task expects.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/structural_alignment.py \\
        --chapters-runs <arm1> <arm2> <arm3> \\
        --catalog-dir src/aegir/ontology/catalog
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("structural-alignment")


# SOTAB-CTA Schema.org labels (82 classes); these are the column types
# the downstream task asks the model to predict. We want chapters that
# produce columns naming these (or close synonyms).
SOTAB_CTA_SCHEMAORG = {
    # Top-level
    "thing", "person", "organization", "place", "event", "creativework",
    "product", "offer", "review", "rating",
    # Person
    "givenname", "familyname", "name", "email", "telephone", "address",
    "jobtitle", "worksfor", "birthdate",
    # Address/Place
    "streetaddress", "postalcode", "addresslocality", "addressregion",
    "addresscountry", "geocoordinates", "latitude", "longitude",
    # Product
    "sku", "brand", "manufacturer", "model", "category", "color", "weight",
    "price", "priceurrency", "availability",
    # Event
    "startdate", "enddate", "location", "duration", "organizer",
    # Date/Time / Numeric
    "datepublished", "datecreated", "datemodified", "datetime", "date",
    "duration", "quantity", "number", "integer",
    # CreativeWork
    "author", "publisher", "headline", "articlebody", "keywords", "genre",
    "isbn", "issn", "language", "license",
    # Offer / commerce
    "currency", "discount", "shippingcost", "validfrom", "validthrough",
    # Review
    "reviewbody", "ratingvalue", "bestrating", "worstrating", "reviewcount",
    # Misc Schema.org common
    "description", "url", "image", "identifier", "alternatename",
    "additionalname", "title", "subtitle",
}


def load_ontology_slot_vocab(catalog_dir: Path) -> set[str]:
    """Extract slot-type vocabulary across all canonical catalog files."""
    vocab: set[str] = set()
    from aegir.ontology.schema import catalog_files
    for p in catalog_files(catalog_dir):
        data = json.loads(p.read_text())
        for t in data.get("templates", []):
            for slot_type in (t.get("slot_types") or {}).values():
                for w in re.split(r"[_\W]+", slot_type.lower()):
                    if w and len(w) > 2:
                        vocab.add(w)
            tid = t.get("template_id", "").lower()
            for w in re.split(r"[_\W]+", tid):
                if w and len(w) > 2:
                    vocab.add(w)
    return vocab


_TABLE_ROW_RE = re.compile(r"^\s*\|[^\n]*\|\s*$")
_TABLE_HEADER_SEP_RE = re.compile(r"^\s*\|[\s|:\-]+\|\s*$")


def parse_chapter_tables(text: str) -> list[list[str]]:
    """Return list of header rows from each markdown table in the chapter."""
    headers = []
    lines = (text or "").split("\n")
    i = 0
    while i < len(lines):
        if (_TABLE_ROW_RE.match(lines[i])
                and i + 1 < len(lines)
                and _TABLE_HEADER_SEP_RE.match(lines[i + 1])):
            cols = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            headers.append(cols)
            i += 2
        else:
            i += 1
    return headers


def normalize_column(col: str) -> list[str]:
    """Return lowercase words from a column header for vocab matching."""
    return [w for w in re.findall(r"[A-Za-z]+", col.lower()) if len(w) > 2]


def overlap_score(col: str, vocab: set[str]) -> float:
    """Fraction of column header words that appear in vocab.

    Generous matching: substring either direction. column 'StreetAddress'
    matches 'streetaddress' (exact), 'street' (substring), 'addresscountry'
    (substring in reverse).
    """
    words = normalize_column(col)
    if not words:
        return 0.0
    matched = 0
    for w in words:
        for v in vocab:
            if w in v or v in w:
                matched += 1
                break
    return matched / len(words)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-runs", required=True, nargs="+")
    p.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Per-column overlap threshold for 'aligned'")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    catalog_dir = REPO / args.catalog_dir
    ontology_vocab = load_ontology_slot_vocab(catalog_dir)
    sotab_vocab = SOTAB_CTA_SCHEMAORG
    logger.info(f"ontology vocab: {len(ontology_vocab)} terms")
    logger.info(f"sotab-cta vocab: {len(sotab_vocab)} terms")

    arm_stats: dict[str, dict] = defaultdict(lambda: {
        "n_chapters": 0, "n_tables": 0, "n_columns": 0,
        "sotab_aligned_cols": 0, "ontology_aligned_cols": 0,
        "all_columns": Counter(),
        "sotab_overlaps": [], "ontology_overlaps": [],
    })

    for run_dir in args.chapters_runs:
        run_dir = Path(run_dir)
        chapters = pq.read_table(run_dir / "chapters.parquet").to_pandas()
        ablation = chapters["ablation"].iloc[0] if "ablation" in chapters.columns else "full"

        for _, ch in chapters.iterrows():
            arm_stats[ablation]["n_chapters"] += 1
            headers = parse_chapter_tables(ch.get("response_text", "") or "")
            for hdr in headers:
                arm_stats[ablation]["n_tables"] += 1
                for col in hdr:
                    arm_stats[ablation]["n_columns"] += 1
                    arm_stats[ablation]["all_columns"][col.lower()] += 1
                    s_ov = overlap_score(col, sotab_vocab)
                    o_ov = overlap_score(col, ontology_vocab)
                    arm_stats[ablation]["sotab_overlaps"].append(s_ov)
                    arm_stats[ablation]["ontology_overlaps"].append(o_ov)
                    if s_ov >= args.threshold:
                        arm_stats[ablation]["sotab_aligned_cols"] += 1
                    if o_ov >= args.threshold:
                        arm_stats[ablation]["ontology_aligned_cols"] += 1

    # Print scorecard
    print()
    print(f"{'arm':<14} {'chap':>5} {'tbl':>4} {'cols':>5} "
          f"{'sotab_aln%':>11} {'mean_sotab':>11} "
          f"{'onto_aln%':>10} {'mean_onto':>10}")
    print("─" * 90)
    for arm in sorted(arm_stats):
        s = arm_stats[arm]
        n_cols = s["n_columns"]
        sotab_pct = 100 * s["sotab_aligned_cols"] / max(n_cols, 1)
        onto_pct = 100 * s["ontology_aligned_cols"] / max(n_cols, 1)
        sotab_mean = float(np.mean(s["sotab_overlaps"])) if s["sotab_overlaps"] else 0
        onto_mean = float(np.mean(s["ontology_overlaps"])) if s["ontology_overlaps"] else 0
        print(f"{arm:<14} {s['n_chapters']:>5d} {s['n_tables']:>4d} "
              f"{n_cols:>5d} "
              f"{sotab_pct:>10.1f}% {sotab_mean:>11.3f} "
              f"{onto_pct:>9.1f}% {onto_mean:>10.3f}")

    print()
    print("Top 12 column headers per arm:")
    for arm in sorted(arm_stats):
        s = arm_stats[arm]
        print(f"\n  {arm}:")
        for col, count in s["all_columns"].most_common(12):
            sotab_hit = "S" if overlap_score(col, sotab_vocab) >= args.threshold else "."
            onto_hit = "O" if overlap_score(col, ontology_vocab) >= args.threshold else "."
            print(f"    [{sotab_hit}{onto_hit}] {count:3d}× {col}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
