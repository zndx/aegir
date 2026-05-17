#!/usr/bin/env python
"""Verifier loop for ontology-grounded chapters.

Scores each generated chapter against four criteria:

  R_topic    — BERTopic-style alignment with the FinePDFs style
               anchors (sentence-transformer cosine).
  R_iri      — fraction of cited ontology templates whose
               verbalization keywords appear in the chapter prose.
  R_density  — tables per chapter and rows-per-table; GLM-style
               wants ≥2 tables, Grok-style wants ≥3 with cross-FKs.
  R_axiom    — structural check that markdown tables have header
               columns matching template slot types. (LLM-judge
               variant for harder cases is a planned v2 layer.)

Composite::

    R = geometric_mean(R_topic, R_iri, R_density, R_axiom)
    status = accepted   if R ≥ τ_accept
           = borderline if τ_review ≤ R < τ_accept
           = rejected   if R < τ_review

Writes a ``raw.chapter_verification`` Iceberg table alongside the
existing ``raw.exchange`` table — same postgres-backed catalog, same
schema-evolution discipline.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/verify_chapters.py \\
        --chapters-run /raid/checkpoints/aegir-artifacts/chapters_v0/11147ff710cbf597 \\
        --audit-run /raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("verify-chapters")


# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-run", required=True,
                   help="Path to a chapters_v0/<run_id>/ directory")
    p.add_argument("--audit-run", required=True,
                   help="Path to a coverage_v0/<run_id>/ for style anchors")
    p.add_argument("--catalog-dir", default="src/aegir/ontology/catalog",
                   help="Ontology catalog dir (for resolving template slot types)")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--tau-accept", type=float, default=0.50,
                   help="Composite R threshold for accepted status")
    p.add_argument("--tau-review", type=float, default=0.30,
                   help="Composite R threshold for borderline status; "
                        "below this is rejected")
    p.add_argument("--write-to-iceberg", action="store_true",
                   help="Also append verification rows to raw.chapter_verification")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Scorers
# ─────────────────────────────────────────────────────────────────────────

_TABLE_ROW_RE = re.compile(r"^\s*\|[^\n]*\|\s*$")
_TABLE_HEADER_SEP_RE = re.compile(r"^\s*\|[\s|:\-]+\|\s*$")
_NAMED_ENT_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|\b[A-Z]{2,}\b)")


def score_topic(chapter_text: str, anchors: list[str],
                embedder) -> float:
    """R_topic: mean cosine sim of chapter to FinePDFs style anchors.

    Higher = chapter looks like the FinePDFs distribution we want
    the byte-LM to learn.
    """
    if not anchors:
        return 0.0
    chapter_text = (chapter_text or "")[:8000]  # cap so a 50K-char chapter
                                                   # isn't compared as a giant doc
    if not chapter_text.strip():
        return 0.0
    embs = embedder.encode([chapter_text] + list(anchors),
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True)
    chapter_vec = embs[0]
    anchor_vecs = embs[1:]
    sims = anchor_vecs @ chapter_vec
    return float(np.mean(sims))


def score_iri(chapter_text: str, cited_templates: list[dict]) -> float:
    """R_iri: fraction of cited templates whose key terms appear in prose.

    Each template has a verbal_template like "X is something signed by Y"
    or "X is a artifact". We extract content keywords (non-stopword,
    non-slot-variable) and check substring presence (case-insensitive)
    in the chapter. Score = matched_templates / total_templates.
    """
    if not cited_templates:
        return 0.0
    text_lc = (chapter_text or "").lower()
    if not text_lc:
        return 0.0

    STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "to", "of", "in", "on", "at", "by", "with", "for",
            "as", "that", "this", "it", "and", "or", "not", "from",
            "min", "max", "exactly", "some", "only", "and", "or",
            "something", "things", "concept"}
    n_matched = 0
    for t in cited_templates:
        vt = (t.get("verbal_template") or "").lower()
        if not vt:
            # No verbalization — fall back to template_id parts
            tid = t.get("template_id", "").lower()
            keywords = [w for w in re.split(r"[_\W]+", tid) if w and w not in STOP]
        else:
            # Strip slot placeholders like {X} {Y:Class}
            vt_clean = re.sub(r"\{[^}]*\}", " ", vt)
            keywords = [w for w in re.findall(r"[a-z]+", vt_clean)
                        if w not in STOP and len(w) > 2]
        if not keywords:
            n_matched += 1  # Vacuous match — empty verbalization
            continue
        # Require any 1-of-K keyword hit; tighten to >=2 if we want stricter
        if any(k in text_lc for k in keywords):
            n_matched += 1
    return n_matched / len(cited_templates)


def parse_markdown_tables(text: str) -> list[dict]:
    """Return a list of {n_rows, n_cols, headers, cells} for each table."""
    tables = []
    lines = (text or "").split("\n")
    i = 0
    while i < len(lines):
        if not _TABLE_ROW_RE.match(lines[i]):
            i += 1
            continue
        # candidate table — next line must be a header separator
        if i + 1 < len(lines) and _TABLE_HEADER_SEP_RE.match(lines[i + 1]):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            data_rows = []
            j = i + 2
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                data_rows.append(cells)
                j += 1
            tables.append({
                "n_rows": len(data_rows),
                "n_cols": len(header),
                "headers": header,
                "cells": data_rows,
            })
            i = j
        else:
            i += 1
    return tables


def score_density(chapter_text: str, prompt_kind: str) -> tuple[float, list[dict]]:
    """R_density: scores chapter for table count + rows.

    GLM-style chapters want ≥2 tables. Grok-style want ≥3 tables, with
    at least one column whose name looks like a foreign-key reference
    to another table's primary key (heuristic: a column name in table A
    that matches a column name in another table).
    """
    tables = parse_markdown_tables(chapter_text)
    if not tables:
        return 0.0, []

    n_tables = len(tables)
    if prompt_kind == "grok":
        min_tables, target_tables = 3, 5
    else:
        min_tables, target_tables = 2, 4

    # Table-count score: 0 below min, linear ramp to 1 at target, plateau.
    count_score = max(0.0, min(1.0, (n_tables - (min_tables - 1)) /
                                     (target_tables - (min_tables - 1))))

    # Row-density score: average rows per table (3 = full credit, 1 = no credit).
    avg_rows = float(np.mean([t["n_rows"] for t in tables]))
    row_score = max(0.0, min(1.0, (avg_rows - 1) / 2))

    # Cross-FK heuristic (Grok bonus): does any header in table A
    # also appear as a header in table B?
    cross_fk = 0.0
    if prompt_kind == "grok" and n_tables >= 2:
        all_headers = [set(h.lower() for h in t["headers"]) for t in tables]
        intersections = 0
        pairs = 0
        for a in range(n_tables):
            for b in range(a + 1, n_tables):
                pairs += 1
                if all_headers[a] & all_headers[b]:
                    intersections += 1
        cross_fk = (intersections / pairs) if pairs else 0.0

    # Composite: count gates, row + cross_fk refine.
    if prompt_kind == "grok":
        density = 0.5 * count_score + 0.3 * row_score + 0.2 * cross_fk
    else:
        density = 0.6 * count_score + 0.4 * row_score
    return float(density), tables


def score_axiom(tables: list[dict], cited_templates: list[dict]) -> float:
    """R_axiom: structural — table headers contain ontology slot vocabulary.

    For each table: how many of its header columns contain keywords drawn
    from the cited templates' slot_types or template_id parts?
    Score = mean fraction across tables.

    This is the "deterministic" version. A v2 LLM-judge for axiom
    traceability is planned; this catches the cheap wins.
    """
    if not tables or not cited_templates:
        return 0.0
    slot_vocab: set[str] = set()
    for t in cited_templates:
        for slot_name, slot_type in (t.get("slot_types") or {}).items():
            for w in re.split(r"[_\W]+", slot_type.lower()):
                if w and len(w) > 2:
                    slot_vocab.add(w)
        tid = t.get("template_id", "").lower()
        for w in re.split(r"[_\W]+", tid):
            if w and len(w) > 2 and w not in {"min", "max", "basic"}:
                slot_vocab.add(w)

    if not slot_vocab:
        return 0.0

    per_table = []
    for tbl in tables:
        headers = [h.lower() for h in tbl["headers"]]
        if not headers:
            continue
        matched = sum(
            1 for h in headers
            if any(v in h or h in v for v in slot_vocab)
        )
        per_table.append(matched / len(headers))
    return float(np.mean(per_table)) if per_table else 0.0


def geometric_mean(scores: list[float]) -> float:
    """Composite R as geometric mean; treats any zero score as a hard fail."""
    if not scores or any(s <= 0 for s in scores):
        return 0.0
    return float(math.exp(sum(math.log(s) for s in scores) / len(scores)))


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def load_ontology_lookup(catalog_dir: Path) -> dict[str, dict]:
    """All templates, keyed by template_id, across all canonical family files."""
    out: dict[str, dict] = {}
    for p in sorted(catalog_dir.glob("0*.json")):
        if "candidate" in p.name:
            continue
        data = json.loads(p.read_text())
        for t in data.get("templates", []):
            t = dict(t)
            t["_family"] = p.stem
            out[t["template_id"]] = t
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    chapters_run = Path(args.chapters_run)
    audit_run = Path(args.audit_run)
    catalog_dir = REPO / args.catalog_dir

    chapters = pq.read_table(chapters_run / "chapters.parquet").to_pandas()
    audit_topics = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    ontology = load_ontology_lookup(catalog_dir)
    logger.info(f"chapters: {len(chapters)}  ontology templates: {len(ontology)}  "
                f"audit topics: {len(audit_topics)}")

    # Style anchors lookup: topic_id -> topic_repr_text
    anchors_by_id = {int(r.topic_id): r.topic_repr_text or ""
                     for _, r in audit_topics.iterrows()}

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"device: {device}")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedding_model, device=device)

    def _list_get(row, key) -> list:
        """Pandas-safe extraction of list-typed columns. Returns [] for
        missing or empty arrays."""
        val = row[key] if key in row.index else None
        if val is None:
            return []
        try:
            return list(val)
        except TypeError:
            return []

    rows = []
    for _, ch in chapters.iterrows():
        chapter_text = (ch["response_text"] if "response_text" in ch.index
                        else "") or ""
        prompt_kind = (ch["prompt_kind"] if "prompt_kind" in ch.index
                        else "") or "glm"
        cited_ids = [str(x) for x in _list_get(ch, "template_ids")]
        cited_templates = [ontology[tid] for tid in cited_ids if tid in ontology]
        style_ids = [int(x) for x in _list_get(ch, "style_topic_ids")]
        anchor_texts = [anchors_by_id.get(t, "") for t in style_ids]
        anchor_texts = [a for a in anchor_texts if a]

        r_topic = score_topic(chapter_text, anchor_texts, embedder)
        r_iri = score_iri(chapter_text, cited_templates)
        r_density, tables = score_density(chapter_text, prompt_kind)
        r_axiom = score_axiom(tables, cited_templates)
        r_composite = geometric_mean([r_topic, r_iri, r_density, r_axiom])

        if r_composite >= args.tau_accept:
            status = "accepted"
        elif r_composite >= args.tau_review:
            status = "borderline"
        else:
            status = "rejected"

        notes_parts = []
        if r_topic < 0.30: notes_parts.append("low_topic")
        if r_iri < 0.50: notes_parts.append("low_iri")
        if r_density < 0.40: notes_parts.append("low_density")
        if r_axiom < 0.30: notes_parts.append("low_axiom")
        if not tables: notes_parts.append("no_tables")

        rows.append({
            "chapter_id": str(ch["chapter_id"]),
            "audit_run_id": str(ch["audit_run_id"] if "audit_run_id" in ch.index else ""),
            "family": str(ch["family"] if "family" in ch.index else ""),
            "model": str(ch["model"] if "model" in ch.index else ""),
            "prompt_kind": prompt_kind,
            "n_tables": len(tables),
            "n_template_ids": len(cited_ids),
            "n_anchors": len(anchor_texts),
            "r_topic": float(r_topic),
            "r_iri": float(r_iri),
            "r_density": float(r_density),
            "r_axiom": float(r_axiom),
            "r_composite": float(r_composite),
            "status": status,
            "notes": ",".join(notes_parts),
            "verified_at": datetime.now(timezone.utc),
        })

    # Write to parquet (always) + Iceberg (if --write-to-iceberg)
    schema = pa.schema([
        ("chapter_id", pa.string()),
        ("audit_run_id", pa.string()),
        ("family", pa.string()),
        ("model", pa.string()),
        ("prompt_kind", pa.string()),
        ("n_tables", pa.int32()),
        ("n_template_ids", pa.int32()),
        ("n_anchors", pa.int32()),
        ("r_topic", pa.float32()),
        ("r_iri", pa.float32()),
        ("r_density", pa.float32()),
        ("r_axiom", pa.float32()),
        ("r_composite", pa.float32()),
        ("status", pa.string()),
        ("notes", pa.string()),
        ("verified_at", pa.timestamp("us", tz="UTC")),
    ])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    out_parquet = chapters_run / "verification.parquet"
    pq.write_table(tbl, out_parquet, compression="zstd")
    logger.info(f"wrote {out_parquet}")

    # Summary
    accepted = sum(1 for r in rows if r["status"] == "accepted")
    borderline = sum(1 for r in rows if r["status"] == "borderline")
    rejected = sum(1 for r in rows if r["status"] == "rejected")
    print()
    print(f"  status: {accepted} accepted / {borderline} borderline / {rejected} rejected"
          f"  (n={len(rows)})")
    print()
    for kind in ("glm", "grok"):
        subset = [r for r in rows if r["prompt_kind"] == kind]
        if not subset:
            continue
        s = lambda k: np.mean([r[k] for r in subset])
        print(f"  {kind} (n={len(subset)}):")
        print(f"    R_topic   mean={s('r_topic'):.3f}")
        print(f"    R_iri     mean={s('r_iri'):.3f}")
        print(f"    R_density mean={s('r_density'):.3f}")
        print(f"    R_axiom   mean={s('r_axiom'):.3f}")
        print(f"    R_comp    mean={s('r_composite'):.3f}")
    print()
    print(f"  most common notes:")
    from collections import Counter
    notes_flat = [n for r in rows for n in r["notes"].split(",") if n]
    for note, count in Counter(notes_flat).most_common():
        print(f"    {note}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
