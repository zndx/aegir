#!/usr/bin/env python
"""Generate an ontology-grounded textbook chapter in FinePDFs style.

The chapter generator wires the four substrates of the v0.3 thesis:

  1. Aegir ontology (deterministic axiom grounding — the WHAT).
  2. FinePDFs topic centroids (style anchor — the HOW).
  3. GLM-4.7 via Cerebras + DSPy (volume-tier LLM — the GENERATOR).
  4. HX-captured Iceberg trace (reproducibility — the AUDIT).

For each invocation:

  a. Pick K templates from a chosen ontology family. Their
     ``verbal_template`` and ``manchester_template`` fields define
     the content to explain.

  b. Cross-reference the coverage_v0 audit's per-topic data: find
     topics whose top-1 template lies in the requested family.
     Pull their ``topic_repr_text`` excerpts (real FinePDFs passages)
     as STYLE ANCHORS — the chapter should *look like* these.

  c. Prompt GLM-4.7 with both: ontology axioms (verbalized) +
     style anchors + an instruction to produce a textbook chapter
     of ~1500-3000 tokens with at least 2 tables whose cells are
     directly derivable from the cited axioms.

  d. Capture the full exchange (prompt + response + reasoning) in
     ``raw.exchange`` (postgres-backed Iceberg) and append the
     generated chapter to a flat parquet for downstream pretrain
     ingestion. Both keyed by a deterministic ``chapter_id``.

The self-reinforcing property: the chapter looks like FinePDFs (so
training on it teaches the model the FinePDFs distribution) and
its tables are ontology-derivable (so training on them teaches
table-aware processing). Both objectives shape the model in the
same direction.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/generate_chapter.py \\
        --family 03_directive_governance \\
        --n-chapters 5 \\
        --audit-run /raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf \\
        --output /raid/checkpoints/aegir-artifacts/chapters_v0/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("generate-chapter")


# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restrict-families", default=None,
                   help="Optional comma-separated family allowlist (e.g., "
                        "'03_directive_governance,07_long_tail'). When set, the "
                        "topic-first sampler only considers templates whose "
                        "family is in this list. Default (None) = all 7 "
                        "families eligible. Replaces the old --family flag.")
    p.add_argument("--n-chapters", type=int, default=5)
    p.add_argument("--budget-usd", type=float, default=None,
                   help="Hard stop once cumulative generation cost reaches this (USD). "
                        "Cost from litellm usage × RATES_USD_PER_M.")
    p.add_argument("--flush-every", type=int, default=100,
                   help="Checkpoint chapters.parquet every N chapters (long-run durability).")
    p.add_argument("--templates-per-chapter", type=int, default=4,
                   help="How many ontology templates to ground a chapter on")
    p.add_argument("--style-anchors", type=int, default=3,
                   help="Number of FinePDFs passages to include as style refs. "
                        "First passage is the target topic's repr_text "
                        "(also drives axiom selection); remainder are style "
                        "siblings picked by the same anti-repetition sampler.")
    p.add_argument("--tau-anchor", type=float, default=0.20,
                   help="Minimum coverage_score for a topic to be eligible as "
                        "a target/anchor. Below this, the audit shows the "
                        "ontology has no good template match — anchoring "
                        "there would force template citations the ontology "
                        "can't support, breaking R_axiom integrity. The "
                        "audit run baseline shows 160/200 topics qualify at "
                        "0.20, covering 86%% of FinePDFs docs.")
    p.add_argument("--tau-template", type=float, default=0.15,
                   help="Minimum per-topic similarity for a candidate "
                        "template to be selectable. Filters the topic's "
                        "top_templates list before family-diverse selection.")
    p.add_argument("--prefer-family-diverse", action="store_true", default=True,
                   help="When ≥2 families pass tau-template, round-robin pick "
                        "one per family (highest similarity first). Forces "
                        "cross-family axiom citations on the ~160/200 topics "
                        "whose top-5 templates already span multiple families.")
    p.add_argument("--no-family-diverse", dest="prefer_family_diverse",
                   action="store_false",
                   help="Disable family-diverse selection; pick top-K by "
                        "similarity regardless of family. Use to A/B against "
                        "the family-diverse run.")
    p.add_argument("--family-complex",
                   default="src/aegir/ontology/family_complex.json",
                   help="Data-driven simplicial complex over family axis. "
                        "After topic-first template selection, the chosen "
                        "family-set is checked against this complex. If it's "
                        "not allowed (outside closure of maximal simplices, "
                        "or in the measured-below-floor puncture set), "
                        "templates are filtered down to the largest allowed "
                        "face. Pass --family-complex '' to disable.")
    p.add_argument("--audit-run", default=None,
                   help="Path to coverage_v0/<run_id>/ for topic_coverage + template_density "
                        "(topic-first sampler). Required unless --from-harvest is given.")
    p.add_argument("--from-harvest", nargs="?", const="build/domain_harvest", default=None,
                   help="CONTENT-FIRST: write one chapter per harvested in-domain doc (the "
                        "content-addressed harvest cache), grounded in the DERIVED ontology (catalog.json) — "
                        "prose + schema co-derived from the same source. Replaces the audit topic sampler. "
                        "Optional path (default build/domain_harvest).")
    p.add_argument("--output", default="/raid/checkpoints/aegir-artifacts/chapters_v0/")
    p.add_argument("--mix",
                   default="cerebras/zai-glm-4.7:0.6,xai/grok-4.3:0.4",
                   help="Comma-separated model:weight pairs. Weighted sampling "
                        "per chapter. Default 60/40 GLM/Grok. Both emit native "
                        "reasoning_content channels — confirmed via probe — so "
                        "both populate HX.response_reasoning uniformly. GLM-4.7 "
                        "on Cerebras = volume + speed (1000+ tok/s); Grok 4.3 "
                        "on xAI = depth + complex cross-joinable LIMS-style "
                        "relational tables. (NOT grok-4-1-fast-non-reasoning — "
                        "that variant emits no reasoning channel and is also "
                        "no longer in the xAI catalog as of 2026-05.)")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Bumped from 4096 — initial run hit truncation at 4096; "
                        "8192 gives room for cross-joinable schema chapters.")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=4649)
    p.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    p.add_argument("--ablation", default="full",
                   choices=("full", "no-ontology", "no-schema"),
                   help="Ablation mode for value-add experiments. "
                        "'full' = ontology axioms + style anchors + schema-rich prompt "
                        "(the production pipeline). "
                        "'no-ontology' = style anchors + schema-rich prompt only "
                        "(topic source = anchor passages, no axiom citations) — "
                        "tests whether ontology grounding adds value. "
                        "'no-schema' = ontology axioms + style anchors + standard "
                        "textbook prompt (no LIMS-schema instruction) — tests "
                        "whether the schema-rich Grok prompt adds value vs "
                        "ontology+style alone.")
    p.add_argument("--realize-schemas", action="store_true",
                   help="write chapters around REALIZED schema subgraphs (EAV / junction / star / "
                        "snowflake via realize.py) instead of flat one-table-per-template — the "
                        "super-linear DDL+views deliverable with real-world structural diversity")
    p.add_argument("--naming", choices=["natural", "semantic"], default="natural",
                   help="C2.5 L2 physical naming of embedded tables/columns/views. 'natural' (default) = the "
                        "CANONICAL trained deliverable (DBA-realistic names from natural_names.json, RI-safe); "
                        "'semantic' = ontology-native names (Atlas/lineage reference). natural also breaks the "
                        "concept-from-header training shortcut.")
    return p.parse_args()


def parse_mix(spec: str) -> list[tuple[str, float]]:
    """Parse 'model_a:w_a,model_b:w_b' into normalized (model, weight) pairs."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        model, _, w = tok.rpartition(":")
        if not model or not w:
            raise SystemExit(f"bad --mix entry: {tok!r}")
        out.append((model, float(w)))
    total = sum(w for _, w in out)
    if total <= 0:
        raise SystemExit(f"--mix weights must sum to positive: {spec!r}")
    return [(m, w / total) for m, w in out]


def select_model(mix: list[tuple[str, float]], rng: np.random.Generator) -> str:
    """Sample one model from the weighted mix."""
    models, weights = zip(*mix)
    return str(rng.choice(models, p=weights))


# ─────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────

def load_all_templates(catalog_dir: Path,
                       restrict_families: set[str] | None = None
                       ) -> dict[str, dict]:
    """Load templates from every family file in the catalog directory.

    Returns a dict keyed by template_id, with `_family` tag added per
    template. Used by the topic-first sampler to enrich the audit's
    top_templates entries (which carry only template_id + family +
    similarity + manchester_template) with full catalog data
    (verbal_template, slot_types, etc.) needed by build_axiom_section.

    `restrict_families` (optional) filters templates to a given allowlist.
    """
    all_templates: dict[str, dict] = {}
    from aegir.ontology.schema import catalog_files
    for p in catalog_files(catalog_dir):
        family = p.stem
        if restrict_families is not None and family not in restrict_families:
            continue
        data = json.loads(p.read_text())
        for t in data.get("templates", []):
            t = dict(t)
            t["_family"] = family
            all_templates[t["template_id"]] = t
    return all_templates


def load_harvest_docs(store: Path, max_chars: int = 4000) -> list[dict]:
    """Content-first inputs: the pre-matched in-domain docs from the harvest cache
    (``harvest_domain_docs``), each as a style/content anchor + its harvested SKOS domain tag."""
    manifest: dict[str, dict] = {}
    mf = store / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text().splitlines():
            try:
                r = json.loads(line)
                manifest[r["hash"]] = r
            except (ValueError, KeyError):
                continue
    docs: list[dict] = []
    for f in sorted((store / "docs").glob("*.txt")):
        m = manifest.get(f.stem, {})
        docs.append({"passage": f.read_text(encoding="utf-8", errors="ignore")[:max_chars],
                     "topic_id": -1, "hash": f.stem, "code": m.get("code"), "label": m.get("label")})
    return docs


def select_content_first(i: int, docs: list[dict], pool: list[dict], k: int):
    """One chapter per harvested doc: the doc is the content/style anchor, grounded in ``k`` derived
    ontology primitives (round-robin over the derived pool). Returns (target_ns, anchors, chosen) in the
    shapes the generation loop expects."""
    from types import SimpleNamespace
    doc = docs[i % len(docs)]
    start = (i * k) % max(1, len(pool))
    chosen = [pool[(start + j) % len(pool)] for j in range(min(k, len(pool)))]
    target = SimpleNamespace(topic_id=-1, coverage_score=1.0, label=doc.get("label"))
    return target, [doc], chosen


def load_audit(audit_run: Path) -> tuple[Any, Any, Any]:
    """Load the three parquets from a coverage audit run."""
    topic_cov = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    template_density = pq.read_table(audit_run / "template_density.parquet").to_pandas()
    family_density = pq.read_table(audit_run / "family_density.parquet").to_pandas()
    return topic_cov, template_density, family_density


def pick_topic_and_anchors(topic_cov, usage: dict[int, int], n_anchors: int,
                           rng: np.random.Generator,
                           tau_anchor: float,
                           ) -> tuple[Any, list[dict]]:
    """Pick a target topic + (n_anchors-1) style siblings via anti-repetition.

    Topic-first sampler. Returns (target_row, anchors) where:
      - target_row is the topic that drives axiom selection (its top_templates
        is the candidate pool for `pick_topic_templates`).
      - anchors is the full list of (target + siblings) for the prompt's
        style section.

    Eligibility filter: ``coverage_score >= tau_anchor``. This keeps the
    sampler in topics where the audit shows the ontology has real template
    support, so anchoring axioms there won't force the LLM to fabricate
    template→topic alignment that doesn't exist.

    Anti-repetition: per-topic weight ∝ 1/(1+usage_count). Within a single
    run, hot topics get pushed down on each subsequent draw, naturally
    spreading the corpus across the eligible pool. Updates ``usage`` in
    place — caller does not need to.
    """
    eligible = topic_cov[topic_cov.coverage_score >= tau_anchor]
    if len(eligible) == 0:
        raise SystemExit(
            f"no audit topics with coverage_score >= {tau_anchor}; "
            f"lower --tau-anchor or extend the ontology"
        )
    if len(eligible) < n_anchors:
        # Pool smaller than requested anchors — sample without replacement
        # against the whole eligible set, no anti-repetition possible.
        n_anchors = len(eligible)

    weights = 1.0 / (1.0 + np.array(
        [usage.get(int(t), 0) for t in eligible.topic_id], dtype=float
    ))
    weights = weights / weights.sum()
    idx = rng.choice(eligible.index.to_numpy(), size=n_anchors,
                      replace=False, p=weights)
    picks = eligible.loc[idx]

    for tid in picks.topic_id:
        tid_int = int(tid)
        usage[tid_int] = usage.get(tid_int, 0) + 1

    anchors = [
        {
            "topic_id": int(r.topic_id),
            "coverage_score": float(r.coverage_score),
            "top_template_id": r.top_template_id,
            "passage": r.topic_repr_text or "",
        }
        for _, r in picks.iterrows()
    ]
    return picks.iloc[0], anchors


def pick_topic_templates(target: Any, all_templates: dict[str, dict],
                         k: int, tau_template: float,
                         prefer_family_diverse: bool,
                         restrict_families: set[str] | None = None,
                         ) -> list[dict]:
    """Pick K templates from the target topic's top_templates list.

    The audit's ``top_templates`` field already encodes "which templates
    best match this topic" (similarity-ranked). We filter to
    ``similarity >= tau_template`` then either round-robin across
    families (default, exposes the ~160/200 topics whose top-5 templates
    span ≥2 families) or take top-K by similarity.

    Each chosen entry is enriched with the full catalog template fields
    (verbal_template, slot_types) since the audit row carries only a
    summary subset.
    """
    candidates = list(target.top_templates) if target.top_templates is not None else []
    candidates = [t for t in candidates if t.get("similarity", 0.0) >= tau_template]
    if restrict_families is not None:
        candidates = [t for t in candidates if t.get("family") in restrict_families]
    if not candidates:
        return []

    if prefer_family_diverse and len(candidates) > 1:
        by_fam: dict[str, list[dict]] = {}
        for t in sorted(candidates, key=lambda t: -t.get("similarity", 0.0)):
            by_fam.setdefault(t["family"], []).append(t)
        chosen: list[dict] = []
        families = list(by_fam.keys())
        while len(chosen) < k and any(by_fam.values()):
            for fam in families:
                if not by_fam[fam]:
                    continue
                chosen.append(by_fam[fam].pop(0))
                if len(chosen) == k:
                    break
    else:
        chosen = sorted(candidates, key=lambda t: -t.get("similarity", 0.0))[:k]

    enriched: list[dict] = []
    for c in chosen:
        full = all_templates.get(c["template_id"])
        if full is None:
            # Template_id missing from catalog (should not happen but is
            # defensive — audit data may pre-date a catalog edit).
            enriched.append({
                "template_id": c["template_id"],
                "manchester_template": c.get("manchester_template", ""),
                "verbal_template": "",
                "slot_types": {},
                "_family": c.get("family", ""),
                "_similarity": float(c.get("similarity", 0.0)),
            })
            continue
        t = dict(full)
        t["_family"] = c.get("family", t.get("_family", ""))
        t["_similarity"] = float(c.get("similarity", 0.0))
        enriched.append(t)
    return enriched


# ─────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────

GLM_PROMPT_TEMPLATE = """You are writing one chapter of a technical textbook. The chapter
must teach a small number of related concepts precisely, with embedded
data tables, in the style of professional technical documentation
(audit reports, compliance handbooks, governance frameworks, regulatory
guides — the kind of dense, evidence-anchored prose you find in
high-quality PDFs).

═══════════════════════════════════════════════════════════════════════
SECTION 1 — STYLE REFERENCES (the chapter should LOOK LIKE these)
═══════════════════════════════════════════════════════════════════════

These are real passages from a technical-document corpus. Match the
register, structure, density, and use of evidence. DO NOT copy their
subject matter — they are style anchors, not content sources.

{style_section}

═══════════════════════════════════════════════════════════════════════
SECTION 2 — ONTOLOGY AXIOMS (the chapter must explain THESE concepts)
═══════════════════════════════════════════════════════════════════════

The following are formal axioms from an OWL ontology, expressed both
in Manchester syntax and natural-language verbalization. Every named
entity in your chapter that maps to an ontology slot must be
introduced via one of these axioms. You may invent specific instances
to illustrate them, but the structural claims must be derivable from
these axioms.

{axiom_section}

═══════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Write a single textbook chapter approximately 1500-3000 words long.

REQUIREMENTS:
- Open with a short scope/preamble paragraph (3-5 sentences).
- Introduce each of the {n_templates} ontology concepts in a dedicated
  numbered section. Each section weaves prose, definitions, and at
  least one short example.
- Include AT LEAST 2 markdown tables in the chapter body. Each table
  must have a header row and 3-7 data rows. Tables should encode
  relationships, instances, or property values that are directly
  consistent with the cited axioms — not generic examples.
- Use chapter-style numbering: a top-level section per concept, with
  sub-sections (1.1, 1.2, ...) where useful.
- Maintain the dense, evidence-anchored register of the style
  references. Avoid casual phrasing, marketing language, or padding.

OUTPUT FORMAT: pure markdown. Begin with a level-1 heading (# Chapter
title). Do not include preamble like "Here is the chapter:". Do not
include explanations of what you generated. Just the chapter.
"""


# Natural/TOPICAL register (C2.5 L2) — paired with --naming natural. Same ontology grounding + tables, but
# the chapter is written as a DOMAIN PRACTITIONER's handbook ABOUT THE SUBJECT (not about the ontology), so
# the corpus accumulates a second, complementary surface from the same input: topical prose + DBA-real names.
GLM_NATURAL_TEMPLATE = """You are writing one chapter of a practitioner's handbook in a specific SUBJECT
AREA — the kind a senior professional writes for their own field (laboratory operations, kernel
observability, data governance, clinical workflows, supply-chain QA, …). You write ABOUT THE SUBJECT,
using its real working data tables as the backbone.

═══════════════════════════════════════════════════════════════════════
SECTION 1 — STYLE REFERENCES (the chapter should LOOK LIKE these)
═══════════════════════════════════════════════════════════════════════

These are real passages from a technical-document corpus. Match the register, structure, density, and
use of evidence. DO NOT copy their subject matter — they are style anchors, not content sources.

{style_section}

═══════════════════════════════════════════════════════════════════════
SECTION 2 — THE DOMAIN'S DATA MODEL + RECORDS (your chapter is built around THIS)
═══════════════════════════════════════════════════════════════════════

Below are the working tables of this subject area — a relational schema with real records — and short
factual statements about how they relate. Treat them as the OPERATIONAL DATA of the field: the tables a
practitioner actually queries. The table and column names are the field's real schema — use them exactly
as given. Build your chapter around these records; every entity you discuss must trace to one of them.

{axiom_section}

═══════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Write a single handbook chapter approximately 1500-3000 words long, IN THE VOICE OF A DOMAIN PRACTITIONER.

REQUIREMENTS:
- Write about the SUBJECT MATTER, not about data modelling. Do NOT use the words "ontology", "axiom",
  "OWL", "schema template", or "concept" — a practitioner writes about specimens, runs, programs, and
  policies, not about formal representation.
- Open with a short scope paragraph framing the subject area (3-5 sentences).
- Organize by TOPIC (the real things in the domain), each in a numbered section with prose and at least
  one short worked example that references the data.
- Embed AT LEAST 2 of the data tables verbatim (markdown, header + 3-7 rows) and refer to their rows
  naturally in the prose (e.g. "as the sample-register table records, …"). Use the given column names as-is.
- Maintain the dense, evidence-anchored register of the style references. No marketing, no padding.

OUTPUT FORMAT: pure markdown. Begin with a level-1 heading (# Chapter title). No preamble, no
meta-commentary. Just the chapter.
"""


# Grok prompt — depth + max ontology grounding + cross-joinable LIMS-style
# relational tables. The Grok output is where the multi-hop relational
# structure lives that lets TAPEX-style table reasoning emerge.
GROK_PROMPT_TEMPLATE = """You are writing one chapter of a technical reference book. This
chapter teaches a small set of related ontological concepts AND
demonstrates them through a relational data model: 3-5 tables that
share primary/foreign keys, queryable as a small schema. Reader should
be able to cross-join across tables to answer multi-hop questions
(e.g., "what is the result of the test on sample S using equipment E
operated by technician T?").

═══════════════════════════════════════════════════════════════════════
SECTION 1 — STYLE REFERENCES (the chapter should LOOK LIKE these)
═══════════════════════════════════════════════════════════════════════

These are real passages from a technical-document corpus. Match the
register, structure, density, and use of evidence. DO NOT copy their
subject matter — they are style anchors, not content sources.

{style_section}

═══════════════════════════════════════════════════════════════════════
SECTION 2 — ONTOLOGY AXIOMS (the chapter must explain THESE concepts)
═══════════════════════════════════════════════════════════════════════

The following are formal axioms from an OWL ontology. Every named
entity in your chapter that maps to an ontology slot must be
introduced via one of these axioms. The tables you produce must
embody the relationships these axioms describe — cell values should
be derivable instances of the slot types.

{axiom_section}

═══════════════════════════════════════════════════════════════════════
SECTION 3 — RELATIONAL SCHEMA REQUIREMENTS (this is the key difference)
═══════════════════════════════════════════════════════════════════════

Pick a concrete domain where cross-joinable relational data is
natural and the chosen domain instances cleanly satisfy the ontology
axioms above. Examples (pick ONE; prefer one most natural for the
axioms):

  - Laboratory Information Management (LIMS): Samples, Tests, Results,
    Methods, Equipment, Technicians — each test references a Sample
    and a Method; each Result references a Test and an Equipment.
  - Audit trail systems: Subjects, Controls, Tests, Findings,
    Evidence — Findings reference Tests, which reference Controls,
    which reference Subjects, with Evidence linked to Findings.
  - Clinical trials: Subjects, Protocols, Visits, Measurements,
    Outcomes — Visits link Subjects to Protocols; Measurements
    reference Visits.
  - Supply chain quality: Products, Lots, Shipments, QualityChecks,
    Auditors — QualityChecks reference Shipments which reference Lots
    which reference Products.

Produce 3-5 tables with these properties:

  TABLE STRUCTURE
  - Each table has a clear PRIMARY KEY column (named explicitly,
    e.g., ``sample_id`` or ``test_id``).
  - At least 3 of the 5 tables have FOREIGN KEY columns referencing
    other tables' primary keys.
  - 4-8 data rows per table (enough to demonstrate cross-joins;
    not so many the chapter becomes a database dump).
  - For each table, write a short paragraph BEFORE the table
    explaining: (a) which ontology axiom(s) it embodies, (b) which
    columns are PK / FK, (c) one example cross-join query the
    reader can run mentally.

  CROSS-JOIN DEMONSTRATION
  - Include AT LEAST ONE worked example after the tables that shows
    how to answer a multi-hop question by walking PK/FK links across
    2-3 tables. Show the resulting joined row(s) inline.

═══════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Write a single chapter approximately 2000-3500 words. Open with a
short scope/preamble paragraph. Number sections (1, 1.1, 1.2, 2, ...).
Maintain the dense register of the style references — no marketing
language, no padding, no exclamation marks.

OUTPUT FORMAT: pure markdown starting with a level-1 heading
(# Chapter title). Do not include preamble like "Here is the chapter:".
Do not include explanations of what you generated. Just the chapter.
(Your chain-of-thought is captured separately via the model's native
reasoning channel; no need to inline it.)
"""

# ─────────────────────────────────────────────────────────────────────────
# Ablation prompts — "no-ontology" keeps schema + style but removes the
# ontology axioms; "no-schema" keeps ontology + style but removes the
# LIMS-style cross-joinable instruction.
# Used by --ablation flag for the load-bearing experiment.

NO_ONTOLOGY_PROMPT_TEMPLATE = """You are writing one chapter of a technical reference book on a
topic that fits the style references below. The chapter demonstrates
your topic through a relational data model: 3-5 tables that share
primary/foreign keys, queryable as a small schema. Reader should be
able to cross-join across tables to answer multi-hop questions.

═══════════════════════════════════════════════════════════════════════
SECTION 1 — STYLE REFERENCES (the chapter should LOOK LIKE these)
═══════════════════════════════════════════════════════════════════════

These are real passages from a technical-document corpus. Match the
register, structure, density, and use of evidence. Identify a topic
that would naturally appear in this kind of corpus, and write a
chapter on it.

{style_section}

═══════════════════════════════════════════════════════════════════════
SECTION 2 — RELATIONAL SCHEMA REQUIREMENTS
═══════════════════════════════════════════════════════════════════════

Pick a concrete domain where cross-joinable relational data is
natural (LIMS, audit trails, clinical trials, supply chain, etc.).
Produce 3-5 tables with these properties:

  - Each table has an explicit PRIMARY KEY column.
  - At least 3 of the 5 tables have FOREIGN KEY columns referencing
    other tables' primary keys.
  - 4-8 data rows per table.
  - For each table, write a short paragraph before it explaining what
    relationship it captures and which columns are PK / FK.
  - Include at least one worked cross-join example after the tables.

═══════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Write a single chapter approximately 2000-3500 words. Open with a
short scope/preamble paragraph. Number sections. Maintain the dense
register of the style references — no marketing language, no padding.

OUTPUT FORMAT: pure markdown starting with a level-1 heading.
"""


NO_SCHEMA_PROMPT_TEMPLATE = """You are writing one chapter of a technical textbook. The chapter
must teach a small number of related ontological concepts precisely,
with embedded data tables, in the style of professional technical
documentation (audit reports, compliance handbooks, governance
frameworks, regulatory guides).

═══════════════════════════════════════════════════════════════════════
SECTION 1 — STYLE REFERENCES (the chapter should LOOK LIKE these)
═══════════════════════════════════════════════════════════════════════

These are real passages from a technical-document corpus. Match the
register, structure, density, and use of evidence. DO NOT copy their
subject matter — they are style anchors, not content sources.

{style_section}

═══════════════════════════════════════════════════════════════════════
SECTION 2 — ONTOLOGY AXIOMS (the chapter must explain THESE concepts)
═══════════════════════════════════════════════════════════════════════

The following are formal axioms from an OWL ontology, expressed both
in Manchester syntax and natural-language verbalization. Every named
entity in your chapter that maps to an ontology slot must be
introduced via one of these axioms.

{axiom_section}

═══════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Write a single textbook chapter approximately 1500-3000 words long.
Open with a short scope paragraph. Introduce each of the {n_templates}
ontology concepts in a dedicated numbered section. Include at least 2
markdown tables in the chapter body, each with a header row and 3-7
data rows.

OUTPUT FORMAT: pure markdown starting with a level-1 heading.
"""


# Default prompt template per provider. Cerebras gets the GLM template
# (general textbook, native reasoning channel). xAI gets the schema-rich
# Grok template (LIMS-style cross-joinable tables, native reasoning).
PROMPT_TEMPLATES = {
    "cerebras": "glm",
    "xai":      "grok",
}

PROMPT_BY_KIND = {
    "glm":           GLM_PROMPT_TEMPLATE,
    "grok":          GROK_PROMPT_TEMPLATE,
    "no-ontology":   NO_ONTOLOGY_PROMPT_TEMPLATE,
    "no-schema":     NO_SCHEMA_PROMPT_TEMPLATE,
}


def prompt_kind_for_ablation(ablation: str, model: str) -> str:
    """Pick prompt template based on (ablation, model). full → model-default;
    no-ontology and no-schema override with their dedicated templates."""
    if ablation == "no-ontology":
        return "no-ontology"
    if ablation == "no-schema":
        return "no-schema"
    return prompt_kind_for_model(model)


def prompt_kind_for_model(model: str) -> str:
    """Pick which prompt template based on model id (provider prefix)."""
    provider = model.split("/", 1)[0]
    return PROMPT_TEMPLATES.get(provider, "glm")


def build_style_section(anchors: list[dict]) -> str:
    lines = []
    for i, a in enumerate(anchors, 1):
        passage = (a["passage"] or "").strip()
        if len(passage) > 700:
            passage = passage[:700] + "..."
        lines.append(f"  STYLE REFERENCE {i}:")
        lines.append(f"  ──────────────────")
        lines.append("  " + passage.replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)


def pick_verbalization(t: dict, rng=None) -> str:
    """Sample one surface form from the template's diverse ``verbal_templates`` set (Comp 3), so different
    chapters citing the same template get different phrasings (corpus diversity). Falls back to the single
    ``verbal_template`` when the set is absent or no ``rng`` is given (preserves legacy behavior)."""
    frames = t.get("verbal_templates") or ([t["verbal_template"]] if t.get("verbal_template") else [])
    if not frames:
        return ""
    return rng.choice(frames) if rng is not None else frames[0]


def build_axiom_section(templates: list[dict], rng=None) -> str:
    lines = []
    for i, t in enumerate(templates, 1):
        manchester = t.get("manchester_template", "")
        verbal = pick_verbalization(t, rng)
        slots = t.get("slot_types", {})
        slots_fmt = ", ".join(f"{k}: {v}" for k, v in slots.items())
        lines.append(f"  AXIOM {i} (template_id: {t['template_id']}):")
        lines.append(f"    Manchester:   {manchester}")
        if verbal:
            lines.append(f"    Verbalization: {verbal}")
        lines.append(f"    Slot types:    {slots_fmt}")
        lines.append("")
    return "\n".join(lines)


def build_axiom_section_with_ddl(templates: list[dict], family_complex=None, rng=None) -> str:
    """Axiom section with the deterministic DDL footprint injected (load-bearing ontology).

    Each cited template carries its rendered CREATE TABLE schema — slot-typed columns +
    family-complex-sanctioned FOREIGN KEY constraints, from the DDL spine — so the model
    populates a *given* relational structure rather than inventing one. This is what makes
    the ontology grounding load-bearing; the no-schema arm (plain ``build_axiom_section``)
    conspicuously lacks it, isolating the DDL's value-add.
    """
    from aegir.ontology.ddl import cross_family_fks, render_ddl, template_to_table
    from aegir.ontology.schema import CatalogTemplate

    spine, by_id = [], {}
    for t in templates:
        ct = CatalogTemplate(
            template_id=t["template_id"],
            manchester_template=t.get("manchester_template", ""),
            slot_types=t.get("slot_types", {}),
            is_complex=t.get("is_complex", False),
            verbal_template=t.get("verbal_template", ""),
            bfo_anchor_path=t.get("bfo_anchor_path", []),
        )
        st = template_to_table(ct, t.get("_family", ""))
        spine.append(st)
        by_id[t["template_id"]] = st

    fks_by_src: dict[str, list] = {}
    if family_complex is not None:
        edges, _ = cross_family_fks(spine, family_complex)
        for e in edges:
            fks_by_src.setdefault(e.src_table, []).append(e)

    lines = [
        "Each axiom below is paired with the RELATIONAL SCHEMA it projects to "
        "(deterministically derived from its slot structure). Populate THESE exact tables "
        "with realistic, axiom-consistent rows grounded in the style passages — do not "
        "invent different columns. Column types follow the slot types (Class/Individual = "
        "entity reference, DataProperty = typed literal); FOREIGN KEY constraints are the "
        "sanctioned cross-table joins. Before each table, explain in one or two sentences "
        "which axiom it embodies and what its primary-key / foreign-key structure means.",
        "",
        "After the chapter body, append a single fenced ```json code block giving the rows "
        "you placed in each table, as: {\"tables\": [{\"name\": \"<exact table name above>\", "
        "\"rows\": [[<one value per column in the schema's column order, including id>], ...]}]}. "
        "This lets the populated relational structure be verified against the deterministic "
        "schema (column slot-types and foreign-key validity).",
        "",
    ]
    for i, t in enumerate(templates, 1):
        st = by_id[t["template_id"]]
        slots_fmt = ", ".join(f"{k}: {v}" for k, v in t.get("slot_types", {}).items())
        lines.append(f"  AXIOM {i} (template_id: {t['template_id']}):")
        lines.append(f"    Manchester:   {t.get('manchester_template','')}")
        _verbal = pick_verbalization(t, rng)
        if _verbal:
            lines.append(f"    Verbalization: {_verbal}")
        lines.append(f"    Slot types:    {slots_fmt}")
        lines.append("    RELATIONAL SCHEMA:")
        lines.append("      " + render_ddl(st, fks_by_src.get(st.table.name, [])).replace("\n", "\n      "))
        lines.append("")
    return "\n".join(lines)


def _md_table(colnames: list[str], rows: list[list[str]], max_rows: int = 12) -> str:
    """Render rows as a GitHub markdown table (pipes in values escaped)."""
    def esc(v) -> str:
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(colnames) + " |"
    sep = "| " + " | ".join("---" for _ in colnames) + " |"
    body = "\n".join("| " + " | ".join(esc(v) for v in r) + " |" for r in rows[:max_rows])
    return "\n".join([head, sep, body])


def serialize_payload_footer(payload) -> str:
    """The exact verifiable JSON footer the chapter must echo — the authoritative (RI-true) base
    tables in the existing ``{"tables":[{"name","rows":[[…]]}]}`` shape (column order incl. id)."""
    tables = [{"name": st.table.name, "rows": [list(r) for r in st.table.rows]}
              for st in payload.base_tables]
    return json.dumps({"tables": tables}, separators=(",", ":"))


def build_axiom_section_with_tables(templates: list[dict], payload, rng=None) -> str:
    """Axiom section with the **fixed, populated** tables injected (Track A correct-by-construction).

    Each axiom carries its CREATE TABLE *and* the authoritative, referentially-consistent, typed,
    ontology-grounded rows — so the model writes prose AROUND a given relational structure rather than
    inventing values (which left RI at ~0.3–0.7). The cross-table views give the joins to narrate; the
    verbatim JSON footer makes RI = 1.0 an audited property of the chapter. The no-schema arm
    (plain ``build_axiom_section``) still lacks all of this — the load-bearing contrast is preserved.
    """
    from aegir.ontology.ddl import render_ddl
    from aegir.ontology.presentation import present_table  # prose render register (≠ schema identity)

    base_by_ref = {st.table.ref: st for st in payload.base_tables}
    fks_by_src: dict[str, list] = {}
    for e in payload.fks:
        fks_by_src.setdefault(e.src_table, []).append(e)

    lines = [
        "Each axiom below is paired with its RELATIONAL SCHEMA and the EXACT, AUTHORITATIVE ROWS of that "
        "table — already materialized: referentially consistent (foreign keys point at real primary keys), "
        "correctly typed, and grounded in the ontology. DO NOT invent, alter, renumber, drop, or add rows "
        "or columns. Write the chapter's prose, definitions, and worked examples AROUND these fixed tables: "
        "introduce each table, explain what its primary-key / foreign-key structure means, narrate at least "
        "one cross-table join using the ACTUAL key values shown, and interpret the values. Reproduce each "
        "table verbatim (as a markdown table) where you discuss it.",
        "",
    ]
    for idx, t in enumerate(templates, 1):
        st = base_by_ref.get(t["template_id"])
        slots_fmt = ", ".join(f"{k}: {v}" for k, v in t.get("slot_types", {}).items())
        lines.append(f"  AXIOM {idx} (template_id: {t['template_id']}):")
        lines.append(f"    Manchester:   {t.get('manchester_template','')}")
        _verbal = pick_verbalization(t, rng)
        if _verbal:
            lines.append(f"    Verbalization: {_verbal}")
        lines.append(f"    Slot types:    {slots_fmt}")
        if st is not None:
            lines.append("    RELATIONAL SCHEMA:")
            lines.append("      " + render_ddl(st, fks_by_src.get(st.table.name, [])).replace("\n", "\n      "))
            # PROSE render register: prettify headers + demote a leading surrogate id (schema/footer
            # keep physical snake_case identity — presentation ≠ identifier). Values untouched → RI +
            # table_fidelity preserved.
            colnames = [c.name for c in st.table.columns]
            slot_refs = [c.slot_ref for c in st.table.columns]
            disp_cols, disp_rows = present_table(colnames, st.table.rows, slot_refs=slot_refs)
            lines.append("    AUTHORITATIVE ROWS (reproduce verbatim):")
            lines.append("      " + _md_table(disp_cols, disp_rows).replace("\n", "\n      "))
        lines.append("")
    joins = [(v, r) for v, r in payload.views if v.fk is not None]
    if joins:
        lines.append("CROSS-TABLE VIEWS (narrate at least one of these joins using the real keys shown):")
        for v, vrows in joins[:4]:
            lines.append(f"  VIEW {v.name}: {v.verbalization}")
            vcols, vdisp = present_table([vc for vc, _, _ in v.columns], vrows)
            lines.append("    " + _md_table(vcols, vdisp).replace("\n", "\n    "))
        lines.append("")
    lines.append("AFTER the chapter body, append EXACTLY this fenced JSON block verbatim — it records the "
                 "authoritative rows you wrote prose around (do not modify it):")
    lines.append("```json")
    lines.append(serialize_payload_footer(payload))
    lines.append("```")
    return "\n".join(lines)


def build_prompt(anchors: list[dict], templates: list[dict], kind: str,
                 family_complex=None, *, seed: int = 0, realize_schemas: bool = False,
                 naming: str = "natural") -> tuple:
    """Build the generation prompt; returns ``(prompt, payload | None)``.

    glm/grok get the **fixed populated tables** injected (Track A: load-bearing ontology + RI=1.0);
    the no-schema arm keeps plain-text axioms WITHOUT the DDL — the load-bearing contrast. The payload
    (or None) is returned so the caller can record the authoritative tables on the chapter row.
    """
    # natural naming pairs with the TOPICAL register (write about the subject, not the ontology); the
    # semantic variant keeps the ontology-discourse template. Both accumulate → two corpus surfaces.
    tpl = GLM_NATURAL_TEMPLATE if (naming == "natural" and kind == "glm") else PROMPT_BY_KIND[kind]
    payload = None
    # Per-chapter rng for verbalization-frame sampling (Comp 3): same seed as the chapter, so the choice
    # is deterministic/reproducible yet varies across chapters → corpus-level surface diversity.
    import random
    vrng = random.Random(seed)
    if kind in ("glm", "grok"):
        try:
            from aegir.ontology.chapter_tables import chapter_relational_payload
            payload = chapter_relational_payload(templates, family_complex, seed=seed,
                                                 realize=realize_schemas)
            if naming == "natural":  # C2.5 L2 — the canonical-deliverable natural physical names (RI-safe)
                from aegir.ontology.natural_naming import load_natural_names, natural_payload
                nn = load_natural_names()
                if nn:
                    payload = natural_payload(payload, nn)
            axiom_section = build_axiom_section_with_tables(templates, payload, rng=vrng)
        except Exception as exc:  # noqa: BLE001 — RI/lowering failure must not abort a long run
            logger.warning(f"  relational payload failed ({type(exc).__name__}: {str(exc)[:100]}); "
                           f"falling back to schema-only axioms")
            payload = None
            axiom_section = build_axiom_section_with_ddl(templates, family_complex, rng=vrng)
    else:
        axiom_section = build_axiom_section(templates, rng=vrng)
    # The Grok template doesn't substitute n_templates (it picks domain freely); GLM does.
    fmt_kwargs: dict[str, Any] = {
        "style_section": build_style_section(anchors),
        "axiom_section": axiom_section,
    }
    if "{n_templates}" in tpl:
        fmt_kwargs["n_templates"] = len(templates)
    return tpl.format(**fmt_kwargs), payload


# Approximate provider pricing ($/M tokens: input, output). ADJUST to current rates —
# litellm's own per-call cost is preferred at runtime when it knows the model.
RATES_USD_PER_M: dict[str, tuple[float, float]] = {
    "cerebras/zai-glm-4.7": (0.50, 1.50),
    "xai/grok-4.3": (3.00, 15.00),
}
_RATES_DEFAULT = (2.0, 10.0)


def resolve_local_model(model: str) -> str:
    """`local/<name>` → <name>; `local/auto` → the (single) model the endpoint serves."""
    name = model.split("/", 1)[1] if "/" in model else "auto"
    if name != "auto":
        return name
    import urllib.request
    base = os.environ.get("OPENAI_API_BASE", "http://localhost:8088/v1/").rstrip("/")
    with urllib.request.urlopen(f"{base}/models", timeout=10) as r:
        data = json.loads(r.read())
    return data["data"][0]["id"]


def usage_and_cost(lm, model: str) -> tuple[int, int, int, float]:
    """(prompt_tokens, completion_tokens, reasoning_tokens, cost_usd) for the last call,
    read from the dspy/litellm history. Cost = litellm's if known, else RATES_USD_PER_M."""
    try:
        h = lm.history[-1]
        u = h.get("usage") or {}
        get = u.get if isinstance(u, dict) else (lambda k, d=None: getattr(u, k, d))
        pt = int(get("prompt_tokens", 0) or 0)
        ct = int(get("completion_tokens", 0) or 0)
        det = get("completion_tokens_details", None)  # a wrapper object OR dict OR None
        rt = getattr(det, "reasoning_tokens", None)
        if rt is None and isinstance(det, dict):
            rt = det.get("reasoning_tokens", 0)
        rt = int(rt or 0)
        cost = h.get("cost")  # litellm's own cost when it knows the model (preferred)
        if not cost:
            # local/ and engine/ are LOCAL (gRPC engine) → $0; the remote-equivalent cost for scaling
            # extrapolation is computed separately (scripts/analyze_generation_stats.py), not charged here.
            rin, rout = (0.0, 0.0) if model.startswith(("local/", "engine/")) else \
                RATES_USD_PER_M.get(model, _RATES_DEFAULT)
            cost = pt / 1e6 * rin + ct / 1e6 * rout
        return pt, ct, rt, float(cost or 0.0)
    except Exception:
        return 0, 0, 0, 0.0


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def compute_chapter_id(prompt: str, model: str, seed_offset: int) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(str(seed_offset).encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    catalog_dir = REPO / args.catalog_dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    content_first = bool(args.from_harvest)
    sampler_tag = "content-first" if content_first else "topic-first"

    # Inputs
    restrict_families = (
        set(s.strip() for s in args.restrict_families.split(","))
        if args.restrict_families else None
    )
    all_templates = load_all_templates(catalog_dir, restrict_families)
    logger.info(
        f"loaded {len(all_templates)} templates across "
        f"{len({t['_family'] for t in all_templates.values()})} "
        f"families (restrict={restrict_families or 'all'})"
    )

    harvest_docs: list[dict] = []
    derived_pool: list[dict] = []
    topic_cov = None
    if content_first:
        # CONTENT-FIRST: one chapter per harvested in-domain doc, grounded in the DERIVED ontology.
        store = Path(args.from_harvest)
        if not store.is_absolute():
            store = REPO / store
        audit_run = store                       # stand-in: row/HX record `audit_run.name` = the harvest store
        harvest_docs = load_harvest_docs(store)
        derived_pool = [t for t in all_templates.values() if t["_family"] == "catalog"] \
            or list(all_templates.values())
        if not harvest_docs:
            raise SystemExit(f"no harvested docs in {store} — run harvest_domain_docs first")
        logger.info(f"content-first: {len(harvest_docs)} harvested docs · "
                    f"grounding pool = {len(derived_pool)} derived primitives")
    else:
        if not args.audit_run:
            raise SystemExit("--audit-run is required for topic-first generation (or pass --from-harvest)")
        audit_run = Path(args.audit_run)
        topic_cov, _template_density, _family_density = load_audit(audit_run)
        eligible_topics = int((topic_cov.coverage_score >= args.tau_anchor).sum())
        logger.info(
            f"loaded audit run from {audit_run}: "
            f"{eligible_topics}/{len(topic_cov)} topics eligible at "
            f"tau_anchor={args.tau_anchor}"
        )

    # Load the family complex if provided. Empty string disables the guard.
    family_complex = None
    if args.family_complex:
        from aegir.ontology.complex import FamilyComplex
        complex_path = Path(args.family_complex)
        if not complex_path.is_absolute():
            complex_path = REPO / complex_path
        if complex_path.exists():
            family_complex = FamilyComplex.from_json(complex_path)
            logger.info(
                f"family complex: {len(family_complex.maximal_simplices)} "
                f"maximal simplices, {len(family_complex.measured_below_floor)} "
                f"punctures, vertices={sorted(family_complex.vertices)}"
            )
        else:
            logger.warning(
                f"family-complex file not found: {complex_path} — "
                f"running without simplex guard"
            )

    # Topic-usage tracker drives the anti-repetition weighting across a run.
    topic_usage: dict[int, int] = {}

    # Lazy LLM import
    import dspy
    from gaius.hx.exchange import ExchangeRecord
    from aegir.hx import append_exchange

    # Parse the mix; build a per-model dspy.LM client cache (instantiating
    # is cheap but caching avoids repeated env-var lookups).
    mix = parse_mix(args.mix)
    logger.info(f"model mix: {mix}")

    provider_keys = {
        "cerebras": os.environ.get("CEREBRAS_API_KEY"),
        "xai":      os.environ.get("XAI_API_KEY"),
    }
    lm_cache: dict = {}

    def get_lm(model: str):
        if model in lm_cache:
            return lm_cache[model]
        provider = model.split("/", 1)[0]
        if provider == "engine":
            # Strict layering: reach the model through the Aegir capability/gRPC engine (the SOLE vLLM
            # client), NOT vLLM directly. `engine/<capability>` (e.g. engine/instruct). The complete
            # thinking trace is retained (reasoning_content → response_reasoning + raw.exchange); local
            # ⇒ cost $0 against the budget cap. (Adapter is duck-typed to dspy.LM — see engine/dspy_lm.)
            from aegir.engine.dspy_lm import EngineLM
            cap = model.split("/", 1)[1] if "/" in model else "instruct"
            lm = EngineLM(capability=cap, max_tokens=args.max_tokens, temperature=args.temperature)
            lm_cache[model] = lm
            return lm
        if provider == "local":
            # Self-hosted OpenAI-compatible endpoint (vLLM). `local/auto` resolves the
            # served model from the endpoint; cost = $0 against the budget cap.
            lm = dspy.LM(
                model=f"openai/{resolve_local_model(model)}", api_key="local",
                api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:8088/v1/").rstrip("/"),
                max_tokens=args.max_tokens, temperature=args.temperature,
                cache=False, timeout=300, num_retries=2,
            )
            lm_cache[model] = lm
            return lm
        api_key = provider_keys.get(provider)
        if not api_key:
            raise SystemExit(
                f"missing API key for provider {provider!r} "
                f"(model {model!r}); set "
                f"{provider.upper()}_API_KEY"
            )
        lm = dspy.LM(
            model=model, api_key=api_key,
            max_tokens=args.max_tokens, temperature=args.temperature,
            cache=False,  # corpus build: fresh generations + real per-call usage (no disk-cache hits)
            timeout=180, num_retries=2,  # bound stalled calls; ride out transient API blips
        )
        lm_cache[model] = lm
        return lm

    # Output schema for the chapters table. With topic-first sampling, the
    # `family` column is the modal family across selected templates (still
    # a useful single-value tag for grouping/filtering); the new
    # `template_families` column carries the full list when chapters span
    # multiple families.
    chapter_schema = pa.schema([
        ("chapter_id", pa.string()),
        ("family", pa.string()),
        ("template_families", pa.list_(pa.string())),
        ("target_topic_id", pa.int32()),
        ("ablation", pa.string()),
        ("template_ids", pa.list_(pa.string())),
        ("style_topic_ids", pa.list_(pa.int32())),
        ("model", pa.string()),
        ("prompt_kind", pa.string()),
        ("temperature", pa.float32()),
        ("seed", pa.int32()),
        ("prompt_chars", pa.int32()),
        ("response_chars", pa.int32()),
        ("response_text", pa.string()),
        ("response_reasoning", pa.string()),
        ("hx_exchange_id", pa.string()),
        ("latency_ms", pa.int32()),
        ("prompt_tokens", pa.int32()),
        ("completion_tokens", pa.int32()),
        ("reasoning_tokens", pa.int32()),
        ("cost_usd", pa.float32()),
        ("finish_reason", pa.string()),   # "stop" (trace complete) vs "length" (truncated)
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("audit_run_id", pa.string()),
        # Track A: the authoritative (RI-true) fixed tables this chapter was written around.
        ("rel_tables_json", pa.string()),
        ("rel_views_json", pa.string()),
        ("rel_fks_json", pa.string()),
        ("ri_ok", pa.bool_()),
    ])
    chapter_rows = []
    cum_cost = 0.0
    mix_seed_rng = np.random.default_rng(args.seed)

    # Run id (encodes sampler+restriction) computed up front so the parquet can be
    # checkpointed periodically — a transient failure or budget stop never loses a long run.
    run_id_inputs = (
        f"{sampler_tag}:{args.restrict_families or 'all'}:{args.n_chapters}:"
        f"{args.seed}:{args.mix}:{args.ablation}:"
        f"tau_anchor={args.tau_anchor}:tau_template={args.tau_template}:"
        f"fam_div={args.prefer_family_diverse}"
        + (f":harvest={Path(args.from_harvest).name}" if content_first else "")
    )
    run_id = hashlib.sha256(run_id_inputs.encode()).hexdigest()[:16]
    out_run_dir = output_dir / run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)

    def flush_chapters():
        if chapter_rows:
            pq.write_table(
                pa.Table.from_pylist(chapter_rows, schema=chapter_schema),
                out_run_dir / "chapters.parquet", compression="zstd",
            )

    # content-first generates exactly one chapter per harvested doc; topic-first uses --n-chapters
    n_to_generate = min(args.n_chapters, len(harvest_docs)) if content_first else args.n_chapters
    for i in range(n_to_generate):
        seed_offset = i
        rng_local = np.random.default_rng(args.seed + seed_offset)

        # Pick model from the weighted mix for this chapter
        model = select_model(mix, mix_seed_rng)
        provider = model.split("/", 1)[0]
        kind = prompt_kind_for_ablation(args.ablation, model)

        if content_first:
            # CONTENT-FIRST: this doc is the content/style anchor; grounded in derived primitives.
            target, anchors, chosen = select_content_first(i, harvest_docs, derived_pool,
                                                           args.templates_per_chapter)
        else:
            # Topic-first: pick the target topic + style siblings via anti-rep weighting on the
            # eligible (>= tau_anchor) topic pool; updates topic_usage in place.
            target, anchors = pick_topic_and_anchors(
                topic_cov, topic_usage, args.style_anchors, rng_local,
                tau_anchor=args.tau_anchor,
            )
            chosen = pick_topic_templates(
                target, all_templates,
                k=args.templates_per_chapter,
                tau_template=args.tau_template,
                prefer_family_diverse=args.prefer_family_diverse,
                restrict_families=restrict_families,
            )
            if not chosen:
                logger.warning(
                    f"chapter {i+1}/{n_to_generate}: target topic "
                    f"{int(target.topic_id)} has no templates above "
                    f"tau_template={args.tau_template}; redrawing"
                )
                continue

        # Family-complex guard (topic-first only — the derived family isn't in the empirical complex):
        # if the chosen family-set isn't allowed, filter templates to the largest allowed face.
        if family_complex is not None and not content_first:
            current_fams = frozenset(t["_family"] for t in chosen)
            allowed_fams = family_complex.best_face(current_fams)
            if allowed_fams != current_fams:
                dropped = current_fams - allowed_fams
                chosen = [t for t in chosen if t["_family"] in allowed_fams]
                logger.info(
                    f"  simplex-guard: filtered {sorted(current_fams)} → "
                    f"{sorted(allowed_fams)} (dropped families: "
                    f"{sorted(dropped)})"
                )
            if not chosen:
                logger.warning(
                    f"chapter {i+1}/{args.n_chapters}: no allowable face "
                    f"for target topic {int(target.topic_id)} families "
                    f"{sorted(current_fams)}; redrawing"
                )
                continue

        template_families = sorted({t["_family"] for t in chosen})
        # Modal family — most common across the chosen templates; ties
        # broken by sort order. Used as the chapter's `family` tag.
        primary_family = Counter(t["_family"] for t in chosen).most_common(1)[0][0]

        prompt, rel_payload = build_prompt(anchors, chosen, kind=kind, family_complex=family_complex,
                                           seed=args.seed + seed_offset,
                                           realize_schemas=args.realize_schemas, naming=args.naming)
        chapter_id = compute_chapter_id(prompt, model, seed_offset)

        logger.info(f"chapter {i+1}/{args.n_chapters} (id={chapter_id}):")
        logger.info(f"  model: {model}  (prompt kind: {kind})")
        logger.info(f"  target topic: {int(target.topic_id)} "
                    f"(coverage_score={float(target.coverage_score):.3f})")
        logger.info(f"  templates: {[t['template_id'] for t in chosen]}")
        logger.info(f"  template families: {template_families}  "
                    f"(primary={primary_family})")
        logger.info(f"  style anchor topics: {[a['topic_id'] for a in anchors]}")

        # Generate via the appropriate model
        lm = get_lm(model)
        t0 = time.time()
        try:
            result = lm(messages=[{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 — a transient API error must not abort a long run
            logger.warning(f"chapter {i+1}/{args.n_chapters}: generation failed "
                           f"({type(exc).__name__}: {str(exc)[:120]}); skipping")
            continue
        latency_ms = int((time.time() - t0) * 1000)

        if isinstance(result, list):
            item = result[0]
            response_text = item.get("text") or ""
            response_reasoning = item.get("reasoning_content")
        else:
            response_text = str(result)
            response_reasoning = None

        logger.info(f"  latency: {latency_ms}ms, "
                    f"response: {len(response_text)} chars, "
                    f"reasoning: {len(response_reasoning or '')} chars")
        pt, ct, rt, cost = usage_and_cost(lm, model)
        cum_cost += cost
        # finish_reason persisted so trace COMPLETENESS is provable across the run ("length" ⇒ the
        # thinking trace was truncated — incomplete, and may bleed into the answer; raise --max-tokens).
        _hist = lm.history[-1] if getattr(lm, "history", None) else {}
        finish_reason = _hist.get("finish_reason", "") if isinstance(_hist, dict) else ""
        logger.info(f"  tokens: {pt} in / {ct} out ({rt} reasoning) | finish={finish_reason or '?'} | "
                    f"cost ${cost:.3f} | cumulative ${cum_cost:.2f}"
                    + (f" / ${args.budget_usd:.0f}" if args.budget_usd else ""))

        # Capture in HX. provider tag matches what's in the model id.
        record = ExchangeRecord(
            provider=provider,
            request_messages=[{"role": "user", "content": prompt}],
            request_model=model,
            request_params={"max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "seed": args.seed + seed_offset},
            response_content=response_text,
            response_model=model.split("/")[-1],
            response_reasoning=response_reasoning,
            latency_ms=latency_ms,
            source_context={
                "aegir_module": "generate_chapter",
                "chapter_id": chapter_id,
                "family": primary_family,
                "template_families": ";".join(template_families),
                "target_topic_id": str(int(target.topic_id)),
                "prompt_kind": kind,
                "ablation": args.ablation,
                "template_ids": [t["template_id"] for t in chosen],
                "style_topic_ids": [str(a["topic_id"]) for a in anchors],
                "audit_run_id": audit_run.name,
                "sampler": sampler_tag,
                "harvest_doc": (anchors[0].get("hash") if content_first else None),
            },
        )
        try:
            append_exchange(record)
        except Exception as exc:  # noqa: BLE001 — HX capture is best-effort; never lose the chapter
            logger.warning(f"  HX capture failed ({type(exc).__name__}); continuing")
        hx_exchange_id = record.id

        # Track A: serialize the authoritative fixed tables / views / FKs (None for ablation arms).
        rel_tables_json = rel_views_json = rel_fks_json = None
        if rel_payload is not None:
            rel_tables_json = serialize_payload_footer(rel_payload)
            rel_views_json = json.dumps([
                {"view_name": v.name, "sql": v.sql, "n_rows": len(r),
                 "columns": [vc for vc, _, _ in v.columns], "rows": r}
                for v, r in rel_payload.views])
            rel_fks_json = json.dumps([
                {"src_table": e.src_table, "src_col": e.src_col, "dst_table": e.dst_table,
                 "dst_col": e.dst_col, "via_slot": e.via_slot} for e in rel_payload.fks])

        # Stage chapter row
        chapter_rows.append({
            "chapter_id": chapter_id,
            "family": primary_family,
            "template_families": template_families,
            "target_topic_id": int(target.topic_id),
            "ablation": args.ablation,
            "template_ids": [t["template_id"] for t in chosen],
            "style_topic_ids": [int(a["topic_id"]) for a in anchors],
            "model": model,
            "prompt_kind": kind,
            "temperature": float(args.temperature),
            "seed": int(args.seed + seed_offset),
            "prompt_chars": len(prompt),
            "response_chars": len(response_text),
            "response_text": response_text,
            "response_reasoning": response_reasoning or "",
            "hx_exchange_id": hx_exchange_id,
            "latency_ms": latency_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "reasoning_tokens": rt,
            "cost_usd": float(cost),
            "finish_reason": finish_reason,
            "created_at": datetime.now(timezone.utc),
            "audit_run_id": audit_run.name,
            "rel_tables_json": rel_tables_json,
            "rel_views_json": rel_views_json,
            "rel_fks_json": rel_fks_json,
            "ri_ok": rel_payload is not None,
        })

        if len(chapter_rows) % args.flush_every == 0:
            flush_chapters()
            logger.info(f"  [checkpoint] flushed {len(chapter_rows)} chapters to parquet")

        if args.budget_usd and cum_cost >= args.budget_usd:
            logger.info(f"budget reached: ${cum_cost:.2f} >= ${args.budget_usd:.2f} "
                        f"after {len(chapter_rows)} chapters — stopping run.")
            break

    # Final checkpoint (run_id / out_run_dir computed up front; flushed periodically above).
    flush_chapters()

    # Also dump each chapter as a standalone markdown file for human inspection
    for row in chapter_rows:
        (out_run_dir / f"chapter_{row['chapter_id']}.md").write_text(
            row["response_text"]
        )

    logger.info("")
    logger.info(f"DONE — wrote {len(chapter_rows)} chapters to {out_run_dir}")
    logger.info(f"  chapters.parquet, plus {len(chapter_rows)} *.md files")
    logger.info(f"  HX exchanges in raw.exchange (postgres-backed Iceberg)")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
