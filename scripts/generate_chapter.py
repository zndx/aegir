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
    p.add_argument("--family", default="03_directive_governance",
                   help="Ontology family file stem (e.g., '03_directive_governance')")
    p.add_argument("--n-chapters", type=int, default=5)
    p.add_argument("--templates-per-chapter", type=int, default=4,
                   help="How many ontology templates to ground a chapter on")
    p.add_argument("--style-anchors", type=int, default=3,
                   help="Number of FinePDFs passages to include as style refs")
    p.add_argument("--audit-run", required=True,
                   help="Path to coverage_v0/<run_id>/ for topic_coverage + template_density")
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

def load_family_templates(catalog_dir: Path, family: str) -> list[dict]:
    """Load all templates from one family file."""
    path = catalog_dir / f"{family}.json"
    if not path.exists():
        raise SystemExit(f"family file not found: {path}")
    data = json.loads(path.read_text())
    templates = data.get("templates", [])
    for t in templates:
        t["_family"] = family
    return templates


def load_audit(audit_run: Path) -> tuple[Any, Any, Any]:
    """Load the three parquets from a coverage audit run."""
    topic_cov = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    template_density = pq.read_table(audit_run / "template_density.parquet").to_pandas()
    family_density = pq.read_table(audit_run / "family_density.parquet").to_pandas()
    return topic_cov, template_density, family_density


def pick_style_anchors(topic_cov, family: str, k: int,
                       rng: np.random.Generator) -> list[dict]:
    """Pick top-K FinePDFs passages whose nearest template is in this family."""
    candidates = topic_cov[topic_cov.top_family == family].sort_values(
        "coverage_score", ascending=False
    )
    if len(candidates) == 0:
        # Fall back: top-K topics overall by coverage_score
        candidates = topic_cov.sort_values("coverage_score", ascending=False)
    n_take = min(k, len(candidates))
    picks = candidates.head(max(n_take * 3, k))  # widen pool, then sample
    if len(picks) > n_take:
        idx = rng.choice(picks.index.to_numpy(), size=n_take, replace=False)
        picks = picks.loc[idx]
    return [
        {
            "topic_id": int(r.topic_id),
            "coverage_score": float(r.coverage_score),
            "top_template_id": r.top_template_id,
            "passage": r.topic_repr_text or "",
        }
        for _, r in picks.iterrows()
    ]


def pick_templates(templates: list[dict], template_density,
                   k: int, rng: np.random.Generator) -> list[dict]:
    """Pick K templates from the family, biased toward ones with topic coverage.

    Coverage-density-weighted sampling: templates that the audit shows
    actually match FinePDFs topics are preferred, since they're the
    ones whose generated content has the strongest style/substrate
    alignment opportunity.
    """
    td = template_density.set_index("template_id")
    # Score each template by its in_topk count + 1 (smoothing)
    ids = [t["template_id"] for t in templates]
    weights = np.array([
        td.loc[tid, "n_topics_in_topk"] + 1 if tid in td.index else 1
        for tid in ids
    ], dtype=float)
    weights /= weights.sum()
    n_take = min(k, len(templates))
    chosen_idx = rng.choice(len(templates), size=n_take, replace=False, p=weights)
    return [templates[i] for i in chosen_idx]


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

# Default prompt template per provider. Cerebras gets the GLM template
# (general textbook, native reasoning channel). xAI gets the schema-rich
# Grok template (LIMS-style cross-joinable tables, simulated <thinking>).
PROMPT_TEMPLATES = {
    "cerebras": "glm",
    "xai":      "grok",
}

PROMPT_BY_KIND = {
    "glm":  GLM_PROMPT_TEMPLATE,
    "grok": GROK_PROMPT_TEMPLATE,
}


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


def build_axiom_section(templates: list[dict]) -> str:
    lines = []
    for i, t in enumerate(templates, 1):
        manchester = t.get("manchester_template", "")
        verbal = t.get("verbal_template", "")
        slots = t.get("slot_types", {})
        slots_fmt = ", ".join(f"{k}: {v}" for k, v in slots.items())
        lines.append(f"  AXIOM {i} (template_id: {t['template_id']}):")
        lines.append(f"    Manchester:   {manchester}")
        if verbal:
            lines.append(f"    Verbalization: {verbal}")
        lines.append(f"    Slot types:    {slots_fmt}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(anchors: list[dict], templates: list[dict], kind: str) -> str:
    tpl = PROMPT_BY_KIND[kind]
    # The Grok template doesn't substitute n_templates (it picks domain
    # freely). The GLM template does.
    fmt_kwargs = {
        "style_section": build_style_section(anchors),
        "axiom_section": build_axiom_section(templates),
    }
    if "{n_templates}" in tpl:
        fmt_kwargs["n_templates"] = len(templates)
    return tpl.format(**fmt_kwargs)


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
    audit_run = Path(args.audit_run)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    templates = load_family_templates(catalog_dir, args.family)
    logger.info(f"loaded {len(templates)} templates from family={args.family}")
    topic_cov, template_density, family_density = load_audit(audit_run)
    logger.info(f"loaded audit run from {audit_run}")

    # Style anchors are picked once and reused across chapters
    # (consistent style; varied content via different template subsets)
    rng = np.random.default_rng(args.seed)

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
    lm_cache: dict[str, dspy.LM] = {}

    def get_lm(model: str) -> dspy.LM:
        if model in lm_cache:
            return lm_cache[model]
        provider = model.split("/", 1)[0]
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
        )
        lm_cache[model] = lm
        return lm

    # Output schema for the chapters table
    chapter_schema = pa.schema([
        ("chapter_id", pa.string()),
        ("family", pa.string()),
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
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("audit_run_id", pa.string()),
    ])
    chapter_rows = []
    mix_seed_rng = np.random.default_rng(args.seed)

    for i in range(args.n_chapters):
        seed_offset = i
        rng_local = np.random.default_rng(args.seed + seed_offset)

        # Pick model from the weighted mix for this chapter
        model = select_model(mix, mix_seed_rng)
        provider = model.split("/", 1)[0]
        kind = prompt_kind_for_model(model)

        anchors = pick_style_anchors(topic_cov, args.family,
                                      args.style_anchors, rng_local)
        chosen = pick_templates(templates, template_density,
                                args.templates_per_chapter, rng_local)
        prompt = build_prompt(anchors, chosen, kind=kind)
        chapter_id = compute_chapter_id(prompt, model, seed_offset)

        logger.info(f"chapter {i+1}/{args.n_chapters} (id={chapter_id}):")
        logger.info(f"  model: {model}  (prompt kind: {kind})")
        logger.info(f"  templates: {[t['template_id'] for t in chosen]}")
        logger.info(f"  style anchor topics: {[a['topic_id'] for a in anchors]}")

        # Generate via the appropriate model
        lm = get_lm(model)
        t0 = time.time()
        result = lm(messages=[{"role": "user", "content": prompt}])
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
                "family": args.family,
                "prompt_kind": kind,
                "template_ids": [t["template_id"] for t in chosen],
                "style_topic_ids": [str(a["topic_id"]) for a in anchors],
                "audit_run_id": audit_run.name,
            },
        )
        append_exchange(record)
        hx_exchange_id = record.id

        # Stage chapter row
        chapter_rows.append({
            "chapter_id": chapter_id,
            "family": args.family,
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
            "created_at": datetime.now(timezone.utc),
            "audit_run_id": audit_run.name,
        })

    # Write chapters parquet
    run_id = hashlib.sha256(
        f"{args.family}:{args.n_chapters}:{args.seed}:{args.mix}".encode()
    ).hexdigest()[:16]
    out_run_dir = output_dir / run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(chapter_rows, schema=chapter_schema),
        out_run_dir / "chapters.parquet",
        compression="zstd",
    )

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
    raise SystemExit(main())
