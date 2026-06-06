"""Binding: ``generate_fn`` → the GLM/Grok mix (``dspy.LM``), per-skill.

Wires the skills' injected ``generate_fn(skill_id, refs, evidence, repair)`` to the
same weighted ``dspy.LM`` mix ``generate_chapter.py`` uses (cerebras/zai-glm-4.7 +
xai/grok-4.3). For each skill it builds a JSON-demanding prompt, calls the sampled
model, and parses the response into the skill's payload:

  S2          → UnitSchema (tables + fk_edges)        — the exact, type-checkable schema
  S1/S4/S6    → (text, [Claim])                       — prose + grounded claims
  S3          → (mermaid, [(src,dst)], [Claim])       — diagram edges for cross-modal

Parsers are defensive: a malformed response degrades to an empty/!-grounded payload
so the engine's hard gates reject (and the repair loop retries) rather than crashing.
Requires CEREBRAS_API_KEY / XAI_API_KEY in env (as generate_chapter does).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from aegir.ontology.schema import Catalog
from aegir.ontology.skills.base import Claim
from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec, UnitSchema

_PROVIDER_ENV = {"cerebras": "CEREBRAS_API_KEY", "xai": "XAI_API_KEY"}


def parse_mix(spec: str) -> tuple[list[str], list[float]]:
    models, weights = [], []
    for tok in spec.split(","):
        name, _, w = tok.strip().rpartition(":")
        models.append(name)
        weights.append(float(w))
    s = sum(weights)
    return models, [w / s for w in weights]


def _json_block(text: str) -> dict | None:
    """Pull the first JSON object out of a response (fenced or bare)."""
    for pat in (r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


# ── prompts ──────────────────────────────────────────────────────────────────


def _ev_block(evidence) -> str:
    return "\n".join(f"- [{s.doc_id}] {s.text}" for s in evidence) or "(no evidence)"


def _slot_block(refs, catalog: Catalog) -> str:
    out = {}
    for r in refs:
        try:
            out[r] = catalog.by_id(r).slot_types
        except KeyError:
            out[r] = {}
    return json.dumps(out, indent=2)


def _prompt_s2(refs, evidence, catalog, repair) -> str:
    fix = ("\nThe previous attempt failed the structural type-check; emit columns "
           "whose values MATCH the declared slot type and add the missing FK edges.\n"
           if repair else "")
    return (
        "Construct relational tables that instantiate these OWL templates, grounded "
        "in the evidence.\n\nTemplates → typed slots:\n" + _slot_block(refs, catalog)
        + "\n\nEvidence (values must be realistic and consistent):\n" + _ev_block(evidence)
        + "\n\nRules: one table per template; columns = the template's NON-ObjectProperty "
        "slots (column slot_ref = slot name, slot_type = its OWL type); object-property "
        "slots become foreign-key edges between class columns; 3 realistic rows." + fix
        + '\n\nReturn ONLY a ```json fenced object:\n'
        '{"tables":[{"name":"...","ref":"<template_id>","columns":'
        '[{"name":"...","slot_type":"Class|DataProperty|...","slot_ref":"<slot>"}],'
        '"rows":[["..."]]}],"fk_edges":[{"src_table":"...","src_col":"...",'
        '"dst_table":"...","dst_col":"...","via_slot":"<object-property slot>"}]}'
    )


def _prompt_prose(skill_id, refs, evidence, catalog, repair) -> str:
    what = {"S1": "Explain these axioms in grounded prose",
            "S4": "Give a concrete worked example instantiating these axioms",
            "S6": "Cross-reference and relate these concepts"}[skill_id]
    fix = ("\nThe previous attempt asserted ungrounded claims; every claim's `source` "
           "MUST be one of the evidence ids above.\n" if repair else "")
    return (
        f"{what}, using ONLY the evidence.\n\nTemplates → typed slots:\n"
        + _slot_block(refs, catalog) + "\n\nEvidence:\n" + _ev_block(evidence) + fix
        + '\n\nReturn ONLY a ```json fenced object:\n'
        '{"text":"<prose>","claims":[{"text":"<assertion>","source":"<evidence id>"}]}'
    )


def _prompt_s3(refs, evidence, catalog, repair) -> str:
    return (
        "Draw an entity-relationship diagram (Mermaid) over these templates' classes.\n"
        "CRITICAL: every edge endpoint MUST be one of the templates' CLASS slot names "
        "EXACTLY as listed below — these are the table's column names, and each edge must "
        "match a real relation. Emit one edge per ObjectProperty slot, connecting the two "
        "class slots it relates (domain class slot → range class slot).\n\n"
        "Templates → typed slots:\n" + _slot_block(refs, catalog) + "\n\nEvidence:\n"
        + _ev_block(evidence)
        + '\n\nReturn ONLY a ```json fenced object (edges use the class slot names verbatim):\n'
        '{"mermaid":"graph TD; ...","edges":[["<class slot>","<class slot>"]],"claims":[]}'
    )


# ── parsers ──────────────────────────────────────────────────────────────────


def _parse_s2(text: str) -> UnitSchema:
    j = _json_block(text) or {}
    tables = []
    for t in j.get("tables", []):
        cols = [ColumnSpec(c.get("name", ""), c.get("slot_type", ""), c.get("slot_ref", ""))
                for c in t.get("columns", [])]
        rows = [[str(v) for v in row] for row in t.get("rows", [])]
        tables.append(TableSpec(t.get("name", ""), t.get("ref", ""), cols, rows))
    fks = [FKEdge(e.get("src_table", ""), e.get("src_col", ""), e.get("dst_table", ""),
                  e.get("dst_col", ""), e.get("via_slot", "")) for e in j.get("fk_edges", [])]
    return UnitSchema(tables, fks)


def _parse_prose(text: str):
    j = _json_block(text)
    if not j:
        return (text.strip()[:2000], [])     # degrade: prose with no grounded claims → fails no-drift
    claims = [Claim(c.get("text", ""), c.get("source")) for c in j.get("claims", [])]
    return (j.get("text", ""), claims)


def _parse_s3(text: str):
    j = _json_block(text) or {}
    edges = [(str(e[0]), str(e[1])) for e in j.get("edges", [])
             if isinstance(e, (list, tuple)) and len(e) >= 2]
    claims = [Claim(c.get("text", ""), c.get("source")) for c in j.get("claims", [])]
    return (j.get("mermaid", "graph TD"), edges, claims)


# ── factory ──────────────────────────────────────────────────────────────────


def make_generate_fn(catalog: Catalog,
                     mix: str = "cerebras/zai-glm-4.7:0.6,xai/grok-4.3:0.4",
                     max_tokens: int = 4096, temperature: float = 0.7, seed: int = 0):
    """Return a ``generate_fn(skill_id, refs, evidence, repair=False)`` bound to the mix."""
    import dspy

    models, weights = parse_mix(mix)
    cache: dict[str, "dspy.LM"] = {}
    rng = np.random.default_rng(seed)

    def get_lm(model: str):
        if model not in cache:
            provider = model.split("/", 1)[0]
            key = os.environ.get(_PROVIDER_ENV.get(provider, ""))
            if not key:
                raise RuntimeError(f"missing {_PROVIDER_ENV.get(provider)} for {model!r}")
            cache[model] = dspy.LM(model=model, api_key=key,
                                   max_tokens=max_tokens, temperature=temperature)
        return cache[model]

    def _call(prompt: str) -> str:
        model = models[int(rng.choice(len(models), p=weights))]
        r = get_lm(model)(messages=[{"role": "user", "content": prompt}])
        if isinstance(r, list) and r:
            item = r[0]
            return (item.get("text") or "") if isinstance(item, dict) else str(item)
        return str(r)

    def generate_fn(skill_id, refs, evidence, repair: bool = False):
        if skill_id == "S2":
            return _parse_s2(_call(_prompt_s2(refs, evidence, catalog, repair)))
        if skill_id in ("S1", "S4", "S6"):
            return _parse_prose(_call(_prompt_prose(skill_id, refs, evidence, catalog, repair)))
        if skill_id == "S3":
            return _parse_s3(_call(_prompt_s3(refs, evidence, catalog, repair)))
        return ("", [])

    return generate_fn
