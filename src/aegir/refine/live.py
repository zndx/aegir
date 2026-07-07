"""inc-1 — `live_draft`: a real chapter construct carrying column→concept provenance.

Builds a construct (the loop's working unit) from the live ontology pipeline (`chapter_relational_payload`)
instead of the controlled inc-0 demonstrator. The audit's discarded provenance (`provenance:"{}"`) is recovered
here: each column's concept is `concept_of(ColumnSpec.slot_ref)` and the value-ontology is bootstrapped from
the admitted taxonomy. Cell `source` is set to the column concept for now — reconstructing the true source of a
*subtree-mixed* cell (to detect real mixing contamination, vs an injected one) needs the mixing pipeline to
retain per-cell provenance, which is the next sub-step.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

CATALOG_DIR = "src/aegir/ontology/catalog"
FAMILY_COMPLEX = "src/aegir/ontology/family_complex.json"

_GROUNDING: tuple[dict, dict] | None = None


def _load_coverage() -> list:
    """The FinePDFs coverage audit (topic_coverage.parquet): the topics + their template alignment + a real
    representative passage. Found via AEGIR_COVERAGE_RUN or the latest coverage_v* artifact."""
    import pyarrow.parquet as pq
    p = os.environ.get("AEGIR_COVERAGE_RUN")
    if not p or not Path(p).exists():
        cands = sorted(glob.glob("/raid/checkpoints/aegir-artifacts/coverage_v*/*/topic_coverage.parquet"),
                       key=lambda f: Path(f).stat().st_mtime)
        p = cands[-1] if cands else None
    return pq.read_table(p).to_pylist() if p and Path(p).exists() else []


def _grounding() -> tuple[dict, dict]:
    """FinePDFs grounding, cached per process: ``template_id → aligned topic_ids`` and ``topic_id → style anchor``
    (a real FinePDFs passage). Lets a refined chapter be GROUNDED in the topic its templates align to — the
    anchor steers the proposer's register + subject toward genuine corpus content (vs the house technical voice)."""
    global _GROUNDING
    if _GROUNDING is None:
        term_topics: dict = {}
        anchors: dict = {}
        for r in _load_coverage():
            tid = int(r.get("topic_id", -1))
            if tid < 0:
                continue
            anchors[tid] = (r.get("topic_repr_text") or "").strip()[:1400]
            tset = set()
            if r.get("top_template_id"):
                tset.add(str(r["top_template_id"]))
            for e in (r.get("top_templates") or []):
                t = e.get("template_id") if isinstance(e, dict) else e
                if t:
                    tset.add(str(t))
            for t in tset:
                term_topics.setdefault(t, []).append(tid)
        _GROUNDING = (term_topics, anchors)
    return _GROUNDING


def _load_templates(catalog_dir: str = CATALOG_DIR) -> dict:
    """Replicates generate_chapter.load_all_templates (family file stem → `_family`), without its heavy deps."""
    out: dict = {}
    from aegir.ontology.schema import catalog_files
    for p in catalog_files(catalog_dir):
        for t in json.loads(p.read_text()).get("templates", []):
            t = dict(t)
            t["_family"] = p.stem
            out[t["template_id"]] = t
    return out


def payload_to_construct(payload, *, template_id: str, prose: str = "") -> dict:
    from aegir.refine.value_ontology import concept_of, value_ontology_from_taxonomy
    fks_by_table: dict = {}
    for fk in payload.fks:
        fks_by_table.setdefault(fk.src_table, []).append(
            {"col": fk.src_col, "ref_table": fk.dst_table, "ref_col": fk.dst_col})
    tables = []
    for st in payload.base_tables:
        ts = st.table
        pk, cols = None, []
        for j, col in enumerate(ts.columns):
            is_pk = col.slot_ref == "__pk__"
            concept = "identifier" if is_pk else concept_of(col.slot_ref)
            if is_pk:
                pk = col.name
            cells = [{"value": str(row[j]), "source": concept} for row in ts.rows if j < len(row)]
            cols.append({"name": col.name, "concept": concept, "cells": cells})
        tables.append({"name": ts.name, "pk": pk, "fks": fks_by_table.get(ts.name, []), "columns": cols})
    return {"template_id": template_id, "prose": prose or "This chapter describes the dataset.",
            "tables": tables, "value_onto": value_ontology_from_taxonomy()}


def live_draft(*, n_templates: int = 2, family: str | None = None, seed: int = 0xAE61,
               realize: bool = False, offset: int = 0) -> dict:
    """One real chapter construct from the ontology pipeline (deterministic for a given seed/selection).
    ``offset`` windows the template list so a scale run iterates DISTINCT chapters (offset 0, n, 2n, …)."""
    from aegir.ontology.chapter_tables import chapter_relational_payload
    from aegir.ontology.complex import FamilyComplex
    tpls = _load_templates()
    cands = sorted((t for t in tpls.values() if family is None or t["_family"] == family),
                   key=lambda t: t["template_id"])
    # offset = chapter_index * n_templates (from run_scale). First pass: ordered contiguous windows cover every
    # template once. Beyond one full pass: seeded-random DISTINCT subsets — so a large run STACKS thousands of
    # unique chapters instead of wrapping back to duplicates (the old `offset %= len` just re-walked the windows).
    import random
    idx = offset // max(n_templates, 1)
    n_windows = max(len(cands) // max(n_templates, 1), 1)
    if not cands:
        chosen = []
    elif idx < n_windows:
        chosen = cands[idx * n_templates:(idx + 1) * n_templates] or cands[:n_templates]
    else:
        chosen = random.Random(seed + idx).sample(cands, min(n_templates, len(cands)))
    fc = FamilyComplex.load_optional(Path(FAMILY_COMPLEX))  # None post-retirement (no pre-wired gate)
    payload = chapter_relational_payload(chosen, fc, seed=seed, realize=realize)
    # UNIQUE per distinct template-set: with diverse sampling many chapters share a first template, so a bare
    # ch_live_<first> id collides and overwrites (~54% loss observed). A short hash of the sorted set keeps the
    # leading template readable while disambiguating distinct chapters; identical sets ARE the same chapter.
    import hashlib
    key = "|".join(sorted(t["template_id"] for t in chosen)) or "empty"
    tid = ("ch_live_" + (chosen[0]["template_id"] if chosen else "empty")
           + "_" + hashlib.sha1(key.encode()).hexdigest()[:6])
    ch = payload_to_construct(payload, template_id=tid)
    ch["template_ids"] = [t["template_id"] for t in chosen]    # lineup: chapter → terms
    ch["family"] = chosen[0]["_family"] if chosen else None
    term_topics, anchors = _grounding()                        # FinePDFs grounding: dominant aligned topic + anchor
    votes: dict[int, int] = {}
    for t in chosen:
        for tp in term_topics.get(t["template_id"], []):
            votes[tp] = votes.get(tp, 0) + 1
    if votes:
        dom = max(votes, key=lambda k: (votes[k], -k))
        ch["target_topic_id"] = dom
        ch["style_anchor"] = anchors.get(dom, "")
    return ch


if __name__ == "__main__":
    from aegir.refine import eval as ev
    ch = live_draft(n_templates=2)
    print(f"live_draft: {ch['template_id']} | {len(ch['tables'])} tables, "
          f"{sum(len(t['columns']) for t in ch['tables'])} columns, "
          f"{sum(len(c['cells']) for t in ch['tables'] for c in t['columns'])} cells")
    for t in ch["tables"]:
        print(f"  {t['name']} (pk={t['pk']}, fks={len(t['fks'])}): "
              f"{[(c['name'], c['concept']) for c in t['columns'][:4]]}")
    print("baseline metrics:", ev.measure(ch))
