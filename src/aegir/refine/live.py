"""inc-1 — `live_draft`: a real chapter construct carrying column→concept provenance.

Builds a construct (the loop's working unit) from the live ontology pipeline (`chapter_relational_payload`)
instead of the controlled inc-0 demonstrator. The audit's discarded provenance (`provenance:"{}"`) is recovered
here: each column's concept is `concept_of(ColumnSpec.slot_ref)` and the value-ontology is bootstrapped from
the admitted taxonomy. Cell `source` is set to the column concept for now — reconstructing the true source of a
*subtree-mixed* cell (to detect real mixing contamination, vs an injected one) needs the mixing pipeline to
retain per-cell provenance, which is the next sub-step.
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG_DIR = "src/aegir/ontology/catalog"
FAMILY_COMPLEX = "src/aegir/ontology/family_complex.json"


def _load_templates(catalog_dir: str = CATALOG_DIR) -> dict:
    """Replicates generate_chapter.load_all_templates (family file stem → `_family`), without its heavy deps."""
    out: dict = {}
    for p in sorted(Path(catalog_dir).glob("0*.json")):
        if "candidate" in p.name:
            continue
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
               realize: bool = False) -> dict:
    """One real chapter construct from the ontology pipeline (deterministic for a given seed/selection)."""
    from aegir.ontology.chapter_tables import chapter_relational_payload
    from aegir.ontology.complex import FamilyComplex
    tpls = _load_templates()
    cands = sorted((t for t in tpls.values() if family is None or t["_family"] == family),
                   key=lambda t: t["template_id"])
    chosen = cands[:n_templates]
    fc = FamilyComplex.from_json(Path(FAMILY_COMPLEX))
    payload = chapter_relational_payload(chosen, fc, seed=seed, realize=realize)
    tid = "ch_live_" + (chosen[0]["template_id"] if chosen else "empty")
    return payload_to_construct(payload, template_id=tid)


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
