"""inc-0(d) refinement-loop metrics + a controlled ch0 demonstrator.

A chapter **construct** is a dict carrying the column→concept *provenance* the current single-shot pipeline
discards (`provenance:"{}"` — the audit's root cause), so the gates can actually run:

    {"template_id": str,
     "prose": str,
     "tables": [{"name": str,
                 "columns": [{"name": str, "concept": str,
                              "cells": [{"value": str, "source": str}]}],   # source = the value's true concept
                 "pk": str|None,
                 "fks": [{"col": str, "ref_table": str, "ref_col": str}]}],
     "value_onto": {"classes": [...], "subclass_of": [[sub,sup]...], "disjoint": [[a,b]...]}}

The five metrics map 1:1 to the audit findings + the plan's pass criterion. inc-1 retains this provenance in
real generation (via the scaffold tools) so the loop runs on the live corpus instead of this demonstrator.
"""
from __future__ import annotations

import re

PLACEHOLDER_TOKENS = {"tbd", "placeholder", "n/a", "na", "value", "xxx", "todo", "example", "lorem",
                      "foo", "bar", "...", "—", "-", "", "null", "none", "{x}", "{y}", "{z}"}


def _cells(construct):
    for t in construct.get("tables", []):
        for c in t.get("columns", []):
            for cell in c.get("cells", []):
                yield t, c, cell


def placeholder_rate(construct) -> float:
    cells = [cell for _, _, cell in _cells(construct)]
    if not cells:
        return 0.0
    n = sum(1 for cell in cells
            if re.sub(r"[\s_]", "", str(cell.get("value", "")).lower()) in PLACEHOLDER_TOKENS
            or re.fullmatch(r"\{[^}]*\}", str(cell.get("value", "")).strip() or "x"))
    return n / len(cells)


def disjointness_violations(construct) -> list[str]:
    """Value-level HermiT over each provenance-bearing column → the inadmissible cell values.

    A cell whose *source* concept is disjoint from its *column* concept (directly or inferred) is a violation
    — the audit's ASHRAE-in-imaging case. Reuses the validated ``value_gate`` (real reasoner)."""
    from aegir.refine.value_gate import check_chapter
    onto = construct.get("value_onto", {})
    kw = dict(classes=onto.get("classes", ()), subclass_of=onto.get("subclass_of", ()),
              disjoint=onto.get("disjoint", ()))
    columns = []
    for t in construct.get("tables", []):
        for col in t.get("columns", []):
            concept = col.get("concept")
            vs = [(str(cell["value"]), cell.get("source", concept)) for cell in col.get("cells", [])]
            if concept and vs:
                columns.append((f"{t['name']}.{col['name']}", concept, vs))
    return check_chapter(columns, **kw)["violations"] if columns else []


def ri_ok(construct) -> bool:
    """Every FK cell value must reference an existing PK value in the referenced table."""
    pk_values = {}
    for t in construct.get("tables", []):
        pk = t.get("pk")
        if pk:
            for col in t["columns"]:
                if col["name"] == pk:
                    pk_values[t["name"]] = {str(c["value"]) for c in col["cells"]}
    for t in construct.get("tables", []):
        bycol = {col["name"]: col for col in t["columns"]}
        for fk in t.get("fks", []):
            col = bycol.get(fk["col"])
            ref = pk_values.get(fk["ref_table"])
            if col is None or ref is None:
                return False
            if any(str(c["value"]) not in ref for c in col["cells"]):
                return False
    return True


def _structural_entities(construct) -> set[str]:
    """Table + column names — the SCHEMA the prose is expected to describe."""
    ents: set[str] = set()
    for t in construct.get("tables", []):
        ents.add(t["name"])
        for col in t["columns"]:
            ents.add(col["name"])
    return {e for e in ents if e and len(e) > 2}


def _value_sample(construct, *, per_col: int = 2) -> set[str]:
    """A capped SAMPLE of representative non-key cell values — so the prose grounds in real data without being
    required to name every one of a realized schema's hundreds of cells."""
    ents: set[str] = set()
    for t in construct.get("tables", []):
        fkcols = {fk["col"] for fk in t.get("fks", [])}
        for col in t["columns"]:
            if t.get("pk") == col["name"] or col["name"] in fkcols:
                continue
            seen = 0
            for cell in col["cells"]:
                v = str(cell["value"]).strip()
                if v and v.lower() not in PLACEHOLDER_TOKENS and not v.isdigit() and len(v) > 2:
                    ents.add(v)
                    seen += 1
                    if seen >= per_col:
                        break
    return ents


def _frac_mentioned(ents: set[str], prose: str) -> float | None:
    if not ents:
        return None
    m = sum(1 for e in ents if re.sub(r"_", " ", e.lower()) in prose or e.lower() in prose)
    return m / len(ents)


def _concepts(construct) -> set[str]:
    """The DOMAIN concepts a chapter teaches — humanized table subjects + column concepts, NOT the schema
    identifiers. Organic, concept-teaching prose mentions these; it need not enumerate the table/column NAMES
    (the requirement that forced the mechanical 'The X table has columns a, b, c…' readout). Off-topic or
    ungrounded prose still fails — it won't mention the domain concepts."""
    ents: set[str] = set()
    for t in construct.get("tables", []):
        subj = re.sub(r"^(t_|dim_|fact_|bridge_|ref_|stg_)", "", t.get("name", "")).replace("_", " ").strip()
        if len(subj) > 2:
            ents.add(subj)
        for col in t.get("columns", []):
            c = (col.get("concept") or "").replace("_", " ").strip()
            if len(c) > 2:
                ents.add(c)
    return ents


def prose_entailment(construct) -> float:
    """Prose↔data correspondence, REBALANCED for organic exposition. The earlier version weighted STRUCTURAL
    correspondence — naming the tables/columns — at 0.7, which forced a mechanical schema readout ("The X table
    has columns a, b, c… a foreign key into Y") and made the loop *select against* the organic, concept-teaching,
    FinePDFs-style prose the corpus exists to provide. Now reward teaching the DOMAIN CONCEPTS (0.6) and grounding
    in representative VALUES (0.4); the embedded tables carry the schema structure, so the prose need not enumerate
    it. Off-topic / ungrounded prose still fails (it won't mention the concepts or the values)."""
    prose = construct.get("prose", "").lower()
    c = _frac_mentioned(_concepts(construct), prose)
    v = _frac_mentioned(_value_sample(construct), prose)
    if c is None and v is None:
        return 1.0
    if c is None:
        return v if v is not None else 1.0
    if v is None:
        return c
    return 0.6 * c + 0.4 * v


def length_ok(construct, *, lo: int = 400, hi: int = 20000) -> bool:
    return lo <= len(construct.get("prose", "")) <= hi


def measure(construct) -> dict:
    return {
        "placeholder_rate": round(placeholder_rate(construct), 4),
        "disjointness_violations": disjointness_violations(construct),
        "ri_ok": ri_ok(construct),
        "prose_entailment": round(prose_entailment(construct), 4),
        "length_ok": length_ok(construct),
        "length": len(construct.get("prose", "")),
    }


def controlled_ch0() -> dict:
    """A single-shot ch0 carrying the audit's exact defects (+ the provenance to gate them):
    an ASHRAE/EPA HVAC contaminant in an imaging-protocol column, two placeholder cells, and thin prose that
    under-mentions the tables. Disjointness is stated at the PARENT level → the leaf violation is INFERRED."""
    return {
        "template_id": "ch0_imaging_study",
        "prose": "This chapter discusses imaging studies.",   # thin: under-entails, under length band
        "tables": [
            {"name": "imaging_study", "pk": "study_id",
             "fks": [{"col": "protocol_id", "ref_table": "imaging_protocol_ref", "ref_col": "protocol_id"}],
             "columns": [
                 {"name": "study_id", "concept": "identifier",
                  "cells": [{"value": "1", "source": "identifier"}, {"value": "2", "source": "identifier"},
                            {"value": "3", "source": "identifier"}]},
                 {"name": "protocol_id", "concept": "identifier",
                  "cells": [{"value": "p1", "source": "identifier"}, {"value": "p2", "source": "identifier"},
                            {"value": "p3", "source": "identifier"}]},
                 {"name": "modality", "concept": "imaging_protocol",
                  "cells": [{"value": "t2_weighted_fat_sat", "source": "imaging_protocol"},
                            {"value": "gradient_echo_t2", "source": "imaging_protocol"},
                            {"value": "ashrae_62_1", "source": "hvac_standard"}]},   # ← disjoint contaminant
                 {"name": "finding", "concept": "clinical_finding",
                  "cells": [{"value": "TBD", "source": "clinical_finding"},          # ← placeholder
                            {"value": "lesion_detected", "source": "clinical_finding"},
                            {"value": "{X}", "source": "clinical_finding"}]},        # ← placeholder
             ]},
            {"name": "imaging_protocol_ref", "pk": "protocol_id", "fks": [],
             "columns": [
                 {"name": "protocol_id", "concept": "identifier",
                  "cells": [{"value": "p1", "source": "identifier"}, {"value": "p2", "source": "identifier"},
                            {"value": "p3", "source": "identifier"}]},
                 {"name": "protocol_name", "concept": "imaging_protocol",
                  "cells": [{"value": "axial_t1", "source": "imaging_protocol"},
                            {"value": "coronal_stir", "source": "imaging_protocol"},
                            {"value": "sagittal_flair", "source": "imaging_protocol"}]},
             ]},
        ],
        "value_onto": {
            "classes": ["imaging_protocol", "hvac_standard", "clinical_descriptor", "facility_descriptor",
                        "identifier", "clinical_finding"],
            "subclass_of": [["imaging_protocol", "clinical_descriptor"],
                            ["clinical_finding", "clinical_descriptor"],
                            ["hvac_standard", "facility_descriptor"]],
            "disjoint": [["clinical_descriptor", "facility_descriptor"]],   # parent-level → leaf is inferred
        },
    }
