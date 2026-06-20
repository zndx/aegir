#!/usr/bin/env python
"""Ontology structural-diversity audit — how much of the OWL/ODP/FHIR expressivity envelope do our
templates actually exercise? (Evidence-first, mirroring the DDL structural-complexity KPI.)

The DDL realizer can only ground structures the ONTOLOGY expresses, so the templates' axiom-pattern
diversity caps everything downstream. This scans each template's ``manchester_template`` + ``slot_types``
and tallies the OWL constructs used, then scores coverage of a TARGET ENVELOPE drawn from OWL2 + the public
Ontology Design Pattern catalog + FHIR RDF (CC0) — the patterns a real-world ontology uses that ours must
grow into. The gap it reveals is the headroom the dynamic primitive-generator (step 3) unlocks.

    uv run --no-sync python scripts/audit_ontology_structure.py [--catalog combined.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402

# Single-axiom OWL constructs, detectable in the Manchester skeleton.
_CONSTRUCTS = {
    "subclassof": re.compile(r"\bSubClassOf\b", re.I),
    "equivalentto (definition)": re.compile(r"\bEquivalentTo\b", re.I),
    "disjoint": re.compile(r"\bDisjoint(With|UnionOf)\b", re.I),
    "some (existential)": re.compile(r"\bsome\b", re.I),
    "only (universal)": re.compile(r"\bonly\b", re.I),
    "min": re.compile(r"\bmin\s+\d+", re.I),
    "max": re.compile(r"\bmax\s+\d+", re.I),
    "exactly": re.compile(r"\bexactly\s+\d+", re.I),
    "hasValue": re.compile(r"\bvalue\b", re.I),
    "and (conjunction)": re.compile(r"\band\b", re.I),
    "or (disjunction)": re.compile(r"\bor\b", re.I),
    "not (negation)": re.compile(r"\bnot\b", re.I),
    "property-chain (o)": re.compile(r"\s+o\s+|SubPropertyChain", re.I),
    "datatype-facet": re.compile(r"\[\s*(length|minLength|maxLength|pattern|totalDigits|>=|<=|<|>)", re.I),
}
# qualified cardinality = a cardinality word followed by N AND a class filler (vs bare number)
_QUALIFIED_CARD = re.compile(r"\b(min|max|exactly)\s+\d+\s+[\{A-Za-z]", re.I)

# Higher-order ODP / FHIR patterns a SINGLE-axiom subClassOf skeleton structurally cannot express today —
# the representational ceiling. (FHIR RDF analogues in parens; FHIR is CC0 — draw extensively.)
_TARGET_PATTERNS = [
    "equivalentto (definition)", "disjoint", "only (universal)", "qualified-cardinality",
    "and (conjunction)", "or (disjunction)", "not (negation)", "property-chain (o)", "datatype-facet",
    "value-partition (FHIR CodeableConcept)", "reified-n-ary-relation (FHIR Reference)",
    "choice-type union (FHIR value[x])", "composite-datatype (FHIR Quantity/Period)",
    "nested-backbone (FHIR BackboneElement)", "object-property-slot", "dataproperty-slot",
]


def audit(catalog_path: Path) -> dict:
    cat = load_catalog(catalog_path)
    templates = list(cat.templates)
    n = len(templates)
    construct_counts: Counter = Counter()
    qualified_card = 0
    slot_meta: Counter = Counter()
    n_restrictions, nesting_depths, multi_clause = [], [], 0
    by_family: dict[str, Counter] = {}

    for t in templates:
        m = t.manchester_template or ""
        fam = (t.bfo_anchor_path[-1] if t.bfo_anchor_path else "?")
        fc = by_family.setdefault(fam, Counter())
        present = set()
        for name, rx in _CONSTRUCTS.items():
            if rx.search(m):
                present.add(name)
                construct_counts[name] += 1
                fc[name] += 1
        if _QUALIFIED_CARD.search(m):
            qualified_card += 1
        for ot in (t.slot_types or {}).values():
            slot_meta[ot] += 1
        # structural metrics
        n_restrictions.append(len(re.findall(r"\b(some|only|min|max|exactly|value)\b", m, re.I)))
        nesting_depths.append(_max_paren_depth(m))
        if "," in m or " and " in m.lower():
            multi_clause += 1

    # envelope coverage: which TARGET patterns are exercised at all
    has = {
        "equivalentto (definition)": construct_counts["equivalentto (definition)"] > 0,
        "disjoint": construct_counts["disjoint"] > 0,
        "only (universal)": construct_counts["only (universal)"] > 0,
        "qualified-cardinality": qualified_card > 0,
        "and (conjunction)": construct_counts["and (conjunction)"] > 0,
        "or (disjunction)": construct_counts["or (disjunction)"] > 0,
        "not (negation)": construct_counts["not (negation)"] > 0,
        "property-chain (o)": construct_counts["property-chain (o)"] > 0,
        "datatype-facet": construct_counts["datatype-facet"] > 0,
        # higher-order patterns the single-axiom skeleton can't represent → False by construction today
        "value-partition (FHIR CodeableConcept)": False,
        "reified-n-ary-relation (FHIR Reference)": False,
        "choice-type union (FHIR value[x])": False,
        "composite-datatype (FHIR Quantity/Period)": False,
        "nested-backbone (FHIR BackboneElement)": False,
        "object-property-slot": slot_meta.get("ObjectProperty", 0) > 0,
        "dataproperty-slot": slot_meta.get("DataProperty", 0) > 0,
    }
    covered = sum(1 for p in _TARGET_PATTERNS if has.get(p))
    return {
        "catalog": str(catalog_path), "n_templates": n,
        "construct_counts": dict(construct_counts.most_common()),
        "qualified_cardinality": qualified_card,
        "slot_meta_type_distribution": dict(slot_meta.most_common()),
        "restrictions_per_template": {
            "mean": round(sum(n_restrictions) / n, 2), "max": max(n_restrictions), "zero": n_restrictions.count(0)},
        "max_nesting_depth": max(nesting_depths), "multi_clause_templates": multi_clause,
        "envelope_target_patterns": len(_TARGET_PATTERNS),
        "envelope_covered": covered,
        "envelope_coverage": round(covered / len(_TARGET_PATTERNS), 3),
        "envelope_missing": [p for p in _TARGET_PATTERNS if not has.get(p)],
        "per_family_constructs": {f: dict(c.most_common(6)) for f, c in sorted(by_family.items())},
    }


def _max_paren_depth(s: str) -> int:
    depth = mx = 0
    for ch in s:
        if ch == "(":
            depth += 1
            mx = max(mx, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return mx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = audit(Path(args.catalog))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))

    print(f"ONTOLOGY STRUCTURAL DIVERSITY — {rep['n_templates']} templates")
    print(f"  OWL constructs exercised:")
    for k, v in rep["construct_counts"].items():
        print(f"    {v:>4}  {k}")
    print(f"  qualified cardinality: {rep['qualified_cardinality']}   "
          f"slot meta-types: {rep['slot_meta_type_distribution']}")
    print(f"  restrictions/template: mean {rep['restrictions_per_template']['mean']} "
          f"max {rep['restrictions_per_template']['max']} (zero-restriction: {rep['restrictions_per_template']['zero']})  "
          f"· max nesting {rep['max_nesting_depth']} · multi-clause {rep['multi_clause_templates']}")
    print(f"\n  EXPRESSIVITY ENVELOPE: {rep['envelope_covered']}/{rep['envelope_target_patterns']} "
          f"patterns = {rep['envelope_coverage']:.0%} covered")
    print(f"  MISSING (the headroom the primitive-generator unlocks):")
    for p in rep["envelope_missing"]:
        print(f"    ✘ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
