"""Axiom-pattern library — the ``realize.py`` of the ONTOLOGY layer.

Clean-room, parameterized OWL/Manchester axiom-pattern primitives, BFO/CCO-anchored, that **level up** the
prototype templates (today: ``subClassOf … some``, 44% of the expressivity envelope — see
``scripts/audit_ontology_structure.py``) toward real-world ontology richness. Tiering:

  * ``fhir_minimum`` — FHIR RDF patterns (CC0); FHIR sets the MINIMUM bar (Reference→junction,
    value[x]→EAV, CodeableConcept→value-partition, Quantity→composite datatype, BackboneElement→nested).
  * ``owl2_core``    — the OWL2 axiom envelope (definition/disjoint/QCR/property-chain/datatype-facet/…).
  * ``odp``          — public Ontology Design Patterns (participation, part-whole, sequence, …).
  * ``sysmlv2``      — SysML v2 / KerML constructs (parts/ports/connections/requirements/constraints/
    states/actions/verification) — massive LIMS value, levels UP from FHIR.

Each pattern carries the structure ``realize.py`` reads, so a generated primitive GROUNDS a concrete DDL
profile (``grounds_ddl``: junction / eav / composite-datatype / nested-child / enum / dimension /
constraint), closing the ontology↔DDL loop (no more heuristic synthesis).

**Patterns are DATA** (``axiom_patterns.json``, produced by the ``ontology-pattern-taxonomy`` workflow);
this module is the MACHINERY (load + validate + instantiate + API), mirroring catalog JSON ↔ ``schema.py``.

**Clean-room (load-bearing, see memory cleanroom_ddl_generation):** FHIR CC0 + BFO CC-BY + CCO BSD + ODP
public + SysMLv2 patterns-as-method. We copy NO proprietary terminology content (SNOMED CT / LOINC codes);
every identifier in a generated primitive is ontology/seed-derived original expression.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_PATTERNS_PATH = Path(__file__).resolve().parent / "axiom_patterns.json"
_SLOT_RE = re.compile(r"\{(\w+):([^}]+)\}")
TIERS = ("fhir_minimum", "owl2_core", "odp", "sysmlv2")
GROUNDS = ("junction", "eav", "composite-datatype", "nested-child", "enum", "dimension",
           "constraint", "subsumption", "relation")
# Clean-room tripwire: proprietary terminology codes / foreign-schema identifiers must never appear.
_FORBIDDEN = re.compile(r"\bsnomed\b|\bloinc\b|catalog_product|\bir_model|res_partner|analysisrequest", re.I)


@dataclass
class AxiomPattern:
    """One parameterized axiom-pattern primitive. ``manchester_skeleton`` uses the slot DSL
    ``{name:Type}`` (Class / ObjectProperty / DataProperty / Individual / xsd:type) plus ``$DESIGN``
    parameters the step-3 generator fills (property IRI, anchor, datatype, cardinality)."""
    name: str
    tier: str
    origin: str
    owl2_construct: str
    manchester_skeleton: str
    bfo_cco_anchor: str
    grounds_ddl: str
    slot_meta_types: dict = field(default_factory=dict)
    verbalization_hint: str = ""
    cleanroom_note: str = ""
    extends_existing: bool = False
    pattern_kind: str = "class_template"   # class_template (instantiable) | property_axiom (supporting)

    def slots(self) -> "dict[str, str]":
        """Declared ``{name:Type}`` slots in the skeleton → {name: meta-type-head}."""
        return {m.group(1): m.group(2).split(":")[0] for m in _SLOT_RE.finditer(self.manchester_skeleton)}

    def design_params(self) -> "list[str]":
        """``$NAME`` design parameters the generator must fill (property IRI / anchor / datatype / N)."""
        return sorted(set(re.findall(r"\$[A-Z_][A-Z0-9_]*", self.manchester_skeleton)))


_LIBRARY: "dict[str, AxiomPattern] | None" = None


def library() -> "dict[str, AxiomPattern]":
    """Load + cache the pattern library from ``axiom_patterns.json`` (empty if absent)."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = {}
        try:
            data = json.loads(_PATTERNS_PATH.read_text())
        except (OSError, ValueError):
            data = {"patterns": []}
        for p in data.get("patterns", []):
            try:
                ap = AxiomPattern(
                    name=p["name"], tier=p.get("tier", "owl2_core"), origin=p.get("origin", ""),
                    owl2_construct=p.get("owl2_construct", ""), manchester_skeleton=p["manchester_skeleton"],
                    bfo_cco_anchor=p.get("bfo_cco_anchor", ""), grounds_ddl=p.get("grounds_ddl", ""),
                    slot_meta_types=p.get("slot_meta_types") or {}, verbalization_hint=p.get("verbalization_hint", ""),
                    cleanroom_note=p.get("cleanroom_note", ""), extends_existing=bool(p.get("extends_existing", False)),
                    pattern_kind=p.get("pattern_kind", "class_template"))
                _LIBRARY[ap.name] = ap
            except KeyError:
                continue
    return _LIBRARY


def by_tier(tier: str) -> "list[AxiomPattern]":
    return [p for p in library().values() if p.tier == tier]


def by_grounding(grounds: str) -> "list[AxiomPattern]":
    return [p for p in library().values() if p.grounds_ddl == grounds]


def validate(p: AxiomPattern) -> "list[str]":
    """Structural validation (Manchester well-formedness + slot/anchor consistency + clean-room). The
    DEEP gate (HermiT consistency + DeepOnto verbalizability + BFO ancestry) is the step-3 membrane."""
    errs: list[str] = []
    sk = p.manchester_skeleton
    if sk.count("{") != sk.count("}"):
        errs.append("unbalanced slot braces")
    if not re.search(r"\b(SubClassOf|EquivalentTo|DisjointWith|DisjointUnionOf|SubPropertyOf|"
                     r"SubPropertyChain|Characteristics)\b", sk):
        errs.append("no recognized axiom keyword")
    if p.tier not in TIERS:
        errs.append(f"unknown tier {p.tier!r}")
    if not p.bfo_cco_anchor:
        errs.append("missing BFO/CCO anchor")
    if _FORBIDDEN.search(sk) or _FORBIDDEN.search(p.bfo_cco_anchor):
        errs.append("clean-room: forbidden proprietary/foreign token")
    # every declared slot should have a meta-type in the DSL or slot_meta_types
    for name, mtype in p.slots().items():
        if mtype not in ("Class", "ObjectProperty", "DataProperty", "Individual") and not mtype.startswith("xsd"):
            errs.append(f"slot {name!r} has non-standard meta-type {mtype!r}")
    return errs


def validate_all() -> dict:
    lib = library()
    issues = {n: validate(p) for n, p in lib.items()}
    return {
        "n_patterns": len(lib),
        "n_valid": sum(1 for e in issues.values() if not e),
        "by_tier": {t: len(by_tier(t)) for t in TIERS},
        "by_grounding": {g: len(by_grounding(g)) for g in GROUNDS if by_grounding(g)},
        "extends_existing": sum(1 for p in lib.values() if p.extends_existing),
        "issues": {n: e for n, e in issues.items() if e},
    }


def instantiate(pattern_name: str, fillers: "dict[str, str]", *, template_id: str, family: str,
                bfo_anchor_path: "list[str]", verbal_template: str = "") -> dict:
    """Fill a pattern's ``$DESIGN`` params (proposed by the step-3 LLM, e.g. ``$PROP`` → ``sdg:hasComponent``)
    → a ``CatalogTemplate``-shaped dict (instance slots ``{X:Class}`` are kept for chapter-gen). The result
    is HermiT/DeepOnto-verified before catalog admission."""
    p = library()[pattern_name]
    manchester = p.manchester_skeleton
    for k, v in (fillers or {}).items():
        manchester = manchester.replace(k, v)
    slots = {m.group(1): m.group(2).split(":")[0] for m in _SLOT_RE.finditer(manchester)}
    return {
        "template_id": template_id, "manchester_template": manchester, "slot_types": slots,
        "is_complex": p.tier in ("sysmlv2", "odp") or len(slots) > 2,
        "verbal_template": verbal_template, "bfo_anchor_path": list(bfo_anchor_path),
        "_pattern": pattern_name, "_tier": p.tier, "_grounds_ddl": p.grounds_ddl,
        "_design_params_unfilled": [d for d in p.design_params() if d in manchester],
    }


# audit-envelope patterns this library is meant to close (see audit_ontology_structure._TARGET_PATTERNS)
_ENVELOPE_GROUNDINGS = {
    "junction": "reified-n-ary-relation (FHIR Reference)", "eav": "choice-type union (FHIR value[x])",
    "composite-datatype": "composite-datatype (FHIR Quantity/Period)", "nested-child": "nested-backbone (FHIR BackboneElement)",
    "enum": "value-partition (FHIR CodeableConcept)",
}


def coverage_vs_envelope() -> dict:
    """Which audit-missing expressivity patterns the library now covers (the KPI it drives up)."""
    grounded = {g for g in _ENVELOPE_GROUNDINGS if by_grounding(g)}
    # scan both the construct text and the pattern name (constructs use varied wording, e.g. SubPropertyChain)
    blob = " ".join(f"{p.name} {p.owl2_construct}".lower() for p in library().values())
    owl = {
        "disjoint": "disjoint" in blob,
        "property-chain": "chain" in blob,
        "datatype-facet": "facet" in blob or "datatyperestriction" in blob,
        "qualified-cardinality": "qualified" in blob or "qualifiedcardinality" in blob or "exactly" in blob,
        "negation": "negation" in blob or "complement" in blob,
        "property-characteristics": "transitive" in blob or "characteristic" in blob,
    }
    return {"ddl_groundings_covered": sorted(_ENVELOPE_GROUNDINGS[g] for g in grounded),
            "owl2_constructs_present": {k: v for k, v in owl.items()},
            "n_patterns": len(library())}


if __name__ == "__main__":  # smoke: load + validate the committed library
    rep = validate_all()
    print(json.dumps(rep, indent=2))
    print("\ncoverage vs audit envelope:", json.dumps(coverage_vs_envelope(), indent=2))
