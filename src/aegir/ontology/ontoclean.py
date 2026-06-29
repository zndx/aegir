"""OntoClean-informed rigor signals — the deterministic fitness/feedback for the rigor-evolution loop.

Survey synthesis (`docs/scratch/2026-06-28/225637_ontoclean_survey_synthesis.md`): the role-vs-kind
discriminator is OntoClean **anti-rigidity**. A ROLE is anti-rigid (~R) AND relationally/externally dependent
(+D); a DISPOSITION/FUNCTION is internally grounded (the VETO); a defined KIND has a genus + sufficient
differentia (→ `equivalentClass`). These constraints are reasoner-INVISIBLE yet checkable — the un-fakeable
taxonomic-correctness signal an LLM cannot mimic.

This module **FLAGS candidates + the specific rationale** (it is NOT a decision oracle). The engine decides
sufficiency / authors the rigorous axiom (GEPA-style reflection over this feedback); HermiT + the metrology
dispose. Pure stdlib, no JVM — usable in the metrology, the membrane, and the evolution loop.
"""
from __future__ import annotations

import re

# ── lexical signals (the survey's role-detection signature) ──
_ROLE_LEXICON = {
    "supplier", "customer", "buyer", "seller", "vendor", "provider", "consumer", "operator", "reviewer",
    "manufacturer", "consignor", "consignee", "distributor", "retailer", "wholesaler", "patient", "employee",
    "employer", "manager", "administrator", "owner", "holder", "applicant", "examiner", "auditor", "inspector",
    "participant", "client", "contractor", "stakeholder", "partner", "subscriber", "donor", "recipient",
    "guardian", "trustee", "beneficiary", "respondent", "candidate", "operator", "custodian", "sponsor",
    "specimen", "sample", "analyte", "subject", "host", "vector", "target", "actor",
}
_ROLE_PREDICATION = re.compile(
    r"\b(acts? as|plays? the role|in the role of|serves? as|functions? as|in the capacity of|qua\b|"
    r"appointed|assigned|designated|in virtue of (?:its|their|the) (?:role|participation|involvement|membership))", re.I)
_ANTIRIGID_MARK = re.compile(
    r"\b(former|ex-|no longer|used to be|temporar|provisional|while it|during which|currently|pending|"
    r"at the time|for the duration)", re.I)
# disposition/function VETO — internally grounded in the bearer's physical make-up / design
_INTERNAL = re.compile(
    r"\b(fragile|soluble|conductive|elastic|brittle|flammable|malleable|the capacity to|the ability to|"
    r"capable of|disposition to|tendency to|propensity|liability to)", re.I)
_FUNCTION = re.compile(r"\b(by design|designed to|in order to|evolved to|for the purpose of|intended to|serves to)", re.I)
# already role/realizable-modelled in the axiom → leave it
_REALIZABLE = re.compile(r"BFO_0000023|BFO_0000017|BFO_0000016|BFO_0000034|bfo:0000023|bfo:0000017|bfo:0000016|bfo:0000034|realizes|inheres|bearer")
# agentive deverbal nominalization — weak; only counts WITH corroboration, and never for these non-role -er/-or nouns
_AGENTIVE = re.compile(r"(er|or|ant|ee|ist)$")  # NB: not -ent (catches -ment nominalizations: commitment/judgment)
_NOT_AGENTIVE = {
    "water", "matter", "computer", "number", "parameter", "diameter", "perimeter", "register", "cluster",
    "filter", "buffer", "container", "barrier", "border", "order", "factor", "vector", "sensor", "motor",
    "indicator", "descriptor", "identifier", "header", "marker", "center", "chamber", "layer", "polymer",
    "member", "system", "document", "event", "component", "instrument", "segment", "element", "statement",
    "requirement", "equipment", "measurement", "assessment", "treatment", "behavior", "behaviour", "color",
    "report", "record", "process", "service", "resource", "structure", "feature", "procedure", "disorder",
}
_RESTRICTION = re.compile(r"\b(some|exactly|only|min|max|value)\b")


def head_noun(name: str) -> str:
    """Last CamelCase / underscored token, lowercased (the concept's head noun)."""
    name = name.split("#")[-1].split(":")[-1]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name.replace("_", " "))
    return (parts[-1] if parts else name).lower()


def classify(label: str, definition: str = "", manchester: str = "", anchor=()) -> dict:
    """Flag a concept's OntoClean meta-properties + suggest the rigorous pattern. Heuristic — flags, not decides.

    suggested ∈ {role, disposition, function, equivalent_class, phase, keep}; the engine resolves the
    equivalent_class candidates (sufficiency) and authors the axiom under HermiT.
    """
    head = head_noun(label)
    text = f"{label}. {definition}"
    mancanchor = f"{manchester} {' '.join(str(a) for a in (anchor or []))}"

    already = bool(_REALIZABLE.search(mancanchor)) or "EquivalentTo" in manchester
    role_lex = head in _ROLE_LEXICON
    role_pred = bool(_ROLE_PREDICATION.search(text))
    antirigid_mark = bool(_ANTIRIGID_MARK.search(text))
    agentive = bool(_AGENTIVE.search(head)) and head not in _NOT_AGENTIVE
    internal = bool(_INTERNAL.search(text))
    is_function = bool(_FUNCTION.search(text))
    has_restriction = bool(_RESTRICTION.search(manchester))
    # relational = the concept is defined via participation/relation to another entity
    relational = role_pred or role_lex or (agentive and has_restriction)
    anti_rigid = role_lex or antirigid_mark or role_pred or (agentive and relational)
    has_differentia = has_restriction or " and " in manchester.lower()

    if already:
        suggested, why = "keep", "already role/realizable-modelled"
    elif internal and not (role_lex or role_pred):
        suggested = "function" if is_function else "disposition"
        why = f"internally grounded ({'design/purpose etiology' if is_function else 'capacity/tendency'}) → realizable, NOT a role"
    elif anti_rigid and relational:
        cue = "role-lexicon" if role_lex else "role-predication" if role_pred else "anti-rigidity marker" if antirigid_mark else "agentive+relational"
        suggested, why = "role", f"anti-rigid ({cue}) ∧ relational → BFO role (bfo:0000023), borne not subclassed"
    elif anti_rigid:
        suggested, why = "phase", "anti-rigid but intrinsic (no external relation) → phase-like (BFO has no phase primitive)"
    elif has_differentia:
        suggested, why = "equivalent_class", "rigid kind with genus + differentia → equivalentClass CANDIDATE (engine decides sufficiency)"
    else:
        suggested, why = "keep", "primitive / partially-characterized"

    return {"label": label, "head": head, "suggested": suggested, "rationale": why,
            "anti_rigid": anti_rigid, "relational": relational, "internal": internal,
            "has_differentia": has_differentia}


def classify_template(t) -> dict:
    """Classify a derived CatalogTemplate by its HEAD slot (first ``{Name:...}`` of the manchester_template)."""
    man = getattr(t, "manchester_template", "") or ""
    m = re.search(r"Class:\s*\{(\w+)", man)
    head_label = m.group(1) if m else getattr(t, "template_id", "?")
    v = classify(head_label, getattr(t, "verbal_template", "") or "", man, getattr(t, "bfo_anchor_path", []) or [])
    v["template_id"] = getattr(t, "template_id", "?")
    return v


def summarize(verdicts) -> dict:
    from collections import Counter
    return dict(Counter(v["suggested"] for v in verdicts))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from aegir.ontology.schema import load_catalog
    cat = sys.argv[1] if len(sys.argv) > 1 else "src/aegir/ontology/catalog/08_derived.json"
    templates = load_catalog(cat).templates
    verdicts = [classify_template(t) for t in templates]
    print(f"=== OntoClean rigor scan · {cat.split('/')[-1]} · {len(templates)} templates ===")
    print("suggested pattern distribution:", summarize(verdicts))
    for s in ("role", "equivalent_class", "disposition", "function", "phase"):
        ex = [v for v in verdicts if v["suggested"] == s][:4]
        if ex:
            print(f"\n  [{s}] ({sum(v['suggested'] == s for v in verdicts)}):")
            for v in ex:
                print(f"    · {v['template_id']:34s} {v['rationale']}")
