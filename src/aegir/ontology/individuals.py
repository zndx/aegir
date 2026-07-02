"""Individual registry — the ontology's ABox, agentically accreted (Convert 1b, [[convert_priority_cas]]).

Cell values for entity columns stop being strings drawn from a frozen pool and become INDIVIDUALS of
ontology classes: seeded per-domain by the engine from content, disposed by a membrane that RETURNS its
reason (the agent re-proposes against it), accreted here with provenance, and emitted as OWL ``Individual:``
frames into the realized ontology so HermiT certifies instance-level consistency (an individual asserted
into disjoint classes is a caught error — the same theory, ABox edition). RI in the materialized DDL then
rests on class-asserted individuals, not string-equality luck: the same individual backs the PK and every
FK cell that references it.

The registry is the committed successor of the frozen ``entity_value_pools.json``: same consumable shape
(``pools_view`` → ``{template_id: {col: [values]}}``) so :func:`chapter_tables.entity_pools_for_spine`
merges it seamlessly (registry wins), plus per-template provenance {domain, run, model, membrane} — the
authenticity audit travels with the artifact. Accretion = inheritance; a frozen global pool = pre-wiring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "individual_registry.json"

# entity-column values that are vacuous filler, not domain instances
_GENERIC_VALUES = {"example", "value", "item", "test", "sample", "data", "entity", "thing",
                   "placeholder", "unknown", "n/a", "tbd", "none", "foo", "bar", "misc", "other"}
# id-like columns legitimately hold mechanical serials (AUTH-8842 …) — exempt from the stem check
_IDLIKE = re.compile(r"(?:^|_)(id|ids|code|codes|number|no|ref|reference|identifier|key|serial|uuid|urn)(?:_|$)")
_SLUG = re.compile(r"[^a-z0-9]+")


# ── registry I/O ────────────────────────────────────────────────────────────────
def load_registry(path: "Path | None" = None) -> dict:
    p = path or REGISTRY_PATH
    if not p.exists():
        return {"version": "1", "templates": {}}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"version": "1", "templates": {}}


def save_registry(reg: dict, path: "Path | None" = None) -> None:
    (path or REGISTRY_PATH).write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")


def pools_view(reg: dict) -> "dict[str, dict[str, list[str]]]":
    """The legacy-compatible ``{template_id: {col: [values]}}`` view rows.materialize_rows consumes."""
    return {tid: rec.get("columns", {}) for tid, rec in reg.get("templates", {}).items()
            if rec.get("columns")}


# ── the membrane (returns its REASON — the agent's feedback channel) ────────────
def check_column(col: str, values: "list[str]") -> "tuple[bool, str, list[str]]":
    """Dispose one column's proposed values → (ok, reason, kept). Every rejection carries the reason the
    proposer re-authors against ([[agent_mediated_feedback_loop]]); an empty reason means admitted."""
    kept = [v for v in values if v.strip().lower() not in _GENERIC_VALUES]
    n_generic = len(values) - len(kept)
    if len(kept) < 5:
        why = f"only {len(kept)} usable values (need ≥5)"
        if n_generic:
            why += f" after dropping {n_generic} generic filler value(s)"
        return False, why + " — give more distinct, concrete domain instances", kept
    if not _IDLIKE.search(col.lower()):
        # mechanical-stem detector: "Alpha Sample | Beta Sample | Gamma Sample" is canned, not domain data.
        # Only multi-word values participate — single-word values ARE their own stem and carry no shared
        # scaffold; requiring ≥2 words also keeps enum-ish pools ('pending', 'running', …) out of scope.
        for pick in (lambda v: v.split()[-1], lambda v: v.split()[0]):
            toks = [pick(v).lower() for v in kept if len(v.split()) >= 2]
            if len(toks) >= 3:
                top = max(set(toks), key=toks.count)
                if toks.count(top) / len(toks) > 0.6:
                    return False, (f"values are mechanical variants sharing the stem '{top}' — vary the "
                                   "surface forms (real instances differ in kind, not just in one word)"), kept
    ratio = len({v.lower() for v in kept}) / len(kept)
    if ratio < 0.9:
        return False, f"near-duplicate values (distinct ratio {ratio:.2f}) — every value must be distinct", kept
    return True, "", kept


def admit(reg: dict, template_id: str, *, columns: "dict[str, list[str]]",
          entity_classes: "dict[str, str]", domain: str, provenance: dict) -> None:
    """Accrete an admitted seeding into the registry (merge per column; provenance recorded per admit).
    ``entity_classes`` maps each Class-typed column → the class local-name its values instantiate (the
    realizer's slot→global-IRI unification) — stored so ABox emission is self-contained."""
    rec = reg.setdefault("templates", {}).setdefault(template_id, {"columns": {}, "entity_classes": {}})
    for col, vals in columns.items():
        merged = list(dict.fromkeys((rec["columns"].get(col) or []) + vals))
        rec["columns"][col] = merged
    rec.setdefault("entity_classes", {}).update(entity_classes)
    rec["domain"] = domain or rec.get("domain", "")
    rec["provenance"] = provenance


# ── ABox emission (the instantiated ontology) ───────────────────────────────────
def _slug(label: str) -> str:
    return _SLUG.sub("_", label.lower()).strip("_")[:60] or "unnamed"


def abox_manchester(reg: dict, *, ns: str = "https://signals360.example.org/sdg#") -> str:
    """Render the registry's ENTITY-column values as OWL ``Individual:`` frames typed by the class the
    column references (each record's stored ``entity_classes``) — the ontology INSTANTIATED. Values of
    data/string columns are literals, not individuals, and are not emitted. The same label reused across
    classes becomes ONE individual with multiple ``Types:`` — deliberately: if those classes are disjoint
    under BFO+π(CCO), HermiT catches the clash at realize (the instance-level membrane).

    FULL IRIs only (measured 2026-07-02): this OWLAPI's Manchester parser rejects prefix-form names in the
    ``Types:`` section and silently degrades the WHOLE document to a vacuous parse (11 axioms, 0 classes,
    trivially consistent). Full-IRI rendering matches the realize doc's own convention everywhere else."""
    types_by_ind: dict[str, set[str]] = {}
    label_by_ind: dict[str, str] = {}
    for rec in reg.get("templates", {}).values():
        for col, cls in (rec.get("entity_classes") or {}).items():
            cls = re.sub(r"[^A-Za-z0-9_]", "", cls or "")
            if not cls:
                continue
            for v in rec.get("columns", {}).get(col, []):
                ind = f"i_{_slug(v)}"
                types_by_ind.setdefault(ind, set()).add(cls)
                label_by_ind.setdefault(ind, v)
    frames = []
    for ind in sorted(types_by_ind):
        types = ", ".join(f"<{ns}{c}>" for c in sorted(types_by_ind[ind]))
        label = label_by_ind[ind].replace('"', "'")
        frames.append(f'Individual: <{ns}{ind}>\n    Annotations: rdfs:label "{label}"\n    Types: {types}')
    return "\n\n".join(frames)


def types_by_individual(reg: dict) -> "dict[str, set[str]]":
    """``{individual_id → {class local-names}}`` over the registry's entity columns — the input to the
    DECOMPOSED ABox certification (build_realized_ontology): with a nominal-free TBox and this Types-only
    ABox (no Facts/SameAs — enforced by construction here), KB consistency ⟺ TBox consistency ∧ each
    asserted type-conjunction satisfiable (disjoint-union of component models; nothing connects
    individuals). Singleton type-sets are covered by the realize's unsat=∅ invariant."""
    out: dict[str, set[str]] = {}
    for rec in reg.get("templates", {}).values():
        for col, cls in (rec.get("entity_classes") or {}).items():
            cls = re.sub(r"[^A-Za-z0-9_]", "", cls or "")
            if not cls:
                continue
            for v in rec.get("columns", {}).get(col, []):
                out.setdefault(f"i_{_slug(v)}", set()).add(cls)
    return out


def n_individuals(reg: dict) -> int:
    return len(types_by_individual(reg))
