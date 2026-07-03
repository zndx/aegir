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

# ── REAL UNIVERSALS, FICTIONAL PARTICULARS (RH 2026-07-03: sensitive nouns escape the training
# arena — 'Apple Inc' typed into the published ABox). The corpus asserts nothing about real-world
# PARTICULARS: organizations, brands, commercial products, people, named facilities are invented in a
# realistic register. Real UNIVERSALS (standards, protocols, units, kinds: HL7, ISO 17025, UTC) stay —
# they are nomenclature, and domain realism needs them. This deterministic denylist is the fast
# REGRESSION tier (the head of the brand distribution); the engine-screened audit is the judgment tier.
_BRANDS_CLEAR = re.compile(
    r"\b(microsoft|google|amazon|facebook|instagram|nvidia|intel|qualcomm|broadcom|samsung|huawei|"
    r"foxconn|tsmc|asml|cisco|lenovo|toshiba|hitachi|siemens|panasonic|ericsson|nokia|motorola|"
    r"ibm|sap|salesforce|workday|servicenow|databricks|snowflake|cloudera|teradata|informatica|"
    r"cerner|meditech|allscripts|athenahealth|epic\s+systems|medtronic|stryker|baxter|danaher|"
    r"agilent|shimadzu|perkinelmer|perkin\s+elmer|bruker|sartorius|mettler[- ]toledo|labware|"
    r"labvantage|starlims|thermo\s+fisher|beckman|biorad|bio-rad|illumina|qiagen|pfizer|novartis|"
    r"astrazeneca|glaxo|gsk|sanofi|bayer|roche|genentech|moderna|biontech|boeing|airbus|lockheed|"
    r"northrop|raytheon|toyota|volkswagen|daimler|nissan|hyundai|ferrari|porsche|autodesk|"
    r"solidworks|dassault|ansys|synopsys|altium|keysight|tektronix|rohde\s*&\s*schwarz|"
    r"halliburton|schlumberger|baker\s+hughes|exxon|exxonmobil|chevron|conocophillips|petrobras|"
    r"gazprom|aramco|verizon|vodafone|t-mobile|comcast|starlink|spacex|at&t|panduit|corning|"
    r"belden|commscope|schneider\s+electric|rockwell\s+automation|honeywell|emerson|yokogawa|abb)\b",
    re.I)
# ambiguous tokens (common words / fruit / mythology) — flagged only in brand-context shapes:
# "<Brand> Inc/Corp/…", "<Brand> <CapitalizedProduct>", or the bare TitleCase token alone.
# NB the [A-Z] product requirement must stay case-SENSITIVE ('target material' is a noun phrase,
# 'Apple RenderKit' is a brand) — the (?i:…) groups scope insensitivity to brand/suffix tokens only.
_BRANDS_AMBIG = ("apple", "oracle", "amazon", "meta", "epic", "waters", "shell", "bp", "ge",
                 "philips", "tesla", "ford", "delta", "target", "adobe", "stripe", "square",
                 "cadence", "palantir", "anthropic", "openai", "deepmind")
_AMBIG_CTX = re.compile(
    r"(?i:\b(" + "|".join(_BRANDS_AMBIG) + r")\b)\s+(?:(?i:inc|corp|llc|ltd|plc|group|"
    r"systems|health(?:care)?|cloud|labs?)\b|[A-Z][A-Za-z0-9]+)")


def real_entity_hits(values: "list[str]") -> "list[tuple[str, str]]":
    """[(value, matched_brand)] — the deterministic real-particular screen (see policy note above).
    Ambiguous tokens require brand context ('Apple Inc', 'Apple RenderKit') or an exact bare
    TitleCase token ('Apple') — lowercase 'apple' in an orchard column is a fruit and passes."""
    out = []
    for v in values:
        m = _BRANDS_CLEAR.search(v)
        if m:
            out.append((v, m.group(1)))
            continue
        m = _AMBIG_CTX.search(v)
        if m:
            out.append((v, m.group(1).lower()))
            continue
        if v.strip() in {b.title() for b in _BRANDS_AMBIG}:
            out.append((v, v.strip().lower()))
    return out
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
    real = real_entity_hits(kept)
    if real:
        named = ", ".join(sorted({f"'{v}'" for v, _ in real})[:4])
        return False, (f"real-world particular(s) named: {named} — the corpus asserts only FICTIONAL "
                       "organizations/products/people (invent plausible counterparts in the same "
                       "register; real standards/protocols/units are fine)"), kept
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


def abox_manchester(reg: dict, *, ns: str = "https://signals.zndx.org/sdg#",
                    declared: "set[str] | None" = None) -> str:
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
            # WRITER-SIDE COHERENCE GATE: the ABox may not reference classes the target document does not
            # declare — one ghost `Types:` name makes the Manchester parser fail the WHOLE doc (measured
            # 2026-07-02: a class withheld from the TBox after seeding, plus a degenerate 'X' slot, turned
            # the artifact into an 11-axiom vacuous parse). Ghost-typed values are skipped here, loudly
            # countable by the caller; the registry entry remains for the re-seed loop to reconcile.
            if declared is not None and cls not in declared:
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
