"""Entity-centric ontology model — the sdg derive's core representation (#141).

RH ruling (2026-07-04): retire the axiom-template-with-`{X:Y:Z}`-slots artifact; the
objective is real ontology → real DDL structure parity with SchemaPile (the Shape EMD).
An :class:`Entity` is a real named class with its FULL profile — genus (BFO/CCO anchor),
definition, typed DataProperties, cardinality-bounded object relations, enumerations — the
exact levers the current ontology lacks (measured: attr_zero 0.83, 0 junctions, 0 lookups).

:func:`to_manchester` renders a set of entities to a real OWL Manchester document a user
could feed to HermiT/DeepOnto, and which the kvasir DDL toolchain lowers into wide,
FK-bearing, junction-and-lookup-carrying tables. No slots, no templates — named classes
and their attributes, the way an organic domain ontology reads.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# BFO/CCO prefixes the deriver anchors into; the doc header a realized ontology carries.
PREFIXES = (
    "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
    "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
    "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
    "Prefix: fhir: <http://hl7.org/fhir/>\n"
    "Prefix: obi: <http://purl.obolibrary.org/obo/OBI_>\n"
    "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "Prefix: skos: <http://www.w3.org/2004/02/skos/core#>\n"
    "Prefix: iao: <http://purl.obolibrary.org/obo/IAO_>\n"
    "Prefix: dcterms: <http://purl.org/dc/terms/>\n"
)

_XSD = {"string", "integer", "int", "decimal", "double", "float", "boolean",
        "date", "dateTime", "long"}
_CARD = re.compile(r"^(some|only|exactly \d+|min \d+|max \d+)$")


def camel(name: str) -> str:
    """A raw concept name → CamelCase local name (``lab sample`` → ``LabSample``)."""
    parts = re.split(r"[^A-Za-z0-9]+", name.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "Thing"


def prop_name(name: str) -> str:
    """A raw property name → lowerCamel, preserving camelCase already inside a token
    (``collectedDate`` → ``collectedDate``; ``stored in`` → ``storedIn``; ``Barcode`` →
    ``barcode``). Only the very first character is lowercased, never a whole word."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name.strip()) if p]
    if not parts:
        return "hasValue"
    head = parts[0][:1].lower() + parts[0][1:]
    rest = "".join(p[:1].upper() + p[1:] for p in parts[1:])
    return head + rest


@dataclass
class DataAttr:
    """A typed data attribute → a DataProperty + a ``some xsd`` restriction (a column).
    An ``enum`` closed value set → a lookup table."""
    name: str
    xsd: str = "string"
    enum: list[str] = field(default_factory=list)
    definition: str = ""

    def iri(self) -> str:
        return "sdg:" + prop_name(self.name)


@dataclass
class Relation:
    """A cardinality-bounded object relation → an FK (some/exactly 1/max 1) or a
    junction (min/max > 1). ``identifying`` marks an identity-bearing relation (a badge/
    license/account that uniquely identifies its bearer) → InverseFunctional → UNIQUE."""
    prop: str
    target: str  # a class concept name (→ sdg:Target) or a prefixed IRI
    card: str = "some"  # some | only | exactly N | min N | max N
    identifying: bool = False

    def iri(self) -> str:
        return "sdg:" + prop_name(self.prop)

    def target_iri(self) -> str:
        return self.target if ":" in self.target else "sdg:" + camel(self.target)


@dataclass
class Entity:
    """A real named class with its full relational profile — the unit of the greenfield
    derive (replaces the slot-template). Renders to a Manchester Class frame with genus,
    definition, typed attributes, and cardinality-bounded relations."""
    name: str                       # concept name (→ sdg:Name)
    label: str = ""
    genus: str = "cco:ont00000995"     # BFO/CCO anchor (prefixed IRI)
    definition: str = ""
    attributes: list[DataAttr] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    domain: dict = field(default_factory=dict)

    def iri(self) -> str:
        return "sdg:" + camel(self.name)


def _quotable(s: str) -> bool:
    return '"' not in s and "#" not in s


def to_manchester(entities: list[Entity]) -> str:
    """Render entities to a real OWL Manchester document (prefixes + Class frames +
    DataProperty frames w/ enum definitions). HermiT/DeepOnto-feedable; kvasir-lowerable.

    Each entity → one ``Class`` frame: ``SubClassOf: <genus>, <prop> some xsd:*, …,
    <rel> <card> <Target>``. Enumerated attributes attach a ``skos:definition`` on the
    DataProperty so the kvasir front-end lowers a lookup table.
    """
    # PUNNING RESOLUTION (measured at 48-passage merge scale): the same prop name modeled as a
    # data attribute in one entity and an object relation in another → illegal OWL punning →
    # OWLAPI drops the redeclarations and the whole doc parses to ZERO frames (vacuous
    # certificate). Deterministic rule: the OBJECT side wins the name; colliding attributes
    # render as `<name>Detail`.
    rel_iris = {r.iri() for e in entities for r in e.relations}
    def _attr_iri(a: "DataAttr") -> str:
        iri = a.iri()
        return iri + "Detail" if iri in rel_iris else iri

    # KEY SEMANTICS IN THE ONTOLOGY (RH 2026-07-05): the SAME deterministic key plan that
    # realizes construct tables also emits OWL 2 key axioms here — natural keys become
    # `HasKey:` (HermiT-checkable; kvasir elects them as PKs), single-valued attributes
    # become Functional DataProperties, and uniformly to-one relations become Functional
    # ObjectProperties. One plan, two projections — the ontology is the source of truth
    # and the distributions are verifiable by counting axioms in the shipped document.
    kp = plan_keys(entities)
    import random as _random
    _oneof_rng = _random.Random(kp["seed"] ^ 0x0EE0F)
    oneof_attrs: dict = {}
    for e in entities:
        for a in e.attributes:
            if a.enum and len(a.enum) >= 2 and all(_quotable(v) and re.match(r"^[A-Za-z][\w-]*$", v)
                                                   for v in a.enum):
                if _oneof_rng.random() < 0.4 and _attr_iri(a) not in oneof_attrs:
                    kind = camel(a.name) + "Kind"
                    oneof_attrs[_attr_iri(a)] = (f"sdg:{kind}", list(a.enum))
    natural_keys = {e.iri(): next((_attr_iri(a) for a in e.attributes
                                   if a.name == kp["plans"][e.iri()]["pk_attr"]), None)
                    for e in entities if kp["plans"][e.iri()]["kind"] == "natural"}
    card_by_prop: dict = {}
    for e in entities:
        for r in e.relations:
            card_by_prop.setdefault(r.iri(), set()).add(r.card)
    functional_ops = {op for op, cards in card_by_prop.items()
                      if cards <= {"exactly 1", "max 1"}}

    # The Ontology: declaration is REQUIRED for OWLAPI's Manchester loader, and every property
    # (object/data/annotation) MUST be declared before use — without them the doc "loads" as
    # zero frames → a VACUOUS HermiT certificate (kvasir tolerates both omissions; measured).
    lines: list[str] = [
        PREFIXES, "", "Ontology: <https://signals.zndx.org/sdg>", "",
        "AnnotationProperty: rdfs:label", "AnnotationProperty: iao:0000115",
        "AnnotationProperty: skos:definition", "",
    ]
    ifp_ops = {r.iri() for e in entities for r in e.relations if r.identifying}
    for op in sorted({r.iri() for e in entities for r in e.relations} | set(oneof_attrs)):
        lines.append(f"ObjectProperty: {op}")
        chars = (["Functional"] if op in functional_ops else []) + \
                (["InverseFunctional"] if op in ifp_ops else [])
        if chars:
            lines.append("    Characteristics: " + ", ".join(chars))
    lines.append("")
    # Declarations BEFORE use (the OWLAPI Manchester parse is effectively single-pass):
    # data properties referenced in restrictions, and external genera (cco:/bfo:/…) that no
    # Class frame in this doc otherwise declares.
    for dp in sorted({_attr_iri(a) for e in entities for a in e.attributes}):
        if dp in oneof_attrs:
            continue
        lines.append(f"DataProperty: {dp}")
    local = {e.iri() for e in entities}
    for ext in sorted({e.genus for e in entities if e.genus} - local):
        lines.append(f"Class: {ext}")
    lines.append("")
    # the WILD enum idiom: a seeded share of enum attributes become enumeration CLASSES
    # (EquivalentTo: { … }) with an object property, exercising the K4 lowering path —
    # the same closed value set in its naturally occurring OWL form
    _seen_inds: set = set()
    for _iri, (kind_iri, vals) in sorted(oneof_attrs.items()):
        names = [f"sdg:{re.sub(r'[^A-Za-z0-9_]', '_', v)}" for v in vals]
        for ind in names:  # OWLAPI requires Individual declarations for oneOf members
            if ind not in _seen_inds:
                _seen_inds.add(ind)
                lines.append(f"Individual: {ind}")
        lines.append(f"Class: {kind_iri}")
        lines.append(f"    EquivalentTo: {{ {', '.join(names)} }}")
        lines.append("")
    dataprops: dict[str, DataAttr] = {}

    for e in entities:
        head = e.iri()
        lines.append(f"Class: {head}")
        anns = []
        if e.label and _quotable(e.label):
            anns.append(f'rdfs:label "{e.label}"')
        if e.definition and _quotable(e.definition):
            anns.append(f'iao:0000115 "{e.definition}"')
        if anns:
            lines.append("    Annotations: " + ", ".join(anns))
        conj: list[str] = [e.genus] if e.genus else []
        for a in e.attributes:
            ai = _attr_iri(a)
            if ai in oneof_attrs:
                conj.append(f"{ai} some {oneof_attrs[ai][0]}")
                continue  # the Kind class carries the closed set; no DataProperty frame
            xsd = a.xsd if a.xsd in _XSD else "string"
            conj.append(f"{ai} some xsd:{xsd}")
            dataprops.setdefault(ai, a)
        for r in e.relations:
            card = r.card if _CARD.match(r.card) else "some"
            conj.append(f"{r.iri()} {card} {r.target_iri()}")
        if conj:
            lines.append("    SubClassOf: " + ", ".join(conj))
        nk = natural_keys.get(e.iri())
        if nk:
            lines.append(f"    HasKey: {nk}")
        lines.append("")

    prop_users: dict = {}
    for e in entities:
        for a in e.attributes:
            prop_users.setdefault(_attr_iri(a), set()).add(e.iri())
    nk_iris = {natural_keys[k] for k in natural_keys
               if natural_keys[k] and len(prop_users.get(natural_keys[k], ())) == 1}
    if nk_iris:
        lines.append("DataProperty: dcterms:identifier")
        lines.append("")
    for iri, a in sorted(dataprops.items()):
        lines.append(f"DataProperty: {iri}")
        lines.append("    Characteristics: Functional")
        if iri in nk_iris:
            # the WILD identifier idiom alongside HasKey — real toolchains (and our own
            # K3 election) recognize dcterms:identifier subproperties as declared keys
            lines.append("    SubPropertyOf: dcterms:identifier")
        d = a.definition
        if a.enum:
            vals = " / ".join(v for v in a.enum if _quotable(v))
            if vals:
                d = (d + " " if d else "") + f"({vals})"
        if d and _quotable(d):
            lines.append(f'    Annotations: skos:definition "{d}"')
        lines.append("")

    return "\n".join(lines) + "\n"


# ── the structured-output schema the engine json_schema enforces (the propose contract) ──
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "genus": {"type": "string"},
                    "definition": {"type": "string"},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "xsd": {"type": "string"},
                                "enum": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name", "xsd"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prop": {"type": "string"},
                                "target": {"type": "string"},
                                "card": {"type": "string"},
                                "identifying": {"type": "boolean"},
                            },
                            "required": ["prop", "target", "card"],
                        },
                    },
                },
                "required": ["name", "genus", "attributes", "relations"],
            },
        }
    },
    "required": ["entities"],
}


_IRI = re.compile(r"([A-Za-z][\w-]*:[\w-]+)")


def clean_iri(tok: str, default: str = "cco:ont00000995") -> str:
    """Extract a prefixed IRI from a token the agent may have decorated (``bfo:0000040
    (Person)`` → ``bfo:0000040``); default if none is present."""
    m = _IRI.search(tok or "")
    return m.group(1) if m else default


def _prefix(name: str) -> str:
    """A short uppercase key prefix from a class name (``LandParcel`` → ``LAND``)."""
    caps = re.findall(r"[A-Z]", camel(name))
    return ("".join(caps[:4]) or camel(name)[:4]).upper()


_NAMEY = re.compile(r"(name|title|label|description)$", re.I)
_CODEY = re.compile(r"(id|code|number|serial|sku|barcode|key|ref|no|identifier)$", re.I)
# domain-neutral word pools for NAMEY columns — varied, title-cased, never stem-echoes
# (the 'paradi-001 everywhere' cell smell; fictional-particulars-safe by construction)
_ADJ = ["Integrated", "Adaptive", "Regional", "Baseline", "Compact", "Extended", "Primary",
        "Seasonal", "Distributed", "Legacy", "Pilot", "Composite"]
_HEAD = ["Framework", "Assessment", "Initiative", "Protocol", "Survey", "Model", "Programme",
         "Corridor", "Cluster", "Standard", "Series", "Review"]


def _gt_value(name: str, xsd: str, i: int, constraint=None, context: str = "") -> "str | None":
    """A real, PROVENANCE-retained GitTables value for a ``(name, xsd)`` cell at row ``i``, or None → the
    mechanical generator. Delegates to the SAME core as ``rows.value_for`` (:func:`rows._gittables_value`) so the
    flow's construct values and the textbook-embedded values are realism-identical (one core, no split-brain):
    ontology-grounding → relational → views → textbooks all carry the same real values. Deterministic per cell
    (blake2b, so stable across processes — unlike ``hash()``). ``context`` is the owning entity identity
    (genus + name) so a bare ``name`` column disambiguates person vs thing. [[gittables_value_realism]]"""
    from aegir.ontology import rows  # noqa: PLC0415 — late import (rows imports nothing from entities)
    xt = xsd if str(xsd).startswith("xsd:") else f"xsd:{xsd}"
    seed = int.from_bytes(hashlib.blake2b(f"{name}|{i}".encode(), digest_size=8).digest(), "big")
    return rows._gittables_value(name, xt, random.Random(seed), constraint, context)


def resolve_construct_constraints(entities: "list[Entity]", domain: str = "", *, engine: bool = True) -> dict:
    """Agent-mediated round-trip for the FLOW path: resolve a plausibility Constraint for each NUMERIC attribute
    (cached, deduped) → ``{attr_name: Constraint}``. Numeric GitTables pools are magnitude-mixed, so this range
    is what makes a flow numeric column REAL rather than mechanical. Graceful per-attr: an engine hiccup is
    LOGGED and that attr stays mechanical (availability is guaranteed by ``rows.require_gittables``; this is the
    quality refinement). Mirrors :func:`rows.resolve_constraints` so both value paths behave identically."""
    from aegir.ontology import rows, value_alignment  # noqa: PLC0415
    out: dict = {}
    seen: dict[str, object] = {}
    for e in entities:
        ectx = f"{e.name} {e.genus}"
        for a in e.attributes:
            sem = rows._semantic_type_of(a.name, f"xsd:{a.xsd}", ectx)
            if sem is None or sem not in rows._GT_NUMERIC:
                continue
            key = f"{sem}|{a.name.lower()}"
            if key not in seen:
                pool = rows._gt_pools().get(sem) or []
                try:
                    seen[key] = value_alignment.constrain(
                        prop_name(a.name), a.definition, sem, pool[:12], domain, engine=engine)
                except Exception as exc:  # noqa: BLE001 — a hiccup degrades THIS attr (logged), not the run
                    logger.warning(f"value-alignment skipped for '{a.name}' ({sem}): {exc}")
                    seen[key] = None
            if seen[key] is not None:
                out[a.name] = seen[key]
    return out


def _cell(a: "DataAttr", i: int, constraint=None, context: str = "") -> str:
    """A domain-plausible sample value for attribute ``a`` at row ``i`` (RI-true rows for
    prose). Value REALISM is a pipeline lever (naturalness_norms): real GitTables values for
    injectable columns (names/orgs always; numerics within the aligner's plausible range),
    else namey columns draw from varied title-case pools; codey columns get prefixed codes;
    dates jitter (no arithmetic series); nothing echoes the column stem into every row.
    ``context`` is the owning entity identity (genus+name) — disambiguates a bare ``name`` column."""
    if a.enum:
        return a.enum[i % len(a.enum)]
    gv = _gt_value(a.name, a.xsd, i, constraint, context)   # real value (strings always; numerics iff a constraint)
    if gv is not None:
        return gv
    x = a.xsd
    h = hash(a.name)
    if x in ("integer", "int", "long"):
        return str((i + 1) * (3 + h % 9) + h % 40)
    if x in ("decimal", "double", "float"):
        return f"{((i + 1) * (1.7 + (h % 13) / 4) + h % 20):.2f}"
    if x == "boolean":
        return "true" if (i + h) % 2 else "false"
    if x == "date":
        return f"20{22 + (h + i) % 4}-{((h + i * 5) % 12) + 1:02d}-{((h // 3 + i * 11) % 27) + 1:02d}"
    if x == "dateTime":
        return (f"20{22 + (h + i) % 4}-{((h + i * 5) % 12) + 1:02d}-"
                f"{((h // 3 + i * 11) % 27) + 1:02d}T{(h + i * 7) % 24:02d}:{(h * 3 + i * 17) % 60:02d}:00")
    if _NAMEY.search(a.name):
        return f"{_ADJ[(h + i * 5) % len(_ADJ)]} {_HEAD[(h // 7 + i * 3) % len(_HEAD)]}"                + (f" {chr(65 + (h + i) % 6)}" if (h + i) % 3 == 0 else "")
    stem = re.sub(r"(?<!^)(?=[A-Z])", " ", prop_name(a.name)).lower().replace("has ", "")
    if _CODEY.search(a.name):
        return f"{stem.split()[0][:3].upper()}-{2000 + h % 800 + i * (1 + h % 7)}"
    return f"{_ADJ[(h + i * 7) % len(_ADJ)].lower()}-{stem.split()[0][:8]}-{(h % 90) + i + 10}"



# ── SchemaPile key-shape norms (mined 2026-07-05, 198,756 tables; mine_schemapile_keys.py) ──
# Embedded fallback = the mined values; build/schemapile_key_norms.json overrides when present.
_KEY_NORMS_DEFAULT = {
    "pk_naming_single": {"bare_id": 0.5227, "natural": 0.2027, "table_id": 0.1457,
                         "other_id": 0.1289},
    "table_naming": {"snake": 0.8335, "camel_or_pascal": 0.1480, "prefixed": 0.0185},
    "audit_column_rate": 0.1489,
}


def load_key_norms() -> dict:
    import json as _json
    from pathlib import Path as _P
    p = _P(__file__).resolve().parents[3] / "build/schemapile_key_norms.json"
    try:
        d = _json.loads(p.read_text())
        return {k: d.get(k, v) for k, v in _KEY_NORMS_DEFAULT.items()}
    except Exception:  # noqa: BLE001
        return dict(_KEY_NORMS_DEFAULT)


def _snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _plural(s: str) -> str:
    if s.endswith("y") and s[-2:-1] not in "aeiou":
        return s[:-1] + "ies"
    if s.endswith(("s", "x", "z", "ch", "sh")):
        return s + "es"
    return s + "s"


_NATURAL_KEY_RX = re.compile(r"(id|code|number|serial|sku|barcode|key|ref|no)$", re.I)


def plan_keys(entities: "list[Entity]") -> dict:
    """Sample per-CONSTRUCT conventions + per-ENTITY key shapes against the mined SchemaPile
    distributions — deterministic (seeded by the sorted entity names) so reruns are stable.
    One convention per chapter, varied across the corpus: exactly how real repos differ."""
    import hashlib
    import random
    norms = load_key_norms()
    seed = int(hashlib.md5("|".join(sorted(e.name for e in entities)).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    def pick(dist: dict) -> str:
        r, acc = rng.random(), 0.0
        for k, v in dist.items():
            acc += v
            if r <= acc:
                return k
        return next(iter(dist))

    tstyle = pick(norms["table_naming"])
    col_style = "snake" if tstyle != "camel_or_pascal" else "camel"
    view_prefix = rng.choice(["v_", "v_", "vw_", ""])  # view naming is unmined; sensible split
    # a prefixed schema uses ITS OWN app prefix, not a universal tic: derive it from the
    # construct's domain vocabulary (first entity's stem, abbreviated), the way real schemas
    # carry wp_/shop_/lims_ prefixes. Uniform t_ across the corpus was the formulaic tell
    # RH's release gate caught (109 t_* tables — one author's habit, not a world).
    _stem0 = _snake(camel(entities[0].name)).split("_")[0] if entities else "app"
    app_prefix = (_stem0[:4] if len(_stem0) > 4 else _stem0) + "_"

    def col(name: str) -> str:
        return _snake(prop_name(name)) if col_style == "snake" else prop_name(name)

    plans: dict = {}
    for e in entities:
        stem = _snake(camel(e.name))
        table = {"snake": _plural(stem), "camel_or_pascal": camel(e.name),
                 "prefixed": app_prefix + _plural(stem)}[tstyle]
        kind = pick(norms["pk_naming_single"])
        nat = next((a for a in e.attributes if _NATURAL_KEY_RX.search(a.name)), None)
        if kind == "natural" and nat is None:
            kind = "table_id"
        if kind == "bare_id":
            pk, pk_attr = "id", None
        elif kind in ("table_id", "other_id"):
            pk, pk_attr = f"{stem}_id" if col_style == "snake" else f"{stem.split('_')[-1]}Id", None
            kind = "table_id"
        else:
            pk, pk_attr = col(nat.name), nat.name
        plans[e.iri()] = {"table": table, "pk": pk, "kind": kind, "pk_attr": pk_attr,
                          "stem": stem}
    return {"table_style": tstyle, "col_style": col_style, "view_prefix": view_prefix,
            "audit_rate": float(norms["audit_column_rate"]), "seed": seed, "plans": plans}


def _fk_col_name(prop: str, tgt_plan: dict, own: "set[str]", col_style: str) -> str:
    """The FK column referencing ``tgt_plan``'s PK, per real conventions: bare-id targets get
    ``<stem>_id``; named/natural PKs join same-name; collisions role-disambiguate via the prop."""
    stem = tgt_plan["stem"].split("_")[-1]
    if tgt_plan["kind"] == "bare_id":
        base = f"{tgt_plan['stem']}_id" if col_style == "snake" else f"{stem}Id"
    elif tgt_plan["kind"] == "table_id":
        base = tgt_plan["pk"]
    else:
        pk = tgt_plan["pk"]
        base = pk if stem in pk else (f"{tgt_plan['stem']}_{pk}" if col_style == "snake"
                                      else stem + pk[:1].upper() + pk[1:])
    if base in own:
        role = _snake(prop_name(prop)) if col_style == "snake" else prop_name(prop)
        base = f"{role}_{base}" if col_style == "snake" else role + base[:1].upper() + base[1:]
    return base


def to_construct(entities: list[Entity], *, n_rows: int = 4, style_anchor: str = "",
                 constraints: dict | None = None) -> dict:
    """Bridge entities → the prose-harness ``construct`` dict: tables with REAL-WORLD KEY
    SHAPES (per the mined SchemaPile distributions — bare ``id`` / ``<table>_id`` / natural
    keys; integer surrogates; per-chapter naming conventions; audit-column idioms), RI-true
    rows, and FK columns referencing the target's ACTUAL pk values. Many-to-many relations
    are left to ``add_views`` (junction tables with composite keys). ``constraints`` is the
    per-attr round-trip map from :func:`resolve_construct_constraints` (numeric plausibility)."""
    constraints = constraints or {}
    kp = plan_keys(entities)
    rng = random.Random(kp["seed"] ^ 0x5EED)
    plans, col_style = kp["plans"], kp["col_style"]

    def col(name: str) -> str:
        return _snake(prop_name(name)) if col_style == "snake" else prop_name(name)

    # pk values per entity: integer surrogates for id-kinds (real dumps count 1,2,3…),
    # the attribute's own cells for natural keys
    pk_vals: dict[str, list[str]] = {}
    for e in entities:
        plan = plans[e.iri()]
        if plan["kind"] == "natural":
            attr = next(a for a in e.attributes if a.name == plan["pk_attr"])
            ectx = f"{e.name} {e.genus}"
            pk_vals[e.iri()] = [_cell(attr, i, constraints.get(attr.name), ectx) for i in range(n_rows)]
        else:
            base = rng.choice([1, 1, 1, 100, 1000])
            pk_vals[e.iri()] = [str(base + i) for i in range(n_rows)]

    tables = []
    for e in entities:
        plan = plans[e.iri()]
        ectx = f"{e.name} {e.genus}"                # owning-entity context → disambiguates a bare `name` column
        own: "set[str]" = set()
        cols = []
        if plan["kind"] != "natural":
            cols.append({"name": plan["pk"], "concept": camel(e.name),
                         "cells": [{"value": v} for v in pk_vals[e.iri()]]})
            own.add(plan["pk"])
        for a in e.attributes:
            cname = col(a.name)
            if cname in own:
                continue
            cols.append({"name": cname, "concept": prop_name(a.name),
                         "cells": [{"value": pk_vals[e.iri()][i] if a.name == plan["pk_attr"]
                                    else _cell(a, i, constraints.get(a.name), ectx)} for i in range(n_rows)]})
            own.add(cname)
        fks = []
        for r in e.relations:
            if r.card.startswith("min") or (r.card.startswith("max") and r.card != "max 1"):
                continue  # many-to-many → junction (add_views)
            tgt = r.target_iri()
            if tgt not in plans or tgt not in pk_vals:
                continue
            fk = _fk_col_name(r.prop, plans[tgt], own, col_style)
            cols.append({"name": fk, "concept": prop_name(r.prop),
                         "cells": [{"value": pk_vals[tgt][i % len(pk_vals[tgt])]}
                                   for i in range(n_rows)]})
            fks.append({"col": fk, "ref_table": plans[tgt]["table"],
                        "ref_col": plans[tgt]["pk"]})
            own.add(fk)
        # audit idiom at the mined rate (deterministic via the construct rng)
        if rng.random() < kp["audit_rate"]:
            ts = "created_at" if col_style == "snake" else "createdAt"
            cols.append({"name": ts, "concept": "created",
                         "cells": [{"value": f"2025-{(i % 12) + 1:02d}-{(i * 5 % 27) + 1:02d} "
                                             f"{(i * 3 % 24):02d}:14:00"} for i in range(n_rows)]})
            if rng.random() < 0.6:
                us = "updated_at" if col_style == "snake" else "updatedAt"
                cols.append({"name": us, "concept": "updated",
                             "cells": [{"value": f"2025-{(i % 12) + 1:02d}-{(i * 7 % 27) + 2:02d} "
                                                 f"{(i * 5 % 24):02d}:41:00"} for i in range(n_rows)]})
        tables.append({"name": plan["table"], "pk": plan["pk"], "fks": fks, "columns": cols})
    return {"tables": tables, "style_anchor": style_anchor,
            "entities": [e.iri() for e in entities], "key_plan": kp}


def _md_table(columns: "list[str]", rows: "list[list[str]]") -> str:
    """Render a fixed markdown table (the payload blocks embedded verbatim in chapters)."""
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def add_views(construct: dict, entities: "list[Entity]", *, n_rows: int = 4) -> dict:
    """Materialize the VIEWS the chapter embeds — join semantics over the REAL key shapes.

    FK join views join on the actual key columns (``ON o.station_id = s.id`` /
    ``ON b.station_code = s.station_code``); junction views synthesize an RI-true junction
    table with a COMPOSITE key (the SchemaPile-real shape for m2m) and 3-way join over it.
    Each view carries {name, kind, sql, columns, rows}."""
    kp = construct.get("key_plan") or plan_keys(entities)
    plans, col_style, vpx = kp["plans"], kp["col_style"], kp["view_prefix"]
    byname = {t["name"]: t for t in construct["tables"]}

    def _cells(t: dict) -> "dict[str, list[str]]":
        return {c["name"]: [x["value"] for x in c["cells"]] for c in t["columns"]}

    def vname(a_stem: str, b_stem: str, suffix: str = "") -> str:
        base = f"{a_stem}_{b_stem}{suffix}"
        return f"{vpx}{base}" if vpx else f"{base}_view"

    ents = {e.iri(): e for e in entities}
    views = []
    for e in entities:
        plan = plans.get(e.iri())
        a = byname.get(plan["table"]) if plan else None
        if a is None:
            continue
        acols = _cells(a)
        apk = plan["pk"]
        for r in e.relations:
            tgt = ents.get(r.target_iri())
            tplan = plans.get(r.target_iri()) if tgt else None
            b = byname.get(tplan["table"]) if tplan else None
            if b is None:
                continue
            bcols = _cells(b)
            bpk = tplan["pk"]
            m2m = r.card.startswith("min") or (r.card.startswith("max") and r.card != "max 1")
            fk = next((f["col"] for f in a.get("fks", []) if f.get("ref_table") == tplan["table"]),
                      None)
            a_show = [c["name"] for c in a["columns"] if c["name"] != fk][:4]
            b_show = [c["name"] for c in b["columns"]][:3]
            bstem = tplan["stem"].split("_")[-1]
            if not m2m and fk and fk in acols:
                name = vname(plan["stem"], tplan["stem"])
                sql = (f"CREATE VIEW {name} AS\nSELECT " +
                       ", ".join([f"a.{c}" for c in a_show] +
                                 [f"b.{c} AS {bstem}_{c}" for c in b_show]) +
                       f"\nFROM {a['name']} a JOIN {b['name']} b ON a.{fk} = b.{bpk};")
                bidx = {v: i for i, v in enumerate(bcols[bpk])}
                rows = []
                for i in range(len(next(iter(acols.values())))):
                    j = bidx.get(acols[fk][i])
                    if j is None:
                        continue
                    rows.append([acols[c][i] for c in a_show] + [bcols[c][j] for c in b_show])
                views.append({"name": name, "kind": "fk_join", "sql": sql,
                              "columns": a_show + [f"{bstem}_{c}" for c in b_show],
                              "rows": rows})
            elif m2m:
                # junction table: composite PK of the two reference columns (the real shape)
                own: "set[str]" = set()
                ja = _fk_col_name("", plan, own, col_style); own.add(ja)
                jb = _fk_col_name(r.prop, tplan, own, col_style)
                jt = f"{_plural(plan['stem'].split('_')[-1])}_{_plural(tplan['stem'].split('_')[-1])}" \
                    if kp["table_style"] == "snake" else plan["table"] + camel(tplan["stem"])
                apk_vals, bpk_vals = acols[apk], bcols[bpk]
                jrows = [[apk_vals[i], bpk_vals[(i + k) % len(bpk_vals)]]
                         for i in range(len(apk_vals)) for k in range(2)]
                if jt not in byname:
                    tbl = {"name": jt, "pk": [ja, jb], "fks": [
                               {"col": ja, "ref_table": a["name"], "ref_col": apk},
                               {"col": jb, "ref_table": b["name"], "ref_col": bpk}],
                           "columns": [
                               {"name": ja, "concept": a["name"],
                                "cells": [{"value": r0} for r0, _ in jrows]},
                               {"name": jb, "concept": b["name"],
                                "cells": [{"value": r1} for _, r1 in jrows]}]}
                    construct["tables"].append(tbl)
                    byname[jt] = tbl
                name = vname(plan["stem"], tplan["stem"], "_detail")
                sql = (f"CREATE VIEW {name} AS\nSELECT " +
                       ", ".join([f"a.{c}" for c in a_show[:3]] +
                                 [f"b.{c} AS {bstem}_{c}" for c in b_show]) +
                       f"\nFROM {a['name']} a\n  JOIN {jt} j ON j.{ja} = a.{apk}\n"
                       f"  JOIN {b['name']} b ON b.{bpk} = j.{jb};")
                aidx = {v: i for i, v in enumerate(apk_vals)}
                bidx = {v: i for i, v in enumerate(bpk_vals)}
                rows = [[acols[c][aidx[x]] for c in a_show[:3]] + [bcols[c][bidx[y]] for c in b_show]
                        for x, y in jrows if x in aidx and y in bidx]
                views.append({"name": name, "kind": "junction_join", "sql": sql,
                              "columns": a_show[:3] + [f"{bstem}_{c}" for c in b_show],
                              "rows": rows})
    construct["views"] = views
    return construct


def render_payload_blocks(construct: dict) -> "dict[str, str]":
    """The FIXED chapter payload: ``{marker: markdown_block}``. Chapters embed these verbatim
    (deterministic injection — the agent never re-types a row; the Track A lesson)."""
    blocks: "dict[str, str]" = {}
    for t in construct.get("tables", []):
        cols = [c["name"] for c in t["columns"]]
        rows = [[c["cells"][i]["value"] for c in t["columns"]]
                for i in range(len(t["columns"][0]["cells"]))]
        blocks[f"TABLE:{t['name']}"] = f"**Table `{t['name']}`**\n\n" + _md_table(cols, rows)
    for v in construct.get("views", []):
        blocks[f"VIEW:{v['name']}"] = (
            f"**View `{v['name']}`**\n\n```sql\n{v['sql']}\n```\n\n" +
            _md_table(v["columns"], v["rows"]))
    return blocks


_XSD_CANON = {"integer": "integer", "int": "integer", "long": "integer",
              "decimal": "decimal", "double": "decimal", "float": "decimal",
              "number": "decimal", "boolean": "boolean", "bool": "boolean",
              "date": "date", "datetime": "dateTime", "string": "string",
              "text": "string", "enum": "string"}


def _norm_xsd(x: str) -> str:
    """Canonicalize the deriver's xsd emissions: strip the prefix ('xsd:integer'), fold
    aliases ('enum', 'text', 'number'), unknown → string. The 431-passage merge produced
    1,489 UNSAT classes because 'integer' vs 'xsd:integer' silently diverged into
    integer-vs-string restrictions on Functional properties (⊥ conjunctions)."""
    t = (x or "string").strip().removeprefix("xsd:").lower()
    return _XSD_CANON.get(t, "string")


def _scrub_enum(values: "list[str]") -> "list[str]":
    """REAL UNIVERSALS, FICTIONAL PARTICULARS: drop enum values naming real orgs/brands/
    products (the deriver sometimes proposes 'Microsoft Word'-style members; they would flow
    verbatim into the omn AND the embedded table cells). <2 survivors → no enum at all."""
    try:
        from aegir.ontology.individuals import real_entity_hits
        bad = {v for v, _brand in real_entity_hits(values)}
    except Exception:  # noqa: BLE001
        bad = set()
    kept = [v for v in values if v not in bad]
    return kept if len(kept) >= 2 else []


def from_json(obj: dict) -> list[Entity]:
    """Parse the engine's json_schema output into :class:`Entity` records (defensive)."""
    out: list[Entity] = []
    for e in obj.get("entities", []):
        if not isinstance(e, dict) or not e.get("name"):
            continue
        out.append(Entity(
            name=str(e["name"]),
            label=str(e.get("label", "")),
            genus=clean_iri(str(e.get("genus") or "cco:ont00000995")),
            definition=str(e.get("definition", "")),
            attributes=[DataAttr(name=str(a["name"]), xsd=_norm_xsd(str(a.get("xsd", "string"))),
                                 enum=_scrub_enum([str(v) for v in (a.get("enum") or [])]))
                        for a in e.get("attributes", []) if isinstance(a, dict) and a.get("name")],
            relations=[Relation(prop=str(r["prop"]), target=str(r["target"]),
                                card=str(r.get("card", "some")),
                                identifying=bool(r.get("identifying", False)))
                       for r in e.get("relations", []) if isinstance(r, dict) and r.get("prop")],
        ))
    return out
