"""Deterministic ontology → SQL DDL spine, verified with ``polyglot``.

This is the **syntactic** half of the corpus's two-axis verifiability::

    DeepOnto : ontology (OWL Manchester)   ::   polyglot : SQL (DDL)
    ─────────────────────────────────────      ──────────────────────
    semantic gate  (R_axiom / families)         syntactic gate (validate)
    semantic coverage (coverage_audit)          SQL-feature coverage (here)

The ontology catalog already carries the relational structure we need (each
template's typed slots), and ``type_check`` already models a unit's schema
(:class:`~aegir.ontology.type_check.TableSpec` / ``ColumnSpec`` / ``FKEdge``).
This module *lowers* a catalog template to that schema model and then **renders
real SQL DDL** for it, validated against the lakehouse dialects we will feed
(Trino, Spark — Iceberg is a table format expressed via Spark/Trino DDL, so an
Iceberg-flavored Spark variant is rendered too). Generation is pure/deterministic
(no LLM, no JVM); only validation + coverage touch ``polyglot``, behind a guarded
import so the generator is usable even before the native extension is built.

Mapping (OWL meta-type → relational role):

==================  ==========================  ===========================
slot ``slot_type``  relational role             SQL
==================  ==========================  ===========================
``Class``           entity / class-IRI ref      ``VARCHAR(255)``
``Individual``      named-instance ref          ``VARCHAR(255)``
``DataProperty``    typed literal               inferred (INT/DECIMAL/…)
``ObjectProperty``  relation (not a column)     drives a FOREIGN KEY / note
==================  ==========================  ===========================

Cross-family foreign keys (the *join structure* the corpus views exploit) are
sanctioned by the empirical :class:`~aegir.ontology.complex.FamilyComplex`: a
candidate FK linking two families is emitted only if their simplex
``is_allowed`` — punctured combinations are suppressed and audited.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from aegir.ontology.schema import CatalogTemplate
from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec, UnitSchema, infer_value_type

# ── dialects ──────────────────────────────────────────────────────────────────
DEFAULT_DIALECTS: tuple[str, ...] = ("trino", "spark")
ICEBERG_DIALECT = "spark"  # Iceberg DDL is expressed via the Spark dialect

# ── OWL meta-type → SQL type ──────────────────────────────────────────────────
_ENTITY_OWL = {"Class", "Individual", "NamedIndividual"}
_DATA_OWL = {"DataProperty", "Datatype", "Literal", "Data"}
_ENTITY_SQL = "VARCHAR(255)"
# Typed DataProperty ranges (xsd) → SQL, keyed to the SchemaPile-frequent types.
_XSD_SQL = {
    "xsd:string": "VARCHAR(255)", "xsd:dateTime": "TIMESTAMP", "xsd:date": "DATE",
    "xsd:decimal": "DECIMAL(38,9)", "xsd:double": "DOUBLE", "xsd:float": "DOUBLE",
    "xsd:integer": "INTEGER", "xsd:int": "INTEGER", "xsd:long": "BIGINT", "xsd:boolean": "BOOLEAN",
}


def sql_type_for_slot(owl_type: str, sample_values: list[str] | None = None) -> str:
    """Map an OWL slot meta-type (or an ``xsd:`` DataProperty range) to a SQL column type.

    Typed DataProperty ranges (``xsd:*``) map directly — referential-integrity-clean, the
    type traces to the ontology range, not sample inference. Entity-like slots
    (Class/Individual) are reference columns (``VARCHAR(255)``); untyped DataProperty slots
    fall back to :func:`type_check.infer_value_type` on samples.
    """
    if owl_type in _XSD_SQL:
        return _XSD_SQL[owl_type]
    if owl_type in _ENTITY_OWL:
        return _ENTITY_SQL
    if owl_type in _DATA_OWL:
        if sample_values:
            inferred = infer_value_type(sample_values)
            if inferred == "Data":
                joined = " ".join(str(v) for v in sample_values)
                if re.search(r"\d{4}-\d{2}-\d{2}", joined):
                    return "DATE"
                if any("." in str(v) for v in sample_values):
                    return "DECIMAL(38,9)"
                if all(re.fullmatch(r"[+-]?\d+", str(v).strip()) for v in sample_values if str(v).strip()):
                    return "INTEGER"
        return _ENTITY_SQL
    return _ENTITY_SQL  # unknown OWL type → safe reference default


# ── identifier sanitizers ─────────────────────────────────────────────────────
def table_name(template_id: str) -> str:
    return "t_" + re.sub(r"\W+", "_", template_id).strip("_").lower()


def col_name(slot: str) -> str:
    return re.sub(r"\W+", "_", slot).strip("_").lower() or "col"


# ── ontology DataProperties → typed attribute columns ─────────────────────────
_VOCAB_PATH = Path(__file__).resolve().parent / "sdg-vocab.ttl"
_NS = {
    "http://purl.obolibrary.org/obo/BFO_": "bfo:",
    "http://www.commoncoreontologies.org/": "cco:",
    "https://signals.zndx.org/sdg#": "sdg:",
    "http://www.w3.org/2001/XMLSchema#": "xsd:",
}
# anchor → parent anchor, so a table inherits its ancestors' typed attributes.
_ANCHOR_PARENTS = {
    "cco:Artifact": "bfo:IndependentContinuant",
    "cco:DescriptiveICE": "cco:InformationContentEntity",
    "cco:DirectiveICE": "cco:InformationContentEntity",
    "cco:DesignativeICE": "cco:InformationContentEntity",
}
_DATAPROP_CACHE: "dict[str, list[tuple[str, str, str]]] | None" = None


def _prefixed(uri) -> str:
    s = str(uri)
    for ns, p in _NS.items():
        if s.startswith(ns):
            return p + s[len(ns):]
    return s


def _dataprop_col(prop_prefixed: str) -> str:
    local = prop_prefixed.split(":")[-1]
    local = re.sub(r"^(has|is)([A-Z])", r"\2", local)        # hasStartTime → StartTime
    return re.sub(r"(?<!^)(?=[A-Z])", "_", local).lower()    # StartTime → start_time


def _data_properties() -> "dict[str, list[tuple[str, str, str]]]":
    """{domain-anchor → [(column_name, xsd_range, property_iri)]} from sdg-vocab.ttl (cached)."""
    global _DATAPROP_CACHE
    if _DATAPROP_CACHE is not None:
        return _DATAPROP_CACHE
    out: "dict[str, list[tuple[str, str, str]]]" = {}
    try:
        import rdflib
        g = rdflib.Graph(); g.parse(str(_VOCAB_PATH), format="turtle")
        for prop in g.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty):
            dom = g.value(prop, rdflib.RDFS.domain)
            rng = g.value(prop, rdflib.RDFS.range)
            if dom is None or rng is None:
                continue
            pp = _prefixed(prop)
            out.setdefault(_prefixed(dom), []).append((_dataprop_col(pp), _prefixed(rng), pp))
        for k in out:
            out[k].sort()
    except Exception:
        out = {}
    _DATAPROP_CACHE = out
    return out


_DATAPROP_META_CACHE: "dict[str, tuple[str, str, str]] | None" = None


def dataprop_meta() -> "dict[str, tuple[str, str, str]]":
    """``{property_iri(prefixed) → (label, xsd_range, definition)}`` from sdg-vocab.ttl (cached).

    Feeds the deterministic row generator (:mod:`aegir.ontology.rows`): a property whose
    ``skos:definition`` enumerates a closed value set (e.g. ``hasStatus`` →
    "pending / running / complete / failed") grounds that column's *values* in the ontology
    itself; otherwise the ``xsd`` range drives a typed generator. Distinct from
    :func:`_data_properties` (domain-keyed, for column discovery) — this is iri-keyed metadata.
    """
    global _DATAPROP_META_CACHE
    if _DATAPROP_META_CACHE is not None:
        return _DATAPROP_META_CACHE
    out: "dict[str, tuple[str, str, str]]" = {}
    try:
        import rdflib
        from rdflib.namespace import SKOS
        g = rdflib.Graph(); g.parse(str(_VOCAB_PATH), format="turtle")
        for prop in g.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty):
            rng = g.value(prop, rdflib.RDFS.range)
            label = g.value(prop, rdflib.RDFS.label)
            defn = g.value(prop, SKOS.definition)
            out[_prefixed(prop)] = (str(label) if label else "",
                                    _prefixed(rng) if rng is not None else "",
                                    str(defn) if defn else "")
    except Exception:
        out = {}
    _DATAPROP_META_CACHE = out
    return out


# PROVISIONAL (Comp 4 de-canning; provisional_scaffolding_not_goals): per-template typed-attribute
# budget. Real tables in a BFO cohort share FEW columns, not an identical canned set — so each
# template draws a SEEDED subset of its (enriched) anchor pool, varying the column-set across
# same-anchor tables. Empirically (per-family=8): budget 2 puts every anchor's column-name entropy
# (h_colset) at/above SchemaPile's median — real-DB-level vocabulary diversity — while keeping tables
# realistically wide. A floor-to-clear, not the destination; ratchet with the gate.
_ANCHOR_ATTR_BUDGET = 2


def anchor_attributes(anchor: str, template_id: str | None = None, *,
                      budget: "int | None" = None) -> "list[tuple[str, str, str]]":
    """Typed attribute columns (column, xsd_range, property_iri) for a table whose
    bfo_anchor is ``anchor`` — the anchor's DataProperties plus its ancestors'.

    With a ``template_id`` (and a ``budget``), returns a SEEDED per-template *stratified
    subset* of that pool (de-canning, Comp 4): same-anchor tables then carry genuinely
    different typed columns rather than one identical canned set. ``budget=None`` reads the
    module default ``_ANCHOR_ATTR_BUDGET`` at call time (so it stays tunable); an explicit
    ``budget`` of ``0``/negative returns the full inherited pool (back-compat)."""
    if budget is None:
        budget = _ANCHOR_ATTR_BUDGET
    by_dom = _data_properties()
    full: "list[tuple[str, str, str]]" = []
    seen: set[str] = set()
    cur = anchor
    while cur:
        for col, rng, iri in by_dom.get(cur, []):
            if col not in seen:
                seen.add(col)
                full.append((col, rng, iri))
        cur = _ANCHOR_PARENTS.get(cur)
    if template_id is None or not budget or budget <= 0 or len(full) <= budget:
        return full
    rnd = random.Random(int.from_bytes(hashlib.blake2b(template_id.encode(), digest_size=8).digest(), "big"))
    idx = list(range(len(full)))
    rnd.shuffle(idx)
    return [full[i] for i in sorted(idx[:budget])]


# ── ObjectProperty / quantified-restriction resolution from Manchester ─────────
# Matches "<prop> (some|only|min N|max N|exactly N) {Y:Class}" where <prop> is
# either a typed slot "{p:ObjectProperty}" or a literal IRI (e.g. "sdg:inService").
_RESTRICTION_RE = re.compile(
    r"(?:\{(?P<pslot>\w+):ObjectProperty\}|(?P<piri>[\w]+:[\w]+))\s+"
    r"(?P<card>some|only|min\s+\d+|max\s+\d+|exactly\s+\d+)\s+"
    r"\{(?P<target>\w+):(?:Class|Individual)\}"
)


@dataclass
class Restriction:
    prop: str            # property slot name or literal IRI
    is_slot: bool        # True if a {p:ObjectProperty} slot
    cardinality: str     # "some" | "only" | "min 1" | ...
    target_slot: str     # the {Y:Class} slot reached via the property


def parse_restrictions(template: CatalogTemplate) -> list[Restriction]:
    """Object-property / quantified restrictions in a template's Manchester axiom.

    Each links the table's subject to a target Class slot via a property; the
    target slot's column is the natural foreign-key carrier, and ``some``/``min 1``
    make it ``NOT NULL``.
    """
    out: list[Restriction] = []
    for m in _RESTRICTION_RE.finditer(template.manchester_template):
        prop = m.group("pslot") or m.group("piri")
        out.append(Restriction(
            prop=prop,
            is_slot=m.group("pslot") is not None,
            cardinality=re.sub(r"\s+", " ", m.group("card")).strip(),
            target_slot=m.group("target"),
        ))
    return out


@dataclass
class CheckNote:
    """A constraint we surface as a comment (not an enforced CHECK — Trino has none)."""
    column: str
    kind: str   # "bound" | "cardinality" | "relation" | "external_target"
    text: str


# ── template → relational table ───────────────────────────────────────────────
@dataclass
class SpineTable:
    template: CatalogTemplate
    family: str
    table: TableSpec
    notes: list[CheckNote] = field(default_factory=list)
    not_null: set[str] = field(default_factory=set)   # column names that are NOT NULL
    # ── schema-realization (realize.py) — kind + materialization hints. Default "entity" keeps the
    # one-table-per-template baseline; the realizer emits eav_registry/eav_value/junction/fact/dimension
    # sub-tables of a richer subgraph, and rows.materialize_rows reads realize_meta to fill them RI-safe.
    kind: str = "entity"
    realize_meta: dict = field(default_factory=dict)


_DP_RANGE_RE = re.compile(
    r"\{(?P<slot>\w+):DataProperty\}\s+(?:some|only|value|exactly\s+\d+|min\s+\d+|max\s+\d+)?\s*"
    r"(?P<rng>xsd:\w+)")


def _dataprop_ranges(manchester: str) -> dict[str, str]:
    """Map each {slot:DataProperty} to the xsd range parsed from its Manchester restriction."""
    return {m.group("slot"): m.group("rng") for m in _DP_RANGE_RE.finditer(manchester)}


# BFO/RO relations carry opaque numeric local names (BFO_0000050); map the ones we use to
# readable column names — a numeric-leading local name is also invalid as an unquoted SQL column.
_OPAQUE_PROP_LABELS = {
    "bfo:0000050": "part_of", "bfo:0000051": "has_part", "bfo:0000066": "occurs_in",
    "bfo:0000063": "precedes", "bfo:0000062": "preceded_by", "bfo:0000055": "realizes",
    "bfo:0000056": "participates_in", "bfo:0000057": "has_participant",
}


def _prop_col(prop: str) -> str:
    """Column name for an object property: a known opaque BFO/RO ID → its readable label;
    a numeric-leading local name → ``rel_<id>`` (valid SQL); else the camel→snake local name."""
    if prop in _OPAQUE_PROP_LABELS:
        return _OPAQUE_PROP_LABELS[prop]
    local = prop.split(":")[-1]
    return f"rel_{local}" if local[:1].isdigit() else _dataprop_col(prop)


# ── SKOS-native subject naming (decorative; slot_ref stays canonical) ──────────
# BFO/CCO anchor IRI → readable column-name stem for a subject-head column.
_CLASS_LABEL = {
    "cco:Artifact": "artifact", "cco:InformationContentEntity": "information_entity",
    "cco:DescriptiveICE": "descriptive_record", "cco:DirectiveICE": "directive",
    "cco:DesignativeICE": "designator", "bfo:Process": "process",
    "bfo:IndependentContinuant": "continuant",
}
_NAME_STOP = {"subclass", "class", "basic", "template", "generic", "with", "via",
              "and", "the", "for", "to", "of", "an", "from"}
# (template_id, slot) → curated column name; the explicit long-tail override hook (empty default).
CURATED_SLOT_NAMES: dict[tuple[str, str], str] = {}


def _class_label(iri: str | None) -> str | None:
    if not iri:
        return None
    if iri in _CLASS_LABEL:
        return _CLASS_LABEL[iri]
    local = iri.split(":")[-1]
    if not local or local[:1].isdigit():          # opaque BFO numeric local name → no good stem
        return None
    return re.sub(r"(?<!^)(?=[A-Z])", "_", local).lower()


def skos_name_for_slot(template: CatalogTemplate, slot: str) -> str | None:
    """An ontology-native, real-world-ish name for a *subject-head* slot (vs bare ``subject``).

    Priority: curated override → the template's own concept token (its identity in the ontology)
    → the BFO/CCO anchor's readable stem. ``slot_ref`` stays the canonical column↔slot identity,
    so this is a decorative, content-only upgrade (and lifts the e6 name↔verbalization overlap).
    """
    key = (template.template_id, slot)
    if key in CURATED_SLOT_NAMES:
        return CURATED_SLOT_NAMES[key]
    # a renamed domain slot (a multi-char concept token like ``AblationProcess`` — vs a bare ``X``/``Y``
    # filler letter) names itself; the deriver now coins these per primitive (class_slot_names).
    if len(slot) > 1 and slot.lower() not in _NAME_STOP:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", slot).strip("_").lower()
    toks = [t for t in re.split(r"[^A-Za-z]+", template.template_id)
            if len(t) >= 3 and t.lower() not in _NAME_STOP]
    if toks:
        return toks[0].lower()
    anchor = template.bfo_anchor_path[-1] if template.bfo_anchor_path else None
    return _class_label(anchor)


def semantic_col_names(template: CatalogTemplate) -> dict[str, str]:
    """Map each non-ObjectProperty slot to a SEMANTIC column name (vs the bare slot letter).

    The catalog templates are ``Class: {X} SubClassOf: cco:…, <property> some {Y}`` — so a slot
    is either the *subject* head or an *object* reached by a named property. We name accordingly:

    - an object slot reached by a named-property restriction takes the PROPERTY's local name —
      the relationship it carries: ``sdg:assignsMassTo`` → ``assigns_mass_to``,
      ``sdg:hasFocalElement`` → ``focal_element`` (``_dataprop_col`` strips ``has``/``is``);
    - the subject (head) Class/Individual slot — not a restriction target — becomes ``subject``;
    - a target reached only via a generic ``{p:ObjectProperty}`` slot becomes ``related``;
    - DataProperty slots keep their attribute name (``col_name``).

    Disambiguated with a numeric suffix; never collides with ``id``. The column ``name`` is
    decorative — ``slot_ref`` stays the canonical column↔slot identity — so this is a safe,
    content-only improvement over the old ``x``/``y`` lowercased-slot names.
    """
    targets = {r.target_slot: r for r in parse_restrictions(template)}
    names: dict[str, str] = {}
    used = {"id"}
    n_subject = 0
    for slot, owl in template.slot_types.items():
        if owl == "ObjectProperty":
            continue
        r = targets.get(slot)
        if r is not None:
            base = "related" if r.is_slot else _prop_col(r.prop)
        elif owl in ("Class", "Individual"):
            n_subject += 1
            base = skos_name_for_slot(template, slot) or (
                "subject" if n_subject == 1 else f"subject_{n_subject}")
        else:
            base = col_name(slot)
        base = base or col_name(slot)
        name, i = base, 2
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        names[slot] = name
    return names


def template_to_table(template: CatalogTemplate, family: str) -> SpineTable:
    """Lower one catalog template to a :class:`TableSpec` + constraint notes.

    Columns: a surrogate ``id`` primary key, then one column per non-ObjectProperty
    slot (Class/Individual/DataProperty). ObjectProperty *slots* are relations, not
    columns. Manchester restrictions (``some``/``min 1``) mark their target column
    ``NOT NULL`` and are recorded as notes; the ``bfo_anchor_path`` and provenance
    ride in the table comment (the semantic / CTA anchor).
    """
    cols: list[ColumnSpec] = [ColumnSpec(name="id", slot_type="Class", slot_ref="__pk__")]
    notes: list[CheckNote] = []
    not_null: set[str] = set()

    colnames = semantic_col_names(template)
    dp_ranges = _dataprop_ranges(template.manchester_template)
    for slot, owl in template.slot_types.items():
        if owl == "ObjectProperty":
            continue  # relation, represented as a FK / note, not a column
        st = dp_ranges.get(slot, owl) if owl == "DataProperty" else owl
        cols.append(ColumnSpec(name=colnames[slot], slot_type=st, slot_ref=slot))

    # Typed attribute columns derived from the ontology's DataProperties whose domain
    # subsumes this template's bfo_anchor (the seed crystal; the generator extends it).
    anchor = template.bfo_anchor_path[-1] if template.bfo_anchor_path else None
    if anchor:
        existing = {c.name for c in cols}
        for col, xsd_range, prop_iri in anchor_attributes(anchor, template.template_id):
            if col not in existing:
                existing.add(col)
                cols.append(ColumnSpec(name=col, slot_type=xsd_range, slot_ref=f"data:{prop_iri}"))

    for r in parse_restrictions(template):
        tgt = colnames[r.target_slot]
        notes.append(CheckNote(tgt, "relation",
                               f"{r.prop} {r.cardinality} {r.target_slot}"))
        if r.cardinality == "some" or r.cardinality.startswith("min "):
            n = r.cardinality.split()[-1] if r.cardinality.startswith("min") else "1"
            if r.cardinality == "some" or n != "0":
                not_null.add(tgt)

    if template.bfo_anchor_path:
        notes.append(CheckNote("id", "bound",
                               f"subject subClassOf+ {template.bfo_anchor_path[-1]}"))

    return SpineTable(
        template=template, family=family,
        table=TableSpec(name=table_name(template.template_id),
                        ref=template.template_id, columns=cols, rows=[]),
        notes=notes, not_null=not_null,
    )


# ── views: the prose-facing artifact + its verbalization (the E6 bridge object) ──
@dataclass
class ViewSpec:
    """A CREATE VIEW over base spine tables + its natural-language verbalization.

    The view is the relational shadow of an axiom chain: a join along an FK derived from
    an ObjectProperty IS that axiom materialized. ``verbalization`` is the bridge object —
    what the chapter's prose should narrate, and what carries the view's topic position
    in the E6 trace (channel A) and its lexicon in channel B.
    """
    name: str
    sql: str
    verbalization: str
    columns: list[tuple[str, str, str]]  # (view_col, base_table, base_col)
    base_tables: list[str]
    fk: "FKEdge | None" = None


def _verbal_clean(t: CatalogTemplate) -> str:
    """Slot-stripped verbalization (lowercased, placeholder-free) for composition."""
    v = re.sub(r"\{[^}]+\}", "", t.verbal_template or "").strip()
    return re.sub(r"\s+", " ", v).strip(" .").lower() or t.template_id.replace("_", " ")


def verbalize_view(src: SpineTable, dst: "SpineTable | None" = None,
                   fk: "FKEdge | None" = None, projected: list[str] | None = None) -> str:
    """Compose the constituent templates' verbalizations along the join path."""
    cols = ", ".join(projected or [])
    if dst is None or fk is None:
        return (f"Each row presents an instance where {_verbal_clean(src.template)}; "
                f"shown attributes: {cols}.")
    rel = re.sub(r"(?<!^)(?=[A-Z])", " ", fk.via_slot).replace("_", " ").lower()
    return (f"Each row pairs an instance where {_verbal_clean(src.template)} with the related "
            f"instance where {_verbal_clean(dst.template)}, joined via its '{rel}' relation; "
            f"shown attributes: {cols}.")


def render_view_ddl(name: str, src: SpineTable, dst: "SpineTable | None" = None,
                    fk: "FKEdge | None" = None,
                    projected: "list[str] | None" = None) -> ViewSpec:
    """CREATE VIEW: a projection of one base table, or a 2-table join along an FK edge.

    View columns inherit base-column identity (the (view_col, base_table, base_col) triples)
    so CTA labels and column lineage carry through — polyglot's openlineage_run_event over
    ``sql``'s SELECT yields the Column DERIVES_FROM edges for the governance graph.
    """
    def pick(st: SpineTable, alias: str) -> list[tuple[str, str, str]]:
        out = []
        for c in st.table.columns:
            if c.slot_ref == "__pk__":
                continue
            if projected and c.name not in projected:
                continue
            out.append((f"{alias}_{c.name}" if dst is not None else c.name,
                        st.table.name, c.name))
        return out

    if dst is None or fk is None:
        cols = pick(src, "a")
        sel = ", ".join(f"{bc} AS {vc}" if vc != bc else bc for vc, _, bc in cols)
        sql = f"CREATE VIEW {name} AS SELECT {sel} FROM {src.table.name}"
    else:
        cols = pick(src, "a") + pick(dst, "b")
        parts = []
        for vc, bt, bc in cols:
            alias = "a" if bt == src.table.name else "b"
            parts.append(f"{alias}.{bc} AS {vc}")
        sql = (f"CREATE VIEW {name} AS SELECT {', '.join(parts)} "
               f"FROM {src.table.name} a JOIN {dst.table.name} b "
               f"ON a.{fk.src_col} = b.{fk.dst_col}")
    verbal = verbalize_view(src, dst, fk, [vc for vc, _, _ in cols])
    return ViewSpec(name=name, sql=sql, verbalization=verbal, columns=cols,
                    base_tables=[src.table.name] + ([dst.table.name] if dst else []),
                    fk=fk)


def build_table_comment(template: CatalogTemplate, family: str) -> str:
    """Single-line JSON semantic anchor stored as the table COMMENT (the CTA label)."""
    payload = {
        "template_id": template.template_id,
        "family": family,
        "bfo_anchor": template.bfo_anchor_path,
        "is_complex": template.is_complex,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


# ── cross-family foreign keys (the join structure), gated by the family complex ─
def _anchor_root(template: CatalogTemplate) -> str | None:
    """Top of the BFO anchor path (e.g. 'bfo:Process'); None if anchorless."""
    return template.bfo_anchor_path[-1] if template.bfo_anchor_path else None


class _NoSanction:
    """The fc=None stand-in since the family-complex retirement: no pre-wired sanction exists, so no
    cross-family edge is emitted — candidates are still audited (the measurement survives the gate).
    Cross-domain relational structure is the deriver's to EARN (Convert 2: relations asserted between
    classes, lowered by theorem), never a name-match heuristic's to wire."""

    def is_allowed(self, simplex) -> bool:  # noqa: ARG002 — uniform interface
        return False


def cross_family_fks(spine: list[SpineTable], fc) -> tuple[list[FKEdge], list[dict]]:
    """Emit cross-family FK edges sanctioned by the :class:`FamilyComplex`.

    A source table with an object-reference column (one reached via a Manchester
    restriction) may FK to a target table in a *different* family whose subject
    shares the source's BFO anchor root (or both anchorless). The edge is emitted
    only if ``fc.is_allowed({src_family, dst_family})`` — punctured/out-of-closure
    simplices are suppressed. Every candidate is returned in the audit list.
    ``fc=None`` (normal since the family-complex retirement) → :class:`_NoSanction`.
    """
    if fc is None:
        fc = _NoSanction()
    edges: list[FKEdge] = []
    audit: list[dict] = []
    by_fam: dict[str, list[SpineTable]] = {}
    by_root: dict[str | None, list[SpineTable]] = {}
    for st in spine:
        by_fam.setdefault(st.family, []).append(st)
        by_root.setdefault(_anchor_root(st.template), []).append(st)
    fams = sorted(by_fam)

    for st in spine:
        refs = parse_restrictions(st.template)
        if not refs:
            continue
        r0 = refs[0]
        src_col = next((c.name for c in st.table.columns if c.slot_ref == r0.target_slot),
                       col_name(r0.target_slot))
        root = _anchor_root(st.template)
        # Prefer a same-anchor-root target in a different, simplex-allowed family
        # (semantic quality); else fall back to any allowed-simplex different family.
        # The FamilyComplex is the principled gate (only sanctioned family joins);
        # anchor-root is a preference, not a hard filter (it over-restricts capped sets).
        target = next((t for t in by_root.get(root, [])
                       if t.family != st.family
                       and fc.is_allowed(frozenset({st.family, t.family}))), None)
        if target is None:
            target = next((by_fam[of][0] for of in fams
                           if of != st.family
                           and fc.is_allowed(frozenset({st.family, of}))), None)
        audit.append({"src": st.template.template_id,
                      "dst": target.template.template_id if target else None,
                      "src_family": st.family,
                      "dst_family": target.family if target else None,
                      "via": r0.prop,
                      "simplex": sorted({st.family, target.family}) if target else [],
                      "allowed": target is not None,
                      "status": "emitted" if target else "no_allowed_cross_family"})
        if target:
            edges.append(FKEdge(src_table=st.table.name, src_col=src_col,
                                dst_table=target.table.name, dst_col="id", via_slot=r0.prop))
    return edges, audit


# ── DDL rendering ─────────────────────────────────────────────────────────────
def render_ddl(st: SpineTable, fks: list[FKEdge], *,
               constraints: bool = True, iceberg: bool = False) -> str:
    """Render a CREATE TABLE for ``st``.

    ``constraints``: include PRIMARY KEY / FOREIGN KEY table constraints (polyglot
    parses these even where an engine would ignore enforcement; disable for a
    columns-only fallback). ``iceberg``: emit a Spark ``USING iceberg`` +
    ``PARTITIONED BY`` variant for the Iceberg-flavored coverage axis.
    """
    sample = None
    lines: list[str] = []
    for c in st.table.columns:
        if c.slot_ref == "__pk__":
            lines.append("  id VARCHAR(255)")
            continue
        sqltype = sql_type_for_slot(c.slot_type, sample)
        nn = " NOT NULL" if c.name in st.not_null else ""
        lines.append(f"  {c.name} {sqltype}{nn}")

    if constraints:
        lines.append("  PRIMARY KEY (id)")
        for e in fks:
            lines.append(f"  FOREIGN KEY ({e.src_col}) REFERENCES {e.dst_table}(id)")

    body = ",\n".join(lines)
    comment = build_table_comment(st.template, st.family).replace("'", "''")
    head = f"CREATE TABLE {st.table.name} (\n{body}\n)"

    if iceberg:
        part = next((c.name for c in st.table.columns if c.slot_ref != "__pk__"), None)
        clause = " USING iceberg"
        if part:
            clause += f" PARTITIONED BY ({part})"
        # Spark's USING-datasource syntax rejects a standalone COMMENT clause; the
        # semantic anchor rides in TBLPROPERTIES instead (validates under Spark).
        return f"{head}{clause}\nTBLPROPERTIES ('comment'='{comment}')"
    return f"{head}\nCOMMENT '{comment}'"


# ── polyglot bridge (guarded) ─────────────────────────────────────────────────
_POLYGLOT = None


def _polyglot():
    global _POLYGLOT
    if _POLYGLOT is None:
        try:
            import polyglot_sql  # type: ignore
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "polyglot_sql is not importable. Build it: "
                "`devenv tasks run polyglot:build` (maturin develop --release in "
                "components/polyglot/crates/polyglot-sql-python)."
            ) from e
        _POLYGLOT = polyglot_sql
    return _POLYGLOT


@dataclass
class ErrorInfo:
    message: str
    line: int | None = None
    col: int | None = None
    code: str | None = None
    severity: str | None = None


@dataclass
class DialectValidation:
    dialect: str
    valid: bool
    errors: list[ErrorInfo] = field(default_factory=list)


def validate_ddl(ddl: str, dialects: tuple[str, ...] = DEFAULT_DIALECTS) -> list[DialectValidation]:
    """Validate ``ddl`` under each dialect via ``polyglot_sql.validate`` (the syntactic gate).

    A statement is "spine-valid" iff it passes *all* target dialects (Iceberg-valid
    = passes Trino ∩ Spark). ``validate`` does not raise; we read ``result.valid``.
    """
    pg = _polyglot()
    out: list[DialectValidation] = []
    for d in dialects:
        res = pg.validate(ddl, dialect=d)
        errs: list[ErrorInfo] = []
        for e in (getattr(res, "errors", None) or []):
            errs.append(ErrorInfo(
                message=getattr(e, "message", str(e)),
                line=getattr(e, "line", None), col=getattr(e, "col", None),
                code=getattr(e, "code", None), severity=getattr(e, "severity", None),
            ))
        out.append(DialectValidation(dialect=d, valid=bool(getattr(res, "valid", False)), errors=errs))
    return out


# ── SQL-feature coverage inventory (the syntactic coverage axis) ───────────────
_TYPE_TOKENS = ("VARCHAR", "INTEGER", "DECIMAL", "DATE", "BOOLEAN", "BIGINT", "DOUBLE")


def coverage_inventory(ddl: str, *, dialect: str = "trino", parse: bool = True) -> dict[str, int]:
    """Count SQL DDL features in a statement.

    Primary signal is string-level over the generated DDL (reliable because we
    emit it); when ``parse`` and polyglot is available, the AST (``to_dict``) is
    scanned too and node kinds are folded in under ``ast:<kind>`` keys so anything
    we did not model is still surfaced.
    """
    up = ddl.upper()
    feats: dict[str, int] = {}

    def bump(k: str, n: int = 1) -> None:
        feats[k] = feats.get(k, 0) + n

    for t in _TYPE_TOKENS:
        n = up.count(t)
        if n:
            bump(f"type:{t.lower()}", n)
    if "PRIMARY KEY" in up:
        bump("constraint:primary_key", up.count("PRIMARY KEY"))
    if "FOREIGN KEY" in up:
        bump("constraint:foreign_key", up.count("FOREIGN KEY"))
    if "REFERENCES" in up:
        bump("constraint:references", up.count("REFERENCES"))
    if "NOT NULL" in up:
        bump("constraint:not_null", up.count("NOT NULL"))
    if "COMMENT" in up:
        bump("table:comment")
    if "USING ICEBERG" in up:
        bump("iceberg:using_iceberg")
    if "PARTITIONED BY" in up:
        bump("iceberg:partitioned_by")
    # cardinality decoded from the relation notes embedded in COMMENT/DDL is added
    # by the driver from CheckNotes; here we only see the rendered text.

    if parse:
        try:
            ast = _polyglot().parse_one(ddl, dialect=dialect)
            _walk_kinds(ast.to_dict(), feats)
        except Exception:
            pass  # validation is the gate; AST augmentation is best-effort
    return feats


def _walk_kinds(node, feats: dict[str, int]) -> None:
    """Recursively tally node type tags in a polyglot ``to_dict()`` structure."""
    if isinstance(node, dict):
        tag = node.get("class") or node.get("kind") or node.get("type")
        if isinstance(tag, str):
            k = f"ast:{tag.lower()}"
            feats[k] = feats.get(k, 0) + 1
        for v in node.values():
            _walk_kinds(v, feats)
    elif isinstance(node, list):
        for v in node:
            _walk_kinds(v, feats)


def schema_to_unit(spine: list[SpineTable], fks: list[FKEdge]) -> UnitSchema:
    """Assemble the full :class:`UnitSchema` (for the optional r_axiom cross-check)."""
    return UnitSchema(tables=[st.table for st in spine], fk_edges=list(fks))


def column_dicts(st: SpineTable) -> list[dict]:
    """Structured per-column schema (name/sql_type/nullable/pk) for the Atlas projector.

    Equivalent to what a parser would recover from the rendered DDL, but taken
    straight from the deterministic lowering so the projector needn't navigate the
    AST. The DDL text remains the published artifact and is independently validated.
    """
    out: list[dict] = []
    for c in st.table.columns:
        pk = c.slot_ref == "__pk__"
        owl = st.template.slot_types.get(c.slot_ref, "Class")
        out.append({
            "name": c.name,
            "sql_type": "VARCHAR(255)" if pk else sql_type_for_slot(owl),
            "nullable": (not pk) and (c.name not in st.not_null),
            "pk": pk,
        })
    return out
