"""Entity-centric ontology model — the greenfield derive's core representation (#141).

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

import re
from dataclasses import dataclass, field

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
    junction (min/max > 1). The lever the current ontology has zero of."""
    prop: str
    target: str  # a class concept name (→ sdg:Target) or a prefixed IRI
    card: str = "some"  # some | only | exactly N | min N | max N

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
    genus: str = "cco:Artifact"     # BFO/CCO anchor (prefixed IRI)
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

    # The Ontology: declaration is REQUIRED for OWLAPI's Manchester loader, and every property
    # (object/data/annotation) MUST be declared before use — without them the doc "loads" as
    # zero frames → a VACUOUS HermiT certificate (kvasir tolerates both omissions; measured).
    lines: list[str] = [
        PREFIXES, "", "Ontology: <https://signals.zndx.org/sdg/greenfield>", "",
        "AnnotationProperty: rdfs:label", "AnnotationProperty: iao:0000115",
        "AnnotationProperty: skos:definition", "",
    ]
    for op in sorted({r.iri() for e in entities for r in e.relations}):
        lines.append(f"ObjectProperty: {op}")
    lines.append("")
    # Declarations BEFORE use (the OWLAPI Manchester parse is effectively single-pass):
    # data properties referenced in restrictions, and external genera (cco:/bfo:/…) that no
    # Class frame in this doc otherwise declares.
    for dp in sorted({_attr_iri(a) for e in entities for a in e.attributes}):
        lines.append(f"DataProperty: {dp}")
    local = {e.iri() for e in entities}
    for ext in sorted({e.genus for e in entities if e.genus} - local):
        lines.append(f"Class: {ext}")
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
            xsd = a.xsd if a.xsd in _XSD else "string"
            conj.append(f"{_attr_iri(a)} some xsd:{xsd}")
            dataprops.setdefault(_attr_iri(a), a)
        for r in e.relations:
            card = r.card if _CARD.match(r.card) else "some"
            conj.append(f"{r.iri()} {card} {r.target_iri()}")
        if conj:
            lines.append("    SubClassOf: " + ", ".join(conj))
        lines.append("")

    for iri, a in sorted(dataprops.items()):
        lines.append(f"DataProperty: {iri}")
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


def clean_iri(tok: str, default: str = "cco:Artifact") -> str:
    """Extract a prefixed IRI from a token the agent may have decorated (``bfo:0000040
    (Person)`` → ``bfo:0000040``); default if none is present."""
    m = _IRI.search(tok or "")
    return m.group(1) if m else default


def _prefix(name: str) -> str:
    """A short uppercase key prefix from a class name (``LandParcel`` → ``LAND``)."""
    caps = re.findall(r"[A-Z]", camel(name))
    return ("".join(caps[:4]) or camel(name)[:4]).upper()


def _cell(a: "DataAttr", i: int) -> str:
    """A domain-plausible sample value for attribute ``a`` at row ``i`` (RI-true rows for prose)."""
    if a.enum:
        return a.enum[i % len(a.enum)]
    x = a.xsd
    stem = re.sub(r"(?<!^)(?=[A-Z])", " ", prop_name(a.name)).lower().replace("has ", "")
    if x in ("integer", "int", "long"):
        return str((i + 1) * 7 + hash(a.name) % 40)
    if x in ("decimal", "double", "float"):
        return f"{((i + 1) * 3.5 + hash(a.name) % 20):.2f}"
    if x == "boolean":
        return "true" if (i + hash(a.name)) % 2 else "false"
    if x == "date":
        return f"2025-{(i % 12) + 1:02d}-{(i * 7 % 27) + 1:02d}"
    if x == "dateTime":
        return f"2025-{(i % 12) + 1:02d}-{(i * 5 % 27) + 1:02d}T{(i * 3 % 24):02d}:00:00"
    return f"{stem.split()[0][:6]}-{i + 1:03d}"


def to_construct(entities: list[Entity], *, n_rows: int = 4, style_anchor: str = "") -> dict:
    """Bridge entities → the prose-harness ``construct`` dict (tables + RI-true sample rows +
    concepts), the shape ``refine/_propose._render_tables`` consumes. Single-cardinality
    relations become FK columns whose cells reference a real target-entity id (RI holds);
    many-to-many relations are omitted from the flat construct (they belong to junction views)."""
    ids = {e.iri(): [f"{_prefix(e.name)}-{i + 1:04d}" for i in range(n_rows)] for e in entities}
    tables = []
    for e in entities:
        cols = [{"name": "id", "concept": camel(e.name),
                 "cells": [{"value": v} for v in ids[e.iri()]]}]
        for a in e.attributes:
            cols.append({"name": prop_name(a.name), "concept": prop_name(a.name),
                         "cells": [{"value": _cell(a, i)} for i in range(n_rows)]})
        fks = []
        for r in e.relations:
            if r.card.startswith("min") or (r.card.startswith("max") and r.card != "max 1"):
                continue  # many-to-many → junction view, not a flat FK column
            tgt_ids = ids.get(r.target_iri())
            if not tgt_ids:
                continue
            col = prop_name(r.prop)
            cols.append({"name": col, "concept": prop_name(r.prop),
                         "cells": [{"value": tgt_ids[i % len(tgt_ids)]} for i in range(n_rows)]})
            fks.append({"col": col})
        tables.append({"name": "t_" + re.sub(r"(?<!^)(?=[A-Z])", "_", camel(e.name)).lower(),
                       "pk": "id", "fks": fks, "columns": cols})
    return {"tables": tables, "style_anchor": style_anchor,
            "entities": [e.iri() for e in entities]}


def from_json(obj: dict) -> list[Entity]:
    """Parse the engine's json_schema output into :class:`Entity` records (defensive)."""
    out: list[Entity] = []
    for e in obj.get("entities", []):
        if not isinstance(e, dict) or not e.get("name"):
            continue
        out.append(Entity(
            name=str(e["name"]),
            label=str(e.get("label", "")),
            genus=clean_iri(str(e.get("genus") or "cco:Artifact")),
            definition=str(e.get("definition", "")),
            attributes=[DataAttr(name=str(a["name"]), xsd=str(a.get("xsd", "string")),
                                 enum=[str(v) for v in (a.get("enum") or [])])
                        for a in e.get("attributes", []) if isinstance(a, dict) and a.get("name")],
            relations=[Relation(prop=str(r["prop"]), target=str(r["target"]),
                                card=str(r.get("card", "some")))
                       for r in e.get("relations", []) if isinstance(r, dict) and r.get("prop")],
        ))
    return out
