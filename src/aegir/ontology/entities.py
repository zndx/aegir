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
    lines: list[str] = [PREFIXES, ""]
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
            conj.append(f"{a.iri()} some xsd:{xsd}")
            dataprops.setdefault(a.iri(), a)
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
