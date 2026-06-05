"""Deterministic structural verifier for generated relational units — ``r_axiom``.

This is the loop's *structure* guardrail (the `r_axiom` term of the re-grounding
invariant; see ``docs/current/src/ontology/skills_loop_spec.md`` §0/§2.2). It scores
how faithfully a generated unit's **explicit** relational schema instantiates the
slot structure of the ontology templates it claims to realize.

The key departure from the prior prose-only heuristic (header-string matching in
``verify_chapters.py``) is that S2 emits an explicit ``column → slot_ref`` map and
``fk_edges``, so the check is *exact*, not a guess:

- a template's **Class / DataProperty / Individual** slots must each appear as a
  column whose inferred value-type is compatible with the declared OWL type;
- a template's **ObjectProperty** slots must each appear as a foreign-key edge
  whose endpoints exist (the relation, represented structurally rather than as a
  column of property-name strings);
- **ungrounded** columns (a ``slot_ref`` not declared by the template) are
  penalised.

``r_axiom = (matched slot checks) / (total slot checks)`` ∈ [0, 1]. The check is
deterministic and auditable (catalog lookup + value-type inference + BFO-anchor-aware
compatibility); it returns a per-slot ``AxiomBreakdown`` so the S2 repair loop can
regenerate only the offending columns/edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aegir.ontology.schema import Catalog

# ── Explicit unit schema (what S2 emits; what r_axiom scores) ────────────────


@dataclass
class ColumnSpec:
    name: str
    slot_type: str  # OWL type of the slot, e.g. "Class" | "DataProperty" | "Individual"
    slot_ref: str   # the template slot this column instantiates (the deterministic label)


@dataclass
class TableSpec:
    name: str
    ref: str                       # template_id this table realizes
    columns: list[ColumnSpec]
    rows: list[list[str]]          # row-major; row[i] aligns with columns[i]


@dataclass
class FKEdge:
    src_table: str
    src_col: str
    dst_table: str
    dst_col: str
    via_slot: str                  # the ObjectProperty slot this edge instantiates


@dataclass
class UnitSchema:
    tables: list[TableSpec]
    fk_edges: list[FKEdge] = field(default_factory=list)


@dataclass
class SlotCheck:
    table: str
    slot: str
    owl_type: str
    ok: bool
    inferred: str | None
    reason: str


@dataclass
class AxiomBreakdown:
    checks: list[SlotCheck]

    @property
    def score(self) -> float:
        return (sum(1.0 for c in self.checks if c.ok) / len(self.checks)
                if self.checks else 0.0)

    def offending(self) -> list[SlotCheck]:
        return [c for c in self.checks if not c.ok]


# ── Value-type inference + OWL compatibility (deterministic) ──────────────────

_NUM = re.compile(r"^[+-]?\d{1,3}(?:[,\d]*)(?:\.\d+)?$|^[+-]?\.\d+$|^[+-]?\d+\.?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_BOOL = {"true", "false", "yes", "no", "t", "f"}

_ENTITY_TYPES = {"Class", "Individual", "NamedIndividual"}
_DATA_TYPES = {"DataProperty", "Datatype", "Literal", "Data", "xsd:string",
               "xsd:integer", "xsd:decimal", "xsd:dateTime"}


def infer_value_type(values: list[str]) -> str:
    """Infer an OWL value-category for a column from its cell values.

    Deterministic heuristic: a column that is mostly numbers / dates / booleans is
    ``Data``; otherwise textual identifiers are entity-like (``Class``). (Object
    properties are represented as FK edges, not columns, so they are not inferred
    here.) Returns ``"Empty"`` for an all-blank column.
    """
    vals = [str(v).strip() for v in values if str(v).strip() != ""]
    if not vals:
        return "Empty"
    data_like = sum(
        1 for v in vals
        if _NUM.match(v.replace(",", "")) or _DATE.match(v) or v.lower() in _BOOL
    )
    return "Data" if data_like / len(vals) >= 0.6 else "Class"


def type_compatible(expected_owl: str, inferred: str,
                    bfo_anchor_path: list[str] | None = None,
                    catalog: Catalog | None = None) -> bool:
    """Is an inferred column value-type compatible with the declared OWL slot type?

    v0 core: OWL *category* match (entity↔entity, data↔data). ``bfo_anchor_path``
    is accepted so a future version can tighten this to BFO/CCO subsumption of the
    column's resolved class under the slot's anchor; today it is advisory only.
    """
    if inferred == "Empty":
        return False
    if expected_owl in _ENTITY_TYPES:
        return inferred == "Class"
    if expected_owl in _DATA_TYPES:
        return inferred == "Data"
    # Unknown/!-handled OWL type: accept any non-empty column (lenient, v0).
    return True


def fk_valid(edge: FKEdge, schema: UnitSchema, catalog: Catalog | None = None) -> bool:
    """An FK edge is valid if both endpoint tables/columns exist in the unit."""
    by_name = {t.name: t for t in schema.tables}
    src, dst = by_name.get(edge.src_table), by_name.get(edge.dst_table)
    if src is None or dst is None:
        return False
    return (edge.src_col in {c.name for c in src.columns}
            and edge.dst_col in {c.name for c in dst.columns})


# ── r_axiom ──────────────────────────────────────────────────────────────────


def r_axiom(schema: UnitSchema, catalog: Catalog) -> tuple[float, AxiomBreakdown]:
    """Score structural fidelity of an explicit unit schema against the catalog.

    For each table's template: every Class/Data slot must be a type-compatible
    column; every ObjectProperty slot must be a valid FK edge; ungrounded columns
    are penalised. Returns ``(score ∈ [0,1], breakdown)``.
    """
    checks: list[SlotCheck] = []

    for table in schema.tables:
        try:
            tmpl = catalog.by_id(table.ref)
        except KeyError:
            checks.append(SlotCheck(table.name, table.ref, "—", False, None,
                                    "table.ref not in catalog"))
            continue
        slot_types = tmpl.slot_types
        anchor = tmpl.bfo_anchor_path

        # slot_ref -> (column_index, column); first occurrence wins
        col_by_slot: dict[str, tuple[int, ColumnSpec]] = {}
        for ci, c in enumerate(table.columns):
            col_by_slot.setdefault(c.slot_ref, (ci, c))
        fk_by_slot = {e.via_slot: e for e in schema.fk_edges if e.src_table == table.name}

        for slot, owl_type in slot_types.items():
            if owl_type == "ObjectProperty":
                e = fk_by_slot.get(slot)
                ok = e is not None and fk_valid(e, schema, catalog)
                checks.append(SlotCheck(
                    table.name, slot, owl_type, ok, "FK" if e else None,
                    "FK present + valid" if ok else "missing/invalid FK for object-property"))
            else:
                hit = col_by_slot.get(slot)
                if hit is None:
                    checks.append(SlotCheck(table.name, slot, owl_type, False, None,
                                            "slot has no column"))
                    continue
                ci, _ = hit
                col_vals = [row[ci] for row in table.rows if ci < len(row)]
                inferred = infer_value_type(col_vals)
                ok = type_compatible(owl_type, inferred, anchor, catalog)
                checks.append(SlotCheck(table.name, slot, owl_type, ok, inferred,
                                        f"inferred {inferred} vs declared {owl_type}"))

        # Penalise ungrounded columns (slot_ref not declared by the template).
        for c in table.columns:
            if c.slot_ref not in slot_types:
                checks.append(SlotCheck(table.name, c.slot_ref or c.name, "—", False, None,
                                        "ungrounded column (slot_ref not in template)"))

    bd = AxiomBreakdown(checks)
    return bd.score, bd
