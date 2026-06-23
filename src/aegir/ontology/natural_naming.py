"""C2.5 L2 — natural physical naming (the dual-materialization).

``natural_names.json`` (seed_natural_names.py) maps each ``template_id → {table, cols}`` of DBA-realistic
physical names. This module applies that map as a CONSISTENT, RI-safe rename over an already-materialized
spine: table names, column names, and FK edges are renamed *together* — so the rename is a bijection on
physical names and referential integrity is preserved (row values + FK structure are untouched; only the
surface identifiers change). The canonical ontology identities — ``ColumnSpec.slot_ref`` and the template
``ref`` — are PRESERVED, so the natural↔semantic↔Data-Element lineage is recoverable.

Two variants, two roles ([[world_v3_augmentation]] naming policy):
  * **natural** — the CANONICAL DELIVERABLE: trains the model, what Atelier/downstream consume (a realistic
    DBA register transfers to real tables; it also breaks the concept-from-header training shortcut).
  * **semantic** — REFERENCE/ELUCIDATION only: the ontology-native names, paired in Atlas so the inscrutable
    conventions stay cognizable. NOT a training input.

Views are REBUILT from the renamed spine (``build_views``), not string-substituted — so view SQL, columns,
and verbalizations are regenerated consistently against the natural names.

Realized sub-tables (eav/junction/star) whose physical names are not in ``natural_names`` keep their
(already-meaningful, SKOS-derived) names; only the base entity table + its columns carry the LLM-coined
natural names. FK surrogate keys stay ``id`` (the seeder leaves them).
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from aegir.ontology.ddl import SpineTable, table_name
from aegir.ontology.type_check import FKEdge

_RESOURCE = Path(__file__).resolve().parent / "natural_names.json"


def load_natural_names(path: "str | Path | None" = None) -> dict:
    """``{template_id: {"table": str, "cols": {semantic_col: natural_col}}}`` — {} if absent.

    The LLM coined names per-template independently, so natural TABLE names can collide across templates
    (``users``, ``records``…). We disambiguate GLOBALLY and DETERMINISTICALLY (sorted by template_id; the
    first claimant keeps the bare name, later ones get ``_2``, ``_3``…) so every template has a STABLE,
    UNIQUE physical table name — required both for RI under rename (no ``by_name`` shadowing) and for a
    stable Atlas natural↔semantic lineage. Column names are already distinct within a table (seeder)."""
    p = Path(path) if path else _RESOURCE
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    seen: dict[str, int] = {}
    for tid in sorted(d):
        base = d[tid].get("table")
        if not base:
            continue
        if base not in seen:
            seen[base] = 1
        else:
            seen[base] += 1
            d[tid]["table"] = f"{base}_{seen[base]}"
    return d


def build_rename_maps(spine: Sequence[SpineTable], natural_names: dict
                      ) -> "tuple[dict[str, str], dict[tuple[str, str], str]]":
    """(table_name → natural_table, (table_name, col) → natural_col). Keyed on the BASE entity table only
    (``natural_names`` covers ``template_to_table``); realized sub-tables pass through unrenamed."""
    tbl_map: dict[str, str] = {}
    col_map: dict[tuple[str, str], str] = {}
    for st in spine:
        nat = natural_names.get(st.template.template_id)
        if not nat:
            continue
        if st.table.name == table_name(st.template.template_id):  # the base entity table
            if nat.get("table"):
                tbl_map[st.table.name] = nat["table"]
            for c in st.table.columns:
                if c.name in nat.get("cols", {}):
                    col_map[(st.table.name, c.name)] = nat["cols"][c.name]
    return tbl_map, col_map


def _rename_tables(spine: Sequence[SpineTable], tbl_map: dict, col_map: dict) -> list[SpineTable]:
    out: list[SpineTable] = []
    for st in spine:
        new_table = copy.deepcopy(st.table)        # TableSpec: name, ref, columns, rows
        old = st.table.name
        new_table.name = tbl_map.get(old, old)
        for c in new_table.columns:
            c.name = col_map.get((old, c.name), c.name)   # slot_ref untouched
        out.append(replace(st, table=new_table))          # share template/family/notes
    return out


def _rename_fks(fks: Sequence[FKEdge], tbl_map: dict, col_map: dict) -> list[FKEdge]:
    return [FKEdge(src_table=tbl_map.get(e.src_table, e.src_table),
                   src_col=col_map.get((e.src_table, e.src_col), e.src_col),
                   dst_table=tbl_map.get(e.dst_table, e.dst_table),
                   dst_col=col_map.get((e.dst_table, e.dst_col), e.dst_col),
                   via_slot=e.via_slot) for e in fks]


def rename_view(v, tbl_map: dict, col_map: dict):
    """Renamed copy of a ViewSpec: base_tables/base_col mapped, view-col aliases re-derived from the natural
    base col (matching render_view_ddl: projection vc==bc, join vc=={alias}_{bc}), SQL regenerated. Rows are
    positional (aligned with v.columns order) so the caller keeps them unchanged."""
    bases = [tbl_map.get(b, b) for b in v.base_tables]
    fk = v.fk if (v.fk is not None and len(v.base_tables) == 2) else None
    new_cols = []
    for (_vc, bt, bc) in v.columns:
        nbt, nbc = tbl_map.get(bt, bt), col_map.get((bt, bc), bc)
        nvc = (("a_" if bt == v.base_tables[0] else "b_") + nbc) if fk else nbc
        new_cols.append((nvc, nbt, nbc))
    if fk is None:
        sel = ", ".join(f"{bc} AS {vc}" if vc != bc else bc for vc, _, bc in new_cols)
        sql = f"CREATE VIEW {v.name} AS SELECT {sel} FROM {bases[0]}"
        nfk = None
    else:
        src_col = col_map.get((v.base_tables[0], fk.src_col), fk.src_col)
        dst_col = col_map.get((v.base_tables[1], fk.dst_col), fk.dst_col)
        parts = [f"{'a' if bt == bases[0] else 'b'}.{bc} AS {vc}" for vc, bt, bc in new_cols]
        sql = (f"CREATE VIEW {v.name} AS SELECT {', '.join(parts)} "
               f"FROM {bases[0]} a JOIN {bases[1]} b ON a.{src_col} = b.{dst_col}")
        nfk = replace(fk, src_table=bases[0], src_col=src_col, dst_table=bases[1], dst_col=dst_col)
    return replace(v, sql=sql, columns=new_cols, base_tables=bases, fk=nfk)


def natural_rename(spine: Sequence[SpineTable], fks: Sequence[FKEdge], natural_names: dict
                   ) -> "tuple[list[SpineTable], list[FKEdge]]":
    """Return natural-named copies of (spine, fks). RI-preserved: deep-copies only the TableSpec (names/cols/
    rows), shares the template (canonical identity), and renames consistently. A no-op (still copies) if
    ``natural_names`` is empty."""
    tbl_map, col_map = build_rename_maps(spine, natural_names)
    return _rename_tables(spine, tbl_map, col_map), _rename_fks(fks, tbl_map, col_map)


def natural_payload(payload, natural_names: dict):
    """Renamed copy of a chapter ``RelationalPayload`` (base_tables + fks + views) — the CANONICAL natural
    variant the corpus is written around. Tables/FKs renamed; each view's refs+SQL regenerated; view rows
    (positional) carried through unchanged. RI is preserved (same values + structure, natural identifiers)."""
    tbl_map, col_map = build_rename_maps(payload.base_tables, natural_names)
    nat_tables = _rename_tables(payload.base_tables, tbl_map, col_map)
    nat_fks = _rename_fks(payload.fks, tbl_map, col_map)
    nat_views = [(rename_view(v, tbl_map, col_map), rows) for v, rows in payload.views]
    return replace(payload, base_tables=nat_tables, fks=nat_fks, views=nat_views)
