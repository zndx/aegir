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
import re
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from aegir.ontology.ddl import SpineTable, table_name
from aegir.ontology.type_check import FKEdge

_RESOURCE = Path(__file__).resolve().parent / "natural_names.json"

_IDENT = re.compile(r"^[a-z][a-z0-9_]{1,38}$")
# suffixes a DBA appends to a stem; stripped when a column alias is reused as a NAME SEGMENT
# (junction/dim composition) so `admin_id` composes to `auth_grants_admin`, not `..._admin_id_id`
_SEG_SUFFIX = re.compile(r"_(id|ids|ref|code|cd|key|no|num|nm|txt|dt|ts)$")


def stem(name: str) -> str:
    """A natural name reduced to a composable segment: conventional suffix stripped, crude de-plural."""
    s = _SEG_SUFFIX.sub("", name)
    if len(s) > 4 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s or name


def _toks(name: str) -> "set[str]":
    return {t for t in name.lower().split("_") if t}


def _echo_toks(name: str) -> "set[str]":
    """Token set for de-echo comparison: the ``t_`` physical prefix dropped (it is register plumbing,
    not concept content — leaving it lets a verbatim concept copy pass containment) and each token
    crudely de-pluralized (``lecture_sessions`` must not escape as a rename of ``lecture_session``)."""
    toks = [t for t in name.lower().split("_") if t]
    if toks and toks[0] == "t":
        toks = toks[1:]
    out = set()
    for t in toks:
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.add(t)
    return out


# ── the membrane (returns its REASON — the agent's feedback channel, Convert 1c) ─
def check_names(semantic_table: str, semantic_cols: "list[str]", proposal: "dict | None"
                ) -> "tuple[bool, str, dict | None]":
    """Dispose one template's proposed natural names → (ok, reason, cleaned).

    The design rule is COOPERATIVE SIGNAL, INDEPENDENT AUTHORSHIP: a DBA names to facilitate usage, so
    natural names should be as informative as helpful naming makes them — the membrane does NOT enforce
    weakness. What it enforces is the authorship separation reality has by construction (schema author ≠
    taxonomy author): our semantic names and our vocabulary labels share an author, so a verbatim/
    containment echo of the ontology surface is the shared-author circularity leaking through — the one
    thing a downstream tagging benchmark must not contain. Echo rejection ≠ obfuscation (machine-generated
    opaque naming is a different population entirely). Every rejection carries the reason the proposer
    re-authors against ([[agent_mediated_feedback_loop]]). Checks: parse/shape, identifier validity,
    distinctness, FULL column coverage (a half-named table is a mixed-register surface — worse than either
    register), table-name de-echo (identity/containment of the semantic tokens), aggregate column echo rate."""
    if not proposal or not proposal.get("table"):
        return False, "no NAT_TABLE line parsed — re-emit the full NAT_TABLE + NAT_COL contract", None
    table = proposal["table"].lower()
    cols: dict[str, str] = {k: v.lower() for k, v in (proposal.get("cols") or {}).items()}
    if not _IDENT.match(table):
        return False, f"table name '{table}' is not a valid snake_case SQL identifier", None
    bad = [n for n in cols.values() if not _IDENT.match(n)]
    if bad:
        return False, f"invalid column identifier(s): {', '.join(sorted(bad)[:4])}", None
    missing = [c for c in semantic_cols if c not in cols]
    if missing:
        return False, (f"missing natural names for column(s): {', '.join(missing[:6])} — every column "
                       "needs one (a partially-renamed table leaks the ontology register)"), None
    if "id" in cols.values():
        clash = [s for s, n in cols.items() if n == "id"]
        return False, (f"'id' is the reserved surrogate-key name — rename column(s) {', '.join(clash[:3])} "
                       "(e.g. a '<stem>_id' business key)"), None
    if len(set(cols.values())) != len(cols):
        dupes = sorted({v for v in cols.values() if list(cols.values()).count(v) > 1})
        return False, f"duplicate natural column name(s): {', '.join(dupes[:4])}", None
    # table-level de-echo: identity or wholesale containment of the semantic tokens (t_-prefix dropped,
    # plural-normalized) = the concept copied through, which defeats the register's purpose
    st_toks, nt_toks = _echo_toks(semantic_table), _echo_toks(table)
    if nt_toks == st_toks or (st_toks and st_toks <= nt_toks):
        return False, (f"table name '{table}' echoes the ontology name '{semantic_table}' — coin a "
                       "DBA-register name decoupled from the concept label (abbreviate or rephrase, "
                       "not just re-inflect)"), None
    ident_cols = [s for s, n in cols.items() if n.replace("_", "") == s.replace("_", "")]
    if ident_cols:
        return False, (f"column(s) named identically to the ontology name: {', '.join(ident_cols[:4])} "
                       "— every natural name must differ from its ontology-derived source"), None
    echo = sum(1 for s, n in cols.items() if _toks(s) & _toks(n))
    if len(cols) >= 3 and echo / len(cols) > 0.8:
        return False, (f"{echo}/{len(cols)} column names share exact tokens with their ontology names — "
                       "too literal overall; abbreviate/rephrase (e.g. 'administrator' → 'admin_id')"), None
    return True, "", {"table": table, "cols": cols}


def provenance_of(rec: dict) -> str:
    """The name_provenance tag for a natural_names record: engine-derived-under-membrane vs the
    pre-membrane static resource (legacy records carry no provenance key)."""
    src = (rec.get("provenance") or {}).get("source")
    return src or "static-legacy"


_VOWELS = set("aeiou")


def degrade_name(name: str, *, max_tok: int = 5) -> str:
    """Deterministic cryptic-DBA abbreviation — the DEPLOYMENT-REALISTIC degraded form of a name.

    Real warehouses never present nameless columns; the degraded form of a name is a cryptic one
    (``accnt_athrz_rcrd``), not ``col3``. Used when no natural-register record exists for a column
    (pre-1c footprints, unresolved templates): keeps the surface name-shaped and the evidence channel
    live while damping exact-token vocabulary matching (``accnt`` ≠ ``account``). Deterministic —
    no dice-roll; the information loss is the honest signal of the missing derivation, stamped
    ``degraded-mechanical`` in name_provenance so consumers can slice by grade."""
    toks = [t for t in name.lower().split("_") if t]
    if toks and toks[0] == "t" and len(toks) > 1:
        toks = toks[1:]
    out = []
    for t in toks:
        if len(t) <= 4:
            out.append(t)
            continue
        body = t[0] + "".join(c for c in t[1:] if c not in _VOWELS)
        out.append(body[:max_tok])
    return "_".join(out) or name


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
