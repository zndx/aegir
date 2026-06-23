"""Per-chapter **relational payload**: the bridge from the topic-first template
selection to the *fixed, populated, referentially-closed* tables a chapter is
written around (Track A).

Given the chapter's chosen templates (+ the family complex), this lowers them to
the DDL spine, discovers the sanctioned FK edges, **materializes RI-true rows**
(:mod:`aegir.ontology.rows`, with values grounded in the ontology's DataProperty
definitions where they enumerate), and builds the corpus **views** (single-table
projections + FK joins) with their rows pre-computed. The returned
:class:`RelationalPayload` is the single source of truth handed to the generator
prompt and echoed in the verifiable footer — so RI = 1.0 is a *constructed*
property of every chapter, not an emergent one.

Pure/deterministic (no LLM/JVM/GPU); view SQL is produced by the deterministic
:func:`ddl.render_view_ddl` and its rows by a tiny in-memory evaluator (no SQL
engine). Optional polyglot validation of the view SQL is left to the spine build.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import re

from aegir.ontology.ddl import (SpineTable, ViewSpec, _prop_col, cross_family_fks, dataprop_meta,
                                render_view_ddl, template_to_table)
from aegir.ontology.rows import assert_referential_integrity, materialize_rows
from aegir.ontology.schema import CatalogTemplate
from aegir.ontology.type_check import FKEdge

_CHAPTER_SEED = 0xAE61


@dataclass
class RelationalPayload:
    """The fixed tables a chapter is written around. ``views`` pairs each ViewSpec with its
    pre-computed projected rows; ``base_tables`` carry the authoritative (RI-true) base rows."""
    base_tables: list[SpineTable]
    fks: list[FKEdge]
    views: list[tuple[ViewSpec, list[list[str]]]] = field(default_factory=list)


def spine_from_dicts(templates: Sequence[dict]) -> list[SpineTable]:
    """Lower the chosen enriched-template dicts (generate_chapter's ``chosen``) to the spine."""
    spine: list[SpineTable] = []
    for t in templates:
        ct = CatalogTemplate(
            template_id=t["template_id"], manchester_template=t.get("manchester_template", ""),
            slot_types=t.get("slot_types", {}), is_complex=t.get("is_complex", False),
            verbal_template=t.get("verbal_template", ""), bfo_anchor_path=t.get("bfo_anchor_path", []),
        )
        spine.append(template_to_table(ct, t.get("_family", "")))
    return spine


def definitions_for_spine(spine: Sequence[SpineTable]) -> dict[str, dict[str, str]]:
    """``{table → {col → skos:definition}}`` for anchor-attribute columns — so enumerated
    value sets (e.g. ``status`` → pending/running/complete/failed) ground the rows in the ontology."""
    meta = dataprop_meta()
    out: dict[str, dict[str, str]] = {}
    for st in spine:
        d: dict[str, str] = {}
        for c in st.table.columns:
            if c.slot_ref.startswith("data:"):
                iri = c.slot_ref[len("data:"):]
                defn = meta.get(iri, ("", "", ""))[2]
                if defn:
                    d[c.name] = defn
        if d:
            out[st.table.name] = d
    return out


_ENTITY_POOLS_PATH = Path(__file__).resolve().parent / "entity_value_pools.json"
_ENTITY_POOLS_CACHE: "dict[str, dict[str, list[str]]] | None" = None


def _entity_value_pools() -> "dict[str, dict[str, list[str]]]":
    """The committed ``template_id → {col → [domain values]}`` resource (LLM-seeded, RI-safe), cached.
    Absent file → empty (the generator falls back to curated pools + type generators)."""
    global _ENTITY_POOLS_CACHE
    if _ENTITY_POOLS_CACHE is None:
        try:
            _ENTITY_POOLS_CACHE = json.loads(_ENTITY_POOLS_PATH.read_text())
        except (OSError, ValueError):
            _ENTITY_POOLS_CACHE = {}
    return _ENTITY_POOLS_CACHE or {}


def entity_pools_for_spine(spine: Sequence[SpineTable]) -> dict[str, dict[str, list[str]]]:
    """``{table → {col → [domain values]}}`` from the committed entity_value_pools.json — concept-specific
    instance values for the entity/name columns that would otherwise be ``"<Concept> NN"`` placeholders
    (Comp 4). Keyed to table_name (via template_id) for :func:`rows.materialize_rows`; non-FK only."""
    pools = _entity_value_pools()
    out: dict[str, dict[str, list[str]]] = {}
    for st in spine:
        rec = pools.get(st.template.template_id)
        if rec:
            cols = {c.name for c in st.table.columns}
            keep = {col: vals for col, vals in rec.items() if col in cols and vals}
            if keep:
                out[st.table.name] = keep
    return out


def eval_view_rows(view: ViewSpec, base_by_name: dict[str, SpineTable]) -> list[list[str]]:
    """Project/equi-join the base rows per ``view.columns`` — render_view_ddl only ever emits a
    single-table projection or a 2-table join on ``a.{fk.src_col} = b.{fk.dst_col}``."""
    def colmap(tname: str) -> tuple[dict[str, int], list[list[str]]]:
        st = base_by_name[tname]
        return {c.name: i for i, c in enumerate(st.table.columns)}, st.table.rows

    if not view.columns:
        return []
    if view.fk is None or len(view.base_tables) == 1:
        bt = view.base_tables[0]
        m, rs = colmap(bt)
        return [[r[m[bc]] for (_vc, _bt, bc) in view.columns] for r in rs]

    a, b = view.base_tables[0], view.base_tables[1]
    am, ar = colmap(a)
    bm, br = colmap(b)
    fk = view.fk
    by_key: dict[str, list[list[str]]] = {}
    for rb in br:
        by_key.setdefault(rb[bm[fk.dst_col]], []).append(rb)
    out: list[list[str]] = []
    for ra in ar:
        for rb in by_key.get(ra[am[fk.src_col]], []):
            row: list[str] = []
            for (_vc, bt, bc) in view.columns:
                src, m = (ra, am) if bt == a else (rb, bm)
                row.append(src[m[bc]])
            out.append(row)
    return out


def _view_stem(st: SpineTable) -> str:
    """A semantically meaningful concept stem for a view name — the table's ontology concept (its
    template_id, already SKOS/LLM-derived for content-first primitives), NOT the opaque ``t_…`` table id."""
    stem = re.sub(r"\W+", "_", st.template.template_id or "").strip("_").lower()
    return stem or st.table.name


def _domain_broader_map() -> "dict[str, str]":
    """member-concept → hypernym from the HermiT-admitted domain taxonomy (L1 subtree-mix source).
    Local build artifact; returns {} if absent → L1 is a graceful no-op."""
    import json as _json
    p = Path(__file__).resolve().parents[3] / "build" / "domain_taxonomy_admitted.json"
    if not p.exists():
        return {}
    from aegir.ontology.subtree_mix import concept_of
    d = _json.loads(p.read_text())
    return {concept_of(m): t["hypernym"] for t in d.get("admitted", []) for m in t["members"]}


def mixed_entity_pools(spine: Sequence[SpineTable]) -> dict:
    """``entity_pools_for_spine`` + C2.5 L1 domain subtree-mixing: a column at a parent concept draws from the
    union of its children's pools (the heterogeneous subtype population the hypernym-CTA tests). Values-only —
    FK cols are overwritten from PK pools in materialize_rows, so RI is untouched. No-op if no taxonomy."""
    pools = entity_pools_for_spine(spine)
    bmap = _domain_broader_map()
    if bmap:
        from aegir.ontology.subtree_mix import subtree_mixed_pools
        pools, _ = subtree_mixed_pools(pools, bmap)
    return pools


def build_views(spine: Sequence[SpineTable],
                fks: Sequence[FKEdge]) -> list[tuple[ViewSpec, list[list[str]]]]:
    """A projection view per base table + a join view per FK edge, each with its rows evaluated.
    View names are semantically meaningful: ``v_<concept>`` and ``v_<src>__<relation>__<dst>`` (the FK's
    relation, not a literal ``x``) — so the embedded corpus views read as domain objects, not slot ids."""
    by_name = {st.table.name: st for st in spine}
    views: list[tuple[ViewSpec, list[list[str]]]] = []
    for st in spine:
        v = render_view_ddl(f"v_{_view_stem(st)}", st)
        if v.columns:
            views.append((v, eval_view_rows(v, by_name)))
    for e in fks:
        src, dst = by_name.get(e.src_table), by_name.get(e.dst_table)
        if src is not None and dst is not None:
            rel = _prop_col(e.via_slot) or "related"   # the relation name, e.g. has_assay / part_of
            v = render_view_ddl(f"v_{_view_stem(src)}__{rel}__{_view_stem(dst)}", src, dst, e)
            if v.columns:
                views.append((v, eval_view_rows(v, by_name)))
    return views


def payload_from_spine(spine: list[SpineTable], family_complex, *,
                       seed: int = _CHAPTER_SEED) -> RelationalPayload:
    """Materialize RI-true rows + build views for an already-lowered spine."""
    fks: list[FKEdge] = []
    if family_complex is not None:
        fks, _ = cross_family_fks(spine, family_complex)
    materialize_rows(spine, fks, seed=seed, definitions=definitions_for_spine(spine),
                     entity_pools=mixed_entity_pools(spine))  # C2 de-leaked + C2.5 L1 domain-mix
    assert_referential_integrity(spine, fks)
    return RelationalPayload(base_tables=spine, fks=fks, views=build_views(spine, fks))


def realized_payload_from_dicts(templates: Sequence[dict], family_complex, *,
                                seed: int = _CHAPTER_SEED, realize_seed: int = 0x5EED) -> RelationalPayload:
    """Realize each chosen template into a stochastic schema SUBGRAPH (EAV/junction/star/snowflake) and
    assemble the chapter payload from the union — so a chapter is written around RICH relational structure
    (the super-linear DDL+views deliverable), not flat tables. RI = 1.0 by construction; clean-room."""
    import hashlib
    import random

    from aegir.ontology import realize as rz
    from aegir.ontology.ddl import table_name

    tables: list[SpineTable] = []
    rfks: list[FKEdge] = []
    rviews = []
    for t in templates:
        ct = CatalogTemplate(
            template_id=t["template_id"], manchester_template=t.get("manchester_template", ""),
            slot_types=t.get("slot_types", {}), is_complex=t.get("is_complex", False),
            verbal_template=t.get("verbal_template", ""), bfo_anchor_path=t.get("bfo_anchor_path", []))
        rseed = int.from_bytes(
            hashlib.blake2b(f"{realize_seed}:{ct.template_id}".encode(), digest_size=8).digest(), "big")
        rs = rz.realize_schema(ct, t.get("_family", ""), rng=random.Random(rseed))
        tables.extend(rs.tables)
        rfks.extend(rs.fks)
        rviews.extend(rs.views)
    primary = [st for st in tables if st.kind == "entity"
               and st.table.name == table_name(st.template.template_id)]
    cross_fks: list[FKEdge] = []
    if family_complex is not None:
        cross_fks, _ = cross_family_fks(primary, family_complex)
        cols = {st.table.name: {c.name for c in st.table.columns} for st in tables}
        cross_fks = [e for e in cross_fks
                     if e.src_col in cols.get(e.src_table, set()) and e.dst_col in cols.get(e.dst_table, set())]
    fks = rfks + cross_fks
    materialize_rows(tables, fks, seed=seed, definitions=definitions_for_spine(tables),
                     entity_pools=mixed_entity_pools(tables))  # C2 de-leaked + C2.5 L1 domain-mix
    assert_referential_integrity(tables, fks)
    views = [(v, []) for v in rviews] + build_views(primary, cross_fks)
    return RelationalPayload(base_tables=tables, fks=fks, views=views)


def chapter_relational_payload(templates: Sequence[dict], family_complex, *,
                               seed: int = _CHAPTER_SEED, realize: bool = False,
                               realize_seed: int = 0x5EED) -> RelationalPayload:
    """The fixed tables a chapter (its chosen templates) is written around. RI = 1.0 by construction.
    ``realize`` expands each template into a schema subgraph (rich structure) instead of one flat table."""
    if realize:
        return realized_payload_from_dicts(templates, family_complex, seed=seed, realize_seed=realize_seed)
    return payload_from_spine(spine_from_dicts(templates), family_complex, seed=seed)


if __name__ == "__main__":  # smoke: materialize + view-join over two real templates, RI=1.0
    from pathlib import Path

    from aegir.ontology.schema import load_catalog

    cat = load_catalog(Path("src/aegir/ontology/catalog/combined.json"))
    t_proc = next(t for t in cat.templates if t.bfo_anchor_path and "Process" in (t.bfo_anchor_path[-1] or ""))
    t_art = next(t for t in cat.templates if t.bfo_anchor_path and "Artifact" in (t.bfo_anchor_path[-1] or ""))
    spine = [template_to_table(t_proc, "fam_a"), template_to_table(t_art, "fam_b")]
    src = spine[0]
    src_col = next(c.name for c in src.table.columns if c.slot_ref != "__pk__")
    fks = [FKEdge(src.table.name, src_col, spine[1].table.name, "id", "rel")]

    materialize_rows(spine, fks, seed=7, definitions=definitions_for_spine(spine))
    assert_referential_integrity(spine, fks)
    views = build_views(spine, fks)
    print(f"{len(spine)} base tables, {len(views)} views, RI=1.0 ✓")
    join = next((v for v, _ in views if v.fk is not None), None)
    if join is not None:
        vrows = next(r for v, r in views if v is join)
        print(f"join view {join.name}:")
        print("  cols:", [vc for vc, _, _ in join.columns])
        for r in vrows[:3]:
            print("  ", r)
