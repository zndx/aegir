"""Stochastic (seed-deterministic) schema-realization — one ontology template → a *schema subgraph*.

The DDL+views are a primary, super-linear deliverable (the joint train/eval substrate for Aegir's
byte-level HNet+RWKV and Atelier's Dempster-Shafer metadata understanding). A flat one-table-per-template
spine can't exercise the structures real metadata systems use, so this realizes each template under a
sampled *structural profile* into a multi-table subgraph + reconstruction views:

  * ``normalized``  — the one-table baseline (kept for diversity; some schemas are flat).
  * ``eav``         — entity + attribute-registry + per-datatype value tables + a reconstruction view.
                      The classic **Entity-Attribute-Value** pattern (Fowler, *PoEAA*).
  * ``junction``    — many-to-many **association-class** tables (Codd/Chen) for relational restrictions,
                      with their own attributes + a denormalized join view.
  * ``star``        — a **dimensional** fact table + dimension tables (Kimball) for measure-bearing concepts.
  * ``snowflake``   — a star whose dimensions are normalized into sub-dimension hierarchies (Kimball).

**Clean-room (load-bearing, see memory cleanroom_ddl_generation):** the *patterns* above are public prior
art (Fowler / Kimball / Codd) — Magento/Odoo/SENAITE are copyleft *instances* we take inspiration from but
copy nothing of. Every identifier, attribute, FK and value here is generated from OUR ontology (BFO/CCO +
SKOS) + the seed — original expression, not their schemas.

Output ``RealizedSchema`` plugs into the existing spine: its ``tables`` are ``SpineTable``s (tagged with
``kind``/``realize_meta``), its ``fks`` are intra-subgraph ``FKEdge``s, and ``rows.materialize_rows`` reads
the metadata to fill EAV/junction/fact tables RI-safe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from aegir.ontology.ddl import (CheckNote, SpineTable, ViewSpec, _dataprop_ranges, _prop_col,
                                anchor_attributes, col_name, parse_restrictions, semantic_col_names,
                                table_name)
from aegir.ontology.schema import CatalogTemplate
from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec

# xsd range → EAV value-table group (the per-datatype split, à la EAV; names are generic SQL, not a port).
_VALUE_GROUP = {
    "xsd:string": "varchar", "xsd:dateTime": "datetime", "xsd:date": "date",
    "xsd:integer": "int", "xsd:int": "int", "xsd:long": "int",
    "xsd:decimal": "decimal", "xsd:double": "decimal", "xsd:float": "decimal", "xsd:boolean": "boolean",
}
_GROUP_XSD = {"varchar": "xsd:string", "datetime": "xsd:dateTime", "date": "xsd:date",
              "int": "xsd:integer", "decimal": "xsd:decimal", "boolean": "xsd:boolean"}
_NUMERIC_XSD = {"xsd:integer", "xsd:int", "xsd:long", "xsd:decimal", "xsd:double", "xsd:float"}
_GENERIC = {"class", "subclass", "basic", "template", "generic", "foundation", "long", "tail",
            "complex", "axiom", "the", "and", "for", "with", "via", "to", "of", "an", "from"}

# Per-family profile mix (the archetype library): which real-world structural shapes each family leans to.
# observation→LIMS (facts+EAV+junctions), long_tail→retail/ERP (EAV+junctions), lineage→junctions, etc.
FAMILY_PROFILES: dict[str, dict[str, float]] = {
    "01_foundation": {"normalized": 0.6, "eav": 0.4},
    "02_observation_measurement": {"star": 0.3, "snowflake": 0.15, "eav": 0.3, "junction": 0.25},
    "03_directive_governance": {"normalized": 0.35, "junction": 0.4, "eav": 0.25},
    "04_ebpf_kernel": {"star": 0.4, "normalized": 0.3, "junction": 0.3},
    "05_provo_lineage": {"junction": 0.5, "normalized": 0.25, "star": 0.25},
    "06_belief_structure": {"eav": 0.45, "junction": 0.3, "normalized": 0.25},
    "07_long_tail": {"eav": 0.4, "junction": 0.35, "star": 0.25},
}
_DEFAULT_MIX = {"normalized": 0.3, "eav": 0.3, "junction": 0.25, "star": 0.15}

# Closes the ontology↔DDL loop: an axiom pattern's grounds_ddl (patterns.py) → the realize profile that
# faithfully materializes it. So a reified-relation primitive deterministically becomes a junction (not a
# family-sampled guess) — the structure the ontology asserts IS the structure the DDL realizes.
GROUNDS_TO_PROFILE = {
    "junction": "junction", "eav": "eav", "dimension": "star", "nested-child": "junction",
    "composite-datatype": "normalized", "enum": "normalized", "constraint": "normalized",
    "subsumption": "normalized", "relation": "junction",
}


def profile_for_grounds(grounds_ddl: str) -> str:
    """The realize profile that grounds an ontology pattern's declared DDL structure."""
    return GROUNDS_TO_PROFILE.get(grounds_ddl, "normalized")


@dataclass
class RealizedSchema:
    template: CatalogTemplate
    family: str
    profile: str
    tables: list[SpineTable]
    fks: list[FKEdge]
    views: list[ViewSpec] = field(default_factory=list)
    complexity: dict = field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────────────
def _concept(template: CatalogTemplate) -> str:
    toks = [t for t in re.split(r"[^A-Za-z]+", template.template_id)
            if len(t) >= 3 and t.lower() not in _GENERIC]
    return (toks[0].lower() if toks else "entity")


def _pk() -> ColumnSpec:
    return ColumnSpec(name="id", slot_type="Class", slot_ref="__pk__")


def _mk(name: str, ref: str, cols: list[ColumnSpec], *, family: str, template: CatalogTemplate,
        kind: str, meta: dict | None = None, not_null: set[str] | None = None,
        notes: list[CheckNote] | None = None) -> SpineTable:
    return SpineTable(template=template, family=family,
                      table=TableSpec(name=name, ref=ref, columns=cols, rows=[]),
                      notes=notes or [], not_null=not_null or set(), kind=kind, realize_meta=meta or {})


def _entity_columns(template: CatalogTemplate) -> list[ColumnSpec]:
    """id + the Class/Individual (subject + relation-target) slot columns — NOT the DataProperties."""
    names = semantic_col_names(template)
    cols = [_pk()]
    for slot, owl in template.slot_types.items():
        if owl in ("Class", "Individual", "NamedIndividual"):
            cols.append(ColumnSpec(name=names[slot], slot_type=owl, slot_ref=slot))
    if len(cols) == 1:
        cols.append(ColumnSpec(name="subject", slot_type="Class", slot_ref="subject"))
    return cols


def _attributes(template: CatalogTemplate) -> list[tuple[str, str]]:
    """(attr_name, xsd_type) for every typed attribute: the full anchor DataProperty pool + DataProperty slots."""
    out: list[tuple[str, str]] = []
    anchor = template.bfo_anchor_path[-1] if template.bfo_anchor_path else None
    if anchor:
        for col, xsd, _iri in anchor_attributes(anchor, template.template_id, budget=0):  # FULL pool for EAV richness
            out.append((col, xsd))
    dp = _dataprop_ranges(template.manchester_template)
    for slot, owl in template.slot_types.items():
        if owl == "DataProperty":
            out.append((col_name(slot), dp.get(slot, "xsd:string")))
    seen: set[str] = set()
    return [(n, t) for n, t in out if not (n in seen or seen.add(n))]


def _complexity(rs_tables, rs_fks, rs_views, *, eav: int = 0, junctions: int = 0, attrs: int = 0) -> dict:
    n = len(rs_tables)
    fk_depth = 0
    # crude FK-depth: longest chain length over the intra-subgraph edges
    succ: dict[str, list[str]] = {}
    for e in rs_fks:
        succ.setdefault(e.src_table, []).append(e.dst_table)

    def depth(t, seen):
        if t in seen:
            return 0
        return 1 + max((depth(d, seen | {t}) for d in succ.get(t, [])), default=0)
    if succ:
        fk_depth = max(depth(t, set()) for t in succ)
    return {"n_tables": n, "n_views": len(rs_views), "n_fks": len(rs_fks),
            "eav_tables": eav, "junction_tables": junctions, "attr_count": attrs,
            "eav_ratio": round(eav / n, 3) if n else 0.0,
            "m2n_density": round(junctions / n, 3) if n else 0.0, "fk_depth": fk_depth}


# ── profile: normalized (baseline) ────────────────────────────────────────────
def _realize_normalized(template: CatalogTemplate, family: str, rng) -> RealizedSchema:
    from aegir.ontology.ddl import template_to_table
    st = template_to_table(template, family)
    return RealizedSchema(template, family, "normalized", [st], [], [],
                          _complexity([st], [], [], attrs=len(_attributes(template))))


# ── profile: EAV (Fowler) ─────────────────────────────────────────────────────
def _realize_eav(template: CatalogTemplate, family: str, rng) -> RealizedSchema:
    attrs = _attributes(template)
    if len(attrs) < 2:                                  # not enough attributes to be worth EAV
        return _realize_normalized(template, family, rng)
    base = table_name(template.template_id)
    concept = _concept(template)
    entity = _mk(base, template.template_id, _entity_columns(template),
                 family=family, template=template, kind="entity")
    reg_name = f"{base}_attr"
    registry = _mk(reg_name, template.template_id, [
        _pk(), ColumnSpec("attr_name", "xsd:string", "data:attr_name"),
        ColumnSpec("attr_type", "xsd:string", "data:attr_type")],
        family=family, template=template, kind="eav_registry", meta={"attributes": attrs})

    groups: dict[str, list[str]] = {}
    for name, xsd in attrs:
        groups.setdefault(_VALUE_GROUP.get(xsd, "varchar"), []).append(name)

    tables = [entity, registry]
    fks: list[FKEdge] = []
    for grp, members in sorted(groups.items()):
        vt = f"{base}_val_{grp}"
        vcols = [_pk(), ColumnSpec("entity_id", "Class", "fk:entity"),
                 ColumnSpec("attr_id", "Class", "fk:attr"),
                 ColumnSpec("value", _GROUP_XSD.get(grp, "xsd:string"), "data:value")]
        tables.append(_mk(vt, template.template_id, vcols, family=family, template=template,
                          kind="eav_value", not_null={"entity_id", "attr_id"},
                          meta={"value_type": grp, "entity_table": base, "registry_table": reg_name,
                                "members": members}))
        fks.append(FKEdge(vt, "entity_id", base, "id", "eav_has_attribute"))
        fks.append(FKEdge(vt, "attr_id", reg_name, "id", "eav_attribute_def"))

    views = [_eav_long_view(base, reg_name, [t.table.name for t in tables if t.kind == "eav_value"],
                            entity, registry, concept)]
    n_eav = 1 + len(groups)  # registry + value tables
    return RealizedSchema(template, family, "eav", tables, fks, views,
                          _complexity(tables, fks, views, eav=n_eav, attrs=len(attrs)))


def _eav_long_view(base, reg_name, value_tables, entity, registry, concept) -> ViewSpec:
    """The canonical EAV reconstruction (long form): UNION ALL of value tables joined to the registry,
    yielding (entity_id, attr_name, value) — the pivot-ready shadow that teaches the EAV↔relational
    equivalence. Valid Trino∩Spark (CAST + UNION ALL)."""
    parts = [f"SELECT v.entity_id, a.attr_name, CAST(v.value AS VARCHAR) AS value "
             f"FROM {vt} v JOIN {reg_name} a ON a.id = v.attr_id" for vt in value_tables]
    sql = f"CREATE VIEW {base}_long AS " + " UNION ALL ".join(parts)
    verbal = (f"Each row is one recorded attribute of a {concept} in entity-attribute-value form: the "
              f"{concept} instance, the attribute's name, and its value — the reconstruction of a wide "
              f"{concept} record from its EAV decomposition.")
    cols = [("entity_id", base, "id"), ("attr_name", reg_name, "attr_name"),
            ("value", value_tables[0] if value_tables else base, "value")]
    return ViewSpec(name=f"{base}_long", sql=sql, verbalization=verbal, columns=cols,
                    base_tables=value_tables + [reg_name], fk=None)


# ── profile: junction / association-class (Codd/Chen) ─────────────────────────
def _realize_junction(template: CatalogTemplate, family: str, rng) -> RealizedSchema:
    restrictions = parse_restrictions(template)
    if not restrictions:
        return _realize_eav(template, family, rng)
    base = table_name(template.template_id)
    concept = _concept(template)
    colnames = semantic_col_names(template)
    # subject entity table (id + subject slot only; relations become junctions)
    subj_cols = [_pk()]
    for slot, owl in template.slot_types.items():
        if owl in ("Class", "Individual") and slot not in {r.target_slot for r in restrictions}:
            subj_cols.append(ColumnSpec(colnames[slot], owl, slot))
    if len(subj_cols) == 1:
        subj_cols.append(ColumnSpec("subject", "Class", "subject"))
    subject = _mk(base, template.template_id, subj_cols, family=family, template=template, kind="entity")
    tables = [subject]
    fks: list[FKEdge] = []
    views: list[ViewSpec] = []
    n_junctions = 0
    # a small ontology-grounded association-attribute pool (the relation itself carries data)
    _assoc_attrs = [("role", "xsd:string"), ("cardinality_note", "xsd:string"), ("since", "xsd:date")]
    for i, r in enumerate(restrictions):
        # the relation (coined property) is the meaningful name; the target slot's SEMANTIC column name
        # (not its bare slot letter X/Y) names the related entity — so no 't_..._x_y' leaks into the corpus.
        rel = _prop_col(str(r.prop)) or f"rel{i}"
        tgt_name = (colnames.get(r.target_slot) or _prop_col(str(r.prop))
                    or re.sub(r"\W+", "_", r.target_slot).strip("_").lower() or f"target{i}")
        tgt_tbl = f"{base}_{tgt_name}"
        # the related entity gets its own table
        tables.append(_mk(tgt_tbl, template.template_id,
                          [_pk(), ColumnSpec(tgt_name, "Class", r.target_slot)],
                          family=family, template=template, kind="entity"))
        # association-class junction (M:N), named by its relation (Codd/Chen), with its own attributes
        jt = f"{base}__{rel}" if tgt_name == rel else f"{base}__{rel}__{tgt_name}"
        aa = _assoc_attrs[: 1 + (i % 3)]
        jcols = [_pk(), ColumnSpec(f"{concept}_id", "Class", "fk:subject"),
                 ColumnSpec(f"{tgt_name}_id", "Class", "fk:target")]
        jcols += [ColumnSpec(n, t, f"data:{n}") for n, t in aa]
        tables.append(_mk(jt, template.template_id, jcols, family=family, template=template,
                          kind="junction", not_null={f"{concept}_id", f"{tgt_name}_id"},
                          meta={"left_table": base, "right_table": tgt_tbl, "relation": rel, "dense": True}))
        fks.append(FKEdge(jt, f"{concept}_id", base, "id", rel))
        fks.append(FKEdge(jt, f"{tgt_name}_id", tgt_tbl, "id", rel))
        n_junctions += 1
        views.append(_junction_view(jt, base, tgt_tbl, concept, tgt_name, rel))
    return RealizedSchema(template, family, "junction", tables, fks, views,
                          _complexity(tables, fks, views, junctions=n_junctions,
                                      attrs=len(_attributes(template))))


def _junction_view(jt, left, right, concept, tgt_name, rel) -> ViewSpec:
    sql = (f"CREATE VIEW {jt}_resolved AS SELECT j.id AS link_id, l.id AS {concept}_id, "
           f"r.id AS {tgt_name}_id FROM {jt} j JOIN {left} l ON j.{concept}_id = l.id "
           f"JOIN {right} r ON j.{tgt_name}_id = r.id")
    verbal = (f"Each row resolves one many-to-many '{rel}' association between a {concept} and a {tgt_name}, "
              f"denormalizing the association-class link into its two endpoints.")
    return ViewSpec(name=f"{jt}_resolved", sql=sql, verbalization=verbal,
                    columns=[("link_id", jt, "id"), (f"{concept}_id", left, "id"),
                             (f"{tgt_name}_id", right, "id")],
                    base_tables=[jt, left, right], fk=None)


# ── profile: star / snowflake (Kimball) ───────────────────────────────────────
def _realize_star(template: CatalogTemplate, family: str, rng, *, snowflake: bool = False) -> RealizedSchema:
    attrs = _attributes(template)
    measures = [(n, t) for n, t in attrs if t in _NUMERIC_XSD]
    restrictions = parse_restrictions(template)
    if not measures and not restrictions:               # nothing to make a fact/dimension of
        return _realize_eav(template, family, rng)
    base = table_name(template.template_id)
    concept = _concept(template)
    colnames = semantic_col_names(template)
    fact_name = f"fact_{concept}"
    fact_cols = [_pk()]
    fks: list[FKEdge] = []
    tables: list[SpineTable] = []
    views: list[ViewSpec] = []

    # dimensions from the relation targets (+ the subject as a degenerate dim)
    dim_targets = [(re.sub(r"\W+", "_", r.target_slot).strip("_").lower() or f"dim{i}", r)
                   for i, r in enumerate(restrictions)] or [(concept, None)]
    for dname, _r in dim_targets:
        dim_tbl = f"dim_{dname}"
        dcols = [_pk(), ColumnSpec(f"{dname}_label", "xsd:string", "data:label"),
                 ColumnSpec(f"{dname}_category", "xsd:string", "data:category")]
        if snowflake:                                   # normalize one level: dim → sub-dim hierarchy
            sub = f"dim_{dname}_category"
            tables.append(_mk(sub, template.template_id,
                              [_pk(), ColumnSpec("category_name", "xsd:string", "data:category")],
                              family=family, template=template, kind="dimension"))
            dcols.append(ColumnSpec("category_id", "Class", "fk:category"))
            fks.append(FKEdge(dim_tbl, "category_id", sub, "id", "rolls_up_to"))
        tables.append(_mk(dim_tbl, template.template_id, dcols, family=family, template=template,
                          kind="dimension"))
        fk_col = f"{dname}_key"
        fact_cols.append(ColumnSpec(fk_col, "Class", f"fk:{dname}"))
        fks.append(FKEdge(fact_name, fk_col, dim_tbl, "id", "dim"))
    # measures on the fact
    for n, t in (measures or [("event_count", "xsd:integer")]):
        fact_cols.append(ColumnSpec(n, t, f"data:{n}"))
    fact = _mk(fact_name, template.template_id, fact_cols, family=family, template=template, kind="fact",
               not_null={c.name for c in fact_cols if c.slot_ref.startswith("fk:")})
    tables.insert(0, fact)
    views.append(_star_view(fact_name, [t.table.name for t in tables if t.kind == "dimension"], concept,
                            [c.name for c in fact_cols if c.slot_ref.startswith("fk:")]))
    prof = "snowflake" if snowflake else "star"
    cx = _complexity(tables, fks, views, attrs=len(attrs))
    cx["dim_tables"] = sum(1 for t in tables if t.kind == "dimension")
    return RealizedSchema(template, family, prof, tables, fks, views, cx)


def _star_view(fact, dims, concept, fk_cols) -> ViewSpec:
    joins = " ".join(f"JOIN {d} d{i} ON f.{fk} = d{i}.id"
                     for i, (d, fk) in enumerate(zip(dims, fk_cols)))
    sql = (f"CREATE VIEW {fact}_denorm AS SELECT f.* FROM {fact} f {joins}".strip())
    verbal = (f"Each row is a denormalized {concept} fact joined to its conformed dimensions — the "
              f"analytical (star-schema) view of the measures in their full dimensional context.")
    return ViewSpec(name=f"{fact}_denorm", sql=sql, verbalization=verbal,
                    columns=[("*", fact, "*")], base_tables=[fact] + dims, fk=None)


# ── dispatcher ────────────────────────────────────────────────────────────────
_GENERATORS = {
    "normalized": _realize_normalized, "eav": _realize_eav, "junction": _realize_junction,
    "star": _realize_star, "snowflake": lambda t, f, r: _realize_star(t, f, r, snowflake=True),
}


def choose_profile(template: CatalogTemplate, family: str, rng) -> str:
    mix = FAMILY_PROFILES.get(family, _DEFAULT_MIX)
    profiles, weights = zip(*mix.items())
    return rng.choices(list(profiles), weights=list(weights), k=1)[0]


def realize_schema(template: CatalogTemplate, family: str, *, profile: str | None = None,
                   rng=None) -> RealizedSchema:
    """Realize a template into a schema subgraph under a structural ``profile`` (sampled by family
    archetype if not given). Falls back gracefully when a profile's preconditions aren't met."""
    import random
    rng = rng or random.Random(0)
    profile = profile or choose_profile(template, family, rng)
    gen = _GENERATORS.get(profile, _realize_normalized)
    rs = gen(template, family, rng)
    return rs
