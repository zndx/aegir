"""C2.5 subtree-mixing — deterministic MULTI-TEMPLATE TABLE COMPOSITION (the width layer).

The one-template-one-table spine medians at ONE classifiable column (Atelier R2 census); real
warehouses median ~5 (SchemaPile) with a long wide tail (p90 15, p99 50, clickstream-class 100s).
Templates are column-poor (2-4 typed slots each), so width cannot come from a single template — it
comes from COMPOSING related templates. This module is that composition, in two honest strata:

  * **3NF core**  (``realize_meta.layer == "core"``) — a deterministic RESERVE of templates realized
    as their normalized schema subgraphs (``realize.realize_schema``: normalized / eav / junction /
    star) PLUS ontology-derived **lookup/code tables** (enumerated value sets → a shared dimension +
    an FK), which deepen the FK chains. Narrow, RI-true, the normalized deliverable. Its per-table
    width is capped by template column-poverty (documented) — the real lever is R1's richer re-derive.
  * **denorm layer**  (``realize_meta.layer == "denorm"``) — domain-grouped **composite workbench
    tables**: k member templates' flat classifiable columns unioned under one surrogate key. This is
    a *denormalization / reporting layer* (mixed-subject by construction — HONESTLY STAMPED, never a
    3NF base table), and it carries the width tail (p90 / p99 / the ≥50 stratum). Each composite also
    gets a projection VIEW so the wide shape lands in the R4-scored view stratum too.

**Determinism (hard constraint).** Grouping is ontology-structural, never a dice-roll: templates
partition by their SKOS ``provenance.domain`` concept (top-level code), and within a group members
are sorted by an ontology proximity key (sub-domain code → BFO anchor → grounds_ddl → id). The
composition-size schedule (``WIDTH_CYCLE``) is a fixed, SchemaPile-calibrated cycle applied by sorted
position with a per-group offset. A seed may ORDER but never INVENT structure — there is no random
draw anywhere in the plan.

**Lineage (load-bearing).** Every composed column keeps its ``ColumnSpec.slot_ref`` (the member
slot / enumerated concept), and the composite records per-column source-template attribution in
``realize_meta.col_source`` (a list aligned with ``table.columns``) so ``build_atelier_release`` can
attribute each column to its origin template. Both physical-name registers (semantic / natural) are
constructed with IDENTICAL structure — the plan and the column order/count depend only on the
templates, so the naming-map twin-zip holds.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from aegir.ontology.ddl import (SpineTable, ViewSpec, _dataprop_col, col_name, render_view_ddl,
                                table_name, template_to_table)
from aegir.ontology.natural_naming import stem
from aegir.ontology.schema import CatalogTemplate
from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec

# ── SchemaPile-calibrated composition-size schedule ────────────────────────────
# Target column-widths for successive composite tables, walked cyclically per group. Calibrated
# against a light SchemaPile mine (2026-07-04, 9,850 tables: median 5 / p90 13 / p99 30 / max 193,
# 53% ≥5): dominant 5-9 (the OLTP body), a periodic 15-30 mid-band (p90), and a rare 45-130 tail
# (p99 + the deliberate clickstream-class ≥50 stratum R2 asks for). Members accumulate into a bin
# until its column count reaches the current target; the schedule is a DETERMINISTIC assignment by
# sorted position, not a distribution draw. Distribution-fit is intentionally light — a standing
# SchemaPile-mining instrument can refine these constants without touching the mechanism.
WIDTH_CYCLE: tuple[int, ...] = (
    5, 6, 7, 5, 8, 6, 7, 5, 9, 6,        # OLTP body (median driver + denorm count)
    5, 7, 6, 8, 5,                       # more body (keeps denorm plentiful → median margin)
    16, 22, 18, 15, 26, 20,              # mid-band (p90)
    55, 45,                              # tail (p99)
    6, 7, 5, 8,                          # body
    70,                                  # tail
    110,                                 # clickstream-class ≥50 stratum
    7, 6, 5,                             # body
)
# Every RESERVE_EVERY-th member of each domain group (by sorted position) is held back as the 3NF
# core (realized as a rich subgraph) instead of composed; the enum/pool-bearing templates are ALSO
# reserved (realized FLAT + a lookup FK). Small by necessity: normalized subgraphs are narrow AND
# numerous (each template → 2-3 tables), so a large reserve would drag the base-table median below the
# R2 floor of 5. This is the honest cap — a richer 3NF core needs R1 to emit wider templates, not a
# bigger reserve. Tuned (2026-07-04) so the overall base census clears median≥5 / p90≥15 / p99≥50 /
# a ≥50-col stratum with margin, while the denorm layer stays the majority.
RESERVE_EVERY: int = 90


# ── grouping (deterministic, ontology-structural) ──────────────────────────────
def domain_group(template: CatalogTemplate) -> str:
    """The composition group key: the template's SKOS domain concept, rolled up to its TOP-LEVEL code
    (``9.1`` / ``9.4`` → ``9``) so sub-concepts of one domain compose together. Domain-less templates
    (the ``define_fillers`` intermediate classes) share the ``_unassigned`` group — a genuine
    structural partition (``provenance.source``), not an invented one."""
    code = (template.provenance or {}).get("domain", {}).get("code")
    if code:
        return str(code).split(".")[0]
    return "_unassigned"


def _proximity_key(fam: str, template: CatalogTemplate) -> tuple:
    """Within-group ordering so composed members are as ontologically close as the catalog permits:
    finer sub-domain code, then shared BFO anchor, then the grounded DDL shape, then id. Pure
    deterministic ordering (the seed may reorder ties, never regroup)."""
    prov = template.provenance or {}
    subcode = str((prov.get("domain") or {}).get("code") or "")
    anchor = (template.bfo_anchor_path or [""])[-1] or ""
    grounds = str(prov.get("grounds_ddl") or "")
    return (subcode, anchor, grounds, template.template_id)


def _group_stem(members: list[tuple[str, CatalogTemplate]], group: str) -> str:
    """A short DBA-real stem for a group's composite table names, from the domain label (``Laboratory
    Information Management`` → ``lab_information``), else the code. Register-invariant."""
    label = ""
    for _fam, t in members:
        label = ((t.provenance or {}).get("domain") or {}).get("label") or ""
        if label:
            break
    if group == "_unassigned":
        return "reference"
    if label:
        stop = {"and", "the", "for", "with", "of", "to", "management"}
        toks = [w for w in re.split(r"[^A-Za-z]+", label.lower()) if len(w) >= 3 and w not in stop][:2]
        if toks:
            return "_".join(toks)
    return "dom" + re.sub(r"\W+", "", group)


# ── the plan ───────────────────────────────────────────────────────────────────
@dataclass
class CompositionPlan:
    denorm: list[list[tuple[str, CatalogTemplate]]]   # each inner list → one composite workbench table
    core: list[tuple[str, CatalogTemplate]]           # realized as a normalized subgraph (+ lookups)
    core_flat_ids: set = field(default_factory=set)   # template_ids realized FLAT (3NF OLTP + lookup FKs)
    groups: dict[str, str] = field(default_factory=dict)   # group key → stem (for naming)


_FLATCOL_CACHE: dict[str, int] = {}


def _flatcol_count(fam: str, template: CatalogTemplate) -> int:
    """Number of classifiable (non-pk) columns the template's flat table contributes to a composite."""
    tid = template.template_id
    if tid not in _FLATCOL_CACHE:
        _FLATCOL_CACHE[tid] = sum(1 for c in template_to_table(template, fam).table.columns
                                  if c.slot_ref != "__pk__")
    return _FLATCOL_CACHE[tid]


def plan_compositions(all_ft: list[tuple[str, CatalogTemplate]], *, reserve_every: int = RESERVE_EVERY,
                      width_cycle: tuple[int, ...] = WIDTH_CYCLE, lookup_cap: int = 0) -> CompositionPlan:
    """Partition templates into (denorm composition bins, 3NF-core reserve). Deterministic: group by
    domain, sort by proximity, hold back every ``reserve_every``-th member (→ rich subgraphs) PLUS the
    first member introducing each not-yet-covered curated pool (→ a FLAT 3NF entity whose lookup then
    materializes — cover-based so the flat reserve is ~one table per distinct pool, not per column),
    then greedily accumulate the rest into bins sized by the (offset) ``width_cycle``. A lone leftover
    (< 2 members) falls to core. ``lookup_cap`` (0 = uncapped cover) bounds the per-group flat reserve."""
    groups: dict[str, list[tuple[str, CatalogTemplate]]] = {}
    for fam, t in all_ft:
        groups.setdefault(domain_group(t), []).append((fam, t))
    denorm: list[list[tuple[str, CatalogTemplate]]] = []
    core: list[tuple[str, CatalogTemplate]] = []
    core_flat_ids: set = set()
    stems: dict[str, str] = {}
    covered_pools: set = set()                        # global: reserve one flat entity per NEW pool only
    for g in sorted(groups):
        mem = sorted(groups[g], key=lambda ft: _proximity_key(*ft))
        stems[g] = _group_stem(mem, g)
        reserved = set(range(0, len(mem), reserve_every))   # baseline reserve → rich subgraphs
        n_extra = 0
        for i, (fam, t) in enumerate(mem):
            if lookup_cap and n_extra >= lookup_cap:
                break
            if i in reserved:
                continue
            new_pools = normalizable_pool_keys(fam, t) - covered_pools
            if new_pools:
                reserved.add(i)
                core_flat_ids.add(t.template_id)      # realized FLAT (3NF OLTP + lookup FKs), 1 table each
                covered_pools |= new_pools
                n_extra += 1
        composable = [mem[i] for i in range(len(mem)) if i not in reserved]
        core += [mem[i] for i in range(len(mem)) if i in reserved]
        off = (len(g) * 7) % len(width_cycle)
        i, ci = 0, off
        while i < len(composable):
            target = width_cycle[ci % len(width_cycle)]
            ci += 1
            acc, bin_members = 0, []
            while i < len(composable) and acc < target:
                fam, t = composable[i]
                acc += _flatcol_count(fam, t)
                bin_members.append(composable[i])
                i += 1
            if len(bin_members) >= 2:
                denorm.append(bin_members)
            else:
                core += bin_members
    return CompositionPlan(denorm=denorm, core=core, core_flat_ids=core_flat_ids, groups=stems)


# ── the composite workbench table (denorm layer) ───────────────────────────────
def _width_noun(width: int) -> str:
    """A DBA-real width-appropriate table noun (a wide analytics rollup is a ``warehouse``, not a
    ``detail``) — register-invariant structural token."""
    if width <= 9:
        return "detail"
    if width <= 26:
        return "profile"
    if width <= 60:
        return "registry"
    return "warehouse"


def compose_denorm_table(members: list[tuple[str, CatalogTemplate]], group: str, group_stem: str,
                         gidx: int, natural_names: dict) -> tuple[SpineTable, ViewSpec, dict]:
    """One composite workbench table (+ its projection view + a complexity row) from ``members``.

    The collision decision (whether a member column needs a member-stem prefix) is made on SEMANTIC
    names ONLY — register-independent — so the semantic and natural twins prefix the same columns in
    the same positions (identical structure). ``slot_ref`` is preserved per column; ``realize_meta``
    records the composition + a ``col_source`` list aligned with ``table.columns``."""
    natural_names = natural_names or {}
    # semantic name multiset across all member columns → which names collide (register-independent)
    sem_count: dict[str, int] = {}
    per_member: list[tuple[str, str, str, list[ColumnSpec]]] = []   # (mid, sem_stem, nat_stem, cols)
    for fam, t in members:
        cols = [c for c in template_to_table(t, fam).table.columns if c.slot_ref != "__pk__"]
        rec = natural_names.get(t.template_id) or {}
        sem_stem = stem(table_name(t.template_id).replace("t_", "", 1)) or "m"
        nat_stem = stem(rec.get("table") or "") or sem_stem
        per_member.append((t.template_id, sem_stem, nat_stem, cols))
        for c in cols:
            sem_count[c.name] = sem_count.get(c.name, 0) + 1

    out_cols: list[ColumnSpec] = [ColumnSpec(name="id", slot_type="Class", slot_ref="__pk__")]
    col_source: list[str] = ["__pk__"]
    used: set[str] = {"id"}
    for mid, sem_stem, nat_stem, cols in per_member:
        rec_cols = (natural_names.get(mid) or {}).get("cols") or {}
        for c in cols:
            need_prefix = sem_count[c.name] > 1
            if natural_names:                                  # natural register
                base = rec_cols.get(c.name, c.name)
                name = f"{nat_stem}_{base}" if need_prefix else base
            else:                                              # semantic register
                name = f"{sem_stem}_{c.name}" if need_prefix else c.name
            name = _safe_ident(name, nat_stem if natural_names else sem_stem)
            final, k = name, 2                                 # per-register dedup (structure preserved)
            while final in used:
                final, k = f"{name}_{k}", k + 1
            used.add(final)
            out_cols.append(ColumnSpec(name=final, slot_type=c.slot_type, slot_ref=c.slot_ref))
            col_source.append(mid)

    width = len(out_cols) - 1
    noun = _width_noun(width)
    if natural_names:
        tname = f"{group_stem}_{noun}_{gidx:03d}"
    else:
        tname = f"t_{group_stem}_wb{gidx:03d}"
    anchor_fam, anchor_t = members[0]
    meta = {"layer": "denorm", "profile": "composite", "group": group,
            "composition": [t.template_id for _f, t in members], "col_source": col_source,
            "n_composed": len(members)}
    st = SpineTable(template=anchor_t, family=anchor_fam,
                    table=TableSpec(name=tname, ref=anchor_t.template_id, columns=out_cols, rows=[]),
                    notes=[], not_null=set(), kind="composite", realize_meta=meta)
    view = render_view_ddl(f"v_{tname}", st)                   # projection view → R4 view stratum
    cx = {"template_id": anchor_t.template_id, "family": anchor_fam, "profile": "composite",
          "profile_source": "composition:domain-group", "layer": "denorm", "n_composed": len(members),
          "n_tables": 1, "n_views": 1, "n_fks": 0, "eav_tables": 0, "junction_tables": 0,
          "attr_count": width, "eav_ratio": 0.0, "m2n_density": 0.0, "fk_depth": 0, "dim_tables": 0}
    return st, view, cx


# ── lookup / code tables (3NF core enrichment) ─────────────────────────────────
def _register_invariant_name(template: CatalogTemplate, c: ColumnSpec) -> "str | None":
    """The SEMANTIC (register-invariant) attribute name of a data-bearing column, from its ``slot_ref`` —
    so lookup detection is identical across the natural/semantic twins. ``None`` for pk/fk/Class columns
    (only literal-valued attributes get normalized into a code table)."""
    sr = c.slot_ref
    if sr == "__pk__" or sr.startswith("fk:") or sr in ("subject", "related"):
        return None
    if sr.startswith("data:"):
        return _dataprop_col(sr[len("data:"):])
    if template.slot_types.get(sr) == "DataProperty":
        return col_name(sr)
    return None


def _normalizable(template: CatalogTemplate, c: ColumnSpec, defn: "str | None"
                  ) -> "tuple[str, str] | None":
    """``(concept_key, code_col_name)`` if column ``c`` is an ENUMERATED attribute worth a lookup table,
    else ``None``. Two ontology-grounded paths, both register-invariant: a SKOS-definition enum (keyed by
    the concept ``slot_ref``, its own dimension) or a curated closed value-set pool (keyed by the pool
    name, SHARED across every column of that kind — a warehouse has one ``status`` dimension)."""
    from aegir.ontology.rows import SEMANTIC_VALUE_POOLS, parse_enum_from_definition
    sem = _register_invariant_name(template, c)
    if sem is None:
        return None
    if defn and parse_enum_from_definition(defn):
        return (f"def:{c.slot_ref}", sem)
    if sem in SEMANTIC_VALUE_POOLS:
        return (f"pool:{sem}", sem)
    return None


def normalizable_pool_keys(fam: str, template: CatalogTemplate) -> set:
    """The curated-pool enumerated-attribute keys in the template's flat form (deterministic, def-free)
    — the reserve signal that steers a template into the 3NF core so its lookup materializes."""
    from aegir.ontology.rows import SEMANTIC_VALUE_POOLS
    out: set = set()
    for c in template_to_table(template, fam).table.columns:
        sem = _register_invariant_name(template, c)
        if sem is not None and sem in SEMANTIC_VALUE_POOLS:
            out.add(sem)
    return out


def build_lookup_tables(core_tables: list[SpineTable], *,
                        definitions: dict[str, dict[str, str]]
                        ) -> tuple[list[SpineTable], list[FKEdge]]:
    """Normalize ontology-**enumerated** columns in the 3NF core into SHARED lookup/code tables (a
    warehouse has ONE ``status`` dimension, not one per table). For each core column whose SKOS
    definition enumerates a value set (``rows.parse_enum_from_definition``), we (1) ensure a shared
    lookup table for that ENUMERATED CONCEPT exists — ``(id, code, label)``, the ``code`` carrying the
    concept's ``slot_ref`` so ``definitions_for_spine`` fills it with the enum at materialize time — and
    (2) turn the column into an FK to it. Deterministic, RI-true (the FK draws from the lookup PK pool),
    and it deepens the FK chains (entity → lookup, and 2-deep where an entity FK targets another entity
    that itself carries a lookup FK). The FK column's ``slot_ref`` is preserved (its reference stays the
    enumerated concept).

    **Register-invariance (twin-safety):** the lookup identity and the set of FK'd columns are keyed by
    ``slot_ref`` (the concept), never by the register-specific column NAME — so the semantic and natural
    twins produce the same lookup count / FK structure, only the physical names differ. The definition
    path is register-invariant too (a column has a def iff its ``slot_ref`` is ``data:<iri>``, name-free).

    Returns the NEW lookup tables + FK edges (dict-insertion order = concept first-seen order, identical
    across twins)."""
    lookups: dict[str, SpineTable] = {}          # keyed by register-invariant concept_key
    new_fks: list[FKEdge] = []
    for st in core_tables:
        if st.kind not in ("entity", "fact", "dimension"):      # normalized subject/measure tables
            continue
        defs = definitions.get(st.table.name, {})
        for c in st.table.columns:
            hit = _normalizable(st.template, c, defs.get(c.name))
            if hit is None:
                continue
            concept_key, code_col = hit
            if concept_key not in lookups:
                lname = _dedupe(f"lk_{re.sub(r'[^a-z0-9]+', '_', code_col).strip('_') or 'code'}",
                                {lk.table.name for lk in lookups.values()})
                lookups[concept_key] = SpineTable(
                    template=st.template, family=st.family,
                    table=TableSpec(name=lname, ref=st.template.template_id, columns=[
                        ColumnSpec("id", "Class", "__pk__"),
                        # code named after the pool so value_for fills it with the enum (pool path); the
                        # slot_ref carries the concept so definitions_for_spine fills it (SKOS-def path)
                        ColumnSpec(code_col, "xsd:string", c.slot_ref),
                        ColumnSpec("label", "xsd:string", "data:label")], rows=[]),
                    notes=[], not_null=set(), kind="lookup",
                    realize_meta={"layer": "core", "profile": "lookup", "concept": concept_key})
            new_fks.append(FKEdge(src_table=st.table.name, src_col=c.name,
                                  dst_table=lookups[concept_key].table.name, dst_col="id",
                                  via_slot=c.slot_ref))
    return list(lookups.values()), new_fks


def _safe_ident(name: str, stem_hint: str = "col") -> str:
    """A valid SQL identifier: a member natural name can be digit-leading (``930nm_variant``) — invalid
    unquoted SQL — so prefix such names with the member stem (deterministic, keeps the token content)."""
    if re.match(r"^[A-Za-z_]", name):
        return name
    return f"{(stem_hint or 'col')}_{name}"


def _dedupe(name: str, seen: set[str]) -> str:
    if name not in seen:
        return name
    k = 2
    while f"{name}_{k}" in seen:
        k += 1
    return f"{name}_{k}"


# ── driver: realize the full spine (core subgraphs + lookups + denorm composites) ─
def build_realized_spine(all_ft: list[tuple[str, CatalogTemplate]], *, natural_names: dict,
                         realize_seed: int, definitions_fn=None, reserve_every: int = RESERVE_EVERY,
                         lookup_cap: int = 0
                         ) -> tuple[list[SpineTable], list[FKEdge], list[ViewSpec], list[dict]]:
    """Build the realized spine under the composition architecture. Returns
    ``(tables, intra-subgraph fks, reconstruction+projection views, complexity rows)`` — the same tuple
    the flat realize path returned, so ``load_spine`` is a drop-in. ``natural_names`` empty ⇒ semantic
    twin (identical structure, semantic names)."""
    from aegir.ontology import realize as rz

    natural_names = natural_names or {}
    plan = plan_compositions(all_ft, reserve_every=reserve_every, lookup_cap=lookup_cap)
    tables: list[SpineTable] = []
    fks: list[FKEdge] = []
    views: list[ViewSpec] = []
    cx: list[dict] = []

    # denorm layer — composite workbench tables (+ projection views)
    for gidx, members in enumerate(plan.denorm):
        group = domain_group(members[0][1])
        st, view, cxrow = compose_denorm_table(members, group, plan.groups.get(group, "dom"), gidx,
                                                natural_names)
        tables.append(st)
        views.append(view)
        cx.append(cxrow)

    # 3NF core — the baseline reserve as rich subgraphs (junction/star/eav + their reconstruction views);
    # the enum/pool reserve FLAT (a normalized OLTP entity, 1 table, that a lookup FK then normalizes).
    core_tables: list[SpineTable] = []
    for fam, t in plan.core:
        seed = int.from_bytes(
            hashlib.blake2b(f"{realize_seed}:{t.template_id}".encode(), digest_size=8).digest(), "big")
        profile = "normalized" if t.template_id in plan.core_flat_ids else None
        rs = rz.realize_schema(t, fam, rng=random.Random(seed), profile=profile,
                               natural=natural_names.get(t.template_id))
        for stab in rs.tables:
            stab.realize_meta.setdefault("layer", "core")
        core_tables.extend(rs.tables)
        fks.extend(rs.fks)
        views.extend(rs.views)
        cx.append({"template_id": t.template_id, "family": fam, "profile": rs.profile, "layer": "core",
                   "n_composed": 0, **rs.complexity})
    tables.extend(core_tables)

    # lookup/code tables — normalize enumerated core columns (3NF depth), shared across the core
    if definitions_fn is not None:
        lookups, lk_fks = build_lookup_tables(core_tables, definitions=definitions_fn(core_tables))
        for lk in lookups:
            tables.append(lk)
            cx.append({"template_id": lk.template.template_id, "family": lk.family, "profile": "lookup",
                       "profile_source": "composition:enum-normalization", "layer": "core", "n_composed": 0,
                       "n_tables": 1, "n_views": 0, "n_fks": 0, "eav_tables": 0, "junction_tables": 0,
                       "attr_count": 2, "eav_ratio": 0.0, "m2n_density": 0.0, "fk_depth": 0, "dim_tables": 0})
        fks.extend(lk_fks)

    return tables, fks, views, cx
