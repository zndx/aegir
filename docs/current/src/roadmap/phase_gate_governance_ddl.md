# Phase Gate — Governance & DDL Spine (v0.3)

**Date:** 2026-06-07 · **Decision: PASS** · **Commits:** `f3795da`, `3337176` (on the
prior AGE-backend hardening, `614ea8f` and the `rch/signals` fork chain).

This gate certifies the **relational-metadata substrate** for the v0.3 corpus: the corpus's
tables are now backed by deterministically-generated, machine-verified **SQL DDL**, catalogued
in Apache Atlas as a first-class `rdbms_*` footprint with column-level lineage. It establishes
the *second verifiable axis* of the corpus thesis.

## The thesis it certifies

> A v0.3 corpus table is a **view on a larger relational footprint**, and that footprint is
> **verifiable on two independent axes**:
>
> | Axis | Source syntax | Deterministic verifier | Coverage measure |
> |------|---------------|------------------------|------------------|
> | **Semantic** | OWL Manchester | DeepOnto (JVM) | BFO/CCO families, axiom kinds |
> | **Syntactic** | SQL DDL / views | **polyglot** (Rust) | SQL features: types, constraints, FKs |
>
> `polyglot : SQL :: DeepOnto : ontology`. Each corpus column carries an ontology type
> (semantic) *and* a SQL type (syntactic); CTA is the map between them.

## Scope delivered

1. **`polyglot` vendored** as a submodule (`components/polyglot` = `zndx/oss-polyglot @
   rch/devenv`) — a Rust/PyO3 SQL parser/validator/transpiler (sqlglot port, 34 dialects).
   Task-built via `devenv tasks run polyglot:build` (off the uv sync path, per the patched-wheel
   convention). The fork branch exists for the near-term **Kudu** dialect extension.
2. **DDL spine** (`src/aegir/ontology/ddl.py`, `scripts/build_ddl_spine.py`): every catalog
   template lowered to a `CREATE TABLE` (reusing `type_check` schema types), cross-family
   foreign keys gated by the empirical `FamilyComplex`, validated against Trino ∩ Spark plus a
   Spark Iceberg-flavored variant, with a SQL-feature **coverage inventory** (the syntactic
   analogue of `ontology_coverage_audit.py`).
3. **DDL-native Atlas projector** (`scripts/project_atlas_ddl.py`, supersedes the hive
   prototype): `rdbms_instance/db/table/column` + `rdbms_foreign_key` entities, corpus tables as
   `VIEW`s, **column-level lineage** via polyglot OpenLineage → `aegir.governance.ol` →
   `aegir_hx` (new `columnLineage`-facet ingestion).

## Gate evidence (verified, restart-durable)

| Criterion | Result |
|---|---|
| DDL spine generates + validates (all 540 templates) | **540/540** canonical (Trino∩Spark) + **540/540** Iceberg variant (Spark) ✅ |
| Relational footprint in Atlas | **91** `rdbms_table` (56 base + 35 views), **252** `rdbms_column` ✅ |
| Join structure explicit | **36** `rdbms_foreign_key` resolving table/key_columns/references_* ✅ |
| Column-level lineage (view ← base) | **94** `DERIVES_FROM` edges ✅ |
| Semantic overlay | **263** classifications (CTA/CPA/domain), `OntologyProvenance` BM on 56, glossary ✅ |
| Durability | survives `devenv` restart; names decode in search ✅ |

## Significance / what this unblocks

- The "views on a footprint" thesis is now **concrete and browsable** in the Atlas UI, not a
  slide — it is the illustration the corpus paper needs.
- The corpus is **engine-verified** for Trino / Spark / Iceberg **without** running those
  engines (that runtime stays in Signals) — Aegir asserts forward-compatibility statically.
- The syntactic-coverage axis is a new, cheap, deterministic signal to drive generation
  toward under-covered SQL features — complementary to the ontology coverage audit.

## Known issues / deferred (none gate-blocking)

- **classification-filter search** (filter entities by CTA/CPA tag) — deferred to a follow-up.
- **Kudu** dialect — the first `polyglot` fork extension (required near-term).
- **Hybrid → forward** construction — corpus views are still synthesized; next is emitting real
  `CREATE VIEW` from the generation pipeline.
- Cosmetic: Atlas soft-delete dev cruft (UI-invisible); glossary term→entity `meanings` display.

## Operational notes (for the next operator)

- A long-running Atlas instance can develop a **stale AGE connection pool** (every create path
  500s with "Failed to execute vertex query" while the graph is healthy) — **restart Atlas**.
- polyglot OpenLineage wants an **object** `outputDataset` and a **SELECT** (not `CREATE VIEW`).
- `rdbms_foreign_key` needs a `name`; a businessMetadataDef's `applicableEntityTypes` is fixed
  at create-time (recreate to retarget).
- `maturin --release` LTO is pathologically slow; iterate with debug builds.
