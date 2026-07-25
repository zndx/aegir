---
name: sector-extension
description: Extend SDG with a new sector-level topic domain — the full arc from in-scope-external census through aperture admission toward a semantic-layer relational projection. Use when adding a sector (finance, healthcare, energy, engineering, manufacturing, chemistry, …) or integrating an external vocabulary/standard as a contributor.
---

# Sector-level ontology extension

> **STATUS: DRAFT (by definition — RH 2026-07-25).** Phases 1–7 carry five field
> executions (FIBO→FinancialSector/FINTECH · FHIR→Healthcare/BIOTECH ·
> Energistics→HeavyIndustries/ENERGY · SysMLv2→Engineering/MBSE · the
> Manufacturing sector + MFG re-aim). Phase 8 — the relational projection
> integrating external-standard-informed formal ontology constructs — is
> **anticipated, not yet executed**: the manufacturing-operations realization
> (task #44, parity vs the banked DDL yardstick) is its first execution and
> WILL revise this document. Treat phase 8 as design intent; refine this skill
> as that arc lands. The material final product of the pipeline is a
> semantic-layer API over the sector — every earlier phase is in its service.

## Doctrines (load-bearing; violating these is how extensions go wrong)

1. **The language represents; the ontology grounds.** External standards
   (SysMLv2, FIBO, FHIR, Energistics, ISA-95…) are IN-SCOPE EXTERNAL: never
   managed nor extended by us, leveraged extensively. Domain semantics ground in
   OUR BFO/CCO-grounded OWL classes and sector-aligned SKOS — never in the
   external's namespace (never mint foreign-namespace IRIs), and never as
   members of a language's scheme.
2. **Instantiation, not specialization, at the sector tier.** The sector forest
   has NO apex: sector tops are members of the `sdg:TopicDomains` Collection and
   instances of the `sdg:TopicDomain` thin class — never `skos:broader` to
   anything. The contract's sync driver measures members ≡ instances.
3. **Chord ≠ point.** The SKOS chord renders the full hierarchy; the qdrant
   point layer stays sparse. A chord-visible concept does NOT compete in MaxSim
   unless ENUMERATED (admission_filter `aperture_include`) or notation-gated in
   the overlay. GEPA composites may draw constituent vocabulary from chord
   members that are not points.
4. **§4.6.3 discipline.** `skos:hasTopConcept` is convention without integrity
   conditions. Exclusion is only ever the enumerated filter set; genus status is
   structural (`broader ∧ narrower`); atypical presentations (top-with-broader)
   are measured, never trusted or forbidden.
5. **Pre-1.0 = identity without versioning.** No version markers, no deprecated
   bridges, no retirement ceremony in Scratch — edit in place; the
   Scratch→Current→Archive procession is the history mechanism. Convenience
   numbers on corpora/strategy releases are free; compatibility promises wait
   for the signals.zndx.org publication.
6. **Measurement and procession IS the discipline.** Every structural change is
   delta-gated before release; every release seeds the strategy and checks
   drift; every surprise gets its contested artifacts READ before iterating.

## Phase 1 — Census the external

- Run an expressiveness census against the external's actual publication
  (what could our axiom-pattern library rebuild? what grain of identifiers does
  it publish?). Record confirmed links only — naming collision ≠ association.
- Decide the mapping grain from the census, never from habit:
  - RDF concept IRIs published (FIBO, FHIR) → concept-level `skos:closeMatch`
    (non-transitive by design; exactMatch only for true 1:1) + Layer-A zero-copy
    membership (`<external-iri> skos:inScheme <our-scheme>`).
  - No RDF (Energistics XSD/UML, OMG SysMLv2/KerML) → family-grain
    `rdfs:seeAlso` (namespace URIs / spec pages) + spec_mappings entities.

## Phase 2 — Sector + scheme structure

- **Tier shape**: `sdg:TopicDomains` Collection member + `a sdg:TopicDomain`
  typing → sector top (natural-cased: FinancialSector, Healthcare,
  HeavyIndustries, Engineering, Manufacturing) → genus chord (ALL-CAPS fragment:
  FINTECH, BIOTECH, ENERGY, MBSE, MFG) → children.
- **One integration file per external/sector**: `src/aegir/ontology/
  <name>_integration.ttl` (the `*_integration.ttl` glob is canonical discovery —
  file discovery is decoupled from point enumeration; a file may contribute
  schemes and chord with zero points). Prefixed subjects are fine here (rdflib
  reads them); the OVERLAY (`domain_concepts.ttl`) uses full-IRI block form
  (its regex parser reads nothing else — a wrong-form file returns junk
  silently, not an error).
- **Scheme-per-standard with a dependency spine**: each standard gets its own
  `skos:ConceptScheme`; shared kernels get their own scheme referenced via
  `dct:requires` (Energistics common ← WITSML/PRODML/RESQML; KerML ← SysMLv2).
  `dct:hasVersion` pins + `skos:editorialNote` version-watches let each external
  advance independently — we adapt selectively.
- **Thin typing**: `sdg:<Name>Concept rdfs:subClassOf skos:Concept` per external
  (the TopQuadrant governance-annotation attachment point).
- Sectors live in integration files, NEVER in the overlay with a notation
  (overlay + notation = aperture point; sectors organize without admitting).

## Phase 3 — Genus authoring

- prefLabel natural ("Model-Based Systems Engineering"); the fragment-equal
  altLabel carries the chord ("MBSE"); notation = `<sector>.1`.
- `skos:definition` crisp; `skos:scopeNote` composed from the ACTUAL census of
  the family's children/curated subset — vocabulary-bearing retrieval mass
  (grounding-anchor doctrine). Target the child token band (~130–200 ColBERT
  tokens of total text). Declare true adjacencies (BIOTECH↔LIMS) — they are
  measured capture paths, not defects.
- Retrieval text = prefLabel + altLabel + definition + scopeNote + rdfs:comment.
  Implementation/representability provenance goes in `skos:editorialNote`
  (EXCLUDED from retrieval text) — never let the language's vocabulary into a
  domain concept's retrieval text (the MFG blend lesson).

## Phase 4 — Aperture entry

- Membership is ENUMERATED: add `{iri, file, code}` to admission_filter.json's
  `aperture_include`. Seeding reads the SoT file; classify-time policy resolves
  the RELEASED strategy (which lags edits by design — don't be surprised).
- The overlay base is NOTATION-GATED (skos:notation = the admission-surface
  discriminator); no-notation concepts are chord organization, never points.
- Append-only id discipline: new overlay concepts go at the overlay's FILE END
  (seed order = file order); enumerated includes take tail ids. Adding points
  shifts include ids once — regenerate the lineup projection after.
- Reseed via `build_index(vocab=DEFAULT_OVERLAY, collection=DEFAULT_APERTURE)`.
  **Never reseed while a harvest is live** (delete-recreate 404s it — measured);
  payload-only refreshes should upsert in place.
- Regenerate `build/aperture_constituents.json`
  (scripts/build_aperture_constituents.py) so the Scratch chord tracks the
  surface, then `python -m aegir.lineup build`. Current stays untouched.

## Phase 5 — Contract

`uv run python scripts/check_aperture_contract.py` — all DRIVERS, not gates:
point count · genus (broader∧narrower) · altLabel≡fragment · S2–S9+S13 (schemes,
Collection disjointness — the standing Atlas-curator signal that richer
frameworks belong in HermiT/kvasir-checked OWL) · topic-domains sync
(members≡typed) · S9-settled two-way delta · atypical presentations. Expect
0 violations; expect the drivers to MOVE (that's what they're for).

## Phase 6 — Delta gate

- Recompose over the FROZEN store (`measure_admission_selectivity.recompose`,
  fresh `out_path` — preserve prior baselines) BEFORE releasing.
- Interpretation discipline: judge by MARGIN STRUCTURE (low-margin fraction,
  tie mass), never share movement. An intentional re-aim SHEDS the old aim's
  borderline catch on a frozen store (survivorship). New anchors on stores
  without their aim point measure ~0 — the fresh stream is their real test
  (FINTECH: 0 frozen → 38 first window).
- **When a delta surprises, READ the contested docs before iterating text.**
  Two blind text iterations lost to one six-document read (measured). Name
  genuine boundary ambiguities as adjudication pairs for the sibling machinery.

## Phase 7 — Release

`python -m aegir.strategy.manifest seed` → commit strategy submodule → commit
aegir (+ pointer) → `manifest drift` must print CLEAN. Scratch note in
`docs/scratch/<date>/HHMMSS_*.md`; memory updated. Integration files ship as
lens components automatically (the glob).

## Phase 8 — Semantic-layer realization [DRAFT — first execution = task #44]

The sector's material product: a relational projection whose rows are the
individuals the domain tracks, with a semantic-layer API over it. Anticipated
shape (to be PROVEN and revised by the manufacturing-operations arc):

- **Author the sector OWL module**, BFO/CCO-grounded (orders/plans ⊑ directive
  ICE · physical things ⊑ artifact · sites ⊑ facility · flows/runs ⊑ process ·
  levels/measures ⊑ quality), HermiT/kvasir-gated, SKOS-aligned to the sector's
  anchors. External-standard constructs inform the design via spec_mappings —
  never by subclassing foreign IRIs that don't exist in OWL.
- **Parity-yardstick method**: bank a hand-authored target DDL (see
  docs/scratch/2026-07-25/162615_mfg_semantic_layer_ddl_yardstick.sql — thin
  external-standard identity backbone + controlled-vocabulary lookups + typed
  domain tables); kvasir-lower the module; diff; converge. The functional
  requirement becomes checkable.
- **Triad by generation**: controlled-vocab tables = the SKOS image (kvasir
  scheme→lookup-DDL lowering, seed rows carrying spec_ref + sdg_concept_iri);
  domain tables = the OWL image (existing projection-total lowering);
  FK/CHECK/UNIQUE lattice = the SHACL image (composite-FK metaclass pattern:
  `UNIQUE(id, metaclass)` + subtype FK on the pair). One source of truth;
  OWL ⊨ SKOS ⊨ SHACL maintained by regeneration, not discipline.
- **One fact, one home**: domain tables authoritative; interchange relations
  are a generated projection. HEAD-state store: history is the procession +
  OL/Atlas lineage, never a second git in the database.

> After #44 lands: replace this phase's "anticipated" framing with the measured
> procedure, record what the yardstick diff taught, and promote the yardstick
> pattern into strategy targets/. Until then, phase 8 is design intent.
