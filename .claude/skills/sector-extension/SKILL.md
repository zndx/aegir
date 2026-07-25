---
name: sector-extension
description: Extend SDG with a new sector-level topic domain — the full arc from in-scope-external census through aperture admission toward a semantic-layer relational projection. Use when adding a sector (finance, healthcare, energy, engineering, manufacturing, chemistry, …) or integrating an external vocabulary/standard as a contributor.
---

# Sector-level ontology extension

> **STATUS: DRAFT (by definition — RH 2026-07-25).** Phases 1–7 carry five field
> executions (FIBO→FinancialSector/FINTECH · FHIR→Healthcare/BIOTECH ·
> Energistics→HeavyIndustries/ENERGY · SysMLv2→Engineering/MBSE · the
> Manufacturing sector + MFG re-aim). Phase 8's **controlled-vocabulary tier is
> now executed once** (SysMLv2, census-verified against the official Ecore —
> the machine-readable census in Phase 1b is PROVEN procedure); the lowering
> and parity steps remain **anticipated**: the manufacturing-operations
> realization (task #44) is their first execution and WILL revise this
> document. The material final product of the pipeline is a semantic-layer API
> over the sector — every earlier phase is in its service.

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
7. **SKOS evolves quickly; ontology updates enforce consistency** (RH 2026-07-25).
   Iterating at the SKOS level first is LEGITIMATE convenience — labels,
   membership, scope notes are a fast authoring surface, and the 2026-07
   sector arc worked exactly this way. But SKOS-ahead-of-OWL is a **gap signal
   co-equal with unsat and underspecification**: the invariant OWL ⊨ SKOS ⊨
   SHACL is only real when the OWL leg exists to do the entailing. Closure is
   an IMPERATIVE, not tidying — see "The gap-closure loop" below. Read
   comments and docs written during SKOS-first phases under this rule: they
   record expedient sequence, never authority.

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

### Phase 1b — The machine-readable census [PROVEN: SysMLv2, 2026-07-25]

For metamodel/schema-publishing externals (SysMLv2/KerML Ecore; **WITSML/
PRODML/RESQML energyML XSDs — same spec flavor, same procedure**; ISA-95 UML),
every vocabulary claim must trace to the external's OFFICIAL machine-readable
artifact — the practitioners of that standard are the scrutiny audience, and
folklore is what they smell first.

**Source hierarchy**: machine-readable metamodel/schema (Ecore/XMI, XSD, RDF)
> normative spec text (clause-cited) > memory/folklore (BANNED — the first
SysMLv2 census run caught three wrong abstract flags written from folklore).

**Procedure**:
1. Acquire the official artifact (SysMLv2: Systems-Modeling pilot
   `SysML.ecore`; WITSML: the energyML XSD set for the pinned version) into
   `build/foreign/<external>/`.
2. Parse mechanically (ecore/XSD are XML — extract per term: presence,
   structural flags like `abstract`, the specialization/extension hierarchy,
   documentation fields). Census OUR claimed vocabulary against it; bank
   `census_*.json` as the evidence artifact.
3. Apply verdicts to the TTL:
   - Confirmed structural facts → recorded (in-set specializations as
     `skos:broader`; out-of-set supertypes in `skos:editorialNote` WITHOUT
     minting concepts — confirmed links only, no scope creep).
   - Wrong claims → corrected, with "(census corrected <date>)" notes.
   - **Terms NOT in the external's vocabulary** → retype honestly as **SDG
     RELATIONAL FLATTENINGS** (ours, in our namespace, marked as such, naming
     the external's actual reified form). A pragmatic edge vocabulary is a
     legitimate engineering choice; a false attribution is not. (Measured:
     Allocation/Connection/ItemFlow are usages in SysMLv2, not edge kinds.)
   - **Version drift** → the census catches superseded names (ItemFlow →
     Flow family); record a version-watch note tied to the scheme's
     `dct:hasVersion` pin.
4. The census verdicts feed the reference tables' honesty: census-verified
   rows vs SDG-extension rows are distinguishable in the lookup's provenance
   columns (`spec_ref` present vs absent).

**The epistemic split, enforced structurally**: OUR namespace authors freely
(BFO/CCO foundations, bare names, HermiT-gated — the logical foundations we
are uniquely positioned to define). THEIR vocabulary is evidence-governed —
census-confirmed or explicitly marked ours. When a domain class claims
representability atop the external (`sdg:InventoryItem` atop `ItemUsage`),
the claim cites a census-confirmed construct, never a guess.

### Phase 1c — The fidelity probe [PROVEN: SysMLv2, 2026-07-25]

History's existence proof (RH): EMF demonstrated that full applications
generate from strongly typed persisted state objects — Ecore/XSD-flavor
externals are therefore LOAD-BEARING generation sources, and our arc
(ontology → triad → kvasir-lowered relational → semantic-layer API) is the
same shape with a logic upgrade. That footing earns an adversarial obligation.

**The question every agent must ask**: *what can be expressed and stored in
the external modeling framework which CANNOT be serialized in our ontology
with equivalent or superior fidelity?* Run it mechanically against the same
artifact the census parsed: enumerate the framework's expressive constructs
(containment references, bidirectional opposites, ordered multi-valued
features, multiplicity bounds, derived/transient features, defaults, enums,
operations; for XSD: choice/substitution groups, xs:key/keyref, facets) and
give each a verdict:

- **EQUIVALENT/SUPERIOR** — record WHICH leg carries it: containment →
  part-whole axioms + SHACL + `ON DELETE CASCADE`; eOpposite pairs → inverse
  object properties + SHACL pair shapes; xs:keyref → FKs (kvasir's native
  ground); enums/defaults → lookup tables / DDL `DEFAULT`; derived features →
  SQL views + OWL entailment (SUPERIOR where the derivation becomes
  HermiT-checkable instead of an OCL/Java implementation — claim superiority
  only with the demonstration).
- **TENSION** — a construct needing real design (e.g. ordered multi-valued
  features: RDF has no native ordering; the relational answer is
  `sequence_number` discipline; decide and record). Tensions enter the
  worklist (escalation organ — never dropped); **resolving them IS the
  quality signal**.
- **GAP-ACKNOWLEDGED** — a deliberate non-goal, documented (e.g. behavioral
  operations: we serialize state, not behavior).

**First run (SysML.ecore, artifact `build/foreign/sysmlv2/fidelity_probe.json`)
— the probe can DISSOLVE apparent tension**: 79% of the metamodel's features
(328/~415) are derived+transient — computed views, not persisted state. The
persisted core is 3 containment refs + non-derived references + 24 defaults +
7 enums: EXACTLY the thin elements+relations backbone the target DDL designs,
independently confirmed by the external's own statistics. Real tensions
isolated: ordered multi-valued (153), operations (70, gap-acknowledged),
the derived layer (superior-candidate via entailment).

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

## Phase 8 — Semantic-layer realization [PARTIALLY EXECUTED — task #44]

The sector's material product: a relational projection whose rows are the
individuals the domain tracks, with a semantic-layer API over it.

**Executed once (SysMLv2, 2026-07-25) — the controlled-vocabulary tier**: the
target DDL's reference-table seed rows become SKOS scheme members, layered by
the spec's own strata (kernel terms → the kernel scheme, language terms → the
integration scheme), grouped by `skos:Collection`s — **Collection = reference
table, member = row**. Each member carries the census verdict (Phase 1b);
`rdfs:seeAlso` feeds the table's `spec_ref` column and the member IRI feeds
`sdg_concept_iri`. Vocabulary-tier members are never aperture points and never
enter the chord (the constituents builder reads overlay + enumerated includes
only).

**Executed twice (the OWL module + lowering, 2026-07-25)** — two rulings now
govern:

- **One unified critical path.** No per-module feature flags: sector modules
  append UNCONDITIONALLY into the realized-ontology assembly (the
  relational-concepts/spec-mappings precedent) and the flow's own gates
  (`just metaflow`: HermiT certificate, kvasir checks) prove them on-path.
  An isolated verify script is a fast dev-loop probe, never a staging gate.
- **The completeness doctrine.** The REQUIREMENT to produce a complete,
  functional relational schema DRIVES ontology completeness and quality: a
  declared class the lowering folds or drops is UNDERSPECIFIED, and
  underspecification is a remediation signal CO-EQUAL with logical
  inconsistency — both inform agentic remediation, neither is ever silently
  accepted (silent folds are the emit-signals-not-drops sin at the lowering
  boundary). kvasir complains NATIVELY (`Plan.underspecified` in the plan
  sidecar + a stderr complaint; inconsistency REFUSES, underspecification
  COMPLAINS while emitting the specified part). Gates CONSUME that signal —
  never re-derive it with a side shim. Remediate with what the domain
  actually tracks, never appeasement (measured: Warehouse folded for lacking
  a distinguishing feature → `facilityCode`, what a WMS tracks →
  materialized; first production contact surfaced 89 genuine finds).

Anticipated shape for the rest (to be PROVEN and revised by the
manufacturing-operations arc):

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
- **Triad by generation, with the AUTHORITY ORDER explicit** (RH 2026-07-25):
  the entailment direction OWL ⊨ SKOS ⊨ SHACL IS the authority direction.
  **The realized OMN is the source of LOGICAL truth** — differentia,
  subsumption, restrictions, HermiT-checkable consistency; it is the one
  artifact certified and lowered. SKOS is the controlled-vocabulary CONTENT
  model only (the Primer's scope: labels, membership, scheme structure —
  `skos:broader` carries no subsumption). SHACL is the validation image.
  Concretely: controlled-vocab tables draw CONTENT from SKOS Collections,
  carried INTO the realized OMN by the vocab-lowering generator
  (definition-enums → kvasir's designed-dormant `@Enum` path → lookup + FK);
  domain tables = the OWL image (projection-total lowering); FK/CHECK/UNIQUE
  lattice = the SHACL image (composite-FK metaclass pattern: `UNIQUE(id,
  metaclass)` + subtype FK on the pair). Any logical claim about vocabulary
  values (distinctness, transitions) authors in OMN, never SKOS. Maintained
  by regeneration, not discipline.
- **One fact, one home**: domain tables authoritative; interchange relations
  are a generated projection. HEAD-state store: history is the procession +
  OL/Atlas lineage, never a second git in the database.

## The gap-closure loop — when SKOS skates ahead of the SDG puck [ACP]

Three remediation signals, one pattern (detect → signal → agentic remediation
→ gate). None is ever silently accepted; each names its fix:

| signal | detector | remediation |
|---|---|---|
| unsat | HermiT — realize REFUSES | re-author the axioms (CAS loop, justification-guided) |
| underspecified | kvasir `Plan.underspecified` | add what the domain actually tracks |
| **SKOS-ahead** | ungrounded anchors (below) | **BACKFILL the OWL module** |

**Detection — declared joins, never name-derived**: a SKOS anchor whose
territory is grounded DECLARES it (`sdg:groundedByModule "<python module>"` on
the anchor concept); the gap worklist = sdg anchors carrying entity semantics
with no grounding declaration. SKOS concepts are TOPICS, OWL classes are
ENTITIES — the join is territory-coverage, which only a declaration can state
honestly. First closed case: `sdg:MFG_LOGISTICS` → `manufacturing_module`
(8 classes). Standing worklist: the FINTECH / BIOTECH / MBSE / ENERGY-children
territories — SKOS-ahead today, modules owed.

**The ACP closure procedure** (what an agent does with a gap):
1. Read the anchor's SKOS content (definition + scopeNote = the TERRITORY
   spec) and the sector's census artifacts (`build/foreign/<external>/`).
2. Author the OWL module in OUR namespace: bare-name classes; BFO/CCO
   grounding via the external index's OPAQUE IRIs only (coined-alias ban);
   representable-atop notes citing census-confirmed constructs; restrictions
   carrying the relational intent (`exactly 1` → NOT NULL FK; enumerable
   vocabularies via `sdg:lowersToProperty` Collections → definition-enums).
3. Append UNCONDITIONALLY to the realized assembly (one critical path, no
   flags); the gates prove it on-path — HermiT (unsat), kvasir
   (underspecified), the dev-loop probe for fast iteration.
4. Declare the grounding on the SKOS anchor — the gap closes MEASURABLY.
5. Any logical claim about SKOS-held vocabulary (status distinctness,
   transition constraints) moves INTO the module as axioms — SKOS keeps
   content only.

**The imperative**: an ungrounded anchor is DEBT against the invariant.
Worklist-tracked under escalation-organ discipline — closure or explicit
deferral with a reason that names the fix, never dropped — because every
downstream product (admission, corpus, the semantic-layer API) inherits
exactly the logic the OWL leg provides and nothing more.

> After #44 lands: replace this phase's "anticipated" framing with the measured
> procedure, record what the yardstick diff taught, and promote the yardstick
> pattern into strategy targets/. Until then, phase 8 is design intent.
