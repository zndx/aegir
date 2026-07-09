# Charter

This is Ægir's internal direction-setter for the ontology scope. It
declares what Ægir publishes outward, names the provenance discipline
and design constraints that follow, records the committed BFO/CCO
branch structure and external-standard anchors, and pins the gate any
ontology change must clear before it ships.

> **Status note.** The operational rigor program — the metric suite,
> the OQuaRE publish gate, the disposal membranes, and the
> agent-mediated propose/dispose loop — is documented canonically in
> the [Authors Guide](./authors_guide.md); this charter does not
> duplicate it. The branch structure and external anchors in
> §§ *Domain commitments* below remain the committed architecture of
> the SDG ontology and are load-bearing. Sections of earlier
> revisions that framed the deliverable as a `vocab_label_map.json`
> contract, a `src/aegir/synth/` generator library, or a fixed
> ~520-template catalog under a GRPO/verifier training program
> describe a superseded plan; the current deliverable is the realized
> HermiT-validated OWL artifact and the ontology-grounded corpus.

## Contract Ægir publishes outward

The outward deliverable is the **realized SDG ontology** — a
HermiT-validated OWL artifact at
`corpora/ontology/sdg-ontology.{omn,owl}` with a consistency
certificate (`HERMIT_CERTIFICATE.md`) — shared through the `corpora`
submodule (zndx/sdg-corpora). It is **versioned and independently
consumable**: any consumer can load the `.omn`/`.owl` in an OWL
reasoner (Protégé/HermiT, ROBOT, owlready2) and re-verify consistency.
The ontology ships alongside the ontology-grounded synthetic corpus,
the relational DDL spine derived from it, the catalog mirror, and the
SKOS vocabulary — the Data Products of the `corpora` submodule.

Publication is **gated** — no `sync --push` of the ontology Data
Product until the OQuaRE quality gate is GREEN (see
[Authors Guide § 4](./authors_guide.md#4-the-oquare-quality-gate)).
This is the entire outward obligation; anything else a consumer wants
is a feature request, not a constraint on Ægir's internals.

## Provenance discipline

The ontology's structure carries its own grounding:

- **Public-namespace IRIs** (`bfo:`, `cco:`, `fhir:`, `iao:`, `skos:`,
  `rdfs:`, `owl:`) are reused directly — their authority comes from the
  namespace. Numeric IRIs (`bfo:0000040`, `cco:ont00000713`) are looked
  up with the grounding-anchor retriever
  (`scripts/grounding_anchors.py`), never invented.
- **SDG-namespace terms** (`sdg:` prefix) are bespoke classes and
  properties authored by the project. Each chains to a BFO 2020 upper
  class (directly or through CCO/FHIR) and carries a definition. These
  are the novel contributions of the work — by construction they do
  not exist in public reference sets, which is the point of a bespoke
  ontology.

The discipline is editorial, not algorithmic. A CI script that tried
to mechanically verify "novel-vs-derived" would either block
legitimate bespoke entities (which by construction appear in no public
reference set) or rubber-stamp around its own checks. Provenance lives
in PR review: a reviewer who recognizes that a candidate term reads as
material lifted from an external source, rather than as the project's
own engineering and conceptual work, raises that the same way they
would raise any other authorship concern.

The mechanical checks Ægir *does* run are about **structural
integrity**, not provenance: that the TTL parses, every term carries a
label and definition, and every `sdg:` term has a BFO subClassOf chain
(`just check-ontology-schema`). The strong, un-fakeable membranes
(HermiT and OntoClean) enforce logical and ontological correctness;
see [Authors Guide § 5](./authors_guide.md#5-the-disposal-membranes-what-rejects-your-extension-and-why).

## Design constraints that follow

1. **The ontology lives in source, not in a database.** The live
   catalog (`src/aegir/ontology/catalog/catalog.json`) and its
   candidate staging are text files in version control; mutations are
   PRs with diffs. The realized `.omn`/`.owl` is build output. If a UI
   ever writes to a DB, the export pipeline reconciles into the
   catalog, not the other way around.

2. **The ontology drives a synthetic corpus, not a service.** The
   ontology-grounded chapter generation and DDL-spine materialization
   (`scripts/generate_chapter.py`, `src/aegir/ontology/ddl.py`,
   `realize.py`) run as importable, seed-deterministic Python that
   emits chapters, tables, and views to disk for downstream
   consumers; they are not a daemon or a network service in Ægir's own
   usage.

3. **Content-first derivation drives coverage.** The ontology is
   fully derived: FinePDFs-content derivation (qdrant/ColBERT
   aperture filtering → engine derives pattern-bound primitives →
   membranes dispose → `scripts/promote_candidates.py` admits into
   `catalog.json`). The hand-authored seed families are retired —
   everything in the catalog earned its way through the
   derive→promote loop. Coverage grows by deriving new
   property-bearing subsumers from text, not by enlarging a fixed
   template count.

4. **One BFO anchor, multiple operational contexts.** SDG forces
   cross-context concepts to be expressed as shared subclasses of
   common BFO/CCO ancestors rather than as discipline-specific aliases
   for the same real-world entity. This *cross-context cousining* is
   the load-bearing architectural invariant (§ *Branch structure*).

## Domain commitments — Signals Data Governance (SDG) Ontology

*Section added 2026-05-09 after collaborative domain choice;
session note at `docs/scratch/2026-05-09/232551_domain_choice.md`.
The branch structure and external anchors below remain the committed
architecture of the SDG ontology.*

*Note (2026-07-09): the `sdg:` leaf classes named below are the v0.1
seed-era population; they were retired with the hand-authored seed
families (`4d200e4`). The **upper-branch commitment and the
cross-context-cousining invariant stand** — the fully-derived catalog
populates the same BFO/CCO branches (at the time of writing 106
realized classes anchor at `bfo:Process`, 68 at `bfo:Role`, with the
ICE branches carrying the record / measurement / directive classes).
The belief branch's reinstatement is a standing publish rider (the
DST belief module, #136).*

### Identity

The bespoke ontology the project authors and publishes is the
**Signals Data Governance (SDG) Ontology** — a vendor-neutral research
artifact that Signals 360 implements and extends. The neutral name
preserves flexibility for open-source release or sovereign
deployments.

- **Ontology IRI prefix:** `sdg:` for bespoke terms;
  `cco:`, `bfo:`, `fhir:`, `iao:`, `skos:`, `rdfs:` for
  public-namespace anchors. (See the namespace table in
  [Authors Guide § 1](./authors_guide.md#namespaces).)
- **Source-of-truth:** the live catalog
  `src/aegir/ontology/catalog/catalog.json`, realized to
  `corpora/ontology/sdg-ontology.{omn,owl}`.
- **Aegir** remains the project / codebase identity; SDG is the
  ontology that the Aegir project hosts.

### Branch structure (committed)

Five primary branches plus a belief branch, all anchored in BFO 2020
+ CCO. Cross-context cousining (e.g., `sdg:Trace` and `sdg:LabRun`
share `sdg:ObservationProcess`) is the load-bearing architectural
invariant.

```
bfo:Continuant
├── cco:IndependentContinuant
│   ├── cco:Artifact ← sdg:Instrument, sdg:Dataset, sdg:SystemBlock,
│   │                  sdg:Program, sdg:Sample, sdg:eBPFProgram,
│   │                  sdg:KernelHook, sdg:Map (eBPF map)
│   └── cco:Person / cco:Organization
└── bfo:GenericallyDependentContinuant
    └── cco:InformationContentEntity
        ├── cco:DesignativeICE ← sdg:Identifier, sdg:AttributeKey,
        │                         sdg:Reference, sdg:Syscall (ID)
        ├── cco:DescriptiveICE ← sdg:Measurement, sdg:Profile,
        │                         sdg:OutlierClaim, sdg:State,
        │                         sdg:Annotation, sdg:AttributeSet,
        │                         sdg:Lift, sdg:Aggregation
        │   └── sdg:BeliefStructure ← sdg:MassFunction,
        │                              sdg:BeliefInterval,
        │                              sdg:Evidence, sdg:Claim
        └── cco:DirectiveICE ← sdg:Requirement, sdg:Control,
                                sdg:Policy, sdg:Constraint
                                (CCO label is "Prescriptive ICE";
                                 SDG names this branch "Directive
                                 ICE" via owl:equivalentClass to
                                 cco:ont00000965 — see naming note
                                 below)

bfo:Occurrent
└── bfo:Process
    ├── sdg:ObservationProcess ← sdg:LabRun, sdg:Trace,
    │                             sdg:Profiling, sdg:OutlierDetection,
    │                             sdg:eBPFEvent
    ├── sdg:DerivationProcess ← sdg:LineageEdge, sdg:Transformation,
    │                            sdg:Allocation
    │                            (PROV-O anchored: subClassOf prov:Activity)
    └── sdg:GovernanceProcess ← sdg:Verification, sdg:Attestation,
                                 sdg:Classification, sdg:Audit
```

### Branch / context mapping

| Professional context | Primary branch hits |
|---|---|
| **LIMS** | `sdg:Sample`, `sdg:Instrument`, `sdg:LabRun`, `sdg:Measurement`, `sdg:Verification`; lineage via `sdg:LineageEdge` |
| **MBSE / SysMLv2** (user-level) | `sdg:SystemBlock`, `sdg:Requirement`, `sdg:State`, `sdg:Verification`, `sdg:Allocation`, `sdg:Constraint` |
| **Database metadata + EAV + open lineage** | `sdg:Dataset`, `sdg:AttributeKey`, `sdg:Identifier`, `sdg:Reference`, `sdg:Profile`, `sdg:Annotation`, `sdg:LineageEdge`, `sdg:Transformation` |
| **Macrobase modernization** | `sdg:OutlierDetection`, `sdg:OutlierClaim`, `sdg:AttributeSet`, `sdg:Lift`, `sdg:Aggregation`, `sdg:Profile` |
| **OTel + eBPF cybersec** | `sdg:Trace` (spans), `sdg:Instrument` (probe/exporter), `sdg:Program`, `sdg:eBPFProgram`, `sdg:KernelHook`, `sdg:eBPFEvent`, `sdg:Syscall`, `sdg:Map`, `sdg:AttributeKey` (SemConv), `sdg:Measurement`, `sdg:Policy`, `sdg:Control` |

### External anchors

| External standard | SDG alignment |
|---|---|
| **BFO 2020** | Upper structure; every leaf has `subClassOf+` to BFO |
| **CCO 2.x** | Mid-tier (Artifact, ICE branches); imported as a reasoning authority so HermiT validates grounding against CCO's disjointness axioms |
| **FHIR R5** | Clinical/record genera, bridged to `cco:InformationContentEntity` (210 types in the grounding index) |
| **OBI / IAO** (OBO Foundry) | `iao:0000115` definition annotations; anchor for `sdg:LabRun`, `sdg:Measurement`, `sdg:Instrument` |
| **PROV-O** (W3C) | OWL-semantics anchor for `sdg:DerivationProcess` lineage |
| **OpenLineage** (LF AI&Data) | Operational runtime surface for `sdg:LineageEdge`; mapped via SSSOM |
| **OpenMetadata** | Operational runtime alignment for `sdg:Dataset`, `sdg:Annotation` |
| **OTel SemConv** | Mapping target for `sdg:AttributeKey` (HTTP, DB, RPC, security conventions) |
| **SysMLv2** (user-level) | Mapping target for `sdg:SystemBlock`, `sdg:Requirement`, `sdg:Allocation`, `sdg:State`, `sdg:Constraint` (KerML metamodel deferred) |
| **NIST PII / ISO 19944** | Public reference for `sdg:Classification` sensitivity tiers |
| **W3C DCAT, Schema.org, DBpedia** | Public mid-tier for benchmark coverage (SOTAB, GitTables) |

### Naming note — DirectiveICE vs CCO's PrescriptiveICE

CCO's canonical IRI `cco:ont00000965` carries `rdfs:label
"Prescriptive Information Content Entity"`. SDG renames this branch
"Directive ICE" because *directive* better captures the normative
sense (requirements, controls, policies, constraints) than
*prescriptive* (which can read as recipe-like). The rename is a
shorthand convention only — the bespoke `sdg:DirectiveICE` is declared
as `owl:equivalentClass cco:ont00000965` so all CCO-side deductions
remain available. Reviewers reading CCO source see the canonical
"Prescriptive ICE" label; reviewers reading SDG see "Directive ICE";
both ground at the same IRI.

### Resolved design decisions (2026-05-09)

The six open questions in
`docs/scratch/2026-05-09/232551_domain_choice.md` resolved as:

- **Q1 — Belief branch:** **include**; `sdg:BeliefStructure` under
  `cco:DescriptiveICE`. Direct alignment with Atelier's DST evidence
  fusion; future-proofs federated-intelligence use cases where
  conflict *K* and epistemic uncertainty must propagate across nodes.
- **Q2 — eBPF / cybersec depth:** **eBPF first-class**; adds
  `sdg:eBPFProgram`, `sdg:KernelHook`, `sdg:Syscall`, `sdg:Map`,
  `sdg:eBPFEvent`. OTel remains the primary runtime surface;
  first-class eBPF preserves semantic grounding without translation
  loss.
- **Q3 — SysMLv2 depth:** **user-level primitives only**; KerML
  metamodel deferred. Block, Part, Action, State, Requirement,
  Allocation, Verification only.
- **Q4 — Lineage anchor:** **PROV-O for OWL semantics + OpenLineage
  for runtime surface**, mapped via SSSOM. Single deductive core;
  preserves operational interop.
- **Q5 — Macrobase:** **pre-anchor lightly** (`sdg:OutlierClaim`,
  `sdg:AttributeSet`, `sdg:Lift`, `sdg:Aggregation`, plus relations).
  Modernization team free to extend.
- **Q6 — Ontology name:** **Signals Data Governance (SDG) Ontology**.
  Vendor-neutral; preserves open-source / sovereign deployment
  optionality.

These decisions are committed at v0.1 of the SDG ontology. Future
revisions require explicit version bumps tracked in
`docs/scratch/YYYY-MM-DD/` session notes.

## What stays out of Ægir

- Dempster-Shafer fusion, belief/plausibility logic, any specific
  classification pipeline shape (Atelier's domain).
- Gateway / UI features that are not directly about Ægir's own view of
  its runs.
- Customer-deployment glue: mid-run watchers, agent loop governance,
  FSM session state. These belong with the consumer that owns the
  deployment lifecycle.
- Storage schemas (Hive / Iceberg / Postgres) that exist only for
  sibling-project governance flows. Ægir publishes the realized OWL
  artifact and the corpus; consumers translate to their own storage
  shape.
