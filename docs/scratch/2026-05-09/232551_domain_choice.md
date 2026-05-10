# Domain choice — composite proposal

*2026-05-09 — collaborative working document for P1a's ontology
domain choice. Companion to the v0.5 concept brief.*

## Inputs

The five professional contexts the bespoke ontology must inform:

1. **LIMS** — laboratory information management; instruments,
   samples, runs, results, batch lineage, QC.
2. **MBSE / SysMLv2** — blocks, parts, actions, states,
   requirements, verifications, allocations.
3. **Database metadata** — column semantics, EAV constructs, open
   lineage (OpenLineage), open metadata (OpenMetadata, Apache
   Atlas, DataHub).
4. **Macrobase modernization** — outlier explanation, statistical
   lift, attribute-set discovery; targeted at consumer
   GPU-at-the-edge networks; modeled in SysMLv2 (so MBSE
   primitives apply directly).
5. **OpenTelemetry + eBPF for cybersecurity** — spans, metrics,
   logs, attributes, baggage, resource (OTel); programs, maps,
   hooks, events (eBPF); semantic conventions, sampling,
   policy.

The ontology must be a *composite* — expressive enough to span
all five, specific enough to be authored without becoming a
generic upper ontology.

## Proposed structure: Observable Information Engineering (OIE)

A five-branch ontology grounded in BFO 2020 + the Common Core
Ontologies (CCO), with optional alignment to PROV-O for lineage
and Atelier's DST machinery for uncertainty.

### Branch sketch

```
bfo:Entity
├── bfo:Continuant
│   ├── bfo:IndependentContinuant
│   │   ├── cco:Artifact
│   │   │   ├── sdg:Instrument        ← lab instruments, sensors,
│   │   │   │                              eBPF probes, OTel exporters
│   │   │   ├── sdg:Dataset           ← tables, columns, datasets,
│   │   │   │                              parquet partitions
│   │   │   ├── sdg:SystemBlock       ← SysMLv2 Block, Part, Item
│   │   │   ├── sdg:Program           ← eBPF programs, query plans,
│   │   │   │                              ML models, pipelines
│   │   │   └── sdg:Sample            ← lab specimens, capture
│   │   │                                  records, packets
│   │   └── cco:Person / cco:Organization
│   │
│   └── bfo:GenericallyDependentContinuant
│       └── cco:InformationContentEntity
│           ├── cco:DesignativeICE
│           │   ├── sdg:Identifier         ← UUIDs, sample IDs,
│           │   │                              span IDs, column names
│           │   ├── sdg:AttributeKey       ← OTel attr keys, EAV
│           │   │                              attribute names,
│           │   │                              SemConv keys
│           │   └── sdg:Reference          ← FKs, lineage refs,
│           │                                  block port refs
│           │
│           ├── cco:DescriptiveICE
│           │   ├── sdg:Measurement        ← lab readings, OTel
│           │   │                              metrics, eBPF counters
│           │   ├── sdg:Profile            ← column statistics,
│           │   │                              KPI snapshots
│           │   ├── sdg:OutlierClaim       ← Macrobase explanations,
│           │   │                              anomaly attributions
│           │   ├── sdg:State              ← SysMLv2 State,
│           │   │                              system state observation
│           │   └── sdg:Annotation         ← Atelier classifications,
│           │                                  CTA/CPA labels
│           │
│           └── cco:DirectiveICE
│               ├── sdg:Requirement        ← SysMLv2 Requirement
│               ├── sdg:Control            ← cybersec controls,
│               │                              governance attestations
│               ├── sdg:Policy             ← sampling, retention,
│               │                              access policies
│               └── sdg:Constraint         ← SQL CHECK, OWL
│                                              cardinality, SysMLv2
│                                              Constraint
│
└── bfo:Occurrent
    └── bfo:Process
        ├── sdg:ObservationProcess
        │   ├── sdg:LabRun               ← LIMS run, sample analysis
        │   ├── sdg:Trace                ← OTel span/trace, eBPF event
        │   ├── sdg:Profiling            ← column profiling, KPI
        │   │                                aggregation
        │   └── sdg:OutlierDetection     ← Macrobase invocation
        │
        ├── sdg:DerivationProcess
        │   ├── sdg:LineageEdge          ← OpenLineage Job/Run/Dataset
        │   ├── sdg:Transformation       ← ETL step, query execution
        │   └── sdg:Allocation           ← SysMLv2 Allocation
        │
        └── sdg:GovernanceProcess
            ├── sdg:Verification         ← SysMLv2 VerificationCase
            ├── sdg:Attestation          ← compliance attestation
            ├── sdg:Classification       ← data classification, tier
            │                                assignment
            └── sdg:Audit                ← review, sign-off
```

Optional belief branch (deferrable; see open question 1):

```
cco:DescriptiveICE
└── sdg:BeliefStructure
    ├── sdg:MassFunction          ← DST mass over focal set
    ├── sdg:BeliefInterval        ← belief / plausibility pair
    ├── sdg:Evidence              ← evidence source + weight
    └── sdg:Claim                 ← supported assertion w/ uncertainty
```

### Mapping the five professional contexts

| Context | Primary branch hits |
|---|---|
| **LIMS** | `sdg:Sample`, `sdg:Instrument`, `sdg:LabRun`, `sdg:Measurement`, `sdg:Verification`; lineage via `sdg:LineageEdge` |
| **MBSE / SysMLv2** | `sdg:SystemBlock`, `sdg:Requirement`, `sdg:State`, `sdg:Verification`, `sdg:Allocation`, `sdg:Constraint` |
| **Database metadata + EAV + open lineage** | `sdg:Dataset`, `sdg:AttributeKey`, `sdg:Identifier`, `sdg:Reference`, `sdg:Profile`, `sdg:Annotation`, `sdg:LineageEdge`, `sdg:Transformation` |
| **Macrobase / outlier explanation** | `sdg:OutlierDetection`, `sdg:OutlierClaim`, `sdg:Profile`; SysMLv2-modeled blocks → `sdg:SystemBlock`; uncertainty → `sdg:BeliefStructure` (if branch included) |
| **OTel + eBPF / cybersecurity** | `sdg:Trace` (spans), `sdg:Instrument` (probe/exporter), `sdg:Program` (eBPF program), `sdg:AttributeKey` (SemConv), `sdg:Measurement` (metrics), `sdg:Policy` (sampling, RBAC), `sdg:Control` (security control) |

Every context has natural ancestors in BFO/CCO and natural cousins
across other contexts (e.g., `sdg:Trace` and `sdg:LabRun` both
specialize `sdg:ObservationProcess`; `sdg:Constraint` covers
both SQL CHECK and SysMLv2 Constraint).

This cross-context cousining is what makes the ontology *useful*
beyond any single domain — annotations from a database column
profile and a SysMLv2-modeled telemetry block can share BFO
ancestors and be reasoned about under one roof.

### Catalog distribution proposal (~500 templates)

| Branch | Templates | Rationale |
|---|---|---|
| `cco:Artifact` (Instrument, Dataset, SystemBlock, Program, Sample) | ~100 | Five primary artifact subbranches; structural relations (part_of, has_role, has_input/output) generate template variation |
| `cco:DesignativeICE` (Identifier, AttributeKey, Reference) | ~70 | Heavy in OTel/EAV/database metadata; sparse complex axioms |
| `cco:DescriptiveICE` (Measurement, Profile, OutlierClaim, State, Annotation) | ~130 | Densest branch — measurement axioms with units, restrictions, Quality-bearing patterns |
| `cco:DirectiveICE` (Requirement, Control, Policy, Constraint) | ~70 | SysMLv2 Requirement + cybersec control + DB constraint; expressive equivalentClass patterns |
| `sdg:ObservationProcess` | ~60 | LabRun, Trace, Profiling, OutlierDetection; existential restrictions on participants |
| `sdg:DerivationProcess` | ~40 | LineageEdge, Transformation, Allocation; PROV-O alignment |
| `sdg:GovernanceProcess` | ~30 | Verification, Attestation, Classification, Audit |
| `sdg:BeliefStructure` (optional) | ~20 (if included) | MassFunction, BeliefInterval, Evidence, Claim |

Total: ~500 with belief branch, ~480 without. The brief
committed "≥ 200 surviving templates" as the P1 exit gate; this
plan is comfortably above that even with conservative drop-rates
on DeepOnto verbalization gates.

## Anchor ontologies and standards

External work the bespoke ontology aligns to (rather than
inherits from):

| External | Aegir alignment |
|---|---|
| **BFO 2020** | Upper structure; every leaf has a `subClassOf+` chain to BFO |
| **CCO 2.x** | Mid-tier (Artifact, ICE branches); equivalentClass / subClassOf bridges |
| **OBI / IAO** (OBO Foundry) | Anchor for `sdg:LabRun`, `sdg:Measurement`, `sdg:Instrument` |
| **PROV-O** (W3C) | Anchor for `sdg:DerivationProcess` lineage; provides Activity, Entity, Agent classes |
| **OpenLineage** (LF AI&Data) | Operational alignment for `sdg:LineageEdge`; non-OWL but mappable via SSSOM |
| **OpenMetadata** | Operational alignment for `sdg:Dataset`, `sdg:Annotation`; non-OWL |
| **OTel SemConv** | Mapping target for `sdg:AttributeKey` (HTTP, DB, RPC conventions); non-OWL |
| **SysMLv2 / KerML** | Mapping target for `sdg:SystemBlock`, `sdg:Requirement`, `sdg:Allocation`, `sdg:State` |
| **NIST PII categories / ISO 19944** | Public reference for `sdg:Classification` sensitivity tiers |
| **W3C DCAT, Schema.org, DBpedia** | Public mid-tier where benchmarks (SOTAB, GitTables) need coverage |

This list is the v1 reference set — the authored vocabulary maps
to it via SSSOM provenance axioms but does not depend on any of
them at runtime.

## Open questions for collaboration

These are decisions I shouldn't make alone; they affect catalog
scope and the brief's downstream commitments.

### Q1 — Belief branch: include or defer?

**For including (~20 templates added):**
- Atelier alignment becomes structural rather than convention-of-API.
- The ontology can express "this measurement has a belief interval
  computed by DST fusion of these evidence sources" directly.
- "Properly uncertain knowledge" in the Signals 360 framing gets a
  first-class home in the ontology, not just in code.

**For deferring (smaller v1, faster authoring):**
- DST/Dempster-Shafer ontology has no widely-adopted public anchor
  (would have to be authored fresh against BFO).
- Adds research scope (uncertainty representation in OWL is its
  own subfield).
- Could land as a v2 ontology extension after P2 validates the
  baseline verifier.

**My read:** include. The Signals 360 mission framing makes belief
representation load-bearing, and 20 templates is a small fraction
of the total. But this is your call.

### Q2 — eBPF / cybersecurity depth

**Option A — OTel-only.** Treat OTel SemConv as the sole
cybersecurity observability surface. eBPF events flow in via OTel
exporters; eBPF-program / kernel-hook / syscall don't get
first-class classes. Saves ~30 templates.

**Option B — eBPF first-class.** Add `sdg:eBPFProgram`,
`sdg:KernelHook`, `sdg:Syscall`, `sdg:Map` as artifact /
process subbranches with their own restrictions. Adds ~30
templates and ~20 cybersecurity-specific axiom patterns.

**My read:** Option B. The accumulated applied work the user
mentioned justifies first-class eBPF representation — and
otherwise we lose the kernel-level granularity that makes eBPF
distinctive vs. OTel-only observability.

### Q3 — SysMLv2 mapping depth

**Option A — User-level primitives only.** Block, Part, Action,
State, Requirement, Allocation, Verification. ~40 templates
across `cco:Artifact` and `sdg:ObservationProcess`.

**Option B — KerML metamodel coverage.** Type, Feature,
Connection, Behavior, plus the SysMLv2 user-level surface.
Doubles to ~80 templates. KerML is the formal foundation of
SysMLv2 and aligns better with BFO via CCO.

**My read:** Option A initially; KerML coverage as a v2
extension if Macrobase modernization or other SysMLv2-heavy work
needs it. Reason: KerML's metamodel-level templates verbalize
poorly (very abstract) and may struggle DeepOnto's verbalization
gate.

### Q4 — OpenLineage vs PROV-O for derivation

**Option A — PROV-O as the anchor.** Rigorous, W3C-standardized,
OWL-native. `sdg:LineageEdge` `subClassOf prov:Activity`, etc.

**Option B — OpenLineage operational + PROV-O conceptual.**
Author against PROV-O for the OWL anchor; provide an SSSOM
mapping table to OpenLineage's JSON-schema entities for
operational use. Both surfaces stay accessible.

**My read:** Option B. PROV-O for the OWL semantics, OpenLineage
for the runtime data exchange. The SSSOM mapping is a small
artifact (~30 entries) and gives downstream consumers the
interop they want without committing the OWL ontology to
OpenLineage's evolving JSON schema.

### Q5 — Macrobase modernization: pre-anchor or co-evolve?

**Pre-anchor.** Author Macrobase-specific templates now
(`sdg:OutlierDetection`, `sdg:OutlierClaim`, statistical-
lift restrictions) so the modernization work plugs into a
ready vocabulary.

**Co-evolve.** Treat Macrobase modernization as a parallel
SysMLv2-modeled engineering effort whose outputs feed back into
v2 of the ontology. Keep v1 ontology agnostic.

**My read:** Pre-anchor lightly. ~15 templates that cover the
core Macrobase concepts (OutlierClaim, AttributeSet, Lift,
Aggregation) — enough that the modernization work has a
vocabulary to extend, not enough that we lock down the
modernization's design choices in v1.

### Q6 — Naming the ontology

The proposal uses "Observable Information Engineering (OIE)" as a
working name. Alternatives:

- **Signals 360 Core Ontology (S360-CO)** — directly product-aligned.
- **Aegir Bespoke Ontology (ABO)** — project-aligned, vendor-neutral.
- **Composite Observation, Provenance, and Belief (COPB)** —
  descriptive of the branches.
- Other?

**My read:** S360-CO if the ontology is meant to be a
product-facing artifact, ABO if it's meant to be vendor-neutral
research output, OIE if "engineering" framing matters. Your
call.

## What I'm proposing

If you agree with the structural proposal (five branches, BFO/CCO
anchors, ~500-template scope), the next concrete unit is:

1. **You answer Q1–Q6** (or redirect any of them).
2. **I draft the ontology charter section** — a 2-page
   addition to `docs/current/ontology/charter.md` that pins the
   committed branch structure, the external-anchor list, and the
   catalog distribution targets.
3. **I scope P1a kickoff** — a session note that breaks the ~500
   templates into authoring batches, with priority ordering by
   benchmark / professional-context coverage.
4. **Track A (P1b verifier shell) starts in parallel** —
   independent of which Q1–Q6 answers come back.

This proposal is a working draft. Tell me where it's wrong.
