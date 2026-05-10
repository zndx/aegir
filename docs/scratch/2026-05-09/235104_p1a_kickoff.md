# P1a kickoff — template authoring batch plan

*2026-05-09 — scopes the ~565 SDG-ontology template authoring
effort committed in
`docs/current/ontology/charter.md#domain-commitments--signals-data-governance-sdg-ontology`
and v0.5 concept brief P1a.*

## Goal

Land `src/aegir/ontology/sdg-vocab.ttl` carrying the bespoke OWL
class definitions, with each SDG-namespaced term anchored to BFO
2020 + CCO and structured as the catalog templates the runtime
verifier consumes. Per the brief P1 exit gate: ≥ 200 surviving
DeepOnto-validated templates spanning ≥ 5 distinct axiom shapes.
Per the SDG charter: target ~565 authored, ~470 surviving
conservatively.

## Authoring batches (priority-ordered)

Batches are sequenced for **maximum verifier-coverage gain per
batch** — earlier batches exercise more axiom-kind variability
and benchmark coverage, so P2 verifier-validation can run on a
partial catalog if the full P1a effort runs long.

### Batch 1 — Foundation skeleton (~80 templates, ~1 week)

The minimum vocabulary that makes the verifier exercisable on
the full slot DSL grammar.

**Scope:**

- All eight axiom-kind families (subClassOf basic, equivalentClass
  with intersection / union, existential `some`, universal `only`,
  cardinality `min` / `max` / `exactly`, enumeration) represented
  ≥ 3× each.
- BFO 2020 upper structure imported by reference; canonical CCO
  IRIs declared.
- `cco:Artifact` populated: `sdg:Instrument`, `sdg:Dataset`,
  `sdg:SystemBlock`, `sdg:Program`, `sdg:Sample` (5 named
  classes; ~30 templates including subclass and relational
  patterns).
- `cco:DesignativeICE` populated: `sdg:Identifier`,
  `sdg:AttributeKey`, `sdg:Reference` (3 named classes; ~25
  templates).
- The two highest-value `sdg:DirectiveICE` classes:
  `sdg:Constraint`, `sdg:Policy` (~15 templates).
- ~10 cross-branch relational templates (e.g.,
  `sdg:Instrument` `producesMeasurement`-some `sdg:Measurement`).

**Exit criterion for batch 1:** verifier runs on the partial
catalog; all four R components produce non-trivial scores on at
least three composition examples.

### Batch 2 — Observation processes + measurement (~110 templates, ~1 week)

Densest branch in the catalog; load-bearing for LIMS, OTel, and
Macrobase contexts.

**Scope:**

- `sdg:ObservationProcess` with subclasses: `sdg:LabRun`,
  `sdg:Trace`, `sdg:Profiling`, `sdg:OutlierDetection`,
  `sdg:eBPFEvent` (5 classes; ~30 templates).
- `cco:DescriptiveICE` populated: `sdg:Measurement`,
  `sdg:Profile`, `sdg:OutlierClaim`, `sdg:State`,
  `sdg:Annotation`, plus Macrobase-specific
  `sdg:AttributeSet`, `sdg:Lift`, `sdg:Aggregation` (8 classes;
  ~80 templates including unit/cardinality restrictions).

**Exit criterion for batch 2:** SOTAB v2 Schema.org CTA labels
that are measurement-flavored (Distance, Duration, Energy, Mass,
QuantitativeValue, …) all have catalog-template paths. Macrobase
core concepts authored.

### Batch 3 — Directive layer + governance (~100 templates, ~5 days)

Normative content for SysMLv2-modeled systems and cybersec
controls.

**Scope:**

- `cco:DirectiveICE` populated: `sdg:Requirement`,
  `sdg:Control`, `sdg:Policy`, `sdg:Constraint` (4 classes; ~70
  templates with strong equivalentClass / intersection patterns).
- `sdg:GovernanceProcess`: `sdg:Verification`,
  `sdg:Attestation`, `sdg:Classification`, `sdg:Audit` (4
  classes; ~30 templates).

**Exit criterion for batch 3:** SysMLv2 user-level primitives
(Block, Part, Requirement, Verification, Allocation, State,
Constraint) all have catalog-template paths; cybersec
control / policy / classification entities expressible.

### Batch 4 — eBPF + low-level cybersec (~50 templates, ~3 days)

The first-class eBPF/cybersec coverage decided in Q2.

**Scope:**

- `sdg:eBPFProgram`, `sdg:KernelHook`, `sdg:Syscall` (under
  `sdg:Identifier`), `sdg:Map` as `cco:Artifact` subclasses (~30
  templates).
- `sdg:eBPFEvent` extension on `sdg:ObservationProcess` axioms
  (~15 templates).
- Templates encoding kernel-syscall-causes-event, eBPF-program-
  attaches-to-hook, and similar relational patterns (~5
  templates).

**Exit criterion for batch 4:** representative cybersec
observability flows (syscall trace, eBPF probe attachment,
observed-event-as-process) all expressible.

### Batch 5 — Lineage + derivation (~50 templates, ~3 days)

PROV-O alignment + OpenLineage SSSOM bridge.

**Scope:**

- `sdg:DerivationProcess` `subClassOf prov:Activity` (1 class;
  ~10 templates).
- `sdg:LineageEdge`, `sdg:Transformation`, `sdg:Allocation`
  (3 classes; ~30 templates including PROV-O-sense restrictions).
- `prov:Entity` / `prov:Activity` / `prov:Agent` reuse
  patterns; SSSOM mapping rows for OpenLineage Job, Run, Dataset,
  Facet (~10 templates that express the bridge).

**Exit criterion for batch 5:** OpenLineage-style lineage edges
between datasets / runs / jobs all resolvable through PROV-O
axioms in the catalog.

### Batch 6 — Belief structure (~30 templates, ~3 days)

DST / Atelier integration; the Q1 commitment.

**Scope:**

- `sdg:BeliefStructure` `subClassOf cco:DescriptiveICE` (1 class;
  ~5 templates).
- `sdg:MassFunction`, `sdg:BeliefInterval`, `sdg:Evidence`,
  `sdg:Claim` (4 classes; ~25 templates including focal-set
  participation, plausibility-belief-interval pairing,
  evidence-source-grounds-claim relational patterns).

**Exit criterion for batch 6:** a Dempster combination over two
evidence sources expressible as a composition over batch-6
templates.

### Batch 7 — Long tail + benchmark coverage gap-filling (~150 templates, ~1.5 weeks)

Whatever remains to hit benchmark-label coverage targets and to
reach the ~565 authored / ~470 surviving target. Likely
candidates surfaced during batches 1–6:

- DBpedia-CTA target classes that aren't in batches 1–4.
- SOTAB v2 `Hotel/name`, `Movie/description`, `Recipe/name`-style
  entity-property pair templates.
- Specialized measurement subtypes (Energy, Mass, Speed,
  Temperature) under `sdg:Measurement`.
- OTel SemConv attribute-key categorical breakdowns
  (`http.method`, `db.system`, `rpc.service`, etc.).

**Exit criterion for batch 7:** P1 exit gate cleared — ≥ 200
surviving DeepOnto-validated templates spanning ≥ 5 axiom shapes;
total catalog content reaches ~470 surviving target.

## Total effort estimate

| Batch | Wall-clock |
|---|---|
| 1 — Foundation skeleton | ~1 week |
| 2 — Observation + measurement | ~1 week |
| 3 — Directive + governance | ~5 days |
| 4 — eBPF + low-level cybersec | ~3 days |
| 5 — Lineage + derivation | ~3 days |
| 6 — Belief structure | ~3 days |
| 7 — Long tail + benchmark gap-filling | ~1.5 weeks |
| **Total** | **~6 weeks of focused authoring** |

This is in line with the v0.5 brief's P3 estimate of "4–8 weeks
for the largest creative effort" — but note that v0.5 framed
the human-authored ontology as a *baseline* in the P1a / P3 split,
distinct from the catalog. Under the SDG charter the catalog
*is* the bespoke ontology authoring effort; P3 (separate
from-scratch hand-authored baseline ontology) becomes a slimmer
"author one ontology entirely outside the catalog as comparison
target" task. Brief should be reconciled to this distinction
when v0.6 lands.

## Order independence within a batch

Templates within a batch can be authored in any sequence;
dependencies are at the batch boundary (e.g., batch 2's
`sdg:Measurement` references batch 1's `sdg:Instrument` via
`producesMeasurement`-some, so batch 1's `sdg:Instrument`
declaration must exist first).

## Tooling for authoring

Each template lives as a row in a `catalog/*.json` file with the
schema from `src/aegir/ontology/schema.py`'s `CatalogTemplate`.
Authoring workflow:

1. Choose a target template family (e.g., "lab instrument
   produces measurement").
2. Write the Manchester-syntax template with typed slots per
   `SLOT_DSL.md`.
3. Fill `slot_types`, leave `is_complex` / `verbal_template` /
   `mean_verbal_length` as placeholders (DeepOnto offline pass at
   batch-end populates them).
4. Run `scripts/check_ontology_schema.py` to validate slot-syntax.
5. (Once P1b-α lands) Run `aegir-verify` on a hand-built
   composition that uses the new template; verify R is in
   sensible range.
6. (At end of each batch) Run the offline DeepOnto pass to
   populate `is_complex` / `verbal_template` / `mean_verbal_length`,
   drop templates that fail to load or verbalize.

## Catalog file organization

Within `src/aegir/ontology/catalog/`, organize by branch:

```
catalog/
├── examples.json                    # placeholder (already exists)
├── 01_foundation.json               # batch 1: artifact + designative
├── 02_observation_measurement.json  # batch 2: observation + descriptive
├── 03_directive_governance.json     # batch 3: directive + governance
├── 04_ebpf.json                     # batch 4: eBPF artifacts + events
├── 05_lineage.json                  # batch 5: derivation + PROV-O
├── 06_belief.json                   # batch 6: BeliefStructure
└── 07_long_tail.json                # batch 7: gap-filling
```

`scripts/check_ontology_schema.py` already iterates `catalog/*.json`
and validates each file independently, so adding batch files
incrementally surfaces errors per-batch rather than at full-catalog
merge time.

## What launches in parallel

Per the v0.5 brief and the user's greenlight:

- **P1b-α verifier shell** (R_A + R_B + R_C without R_D, against
  placeholder `examples.json`) starts immediately, parallel-safe
  to all batches above. Once it lands, batch authors can
  exercise newly authored templates against a working scorer
  during authoring rather than at batch-end.
- **DeepOnto offline harness** is the per-batch closeout step;
  it can be authored in parallel with batches 1–2 so it's ready
  by batch 1 closeout.

## Open authoring questions for batches 4–6

The eBPF, lineage, and belief branches are the least-precedented
in public ontology work. P1a authoring may surface design choices
that want a separate decision pass:

- **eBPF:** does `sdg:eBPFProgram` carry the program type
  (kprobe, uprobe, tracepoint, XDP, …) as `is_a` subclasses, as
  a property, or as a slot? Affects template count by ~10.
- **Lineage:** does `sdg:LineageEdge` get separate subclasses
  per OpenLineage `eventType` (START, COMPLETE, FAIL, ABORT) or
  fold into a single class with a typed property? Affects
  template count by ~5.
- **Belief:** is the focal-set frame enumerated explicitly per
  classification taxonomy, or kept as an open slot? Affects
  template count by ~5 and may surface an open OWL-modeling
  question (open-world DST representation).

These don't gate batches 1–3 starting; flagged here so they're
visible when P1a authoring reaches them.
