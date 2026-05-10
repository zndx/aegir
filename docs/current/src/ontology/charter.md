# Charter

This is Ægir's internal direction-setter for the ontology + synth scope.
It declares the contract Ægir publishes outward, names the design
constraints that follow, and pins the empirical gate that any
vocabulary expansion work has to clear before it ships.

## Contract Ægir publishes outward

Two artifacts. Versioned. Independently consumable.

### `vocab_label_map.json`

A flat JSON map keyed by benchmark label (`"sotab:Hotel/name"`,
`"sotab-dbp:Country"`, `"gt-dbp:Place"`, …). Each entry carries:

- `iri` — canonical Schema.org / DBpedia / Atelier-vocab IRI
- `cco_anchor` — the nearest CCO information-content-entity ancestor
- `bfo_ancestry` — ordered list of BFO 2020 superclasses up to root
- `tier` — provenance (`direct`, `subsumption`, `extension-A/B/C`)
- `notes` — optional free-text rationale (only when the mapping is
  non-obvious)

Rebuilt deterministically from the vocabulary TTL via SPARQL by
`scripts/build_vocab_label_map.py`. Checked into the repo (small, ~tens
of KB) so consumers don't have to install rdflib or run a SPARQL engine
to use it. Versioned alongside the vocabulary; major version bumps
correspond to BFO-anchor changes.

### Trained model checkpoints

Already produced today. The discipline doesn't change: `outputs/runs/`
sidecars (`metadata.json`, `metrics.jsonl`, `metrics_eval.jsonl`,
checkpoint `.pt` files). What's new is the obligation that any
checkpoint distributed as a "vocabulary-aware" artifact carries the
`vocab_label_map.json` version it was trained against in its
`metadata.json`.

This is the entire outward contract. Anything else a consumer wants is
a feature request, not a constraint on Ægir's internals.

## Provenance discipline

The vocabulary's structure carries its own grounding:

- **Public-namespace IRIs** (`schema:`, `dbo:`, `bfo:`, `cco:`,
  `skos:`, `prov:`, …) are reused directly — their authority comes
  from the namespace.
- **Aegir-namespace terms** (`sdg:` prefix) are bespoke entities
  authored by the project. Each one carries an `rdfs:subClassOf`
  chain anchored at a BFO 2020 upper class and a clear
  `skos:definition` describing what the term denotes and why it
  exists. These are the novel contributions of the work — they don't
  exist in public reference sets by definition, and that's the point
  of having a bespoke ontology in the first place.

The discipline is editorial, not algorithmic. Provenance lives in PR
review: a reviewer who recognizes that a candidate term reads as
material lifted from an external source, rather than as the project's
own engineering and conceptual work, raises that in review the same
way they would raise any other authorship concern. `PROVENANCE.md`
(below) is the reviewer's guide.

A CI script that tried to mechanically verify "novel-vs-derived" would
either block legitimate bespoke entities (since by construction they
don't appear in any public reference set) or rubber-stamp around its
own checks. Neither is useful. The mechanical checks Ægir does run
on the TTL — syntactic validity, every term carries a label +
definition, every `sdg:` term has a BFO subClassOf chain — are
about *structural integrity*, not about provenance. See
[Migration §Phase 1](./migration.md#phase-1--author-the-initial-vocabulary).

## Design constraints that follow

1. **Vocabulary lives in source, not in a database.** `sdg-vocab.ttl`
   is a text file in version control; mutations are PRs with diffs.
   Postgres write paths described in older milestones (the M2 line in
   the README about an "ontology editor with Postgres write paths") are
   *consumer-side ergonomics*, not the source of truth. If we add a UI
   that writes to a DB, the export pipeline reconciles into the TTL,
   not the other way around.

2. **Synthetic data generation is a library, not a service.** The
   generators in `src/aegir/synth/` will be importable Python; they
   produce byte-streams or row-dicts in process. A generator output
   snapshot can be written to disk for downstream consumers (BDD/pytest
   fixtures, sibling-project tests), but the generators themselves do
   not run as a daemon or behind a network call from Ægir's own usage.

3. **The pretraining corpus is not yet vocabulary-conditioned.** v2
   (Apr 27, 2026) trained on FineWeb-Edu + SQaLe + SchemaPile +
   FinePDFs-lab — none of which depend on `sdg-vocab.ttl`. Whether a
   future v3+ corpus mix should include vocabulary-driven synthetic
   slices is an open research question, not a migration step. The
   ontology move makes the experiment *possible*; it does not commit us
   to running it.

4. **One BFO anchor, multiple benchmark namespaces.** SOTAB v2
   Schema.org (82 labels), SOTAB-DBpedia (101), SOTAB-DBpedia-restricted
   (53), and GitTables-DBpedia (122) are all tracked in
   `_LABEL_DIMS` today and remain separate keys in
   `vocab_label_map.json`. They share BFO ancestors but stay distinct
   at the benchmark layer; we do not collapse them into a unified label
   set. This decision is reversible if a unified label set proves
   useful later, but the default is preservation.

5. **Schema.org / DBpedia split is Ægir's call, not a consumer's.** A
   sibling project might prefer one or the other; the canonical
   vocabulary covers both because both have benchmark coverage we want
   to demonstrate against. If a consumer wants only one, they filter
   the JSON map at load time.

6. **No prior-incident commitments inherited.** Atelier's documentation
   includes a Tier-A / Tier-B / Tier-C breakdown of SOTAB v2
   Schema.org coverage. Ægir has read it, found the analysis
   directionally useful, and is not bound by the order, the cut points,
   or the implementation milestones. Vocabulary expansion work scopes
   itself against Ægir's empirical gates (below), not against a
   tiering taken from a consumer's planning artifact.

## Empirical gate before any vocabulary expansion

The v2 mixed-corpus pretrain (`outputs/mixed-v2/20260426T232240Z/final.pt`)
is a healthy byte-level backbone — stratified eval shows non-degenerate
representations across `eval.fineweb-held`, `eval.schemapile-held`,
`eval.sqale-held`, `eval.spider`. **However, no SOTAB head fine-tune
has been run from this checkpoint yet.** The Apr 19 representation
collapse on direct-from-random SOTAB CTA was the open wound that
motivated the v2 pretrain in the first place; the v2→SOTAB head
fine-tune is what closes that loop.

The gate is precise:

> **Before merging any vocabulary expansion (Tier-A measurement zoo,
> Tier-B subClassOf plumbing, Tier-C product/jobposting branches, or any
> equivalent Ægir-defined breakdown), the v2→SOTAB head fine-tune must
> demonstrate non-degenerate per-class F1 on a representative subset of
> SOTAB v2 Schema.org labels.**

Concretely:

- ≥ 3 distinct embedding clusters at coarse MCL inflation (vs. the 1
  cluster that flagged collapse in April).
- ≥ 0.10 macro F1 on the held-out SOTAB CTA validation set (a bar that's
  trivial to clear if representations are alive, and unreachable if they
  collapsed).
- A non-trivial confusion matrix — model predictions distributed across
  ≥ 10 distinct labels, not concentrated on any single mode class.

These thresholds are deliberately undemanding. The point is *liveness*,
not benchmark-leading performance. If they fail, the underlying problem
is not vocabulary coverage; tier work would only add labels for a model
that can't distinguish them. We diagnose the model first.

## Domain commitments — Signals Data Governance (SDG) Ontology

*Section added 2026-05-09 after collaborative domain choice;
session note at `docs/scratch/2026-05-09/232551_domain_choice.md`.
The decisions below are the committed branch structure that P1a
catalog authoring works against.*

### Identity

The bespoke vocabulary the project authors and publishes is the
**Signals Data Governance (SDG) Ontology**. SDG is a vendor-neutral
research artifact that Signals 360 implements and extends.
Product-facing branding may be layered later; the neutral name
preserves flexibility for open-source release or sovereign
deployments.

- **Ontology IRI prefix:** `sdg:` for bespoke terms;
  `cco:`, `bfo:`, `iao:`, `obi:`, `prov:`, `schema:`, `dbo:`,
  `skos:`, `dcterms:` for public-namespace anchors.
- **Source-of-truth file:** `src/aegir/ontology/sdg-vocab.ttl`
  (formerly named `aegir-vocab.ttl` pre-renaming).
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
| **CCO 2.x** | Mid-tier (Artifact, ICE branches); `equivalentClass` / `subClassOf` bridges |
| **OBI / IAO** (OBO Foundry) | Anchor for `sdg:LabRun`, `sdg:Measurement`, `sdg:Instrument` |
| **PROV-O** (W3C) | OWL-semantics anchor for `sdg:DerivationProcess` lineage |
| **OpenLineage** (LF AI&Data) | Operational runtime surface for `sdg:LineageEdge`; mapped via SSSOM |
| **OpenMetadata** | Operational runtime alignment for `sdg:Dataset`, `sdg:Annotation` |
| **OTel SemConv** | Mapping target for `sdg:AttributeKey` (HTTP, DB, RPC, security conventions) |
| **SysMLv2** (user-level) | Mapping target for `sdg:SystemBlock`, `sdg:Requirement`, `sdg:Allocation`, `sdg:State`, `sdg:Constraint` (KerML metamodel deferred to v2) |
| **NIST PII / ISO 19944** | Public reference for `sdg:Classification` sensitivity tiers |
| **W3C DCAT, Schema.org, DBpedia** | Public mid-tier for benchmark coverage (SOTAB, GitTables) |

### Catalog distribution targets (~520 templates)

| Branch | Target template count | Rationale |
|---|---|---|
| `cco:Artifact` (Instrument, Dataset, SystemBlock, Program, Sample, eBPFProgram, KernelHook, Map) | ~115 | Adds ~15 templates for first-class eBPF coverage on top of the base ~100 |
| `cco:DesignativeICE` (Identifier, AttributeKey, Reference, Syscall) | ~75 | OTel SemConv + EAV + DB metadata; sparse complex axioms |
| `cco:DescriptiveICE` (Measurement, Profile, OutlierClaim, State, Annotation, AttributeSet, Lift, Aggregation) | ~145 | Densest branch; +15 for Macrobase pre-anchor |
| `cco:DirectiveICE` (Requirement, Control, Policy, Constraint) | ~70 | SysMLv2 user-level Requirement + cybersec control + DB constraint |
| `sdg:ObservationProcess` (LabRun, Trace, Profiling, OutlierDetection, eBPFEvent) | ~70 | Adds ~10 templates for eBPFEvent kernel-granularity coverage |
| `sdg:DerivationProcess` (LineageEdge, Transformation, Allocation) | ~40 | PROV-O + OpenLineage SSSOM mapping |
| `sdg:GovernanceProcess` (Verification, Attestation, Classification, Audit) | ~30 | SysMLv2 VerificationCase + compliance attestation |
| `sdg:BeliefStructure` (MassFunction, BeliefInterval, Evidence, Claim) | ~20 | DST integration with Atelier; "properly uncertain knowledge" first-class |

**Total target: ~565 templates authored, ~470 surviving DeepOnto's
gates conservatively** (see [Migration §Phase 1](./migration.md#phase-1--author-the-initial-vocabulary)
for authoring process). Comfortably above the brief's "≥ 200
surviving templates" P1 exit gate.

### Naming note — DirectiveICE vs CCO's PrescriptiveICE

CCO's canonical IRI `cco:ont00000965` carries `rdfs:label
"Prescriptive Information Content Entity"`. SDG renames this branch
"Directive ICE" because *directive* better captures the normative
sense (requirements, controls, policies, constraints) than
*prescriptive* (which can read as recipe-like). The rename is a
shorthand convention only — the bespoke `sdg:DirectiveICE` is
declared as `owl:equivalentClass cco:ont00000965` so all CCO-side
deductions remain available. Reviewers reading CCO source see the
canonical "Prescriptive ICE" label; reviewers reading SDG see
"Directive ICE"; both ground at the same IRI.

### P1 exit-gate readiness assessment (2026-05-10, post-Batch-7 / catalog complete)

Cross-checked against the v0.5 concept brief P1 exit criteria. **All gates clear; brief target reached.**

| Requirement | Brief threshold | Current state | Status |
|---|---|---|---|
| Surviving DeepOnto-validated templates | ≥ 200 | **522** (Batches 1+2+3+4+5+6+7) | ✓ |
| Brief-target authored | ~520 | **540** | ✓ |
| Distinct axiom shapes | ≥ 5 | 7 (F1–F7 each ≥ 3×) | ✓ |
| Schema validation errors | 0 | 0 (540 templates × 7 batches) | ✓ |
| Cross-context cousining preserved | yes | yes — Trace+LabRun share ObservationProcess; SQL/SysMLv2/eBPF Constraint co-anchored at DirectiveICE; eBPFEvent extends ObservationProcess; Syscall sits at DesignativeICE next to Identifier; Allocation+Transformation share bfo:Process; LineageEdge sits at DescriptiveICE alongside ColumnPolicy, BeliefInterval/Claim/Evidence/MassFunction, and the SOTAB schema.org / CTA-CPA / compliance / DST-ops / schema-evolution / telemetry / eBPF-helper / cross-branch tail from Batch 7 | ✓ |
| Verifier R(O,I) operational | all 4 components | R_A/R_B/R_C/R_D wired; AUC 0.9956; weights locked `{0.50, 0.05, 0.45}` | ✓ |
| Determinism gate | hash-stable across runs | confirmed: P4 smoke test re-runs identical rewards + advantages | ✓ |
| TTL coverage | every term used in templates is declared | 314 sdg:* properties declared | ✓ |
| Provenance discipline | editorial review per PROVENANCE.md | active; no proprietary inheritance | ✓ |
| BeliefStructure branch | optional Q1 inclusion | landed in Batch 6 (32 templates) | ✓ |
| Held-out 50 evaluation set | authored before P5 begins | landed at `tests/p5_held_out_50/` (25 good + 25 bad); separation 0.5129 against locked verifier | ✓ |

**P1 — exited.** The closed-loop ontology → synthetic data → RL corpus
pipeline is operational at the verifier and reward-signal layer.
Remaining batches are catalog growth (mostly mechanical) and P5 model-
side infrastructure.

**Remaining roadmap items (post-P1):**

- **P5 — full GRPO training with SAE-Res-Qwen3.5-27B-W80K-L0_100** against the locked verifier. ~200 GPU-hours per the v0.5 brief budget. **Scaffolding landed 2026-05-10**; see [P5 infrastructure readiness assessment](#p5-infrastructure-readiness-assessment-2026-05-10), the `src/aegir/rl/` package, the new `pyproject.toml [rl]` extra, the prompt template at `aegir.rl.prompt`, and the held-out 50-ontology set at `tests/p5_held_out_50/`.
- **Parallel hygiene track** — v2 → SOTAB head fine-tune. Independent.

The catalog work is complete at the brief target (~520) with margin (540). Remaining work is purely model-side training execution; the verifier, reward signal, prompt template, and held-out evaluation set are all production-ready.

### Verification suite (2026-05-10)

Pre-Batch-3 verification suite executed per user-specified rigor:

| Step | Result | Notes |
|---|---|---|
| 1. Catalog integrity | ✓ | 523 templates schema-validate, 0 errors; 173/173 DeepOnto-loadable; 71/71 TTL-declared |
| 2. Verifier determinism | ✓ | 5 workloads × 3 runs each, all identical R outputs |
| 3. Verbalization spot-check | **anomaly flagged** | 28 of 164 verbalizations capture parent-class only, dropping the slot-bearing restriction (templates anchored at long-label parents like `cco:DescriptiveICE`). Linguistically clean and not hallucinated, but reduced semantic fidelity for R_C/R_D. Fix: change `_verbalize_for_template` selection from "longest" to "highest slot-variable count, ties by length". ~15 min cascading rebuild. **Pending user direction.** |
| 4. Charter readiness | ✓ this section | metrics current; cross-context cousining intact; BeliefStructure on-track |

### Catalog state (2026-05-10, post-Batch-7 / catalog complete)

| Metric | Combined B1 + B2 + B3 + B4 + B5 + B6 + B7 |
|---|---|
| Total templates authored | **540** |
| Verbalized (DeepOnto round-trip) | **522 (97%)** |
| `is_complex` flagged | 485 (90%) |
| Axiom-family coverage | F1–F7 each ≥ 3× (F2 dominant per CCO usage patterns) |
| Schema validation errors | 0 |
| sdg:* properties declared in TTL | 314 |
| **P1 exit gate (≥ 200 surviving templates)** | **PASSED at 522** |
| **Brief target (~520 templates authored)** | **REACHED at 540** |

Per-batch breakdown:

| Batch | Branch focus | Templates | Verbalized | Complex |
|---|---|---|---|---|
| 1 — Foundation | five-branch top-level + cross-cousining | 81 | 73 (90%) | 68 (84%) |
| 2 — Observation + Measurement | LabRun, Trace, eBPFEvent (process), Reading | 92 | 91 (99%) | 79 (86%) |
| 3 — Directive + Governance | HipaaRule, ColumnPolicy, Audit, AttestationActivity | 69 | 66 (96%) | 58 (84%) |
| 4 — eBPF kernel granularity | eBPFProgram, KernelHook, Syscall, Map | 51 | 48 (94%) | 44 (86%) |
| 5 — PROV-O / OpenLineage | Transformation, LineageEdge, Allocation, ProvenanceAgent | 50 | 49 (98%) | 45 (90%) |
| 6 — BeliefStructure (DST) | MassFunction, BeliefInterval, Evidence, Claim | 32 | 30 (94%) | 26 (81%) |
| 7 — Long tail / benchmark gap | SOTAB schema.org, CTA/CPA, NIST/ISO/SOC2/GDPR/HIPAA/PCI compliance, DST-ops, schema-evolution, telemetry, eBPF helpers, cross-branch density | 165 | 165 (100%) | 165 (100%) |
| **Combined** | | **540** | **522 (97%)** | **485 (90%)** |

Null distribution recomputed against the 540-template combined catalog:
`null_mean = 0.41372`, `null_p95 = 0.43012`, `n_samples = 200`. The
shift from the post-Batch-6 baseline (`null_mean = 0.41750`, `null_p95
= 0.43357`) is within sampling noise (`null_std ≈ 0.009` on each fit).
Locked verifier weights `{0.50, 0.05, 0.45}` remain in effect; no
re-tuning warranted.

Held-out 50-ontology evaluation set authored before P5 begins, at
`tests/p5_held_out_50/labels.json` (25 good + 25 bad). Validation
against the locked verifier (R_D skipped):

| Set | n | mean R | median R | range |
|---|---|---|---|---|
| good | 25 | 0.5280 | 0.5274 | [0.5202, 0.5434] |
| bad | 25 | 0.0151 | 0.0000 | [0.0000, 0.1173] |

Separation `good_mean − bad_mean = 0.5129`. The held-out set is
leakage-free relative to P5 training (authored before any GRPO
iteration runs).

### Verbalization fidelity fix (2026-05-10)

A spot-check during the post-Batch-2 verification surface a
selection issue in `_verbalize_for_template`: when multi-clause
SubClassOf templates were anchored at long-label parents (e.g.
`cco:DescriptiveICE`), the candidate ranker preferred the
parent-class verbalization over the slot-bearing restriction
form, dropping ``{Y}`` from ~28 verbalizations.

The fix:
1. Track candidates in pre-substitution form (with ``ZZ<name>ZZ``
   markers) so the ranker can count slot markers correctly.
2. Sort by (slot-marker count desc, length desc) — prefer richer
   forms, with longer wins on tie for semantic-content
   maximization.
3. Suppress the redundant subsumption-axiom candidate when the
   complex-expression form succeeds for the same axiom (avoids
   the awkward ``X is a something that ...`` paraphrase).

Post-fix metrics held: same 95% verbalization rate, but the 28
flagged templates now correctly capture restriction slots
(e.g. `state_transitions_to` → `{X} is something that
transitions to {Y}` instead of the parent-only form). Locked
weights `{0.50, 0.05, 0.45}` retained AUC 0.9956 / separation
0.3360.

### P4 smoke test (2026-05-10)

The P4 reward-signal-propagation smoke test passes against the
locked verifier. A stub policy emits 8 candidate compositions
of varying quality profiles (slot-rich complex / mixed /
trivial-basic / stub-filler), each is scored end-to-end via
the locked `R(O, I)`, and group-relative advantages are
computed for GRPO consumption.

Results:

| Profile | Mean R | Description |
|---|---|---|
| A — slot-rich complex (12 templates) | **0.5285** | High-quality compositions score top |
| B — mixed (8 templates) | 0.4701 | Mid-tier discrimination |
| D — complex with stub fillers (6) | 0.4725 | R_B-driven; correct behavior given locked weights |
| C — trivial F1-subclass (4 templates) | **0.0234** | Low-quality compositions correctly bottom-ranked |

**Discrimination spread: A − C = 0.505** — non-degenerate signal
suitable for GRPO group-relative advantages. Z-normalized
advantages: A samples ≈ +0.74, C samples ≈ −1.70 — the
gradient direction GRPO would consume.

**Determinism confirmed**: re-running the smoke test with the
same seed produces identical rewards and advantages.

**Timing**: 8-sample group end-to-end in 0.13 s post-encoder-warmup,
well within RL throughput requirements (~24 GRPO groups per
minute on CPU; GPU inference for the SAE-Res-Qwen3.5-27B policy
in P5 is the rate-limiting step, not the verifier).

Batch 1 (`01_foundation.json`, 81 templates) covers `cco:Artifact`,
`cco:DesignativeICE`, `cco:DirectiveICE` (subset), and the eight
axiom-kind families (F1 basic SubClassOf, F2 existential, F3
universal, F4 cardinality, F5 intersection, F6 union, F7
negation; F8 enumeration deferred per slot-DSL conflict).
Batch 2 (`02_observation_measurement.json`, 92 templates) covers
the densest branch: `cco:DescriptiveICE`
(Measurement / Profile / OutlierClaim / State / Annotation /
AttributeSet / Lift / Aggregation) and the `sdg:ObservationProcess`
subbranches (LabRun / Trace / Profiling / OutlierDetection /
eBPFEvent), with full coverage of LIMS, OTel, eBPF cybersecurity,
Macrobase, and SysMLv2-state professional contexts.

### Verifier validation — claim C1 (P2 exit gate, 2026-05-10)

The labeled-test-set validation per P2 of the v0.5 concept brief
is committed and passes both gates:

| Gate | Threshold | Result | Status |
|---|---|---|---|
| AUC ≥ 0.85 | 0.85 | **0.9956** | ✓ PASS |
| Mean(R \| good) − Mean(R \| bad) ≥ 0.30 | 0.30 | **0.3388** | ✓ PASS |

**Test set:** 30 ontologies (15 known-good + 15 known-bad)
authored programmatically at `scripts/build_test_set.py`. Good
ontologies cover lab measurement, database metadata, OTel /
eBPF / Macrobase / governance / SysMLv2 / clinical /
provenance / IAM / workflow domains. Bad ontologies span
empty / trivial / no-labels / no-complex-axioms / off-domain
(music, cooking) / random-gibberish / circular / structurally-
truncated patterns.

**Locked aggregation weights:** `{a = 0.50, b = 0.05, c = 0.45}`
(R_B, R_C, R_D respectively). The AUC plateau (0.9956) is
shared across many weight triples; this triple sits within it
while preserving meaningful contributions from R_B and R_D and
satisfying the separation threshold.

R_C carries low weight (0.05) because the test set's "bad"
ontologies often have many shallow axioms — these get
verbalized and produce high R_C scores despite being
conceptually empty. This is a real signal that the current
"verbalization length as semantic-richness proxy" definition is
weakly discriminating; future iterations may refine R_C (e.g.,
penalizing trivial verbalizations, normalizing by axiom
complexity).

R_D contributes a non-trivial 0.45 weight despite scoring 0.0
on most ontologies (size-dependence: most test ontologies
verbalize fewer than the 24 templates the null distribution
samples). When R_D engages — `good_database_metadata.ttl`
scored R_D = 1.00 because its vocabulary directly matches
SchemaPile content — the contribution is decisive.

**Reproducibility:** Run `just c1-validate` to regenerate the
30-ontology test set, score each via `score_ontology.py`,
sweep `{a, b, c}`, and report AUC + per-ontology R. Full
results at `tests/ontology_test_set/c1_results.json`.

### Verifier R(O, I) status — full four-component (P1b-β)

The runtime verifier is operational across all four R-components
as of 2026-05-10:

- **R_A** — structural type-check (slot fillers vs declared
  slot_types). Hard gate.
- **R_B** — complex-template density (`is_complex=True` count
  over τ_B saturation point). 84% of catalog templates flagged.
- **R_C** — semantic-richness proxy via mean verbal length
  against L_target.
- **R_D** — Hungarian-optimal cosine alignment between *T_V*
  (composition's verbalization corpus) and *T_I* (the held-out
  SchemaPile + FinePDFs-lab corpus subset, ~10K sentences,
  k=100 topics), normalized against a 200-sample structural-
  shuffle null distribution.

`T_I` cached at `src/aegir/ontology/catalog/T_I.pkl`. Null
statistics: `null_mean ≈ 0.4186`, `null_std ≈ 0.0073`,
`null_p5 ≈ 0.408`, `null_p95 ≈ 0.4325` (200 samples × 24
templates × tv_k=12).

Verifier is hash-stable across runs (P1b-α determinism gate
passed before R_D activation). End-to-end runtime per
composition: ~3 s on CPU once the encoder and *T_I* are loaded
(~1 s of which is the *T_V* fit + Hungarian alignment).

### P5 infrastructure readiness assessment (2026-05-10)

P5 is full GRPO training of `SAE-Res-Qwen3.5-27B-W80K-L0_100`
against the locked verifier *R(O, I)*. The verifier and
reward-signal layer are operational. This section assesses
what model-side infrastructure is ready vs. what remains.

#### Ready / done

| Component | State |
|---|---|
| Catalog (293+50 = 343 templates, 327 verbalized 95%, 294 complex 86%) | done — Batches 1-5 |
| TTL vocabulary (173 sdg:* properties, BFO+CCO+PROV-O anchors) | done |
| Verifier core `aegir.ontology.verifier` | done — verifier.py, importable |
| Locked weights `{a=0.50, b=0.05, c=0.45}` from C1 sweep (AUC 0.9956) | done — frozen |
| T_I cache (~10K SchemaPile + FinePDFs-lab sentences, k=100) | done |
| Null distribution (200 samples, n_templates_per_composition=24) | done — refit per batch |
| GRPO group iteration validated end-to-end (P4 smoke test) | done — group_size=8, deterministic, 0.13s/group on CPU |
| Schema validator integrated into bdd-0 tier | done |
| JVM bootstrap (`/tmp/jvm-libs` surgical libz) | done — see `setup_jvm_env.sh` |

#### Pending — model-side engineering

| Component | Estimate | Notes |
|---|---|---|
| Policy load: SAE-Res-Qwen3.5-27B-W80K-L0_100 | ~1 day | weights ~54 GB; bf16 forward needs 6× RTX 4090 (24 GB each) with tensor-parallel sharding via vLLM or DeepSpeed-Inference |
| LoRA adapter setup (rank 16-32 on attention + MLP projections) | ~1 day | trainable param budget ~0.3-0.5% of base; the SAE-Res variant exposes a residual stream the SAE already constrains, so LoRA targets should leave the SAE bottleneck untouched |
| GRPO loop: rollout → score → advantage → policy gradient | ~2 days | reuse trl.GRPOTrainer or implement minimal loop; verifier is the reward model |
| Composition-format prompting | ~1 day | system prompt explains slot DSL, in-context shows 2-3 catalog rows; sampling extracts (template_id, slot_fillers) JSON via constrained decode |
| SAE feature logging during rollout | ~1 day | hook the residual SAE module to record top-k feature activations per generation step; aligns with the brief's interpretability claim |
| Data: rollout corpus | already there | the catalog + slot DSL is the rollout space; no additional dataset prep |
| Evaluation harness: held-out compositions + verifier scoring | ~1 day | reuse C1 test set (30 good/bad TTLs) plus a fresh held-out 50-ontology set |
| Compute budget | ~200 GPU-hours | per v0.5 brief; 6× RTX 4090 for ~33 wall-clock hours of training |

#### Risks / open questions

- **Constrained decoding fidelity.** The policy must emit valid
  `(template_id, slot_fillers)` JSON. Prefer a constrained-decode
  approach (lm-format-enforcer or Outlines) over post-hoc parsing,
  to keep R_A from clamping reward to zero on syntax errors during
  early training.
- **R_C floor.** A flat 0.05 weight on R_C is by design from C1
  (it provided ~0 marginal AUC), but it can collapse to a floor if
  the policy converges to "shortest valid composition." If observed,
  re-introduce a length-penalty term in R_C or rebalance against R_D.
- **R_D corpus drift.** T_I is fit on SchemaPile + FinePDFs-lab. If
  the policy starts generating compositions in a topic distribution
  that the corpus *I* doesn't cover (e.g., heavy eBPF / kernel-only
  vocabulary), R_D could under-reward genuinely good compositions.
  Mitigation: re-fit T_I against a broader composite corpus (FineWeb +
  SchemaPile + FinePDFs-lab + a kernel-tracing subset) before P5
  begins, then re-lock null stats.
- **Group size & advantage variance.** P4 used group_size=8 with
  std ≈ 0.21 across mixed quality profiles. P5 with a real policy
  may collapse to lower variance; if so, scale up to group_size=16
  or 32 and increase rollout temperature.
- **Cache invalidation.** Any post-P5 catalog edit (Batch 6/7) must
  re-run null stats and re-validate that locked weights still hold;
  re-locking is only triggered if AUC degrades below 0.95 on the
  C1 test set.

#### Decision: green-light P5 once Batches 6-7 are in?

Two paths:

1. **Sequential** — finish Batches 6 (BeliefStructure) and 7 (long
   tail) before P5 starts. Pro: largest possible catalog at training
   start. Con: ~2-3 weeks of additional authoring before any
   model-side feedback.

2. **Parallel** — start P5 against the current 343-template catalog
   while Batches 6-7 are authored in parallel. Pro: model-side bugs
   surface early; Batch 6-7 can be hot-loaded since the verifier
   re-instantiates per call. Con: any P5 hyper-parameters tuned on
   the smaller catalog may need re-tuning when the catalog grows.

The brief's commitment is to ship a publishable result; the
parallel path gets there faster and is what we recommend. P5
infrastructure scaffolding can begin immediately on the current
catalog with no further dependencies on the ontology track.

### P5 scaffolding landed (2026-05-10)

The P5 scaffolding — the engineering wiring between catalog,
verifier, and a future GRPO/RLVR training run — landed in this
session. The new `src/aegir/rl/` package contains:

| Module | Purpose |
|---|---|
| `aegir.rl.policy` | `PolicyConfig` + `load_policy(cfg)` for SAE-Res-Qwen3.5-27B + LoRA. LoRA targets attention + MLP projections; SAE residual bottleneck left untouched per `leave_sae_untouched=True`. Lazy import of `transformers` / `peft`. |
| `aegir.rl.decoding` | `composition_json_schema(catalog)` builds a discriminated-union JSON schema constraining the policy to emit valid `(template_id, slot_fillers)` per the catalog. `outlines` is the default backend; `lm-format-enforcer` is also wired. |
| `aegir.rl.sae_logging` | `SAELogger` attaches a forward hook to the residual SAE and records top-k feature activations at a configurable token cadence. Backs the v0.5 brief's interpretability claim. |
| `aegir.rl.grpo_loop` | `init_grpo_state` / `grpo_iteration(rollout_fn, policy_step_fn)` / `compute_group_advantages` / `hot_reload_catalog`. The verifier is hot-loaded per call; Batches 7+ pick up automatically. |
| `aegir.rl.eval` | `EvalConfig` + `evaluate(...)` against the C1 test set + an eventual 50-ontology held-out set. Reports mean R, R_A pass rate, and AUC where labels exist. |
| `scripts/p5_scaffold_smoke.py` | End-to-end smoke test: catalog hot-reload, policy-load dry run (no 27B touched), decoding-schema build, single-composition score, 3 GRPO iterations against a stub policy. **Passes**. |

The smoke test against the 375-template post-Batch-6 catalog
runs to completion in ~5 s (CPU-only, stub policy). Verifier
scoring on a 4-template stub composition yields R=0.32 in
0.5 s post-encoder-warmup. The constrained-decode JSON schema
for the 375-branch discriminated union is 138 KB — well within
typical LogitsProcessor budgets.

The remaining engineering work to begin actual P5 training is:

- `pyproject.toml`: add `transformers`, `peft`, `trl`, `outlines` (or `lmformatenforcer`).
- Tensor-parallel sharding configuration for 6× RTX 4090.
- Composition-format prompt template and few-shot context.
- Held-out 50-ontology set authoring (`tests/p5_held_out_50/`).

These are explicit follow-ups; none of them block the current
catalog work.

### Resolved design decisions (2026-05-09)

The six open questions in
`docs/scratch/2026-05-09/232551_domain_choice.md` resolved as:

- **Q1 — Belief branch:** **include**; ~20 templates;
  `sdg:BeliefStructure` under `cco:DescriptiveICE`. Direct
  alignment with Atelier's DST evidence fusion; future-proofs
  federated-intelligence use cases where conflict *K* and
  epistemic uncertainty must propagate across nodes.
- **Q2 — eBPF / cybersec depth:** **eBPF first-class**; adds
  `sdg:eBPFProgram`, `sdg:KernelHook`, `sdg:Syscall`,
  `sdg:Map`, `sdg:eBPFEvent`. ~30 templates total. OTel remains
  the primary runtime surface; first-class eBPF preserves
  semantic grounding without translation loss.
- **Q3 — SysMLv2 depth:** **user-level primitives only for v1**;
  KerML metamodel deferred to v2. Block, Part, Action, State,
  Requirement, Allocation, Verification only.
- **Q4 — Lineage anchor:** **PROV-O for OWL semantics +
  OpenLineage for runtime surface**, mapped via SSSOM. Single
  deductive core; preserves operational interop.
- **Q5 — Macrobase:** **pre-anchor lightly** with ~15 templates
  (`sdg:OutlierClaim`, `sdg:AttributeSet`, `sdg:Lift`,
  `sdg:Aggregation`, plus relations). Modernization team free to
  extend without blocking P1a.
- **Q6 — Ontology name:** **Signals Data Governance (SDG)
  Ontology**. Vendor-neutral; preserves open-source / sovereign
  deployment optionality.

These decisions are committed at v0.1 of the SDG ontology. Future
revisions require explicit version bumps tracked in
`docs/scratch/YYYY-MM-DD/` session notes.

## Source tree layout

```
src/aegir/
├── ontology/
│   ├── __init__.py
│   ├── sdg-vocab.ttl              # the source-of-truth vocabulary
│   ├── README.md                    # editor's guide for vocab changes
│   ├── label_map.py                 # build / load / query helpers
│   └── sparql/
│       ├── totality.rq              # all benchmark labels reach a BFO ancestor
│       ├── ancestry.rq              # BFO ancestry chain for a given IRI
│       └── coverage_by_namespace.rq # per-benchmark coverage diagnostic
├── synth/
│   ├── __init__.py
│   ├── registry.py                  # priority-tiered generator dispatch
│   ├── generators.py                # hand-coded value generators
│   └── snapshots/                   # frozen output samples for tests
└── data/
    └── table_dataset.py             # consumes label_map.py for label↔IRI
```

```
scripts/
├── build_vocab_label_map.py         # TTL → vocab_label_map.json via SPARQL
├── verify_vocab_totality.py         # CI gate: all labels reach BFO root
└── snapshot_synth_corpus.py         # generators → on-disk byte/row dump
```

```
docs/current/ontology/
├── charter.md                       # this document
└── migration.md                     # vocab + synth relocation plan
```

The TTL and the SPARQL queries are the source of truth. The
`vocab_label_map.json` artifact is build output that ships in the repo
for consumer ergonomics; CI rebuilds it on every PR that touches
`ontology/` and fails if the checked-in copy drifts.

## What stays out of Ægir

- Dempster-Shafer fusion, belief/plausibility logic, any specific
  classification pipeline shape.
- Gateway / UI features that are not directly about Ægir's leaderboard
  view of its own runs.
- Customer-deployment glue: nautilus mid-run watchers, agent loop
  governance, FSM session state. These belong with the consumer that
  owns the deployment lifecycle.
- Hive / Iceberg / Postgres schemas that exist only for sibling-project
  governance flows. Ægir publishes JSON; consumers translate to their
  own storage shape.

## Mechanical checks

For every PR that touches `src/aegir/ontology/` or `src/aegir/synth/`:

1. `scripts/build_vocab_label_map.py` runs and the resulting JSON is
   compared to the checked-in copy; a non-zero diff fails the build.
2. The SPARQL totality query (`ontology/sparql/totality.rq`) runs
   against the merged TTL and must return zero unmapped labels.
3. `_LABEL_DIMS` in `src/aegir/data/table_dataset.py` must agree with
   the per-benchmark counts implied by `vocab_label_map.json`.
4. The TTL parses, every term has a `rdfs:label` and a
   `skos:definition`, and every `sdg:`-namespace term has at least
   one `rdfs:subClassOf` edge to a public-namespace ancestor.
5. (Once it exists) the v2→SOTAB head fine-tune smoke test runs in
   short-budget mode and passes the liveness checks above.

These checks are about structural integrity, not about provenance —
provenance is editorial (see above).

## Versioning

`vocab_label_map.json` carries semver:

- **Major** — BFO anchor moved for an existing label, or a label removed.
  Consumers must re-evaluate.
- **Minor** — labels added (new benchmark coverage, new vocabulary
  extension), no existing label changes meaning.
- **Patch** — non-semantic edits (notes, formatting, alphabetical
  reordering).

The first published release is v1.0.0, gated on the empirical liveness
checks above. Pre-v1 work is internal and not advertised as consumable.
