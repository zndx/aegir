# Aegir's semantic engine: an authoritative reference

*Last updated 2026-05-12. Supersedes the 2026-05-10 lift extract.
The closed-loop ontology pipeline and verifier described in §§ 3–5
are operational; the warm-start procedure for the GRPO policy is
currently under empirical test (§ 6) and the procedural claims in
§ 6 are subject to revision when the in-flight run produces its
first held-out evaluation.*

## Abstract

Aegir's semantic engine couples a bespoke OWL ontology — the
**Signals Data Governance (SDG)** ontology, grounded in Basic
Formal Ontology 2020 [BFO; Arp et al. 2015] and the Common Core
Ontologies (CCO) — with a deterministic four-component verifier
*R(O, I)* over (ontology composition, target text corpus) pairs,
and a reinforcement-learning-with-verifiable-reward (RLVR) policy
trained against *R* via Group Relative Policy Optimization [GRPO;
Shao et al. 2024]. The verifier comprises a structural hard gate
*R_A* and three continuous components — *R_B* complex-class
density, *R_C* semantic-richness proxy, *R_D* corpus topic
alignment — with aggregation weights `{a, b, c} = {0.50, 0.05,
0.45}` locked from a 30-ontology hand-authored discrimination
sweep. On that sweep the locked aggregator achieves
**AUC 0.9956** with mean *R*-separation 0.336 between known-good
and known-bad ontologies. A leakage-free held-out evaluation set
of 50 scenarios (25 good, 25 bad), authored before any RL
training began, gives separation **0.5129** under the same
verifier. The verifier is deterministic, hash-stable, and has no
JVM or Java dependencies in its runtime hot path: DeepOnto
[He et al. 2023] is invoked only at catalog construction time.

This document describes (i) the SDG ontology and its procedural
catalog of 540 Manchester-syntax templates [§ 3]; (ii) the
verifier and its empirical validation [§ 4]; (iii) the
closed-loop pipeline that converts organic text into a
verifier-scored RL training corpus [§ 5, Figure 1];
(iv) the current empirical test of the GRPO policy warm-start,
including the recently observed zero-reward failure mode that
motivated a methodological intervention now under evaluation [§ 6];
(v) the current operational state and reproducibility
footing [§ 7]; and (vi) limitations and threats to validity [§ 8].

The contribution of the system is the combination — a
procedurally validated OWL catalog, a deterministic
structurally-and-semantically grounded verifier, and an RLVR
policy targeting that verifier — applied to a setting where
prior work has demonstrated each piece on adjacent artifacts but
not the full chain on an OWL artifact whose verifier acquires
description-logic deductive semantics.

## 1. Background

Aegir's semantic engine sits at the intersection of three lines
of work: ontology engineering for enterprise metadata, the
verbalization of structured knowledge for language-model
training, and reinforcement learning with verifiable rewards on
graph-structured outputs. The substantive engineering claim is
that a bespoke ontology grounded in upper-level formal
foundations (BFO 2020 + CCO) produces verifiable
*cross-context cousining* — a relation between concepts from
disparate operational domains that share an upper-class
ancestry. The methodological claim is that a deterministic
verifier over OWL composition–corpus pairs supplies dense,
hash-stable, semantically grounded reward signal sufficient for
a GRPO-trained language-model policy to discover high-quality
ontology compositions on a downstream target corpus.

Two empirical anchor points exist at the time of writing.
First, on a 30-ontology hand-authored test set (15 known-good,
15 known-bad designed to fail one or more components), the
verifier with locked weights yields AUC 0.9956 and mean
*R*-separation 0.336. Second, on a 50-scenario held-out set
authored before any policy-side RL work began, the same
verifier yields separation 0.5129. These two results establish
that *R* discriminates ontology quality at the verifier level.
The complementary question — whether a GRPO-trained policy can
produce compositions whose *R*-distribution exceeds prompt-
evolved and human-authored baselines on the held-out set — is
the experimental claim currently under test (§ 6).

The system was designed to be hash-stable end-to-end. Every
verifier output is reproducible from a known catalog version,
locked weights, and null-statistics snapshot; the verifier
itself has no JVM or Java dependencies in its hot path. This
determinism is load-bearing for the RL training loop (without
it, group-relative advantage estimation would be confounded by
verifier noise) and for the reproducibility of any downstream
evaluation against archived runs.

## 2. Related work

Each component recipe of the system has prior art on adjacent
artifacts. The combination — a graph-structured output with a
*semantically grounded* deterministic verifier targeting an LLM
policy under RLVR, plus the verbalization-corpus pretraining
application — is the gap addressed here.

**KELM / TEKGEN** [Agarwal et al. 2021]. The closest
verbalization-corpus recipe: verbalize a structured knowledge
source into natural-language sentences, integrate as a model
training corpus, measure downstream effect. KELM verbalizes
ABox triples from Wikidata into a retrieval corpus evaluated on
QA. The present system verbalizes TBox axioms from a bespoke
OWL ontology into a flat byte-level pretraining slice evaluated
on stratified held-out bits-per-byte. The structural content of
the verbalized text differs (taxonomic and logical class
expressions vs. instance-level facts); the integration mechanism
differs (pretrain mix vs. retrieval); and the evaluation
isolates ontology-specific contribution rather than generic QA
lift.

**OLLM** [Lo et al. 2024]. The closest end-to-end LLM ontology
generation approach. OLLM fine-tunes an LLM with a custom
regulariser that reduces overfitting on high-frequency concepts
and produces taxonomic backbones from scratch. The present
system trains via GRPO against a deterministic verifier that
scores OWL-reasoner consistency, axiom complexity, and topic
alignment with a target corpus; the policy emits full OWL
compositions with restrictions and equivalentClass
intersections rather than taxonomic backbones only.

**AutoGraph-R1** [Tsang et al. 2026]. The closest RL-trained
graph-emitting policy: a GRPO-trained LLM emits a knowledge
graph from text, with reward computed from the graph's
downstream RAG utility under an extrinsic LLM-judge. The
present policy emits OWL TBox compositions with class axioms
and uses an *intrinsic*, *semantically grounded* deterministic
verifier (DL-reasoner consistency check at catalog construction
+ programmatic structural property checks + topic-model
alignment). Sound-and-complete deductive reasoning is
unavailable to AutoGraph-R1's flat-triple output by
construction; OWL's class-axiom expressivity is what makes the
DL-reasoner verifier shape possible.

**K2V — Knowledge-to-Verification** [Yuan et al. 2026]. The
closest RLVR + KG-derived reward methodology. K2V builds a KG
from text and frames KG completion as a verifiable QA task to
derive dense rule-based rewards for LLM reasoning. K2V's policy
emits QA reasoning traces, not OWL; the verifier checks subtask
correctness, not ontology loadability / axiom density / topic
alignment. K2V establishes that the RLVR-with-KG-derived-reward
shape works; the present system applies that shape to OWL
generation directly, with the verifier targeting structural and
semantic properties of the artifact rather than QA accuracy on
downstream tasks.

**OntoTune** [Liu et al. 2025]. Iteratively refines an LLM
against an ontology-grounded objective via SFT, with the reward
implicit in does-LLM-already-know-this gating. The model emits
natural-language answers, not OWL. The present system trains
the LLM policy via GRPO with an explicit deterministic verifier
whose output is a continuous reward in [0, 1], and the policy
emits OWL ontology compositions whose well-formedness is
guaranteed by the catalog's typed-slot grammar.

**Zaitoun, Sagi, Peleg** [AAAI Symposium Series 2024]. The
closest OWL-specific verbalization-derived training data
recipe: LLM-assisted verbalization of OWL axioms creates
text → OWL supervised fine-tuning pairs. The present system
treats the verbalizations as a flat self-supervised byte-level
corpus mixed with general pretraining text, with no
instruction-pair framing.

**OnT — Language Models as Ontology Encoders** [Yang et al.
2024]. The closest TBox-axiom-aware embedding approach.
Compositional verbalization of OWL class expressions feeds a
pretrained Sentence Transformer, re-trained via hyperbolic-space
hierarchy / role / conjunction losses. OnT verbalizes TBox
axioms but uses them as auxiliary-objective embedding training;
the underlying LM is not pretrained on the verbalizations. The
present system treats the verbalizations as flat next-token
pretraining bytes mixed with a general-purpose corpus.

Secondary methodological precedents — symbolic RL on EL++
concepts [Lehmann & Haase 2012], JSON-Schema-validator RL
[SRL; RL-Struct], execution-validator RL [CodeRL,
Reasoning-SQL], retrieval-augmented graph reasoning [DRAGON] —
are listed in the project's lit-review document and are not
elaborated here.

A separate methodological precursor relevant to the warm-start
procedure (§ 6) is **Self-Distillation Fine-Tuning (SDFT)**
[Shenfeld et al. 2026]. SDFT uses a demonstration-conditioned
model as its own teacher to generate on-policy training signals
from a small set of demonstrations, with empirical evidence
that it preserves prior capabilities and reduces catastrophic
forgetting relative to vanilla off-policy SFT. SDFT is
methodologically adjacent to the rejection-sampling step
described in § 6.2 — both leverage the model as a partial
source of its own training signal — but the procedures differ
in critical respects: SDFT is on-policy; rejection-sampling
followed by standard SFT is off-policy. SDFT is a candidate
revision of the current warm-start procedure rather than a
method that has been evaluated to date.

## 3. The Signals Data Governance ontology

SDG is a bespoke OWL ontology designed to underwrite metadata
tagging across five operational contexts that enterprise
data-governance teams ordinarily treat as separate disciplines:
LIMS sample tagging, MBSE/SysMLv2 system design, database
metadata governance, kernel-trace observability, and PROV-O /
OpenLineage data lineage. SDG is grounded in BFO 2020 and CCO
and is organized into five top-level branches:

- **Artifact** (CCO) — material things, datasets, programs.
- **DesignativeICE** (CCO) — names, identifiers, designators.
- **DescriptiveICE** (CCO) — measurements, claims, lineage
  records.
- **DirectiveICE** (CCO; alias of `cco:ont00000965`
  "Prescriptive ICE") — requirements, controls, policies,
  constraints.
- **Process** (BFO 2020 `bfo:0000015`) — observation,
  derivation, governance activity.

The five branches are *cross-cousined*: every domain context
contributes templates to multiple branches, anchored at shared
upper-level parents. This is the load-bearing architectural
invariant of SDG — the ontology is forced to express
cross-context concepts as shared subclasses of common
BFO/CCO ancestors rather than as discipline-specific aliases
that happen to refer to the same real-world entity. § 3.2
lists concrete instances.

### 3.1 The procedural catalog

The SDG ontology is compiled into a procedural *catalog* — a
list of OWL axiom templates in Manchester syntax, each with a
typed slot DSL specifying the kinds of fillers (classes, object
properties, individuals) the template admits. The catalog
comprises 540 templates authored across seven batches over
April–May 2026:

| Batch | Branch focus | Templates | Verbalized |
|---|---|---|---|
| 1 — Foundation | five-branch top-level + cross-cousining | 81 | 73 (90%) |
| 2 — Observation + Measurement | LabRun, Trace, eBPFEvent (process), Reading | 92 | 91 (99%) |
| 3 — Directive + Governance | HipaaRule, ColumnPolicy, Audit, AttestationActivity | 69 | 66 (96%) |
| 4 — eBPF kernel granularity | eBPFProgram, KernelHook, Syscall, Map | 51 | 48 (94%) |
| 5 — PROV-O / OpenLineage | Transformation, LineageEdge, Allocation, ProvenanceAgent | 50 | 49 (98%) |
| 6 — BeliefStructure (DST) | MassFunction, BeliefInterval, Evidence, Claim | 32 | 30 (94%) |
| 7 — Long tail / benchmark gap | SOTAB schema.org, CTA/CPA, NIST/ISO/SOC2/GDPR/HIPAA/PCI compliance, DST-ops, schema-evolution, OpenTelemetry, eBPF helpers, cross-branch density | 165 | 165 (100%) |
| **Combined** | | **540** | **522 (97%)** |

A *composition* is a list of `(template_id, slot_fillers)`
pairs which, when materialized, forms a small ontology fragment.
The catalog is paired with 314 declared `sdg:*` object
properties in the SDG vocabulary TTL, each with both
`rdfs:label` and `skos:definition`; the TTL is closed in the
sense that every term referenced by any template is declared.
DeepOnto verbalization is run once per template at catalog
construction time, producing the `verbal_template` and
`mean_verbal_length` fields; 522 of 540 templates verbalize
cleanly (97% coverage), with the residual 18 flagged for
follow-up. 485 templates (90%) carry the `is_complex` flag,
DeepOnto-determined by running
`onto.get_asserted_complex_classes()` on the rendered template:
the long tail and Batch 7's cross-branch density push
complexity high.

### 3.2 Cross-context cousining — concrete instances

Cousining is verifiable directly from the catalog. Four
representative instances:

- `bfo:Process` (BFO 2020 `bfo:0000015`) is shared by
  `sdg:LabRun` (LIMS), `sdg:Trace` (database / MBSE),
  `sdg:eBPFEvent` (kernel observability), `sdg:Transformation`
  + `sdg:Allocation` (lineage), and `sdg:Audit` +
  `sdg:AttestationActivity` (governance). Any composition that
  pulls observation, lineage, and audit templates is grounded
  at the same BFO upper class.
- `cco:DescriptiveICE` is shared by `sdg:Reading` (LIMS
  measurement), `sdg:ColumnPolicy` (database governance),
  `sdg:LineageEdge` (PROV-O), and the DST primitives
  `sdg:Evidence` / `sdg:Claim` / `sdg:BeliefInterval` /
  `sdg:MassFunction`. The same `sdg:Evidence → sdg:Claim →
  sdg:BeliefInterval` triplet covers LIMS quality-tier
  evidence, audit findings supporting or refuting compliance
  claims, lineage-edge plausibility records derived from
  PROV-O traces, and column-tag claims at calibrated
  confidence levels.
- `cco:DirectiveICE` (alias of `cco:ont00000965`) is shared by
  `sdg:HipaaRule`, `sdg:ColumnPolicy`, `sdg:SqlConstraint`,
  `sdg:SysMLv2Constraint`, and the eBPF security-policy
  templates from Batch 4. SQL `CHECK` clauses and SysMLv2
  `constraint` blocks land at the same upper class as a HIPAA
  Privacy Rule provision.
- `cco:DesignativeICE` is shared by `sdg:Identifier` (database
  primary keys), `sdg:Syscall` (kernel-API surface), and the
  Batch 7 `schema.org` alignment properties (`Person.email`,
  `Place.geoCoordinates`, etc.). Syscalls and database
  identifiers are treated as cousins, not as separate
  disciplines.

### 3.3 Atelier ↔ Aegir state-fusion via DST

The Batch 6 BeliefStructure primitives — `sdg:MassFunction`,
`sdg:BeliefInterval`, `sdg:Evidence`, and `sdg:Claim` —
provide shared structural vocabulary for Dempster-Shafer
evidence-fusion pipelines. The Aegir agent-swarm state-fusion
layer consumes belief structures emitted by **Atelier**, a
sibling project providing DST-based evidence fusion for
enterprise data-classification pipelines, using these property
names directly and without translation. The cousining is at the
explicit-uncertainty layer: the same `sdg:Evidence` →
`sdg:Claim` → `sdg:BeliefInterval` triplet covers LIMS
quality-tier evidence, audit findings, lineage-edge
plausibility, and column-tag claims at calibrated confidence
levels.

## 4. The four-component verifier R(O, I)

### 4.1 Formal definition

Let *O* = compose(*C*, σ) denote an OWL ontology composition
produced from the procedural catalog *C* under slot-fill σ.
Let *I* denote a fixed input text corpus. The verifier
*R*: OntologyComposition × Corpus → [0, 1] is defined as

> *R*(*O*, *I*) = *R_A*(*O*) · ( *a* · *R_B*(*O*) + *b* · *R_C*(*O*) + *c* · *R_D*(*O*, *I*) )

where (*a*, *b*, *c*) are non-negative aggregation weights
summing to 1, *R_A* is a hard structural gate, and *R_B*, *R_C*,
*R_D* are continuous components in [0, 1].

**R_A — Well-formedness (hard gate).** *R_A*(*O*) = 1 iff every
(template, σ_i) pair in *O* type-checks against the catalog's
typed slot DSL; *R_A*(*O*) = 0 otherwise. Because every catalog
template was DeepOnto-validated offline, *R_A* = 1 implies the
materialized OWL also passes DeepOnto loadability —
well-formedness at runtime is equivalent to slot-fill
type-checking against *C*. *R_A* is a hard gate by construction
(*R* = 0 whenever *R_A* = 0).

**R_B — Complex-class density.**

> *R_B*(*O*) = min( complex_count(*O*) / τ_*B*, 1 )

where complex_count(*O*) is the number of templates in *O*
whose `is_complex` flag is true in *C*, and τ_*B* is the 95th
percentile complex-count of a structural-shuffle null
distribution computed once per catalog.

**R_C — Semantic richness, proxied via cached verbalization
length.**

> *R_C*(*O*) = clip( mean_verbal_length(*O*) / L_target, 0, 1 )

where mean_verbal_length(*O*) is the mean character length of
the pre-cached verbalizations of templates in *O*, and L_target
is calibrated so that the median template in *C* produces
*R_C* ≈ 0.5. By construction, every template in *C* verbalizes
cleanly (offline gate), so the binary "does it verbalize"
question is uninformative at runtime; *R_C* is repurposed as a
coarse continuous semantic-richness proxy.

**R_D — Topic alignment with corpus *I*.** Let *V(O)* denote
the verbalization corpus of *O* (concatenated cached
verbalizations for the templates in *O*); let *T_I* denote a
fixed BERTopic [Grootendorst 2022] topic model fit to a held-out
subset of *I* (≈ 10K sentences drawn from SchemaPile and the
FinePDFs-lab subset, encoded with
`sentence-transformers/all-MiniLM-L6-v2` [Reimers & Gurevych
2019], k = 100 topics); and let *T_V* denote the topic
distribution induced by mapping *V(O)* through the same encoder.
*R_D* is the Hungarian-optimal cosine alignment between *T_V*
and *T_I*, normalized against a 200-sample structural-shuffle
null distribution computed once per catalog (`null_mean`
0.41372, `null_p95` 0.43012, n = 200 against the 540-template
combined catalog).

The aggregation weights `{a, b, c} = {0.50, 0.05, 0.45}` were
selected as described in § 4.2.

### 4.2 Discrimination on the C1 test set

The aggregation weights were locked from a sweep over the
unit simplex against a 30-ontology hand-authored test set:
15 known-good ontologies designed to satisfy the verifier's
intent (well-typed, mixed-complexity, on-corpus topic
alignment) and 15 known-bad ontologies designed to fail one or
more components (single-template repetition, anti-aligned
topic distribution, ill-typed slots, structurally minimal). The
selected weights `{0.50, 0.05, 0.45}` yield

- **AUC = 0.9956** for discriminating known-good from
  known-bad,
- **mean *R*-separation = 0.336** between the two groups.

The weight sweep is fully reproducible from the committed test
set and verifier implementation. The choice to give *R_C* the
smallest weight reflects its construction-time saturation
(every template verbalizes, so verbalization length is the
weakest discriminator of composition quality); *R_D* and *R_B*
are roughly co-dominant.

### 4.3 Held-out 50 evaluation

A separate evaluation set of 50 scenarios (25 good, 25 bad)
was authored before any policy-side RL infrastructure was
constructed and is therefore leakage-free with respect to any
subsequent policy trained against *R*. Under the locked
verifier this set produces an *R*-separation of 0.5129 —
larger than the C1 sweep's 0.336, consistent with the held-out
set having been authored with sharper distinctions in mind
once *R*'s mechanics were better understood. The held-out 50
is the primary evaluation surface for the GRPO policy (§ 6).

### 4.4 Determinism and hot-path properties

The runtime verifier is deterministic and hash-stable: identical
inputs produce bit-identical *R* values across re-runs, with
stability gated by an automated smoke test. End-to-end verifier
scoring on an 8-sample GRPO group takes approximately 0.13 s on
CPU once *T_I* and the encoder are loaded, of which 0.02 s is
per-sample *R* computation. The runtime verifier requires no
JVM and no Java dependencies; DeepOnto is invoked only by the
catalog-build script and writes `verbal_template` and
`mean_verbal_length` into the JSON catalog. With a 9B-parameter
policy on a single 24 GB GPU, the policy's forward and sampling
passes are the rate-limiting steps of the RL training loop, not
the verifier.

## 5. The closed-loop synthetic-data pipeline

The pipeline (Figure 1) converts organic input corpora into a
verifier-scored synthetic training corpus through a BFO/CCO-
grounded intermediate representation. Input corpora are first
mapped onto SDG catalog concepts; concepts feed procedural
verbalization via DeepOnto's recursive verbaliser; verbalizations
seed LLM generative text and synthetic relational tables that
together constitute the output corpora. Topic preservation —
BERTopic alignment between input and output corpora — supplies
the *R_D* component of the verifier and feeds back into concept
selection as a side-loop signal. The output corpora become the
held-out RL corpus for the H-Net + RWKV-7 byte-level
pretraining slice [Paper 2; deferred to that paper's scope].

```d2
direction: down

input: Input Corpora\nFineFineWeb-edu, GitTables, SOTAB,\norganic PDFs {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"
}

concepts: Concepts\nBFO/CCO-grounded intermediate representation\nSDG catalog (540 templates) {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"
}

topic: Topic Preservation\nBERTopic alignment\nR_D verifier component {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"
}

verbalize: Procedural Ontology Verbalization\nDeepOnto recursive verbaliser {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"
}

output: Output Corpora\nverbalized ontology compositions\n+ synthetic relational tables {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
}

input -> concepts: ontology extraction
concepts -> verbalize: BFO/CCO IR
verbalize -> output: LLM generative text
input -> topic: T_I held-out corpus
output -> topic: T_V verbalizations
topic -> concepts: R_D signal {
  style.stroke-dash: 3
}
output -> input: ground-truth RL corpus\nfor H-Net + RWKV-7 pretraining {
  style.stroke-dash: 5
  style.stroke: "#9e9e9e"
}
```

**Figure 1 — The Aegir closed-loop synthetic-data pipeline.**
Raw input corpora are mapped to SDG catalog concepts; concepts
feed procedural verbalization (DeepOnto) and LLM-driven
generative text; topic preservation between input and output
supplies the *R_D* verifier component. The output corpora
become a verifier-grounded RL training corpus. The dashed gray
arrow indicates the downstream byte-level pretraining slice
addressed in the application paper (out of scope here).

### 5.1 Reward-signal flow

The reward signal flows from procedurally-cached catalog
verbalizations into GRPO advantage estimates:

```
composition (template_id + slot_fillers per entry)
   ↓  verifier — R = R_A · (0.50·R_B + 0.05·R_C + 0.45·R_D)
   ↓  GRPO group advantages — group-relative z-normalized
   ↓  policy gradient — LoRA adapter on attention + MLP
       projections; residual-stream SAE adapter held untouched
```

The residual-stream SAE adapter — a sparse-autoencoder
[Cunningham et al. 2024; Bricken et al. 2023] decomposition of
the policy's residual stream into an interpretable feature
dictionary — is held fixed during policy updates so that
interpretability claims about the trained policy's
representations can be made against a stable feature
dictionary.

## 6. The current empirical test (under revision)

This section describes work-in-progress. The procedural choices
described here are subject to revision pending the in-flight
run and any subsequent comparison against the alternatives in
§ 6.2.

### 6.1 Diagnostic — the 2026-05-11 zero-reward run

A reference run of the GRPO policy was conducted on 2026-05-11
using Qwen3.5-9B-Base with a LoRA adapter on attention and MLP
projections, the corresponding `SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`
residual-stream adapter, and a constrained-decode JSON Schema
covering all 540 catalog branches via lm-format-enforcer
[Gani 2024]. The run produced reward 0.0 with reward standard
deviation 0.0 throughout 1690 steps and was halted. Post-hoc
diagnosis identified two compounding causes: (i) the
constrained-decode schema was constructed but not wired into
the generation path used by the TRL [von Werra et al. 2020]
GRPOTrainer (`GRPOConfig.generation_kwargs` does not accept a
`prefix_allowed_tokens_fn`, so the schema was unreachable from
TRL's internal `model.generate(..., generation_config=...)`
call), and (ii) the per-token state traversal of the
540-branch discriminated-union schema, at the policy's 248K
vocabulary, dominated CPU and slowed generation 5–15×. Both
issues have been addressed: the constraint is now wired through
a monkey-patch on the policy's `model.generate` that survives
the TRL/`accelerator.unwrap_model` indirection, and a relaxed
JSON Schema enforcing only the structural array-of-objects
shape (`template_id`, `slot_fillers`) has been introduced as
an alternative. The relaxed schema is sound under *R_A*:
because *R_A* is a hard gate that returns 0 whenever a
composition contains a `template_id` not in *C*, allowing the
policy to emit hallucinated identifiers does not perturb the
reward landscape — it costs sample budget but preserves
correctness. A diagnostic batch under the relaxed schema
produced *R* = 0.318 on 3 of 4 sampled compositions, above the
SFT inclusion threshold of 0.3.

### 6.2 Three candidate warm-start procedures

The diagnostic above is necessary but not sufficient to produce
non-zero reward variance for GRPO. A separate observation is
that Qwen3.5-9B-Base emits zero-reward random JSON when asked
cold for a verifier-passing composition: the Base model has not
been instruction-tuned on a JSON-emission target and lacks a
prior anchoring its generations to the catalog's typed slot
DSL. Three candidate warm-start procedures are on the table:

- **Option A — Rejection-sampling SFT (in-flight).** Sample
  compositions from the Base model under constrained decoding
  across rotated few-shot prompt variations; score each with
  the verifier; retain samples above *R* ≥ 0.3; supervised-
  fine-tune the Base model on the retained corpus; begin GRPO
  from the SFT checkpoint. The few-shot examples are rotated
  per prompt variation via a seeded shuffle to prevent the
  Base model from pattern-completing the same three templates
  across the SFT corpus.
- **Option B — Use the Instruct variant.** Qwen3.5-9B-Instruct
  has been instruction-tuned on a general corpus and a
  corresponding Instruct-paired residual-stream SAE adapter is
  publicly available. Instruct produces structured JSON
  outputs from cold prompts, eliminating the warm-start need
  that motivates Option A. Option B was *not* considered when
  Option A was committed; the SAE compatibility argument
  ("Instruct would invalidate the Base SAE adapter") that
  motivated Option A is not load-bearing in the presence of
  an Instruct-paired SAE adapter.
- **Option C — Self-Distillation Fine-Tuning (SDFT).**
  Shenfeld et al. (2026) propose SDFT as an on-policy
  alternative to vanilla SFT for continual learning from
  demonstrations, with the demonstration-conditioned model
  acting as its own teacher. SDFT is reported to outperform
  off-policy SFT on skill acquisition while substantially
  reducing catastrophic forgetting. The rejection-sampling
  step of Option A is partially in SDFT territory (the
  demonstrations come from the model itself, conditioned on
  the catalog schema and few-shot exemplars), but the
  subsequent fitting step in Option A is off-policy. An
  on-policy revision of Option A along SDFT lines would
  better preserve the Base model's residual-stream
  distribution — the property the SAE adapter depends on for
  interpretability — than vanilla SFT.

### 6.3 What the in-flight run does and does not settle

The in-flight Option A run will settle whether the
rejection-sampling SFT warm-start is sufficient to produce
non-zero reward variance for GRPO under the current verifier,
and whether a measurable lift on the held-out 50 evaluation
follows. It will *not* settle whether Option A is the best
warm-start choice; that comparison requires running at least
one of (Option B, Option C) under matched conditions. A
direct comparison against Option B is the next-priority
experimental step once Option A produces an initial held-out
evaluation, and is independently motivated by the fact that
Option B avoids the bootstrap step entirely. The
methodological lift from Option C is conditional on the
underlying SFT step in Option A proving to be a meaningful
contributor to performance, which the in-flight run will help
establish.

The reward signature that would falsify Option A is the same
signature that flagged the 2026-05-11 run: persistent
`reward_mean ≈ 0`, `reward_std ≈ 0`, and
`frac_reward_zero_std ≈ 1` over more than a few hundred GRPO
steps after the constrained-decode wiring is in place. If that
signature recurs after the SFT warm-start, the bottleneck is
not the warm-start (which is what Option A addresses) but
something upstream — most plausibly the prompt template's
ability to communicate the schema to the Base model, or the
choice of policy capacity for this task.

## 7. Repository state and reproducibility

### 7.1 Locked artifacts

The following artifacts are versioned in the repository and
together constitute the locked surface against which any
*R*-value can be reproduced:

| Artifact | Identity | Hash basis |
|---|---|---|
| Procedural catalog | 540 templates, 7 batches | `catalog_version` field + content hash of `combined.json` |
| `sdg:*` vocabulary | 314 properties with `rdfs:label` + `skos:definition` | TTL file content hash |
| Verifier weights | `{a, b, c} = {0.50, 0.05, 0.45}` | 16-character `locked_weights_hash` |
| Null distribution | `null_mean` 0.41372, `null_p95` 0.43012, n = 200 | content hash of `null_stats.json` |
| C1 test set | 15 good + 15 bad | committed before the locked weight selection |
| Held-out 50 | 25 good + 25 bad | committed before any policy-side RL work began |

Every RL run records the four locked-artifact hashes in its
run metadata sidecar; a `verify_resume_metadata` policy guard
refuses to resume any run whose locked artifacts have drifted
from the originating run's hashes.

### 7.2 Verification gates

Four automated gates verify the operational state of the
ontology + verifier infrastructure:

1. **Catalog schema check.** Validates the seven catalog files
   (1624 templates total across all sources) against the
   typed-slot schema with zero errors.
2. **C1 regeneration.** Regenerates the C1 sweep AUC 0.9956
   against the committed locked weights from the committed
   test set.
3. **Verifier determinism.** A re-run of the verifier on a
   fixed input produces bit-identical *R* values; group
   advantages have non-degenerate standard deviation; a
   slot-rich complex profile outscores a trivial basic profile
   by ≥ 0.5 *R*-units.
4. **End-to-end scaffold.** A 3-iteration GRPO loop runs
   against a stub policy and exercises catalog hot-reload,
   policy-load dry run, decoding-schema build, and verifier
   scoring.

The four gates together verify the closed-loop ontology →
synthetic data → RL corpus pipeline is operational end-to-end
at the verifier and reward-signal layer.

### 7.3 Code and data availability

The Aegir repository contains: the SDG ontology TTL and
catalog JSON, the verifier implementation, the C1 test set,
the held-out 50 evaluation set, the GRPO loop scaffolding,
the rejection-sampling SFT pipeline, the constrained-decode
wiring, the SAE adapter attachment and feature-spill
infrastructure, and the four verification gates above.
External datasets cited in this document — SchemaPile,
FinePDFs-lab, SOTAB v2, GitTables, FineWeb-Edu — are obtained
via documented download scripts and have stable public
distributions. Run metadata (catalog version, locked weight
hash, null-stats hash, run id) is preserved alongside every
checkpoint to make any subsequent *R*-value reproducible.

## 8. Limitations and threats to validity

**Bounded expressivity.** The policy can only compose what the
catalog *C* contains. Templates outside *C* are unreachable;
genuine novel-axiom emission is not part of the present
system's claim. *R_C*'s saturation against a calibrated
*L_target* further means that *R_C* is a coarse continuous
signal rather than a sharp quality discriminator.

**Single-corpus topic alignment.** *R_D* is computed against
a fixed *T_I* derived from SchemaPile + FinePDFs-lab. The
choice of *I* is a load-bearing decision; a different choice
of *I* would in general produce a different policy. Whether
the trained policy's outputs are useful on a corpus other than
the one shaping *R_D* is an open question.

**C1 sweep n = 30.** The C1 test set is small (15 good,
15 bad). The AUC of 0.9956 should be read against this size;
it is consistent with strong separation but leaves room for
sampling variation. The held-out 50 widens the surface but
does not eliminate the small-n caveat. A larger external
validation set is a natural next step but is not load-bearing
for the current paper-1 claim.

**Warm-start choice is unsettled.** As described in § 6, the
choice among Options A, B, and C is not yet empirically
grounded. The current run tests Option A; Options B and C are
defensible alternatives that have not been evaluated. Any
claim built on the *specific* warm-start procedure (e.g., "SFT
bootstrap is the right warm-start path") is unsupported until
at least one direct comparison is available.

**Verifier signal vs. downstream utility.** The C1 and
held-out separations establish that *R* discriminates known-
good from known-bad ontologies on the test surfaces. They do
*not* establish that improving *R* improves downstream
pretraining utility on the H-Net + RWKV-7 byte-level
pretraining slice. That claim is the subject of the
application paper (out of scope here) and is conditional on
the paper-1 RLVR claim landing first.

**Hardware envelope choice.** The current operational run uses
a 9B-parameter policy on a single 24 GB GPU, downscaled from
the 27B target in earlier planning. The downscale is a
practical accommodation, not a methodological choice; whether
verifier-passing compositions at 9B generalize at higher
capacity is an open question to be answered once the 9B run
produces an initial result.

## References

**Verifiable reward in language models.**

- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the limits of
  mathematical reasoning in open language models.* (Source of
  the GRPO algorithm.)
- DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning
  capability in LLMs via reinforcement learning.*
- Lambert, N., et al. (2024). *Tülu 3: Pushing frontiers in
  open language model post-training.*

**Self-distillation and continual learning.**

- Shenfeld, I., Damani, M., Hübotter, J., Agrawal, P. (2026).
  *Self-Distillation Enables Continual Learning.* arXiv:2601.19897.

**Ontology engineering with deep learning.**

- He, Y., Chen, J., Antonyrajah, D., Horrocks, I. (2023).
  *DeepOnto: A Python package for ontology engineering with
  deep learning.*

**RL on graph-structured / structured outputs.**

- Tsang, et al. (2026). *AutoGraph-R1.* arXiv:2510.15339, ICLR
  2026 submission.
- Yuan, et al. (2026). *K2V — Knowledge-to-Verification.* ICLR
  2026 submission.

**Adjacent LLM ontology / verbalization work.**

- Lo, A., Jiang, A. Q., Li, W., Jamnik, M. (2024). *OLLM:
  Generating ontologies from texts.* NeurIPS 2024.
- Liu, et al. (2025). *OntoTune.* WWW 2025.
- Zaitoun, A., Sagi, T., Peleg, M. (2024). LLM-assisted
  verbalization of OWL axioms. AAAI Symposium Series 2024.
- Yang, Z., Chen, J., He, Y., Gao, F., Horrocks, I. (2024).
  *OnT — Language Models as Ontology Encoders.*
  arXiv:2507.14334.
- Agarwal, O., Ge, H., Shakeri, S., Aharoni, R. (2021).
  *Knowledge graph based synthetic corpus generation for
  knowledge-enhanced language model pre-training (KELM /
  TEKGEN).* NAACL 2021.

**Sparse-autoencoder interpretability.**

- Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L.
  (2024). *Sparse Autoencoders Find Highly Interpretable
  Features in Language Models.* ICLR 2024.
- Bricken, T., et al. (2023). *Towards monosemanticity:
  Decomposing language models with dictionary learning.*
  Anthropic.

**Topic modeling.**

- Grootendorst, M. (2022). *BERTopic: Neural topic modeling
  with a class-based TF-IDF procedure.*
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence
  embeddings using Siamese BERT-networks.* EMNLP 2019.
- Blei, D., Ng, A., Jordan, M. (2003). *Latent Dirichlet
  allocation.* (Legacy reference; not the headline method.)

**Constrained decoding and structured generation.**

- Willard, B. T., Louf, R. (2023). *Efficient Guided
  Generation for Large Language Models* (outlines).
- Gani, N. (2024). *lm-format-enforcer.* (TokenEnforcer-based
  JSON Schema enforcement via `prefix_allowed_tokens_fn`.)
- von Werra, L., et al. (2020–present). *TRL: Transformer
  Reinforcement Learning.* HuggingFace.

**Data sources.**

- Penedo, G., et al. (2024). *FineWeb: Decanting the web for
  the finest text data at scale.*
- Albalak, A., et al. (2023). *A survey on data selection for
  language models.*

**Ontological foundations.**

- Arp, R., Smith, B., Spear, A. (2015). *Building Ontologies
  with Basic Formal Ontology.* MIT Press.
- Common Core Ontologies (CCO). github.com/CommonCoreOntology/CommonCoreOntologies.

**Internal references.**

- Aegir [Charter](./charter.md) — outward contract and
  provenance discipline.
- Aegir [Migration](./migration.md) — vocabulary authorship
  history.
- Aegir [Concept brief — RLVR for ontology generation
  (v0.5)](./concept_brief.md) — research design for the
  paper-1 / paper-2 split this document operationalizes.
- Aegir [Training Regime § 10](../training_regime.md#10-v2-mixed-corpus-pretrain-2026-04-27)
  — v2 byte-level pretrain baseline against which any
  application-paper lift will be measured.

---

*Revision history.* This document supersedes "Production state
of Ægir's semantic engine — 2026-05-10," which is preserved as
a historical snapshot at
`docs/scratch/2026-05-12/033725_production_state_2026-05-10_snapshot.md`.
The 2026-05-10 document was a lift-ready extract for an
external book draft; the present document is the
authoritative reference for external/advisory audiences and is
the canonical statement of the system's operational state and
the empirical claims it currently supports. A revision will
follow the in-flight Option A GRPO run's first held-out
evaluation.
