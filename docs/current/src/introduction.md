# Introduction

Aegir is a hierarchical byte-level sequence model for semantic
annotation of relational data. Given one or more tables, it predicts
the semantic type of each column (Column Type Annotation, **CTA**),
the relationships between columns (Column Property Annotation,
**CPA**), and the cross-table groupings that constitute coherent
real-world *data elements* — for example, a `PaymentCard` data
element spanning `card_number`, `expiry`, and `cardholder` columns
across billing, transaction, and customer tables. The model is paired
with the **Signals Data Governance (SDG) ontology** — a BFO 2020 /
CCO-grounded, HermiT-validated OWL artifact whose classes *are* the
CTA/CPA annotation vocabulary; together they constitute a closed loop
between the model and the structured-knowledge representation it
learns from.

## Two coupled research outputs

The project produces two outputs that are cited together:

1. **A hierarchical byte-level sequence model.** All-RWKV-7
   time-mixing with H-Net dynamic chunking, trained byte-level on a
   mixed corpus and fine-tuned for the column-annotation tasks above.
   The architecture is described in [Architecture](./architecture.md);
   the operational pretraining work is described in
   [Pretraining](./pretraining.md) and
   [Training Regime](./training_regime.md).

2. **An ontology-grounded synthetic corpus and the SDG ontology that
   generates it.** A BFO/CCO-grounded domain ontology, content-derived
   from FinePDFs and realized to a HermiT-validated OWL artifact,
   drives LLM generation of deterministically-verifiable,
   attribution-clean textbook chapters and a relational DDL spine. The
   corpus is byte-level pretraining data *and* an independent
   publishable deliverable (the `corpora/` submodule). The ontology,
   its quality gates, and the disposal membranes that enforce them are
   documented in the [SDG ontology chapter](./ontology.md) and the
   [Ontology Authors Guide](./ontology/authors_guide.md).

The two outputs share substrate — the SDG ontology, the family
catalog of Manchester-syntax templates, the verbalization pipeline —
and are coupled downstream: the ontology produces verbalized,
verified chapters that feed back into the byte-level pretraining
corpus. The ontology is treated as a primary research output in its
own right, not as plumbing; its rigor program is described below.

### The ontology rigor program

The SDG ontology is governed by **propose / dispose**: an agent
proposes axioms and a stack of deterministic membranes disposes —
admitting only what is well-formed (a Manchester/OWLAPI parse
membrane), logically consistent under the reasoner (HermiT, with CCO
imported as a reasoning authority so grounding is validated against
CCO's disjointness axioms), and ontologically clean (an OntoClean
meta-property membrane). The two strongest membranes are
*un-fakeable*: you cannot talk past a contradiction or an
anti-rigidity violation. Quality is measured by a metric suite —
IOF-derived rigor dimensions (`definitional_completeness`,
`bfo_grounded`, `realizable_machinery`, `def_annotation_coverage`),
field-standard OntoQA/OQuaRE structural metrics, and OntoClean
taxonomic-correctness proxies — and a formal **OQuaRE publish gate**
(six SQuaRE characteristics; floors of `oquare_aggregate ≥ 3.5` and
`FunctionalAdequacy ≥ 3.0` plus HermiT consistency), wired hard into
the publish path so a regression cannot ship. The
[Authors Guide](./ontology/authors_guide.md) is the canonical
reference for every metric, band, and threshold.

There is also an in-flight **RLVR research sub-track** — a
language-model policy trained with Group Relative Policy Optimization
(GRPO) against a deterministic four-component verifier *R(O, I)* over
OWL compositions — documented in the
[concept brief](./ontology/concept_brief.md) and the
[RLVR chapter](./ontology/rlvr.md). It shares the catalog and the
verbalization pipeline with the rest of the ontology track but is
distinct from the production realization-and-gate path above; its
status is tracked in `EVIDENCE.md`.

## Problem setting

Enterprise data warehouses contain thousands of tables with columns
whose meaning is often opaque: generic names (`col0`, `field_42`),
inconsistent conventions across teams, no machine-readable metadata.
Understanding what each column represents — and which columns across
different tables refer to the same real-world concept — is foundational
to data governance, privacy compliance, and integration.

Two families of prior approaches exist. **Pattern and heuristic
methods** identify column types through regex detectors, name
matching, embedding similarity, and gradient-boosted classifiers on
hand-engineered features. They work well for structurally distinct
types but struggle with *confusable pairs* — columns whose value
distributions are nearly identical but whose semantic types differ
(advertising IDs versus GUIDs, bank account numbers versus payment
card numbers). They also require manual enumeration of data-element
patterns and do not generalize to novel relationship types. **Learned
sequence models** — DODUO, RECA, REVEAL — treat the table as a token
sequence and classify columns via fine-tuned transformers. REVEAL's
central insight is that *context-column selection matters*: choosing
the right neighboring columns (via MMR diversity sampling) materially
improves annotation accuracy. These models operate on single tables
in isolation and use fixed subword tokenizers that fragment tabular
data unpredictably.

Aegir bridges the two families. It is trained byte-level — no fixed
tokenizer — and is designed to be deployed *in situ* alongside
evidence-based classification pipelines: consuming the same serialized
table representations as the surrounding stack, but learning
cross-column and cross-table relationships end-to-end rather than
relying on enumerated patterns. Standard column-annotation benchmarks
remain reference points — **SOTAB** (Schema.org types over web
tables), **GitTables** (large-scale column type detection across 1M+
CSV tables from GitHub), and **WikiTables** (Wikipedia HTML tables) —
but the project's own thesis is that the ontology is load-bearing for
*property*-prediction (CPA) while it trades raw web-table
distribution alignment, so the headline evaluation targets relational
/ data-element skill rather than assuming a single web-CTA benchmark
is the goal (see [Roadmap](./roadmap.md) and `EVIDENCE.md`).

## Methodological contributions

**Algorithms.** Byte-level dynamic chunking as differentiable
tokenization: a routing module predicts boundary probabilities from
consecutive hidden-state cosine similarity, and chunk representatives
propagate to the next hierarchical stage. The H-Net primitive treats
tokenization as a learned property of the architecture rather than a
fixed preprocessing decision. Chunked-mode RWKV-7 time-mixing through
flash-linear-attention Triton kernels provides constant-state
recurrent computation with parallel training throughput. The ontology
side contributes a *propose / dispose* discipline in which rigor is
enforced by deterministic membranes (parse → HermiT/CCO → OntoClean)
rather than asserted, and measured by an IOF-anchored OQuaRE gate — an
intrinsic, reasoner-validated quality signal rather than a
judge-mediated one.

**Architecture.** A recursive hierarchy in which each stage selects
its own block-type mix from a block factory. Every current default
arch_layout uses RWKV-7 time-mix at every stage; ROSA (a RWKV-8
suffix automaton for exact substring retrieval) and Mamba-2 SSD are
additional block codes that the factory supports for hybrid
configurations and ablations. The recursion alternates encoding,
dynamic chunking, recursive inner processing, EMA dechunking, and
decoding; the recurrent state at every RWKV-7 stage is constant in
sequence length. This makes the recurrent state a fixed-size object
that can be serialized, transmitted, and algebraically combined
across agents — the substrate for the multi-agent state-fusion
infrastructure described in [Agent Swarm](./agent_swarm.md).

**Objectives.** The system pursues two complementary objectives. The
first is competitive accuracy on column-annotation tasks, with an
emphasis on relational / data-element skill — the application of the
byte-level model. The second is a measurably rigorous SDG ontology and
the verifiable, attribution-clean corpus it generates — evaluated by
intrinsic, reasoner-validated gates rather than by a judge. The two
objectives share the SDG ontology and the family catalog as substrate;
the second feeds the first downstream as pretraining data.

## Reading this document

- A reader interested in the **model architecture** should read
  [Architecture](./architecture.md) and its sub-pages on RWKV-7 time
  mixing, dynamic chunking, ROSA, and the block factory.
- A reader interested in the **ontology, its rigor program, and how to
  extend it** should read the [SDG ontology](./ontology.md) chapter
  and the [Ontology Authors Guide](./ontology/authors_guide.md); the
  in-flight RLVR sub-track is in the
  [concept brief](./ontology/concept_brief.md) and the
  [RLVR chapter](./ontology/rlvr.md).
- A reader interested in **byte-level pretraining and downstream
  fine-tune** should read [Pretraining](./pretraining.md) and
  [Training Regime](./training_regime.md).
- A reader interested in **who the system is built for and what
  workflows it commits to** should read [Personas](./personas.md).
- A reader interested in **the operational milestones and what has
  been delivered** should read [Roadmap](./roadmap.md).

The [development guide](./development.md) and the [worktree-aware
development](./worktree_aware.md) chapter cover operational concerns
and the CUDA-extension build path that Aegir's runtime depends on.
