# Introduction

Aegir is a hierarchical byte-level sequence model for semantic
annotation of relational data. Given one or more tables, it predicts
the semantic type of each column (Column Type Annotation, **CTA**),
the relationships between columns (Column Property Annotation,
**CPA**), and the cross-table groupings that constitute coherent
real-world *data elements* — for example, a `PaymentCard` data
element spanning `card_number`, `expiry`, and `cardholder` columns
across billing, transaction, and customer tables. The model is paired
with the **Signals Data Governance (SDG) ontology** and a
deterministic four-component verifier *R(O, I)* over OWL ontology
compositions; together they constitute a closed loop between the
model and the structured-knowledge representation it learns from.

## Two coupled research outputs

The project produces two outputs that are cited together:

1. **A hierarchical byte-level sequence model.** All-RWKV-7
   time-mixing with H-Net dynamic chunking, trained byte-level on a
   mixed corpus and fine-tuned for the column-annotation tasks above.
   The architecture is described in [Architecture](./architecture.md);
   the operational pretraining work is described in
   [Pretraining](./pretraining.md) and
   [Training Regime](./training_regime.md).

2. **An RLVR-trained policy for OWL ontology generation.** A
   language-model policy trained with Group Relative Policy
   Optimization (GRPO) that emits OWL compositions scored by a
   deterministic four-component verifier with locked aggregation
   weights. The verifier, the policy, and the procedural catalog they
   share are described in the [SDG ontology chapter](./ontology.md)
   and the [semantic-engine authoritative
   reference](./ontology/production_state.md).

The two outputs share substrate — the SDG ontology, the 540-template
procedural catalog, the verbalization pipeline — and are coupled
downstream: the RLVR policy produces ontology compositions whose
verbalizations feed back into the byte-level pretraining corpus. The
[concept brief](./ontology/concept_brief.md) documents the two-paper
research structure in full.

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
relying on enumerated patterns. Target benchmarks are **SOTAB**
(Schema.org types over web tables), **GitTables** (large-scale column
type detection across 1M+ CSV tables from GitHub — the hardest regime
for generic column names), and **WikiTables** (column annotation on
Wikipedia HTML tables).

## Methodological contributions

**Algorithms.** Byte-level dynamic chunking as differentiable
tokenization: a routing module predicts boundary probabilities from
consecutive hidden-state cosine similarity, and chunk representatives
propagate to the next hierarchical stage. The H-Net primitive treats
tokenization as a learned property of the architecture rather than a
fixed preprocessing decision. Chunked-mode RWKV-7 time-mixing through
flash-linear-attention Triton kernels provides constant-state
recurrent computation with parallel training throughput. A
four-component deterministic verifier *R(O, I)* scores OWL ontology
compositions across structural (*R_A*), complexity-density (*R_B*),
semantic-richness (*R_C*), and corpus-alignment (*R_D*) axes, with
aggregation weights locked from a 30-ontology authored discrimination
sweep at AUC 0.9956. The verifier supplies the dense, hash-stable
reward signal that the GRPO policy targets.

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
first is competitive accuracy on stratified column-annotation
benchmarks — the application of the byte-level model. The second is
a measurably high-quality stream of OWL ontology compositions,
evaluated by an intrinsic (rather than judge-mediated) deterministic
verifier — the research output of the RLVR policy. The two objectives
share the SDG ontology and procedural catalog as substrate; the
second feeds the first downstream as a training corpus once the RLVR
policy's outputs are verifier-passing at scale.

## Reading this document

- A reader interested in the **model architecture** should read
  [Architecture](./architecture.md) and its sub-pages on RWKV-7 time
  mixing, dynamic chunking, ROSA, and the block factory.
- A reader interested in the **ontology, verifier, and RLVR program**
  should read the [SDG ontology](./ontology.md) chapter and the
  [semantic-engine authoritative
  reference](./ontology/production_state.md).
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
