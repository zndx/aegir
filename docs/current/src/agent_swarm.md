# Agent Swarm Architecture

The agent-swarm modules in `src/aegir/swarm/` are architectural
substrate for the multi-agent operational pattern that the project
will reach for as the metadata landscape scales beyond what a
single-policy loop can address. The system's current operational
loop — the membrane-gated content-first pipeline (`just metaflow`,
`src/aegir/flows/sdg_corpora_flow.py`) described in the
[semantic-engine authoritative reference](./ontology/production_state.md)
— is single-policy: one capability engine proposes, deterministic
membranes dispose. This chapter documents the swarm modules' design,
the engineering rationale for landing them in the codebase ahead of
an operational multi-agent task, and the optimization layers
(prompt evolution, agent RL) that will target the same verifier
*R(O, I)* — realized today as the deterministic membrane stack
(see the [RLVR chapter](./ontology/rlvr.md)'s status note) — once
the swarm becomes operational.

## When the swarm becomes operational

Two concrete forcing functions move the project from single-policy
training to a multi-agent architecture:

- **Scaling to large metadata landscapes.** GitTables (≈ 1M tables,
  100% generic column names) and the WikiTables corpus together
  represent more conceptual breadth than a single LoRA-fine-tuned
  policy of fixed capacity can hold without forgetting. Splitting
  the work across multiple agents — each specialized to a region
  of the metadata landscape, sharing a stable verifier — is the
  architectural answer.
- **Streaming source tagging (Flink SQL, Spark SQL).** Streaming
  query engines produce schemas continuously rather than as static
  batches; new domains arrive as new data products are stood up.
  A full GRPO retrain on every new domain is impractical. An
  online optimization loop that adapts agent prompts and routing
  faster than a weight retrain is the practical alternative.

The agent-swarm scaffolding exists in the codebase today so the
infrastructure is ready when those scaling pressures arrive. The
LatentMAS-informed RWKV-state-sharing design described below is
the substrate; the GEPA- and Agent-Lightning-class optimization
loops further down are the methodological frameworks the project
will adopt for operating it.

## The architectural substrate

The swarm shares compact RWKV recurrent state tensors between
agents rather than exchanging text messages or attention KV
caches — a communication medium that is uniquely efficient for
recurrent architectures.

### Why RWKV state sharing

RWKV's recurrent state is **constant in sequence length**. Each
layer's state is a matrix of shape `(H, K, V)` where `H` is the
number of heads and `K = V = head_size`. The total state size per
layer is

```
O(H * head_size^2) = O(d_model * head_size) = O(d^2)
```

independent of how many tokens the agent has processed. For a
swarm of `N` agents, the cost of sharing all recurrent states is

```
RWKV:        O(N * d^2)          -- constant in sequence length
Transformer: O(N * n * d)        -- linear in sequence length n
```

At context lengths of 4k–128k tokens with typical `d = 512–4096`,
RWKV state sharing is orders of magnitude cheaper. The LatentMAS
paper (arXiv:2511.20639) quantifies this as **235–471× more
information-dense** than text-based inter-agent communication,
since the recurrent state encodes a compressed summary of the
entire processing history.

For Aegir's column-annotation task, this means a specialist
trained on (say) geographic column types can share its accumulated
understanding of a table's structure through a single `(H, K, V)`
tensor per layer, rather than generating and parsing
natural-language explanations.

### Swarm components

The swarm consists of four modules:

| Module | File | Purpose |
|---|---|---|
| `RWKVStateFusion` | `src/aegir/swarm/state_fusion.py` | Combine *N* agent states into one |
| `AlignmentProjection` | `src/aegir/swarm/alignment.py` | Map states between different-sized agents |
| `FrozenSpecialist` | `src/aegir/swarm/specialist.py` | Wrap pre-trained models as frozen agents |
| `SwarmOrchestrator` | `src/aegir/swarm/orchestrator.py` | Routing + reward shaping |

### State fusion modes

`RWKVStateFusion` supports three strategies for combining agent
states:

1. **`weighted_sum`** — Attention-weighted combination using
   learnable query/key projections. The orchestrator learns which
   agents to trust per head.
2. **`gated`** — Per-agent softmax gates. Simpler than attention
   but still differentiable. A reasonable baseline for initial
   experiments.
3. **`concat_project`** — Concatenate all agent states and project
   back to single-agent dimensions. Most expressive but `O(N)` in
   parameter count.

See [RWKV State Fusion](./agent_swarm/state_fusion.md) for
mathematical details.

## Optimization layers for a deployed swarm

The agent swarm is the **architectural substrate**. The verifier
*R(O, I)* described in [RLVR for ontology generation](./ontology/rlvr.md)
is the **reward signal**. The remaining question is which
optimization loop adjusts the swarm against that reward. Three
candidate layers are available today, and the project's plan is
to adopt them in order as operational pressure justifies each.

### Weight-level: GRPO (long-horizon — Signals M4)

The weight-level layer updates a single policy's weights via Group
Relative Policy Optimization [Shao et al. 2024] against *R(O, I)*.
This is the appropriate choice when there is one policy, the corpus
is bounded, and training compute is available in chunks. The
GRPO/RLVR program is positioned as the long-horizon **Signals M4
apparatus** (see the [RLVR chapter](./ontology/rlvr.md)'s status
note and the [Signals Programme](./signals_programme.md)); its
verifier is realized today as the deterministic membrane stack
(HermiT/CCO, OntoClean, OQuaRE) that gates the generation
pipeline.

### Prompt-level: GEPA

When the swarm is operational and the optimization target is the
*prompts* of the agents rather than their weights, the project
will reach for **GEPA** [Agrawal et al. 2025; arXiv:2507.19457;
ICLR 2026 Oral]. GEPA is a Genetic-Pareto reflective prompt
optimizer: it samples trajectories from an LLM-based system,
uses an LLM to reflect on those trajectories in natural language
to diagnose failures, proposes prompt updates targeted at the
real observed failure modes, and combines complementary
improvements along the Pareto frontier of its own attempts. The
paper reports that reflective prompt evolution **outperforms GRPO
using up to 35× fewer rollouts** on the agentic tasks the authors
evaluated.

Two GEPA properties are directly relevant to a deployed swarm:

- **Compound-system support.** GEPA optimizes the prompts of an
  arbitrary LLM-based system — including multi-agent pipelines
  with retrieval, generation, reranking, and synthesis stages.
  The DSPy implementation (`dspy.GEPA`) exposes this for any
  DSPy module, and the same shape applies to a custom swarm with
  `FrozenSpecialist` agents.
- **Actionable Side Information (ASI).** GEPA's feedback channel
  is not just a scalar reward; it accepts structured error
  messages, profiling data, and reasoning traces. The membrane
  stack already produces this kind of structured feedback —
  per-component scores, hard-gate failure reasons, congruence and
  topic-layer diagnostics (`src/aegir/ontology/congruence.py`,
  `src/aegir/ontology/topic_layer.py`) — which is the feedback
  shape GEPA is designed to consume, and which the
  agent-mediated-feedback doctrine already requires membranes to
  return.

A GEPA-style loop is in fact already operational on the ontology
side: `scripts/evolve_rigor.py` reflectively evolves the
ontology's definitional rigor against the OQuaRE/metrology gates
(single-policy, membrane-disposed — no swarm involved). For the
swarm, GEPA becomes the operational optimization loop when it is
composing ontology fragments and the goal is to adapt the
system's behavior to a new domain (a new streaming source, a new
compliance regime) faster than a full GRPO retrain can deliver.

### Agent-level RL: Agent Lightning

When the optimization target is *agent behavior* — including tool
use, retrieval choices, multi-step interaction, and delayed
reward — the project will reach for **Agent Lightning**
[Microsoft Research 2025; arXiv:2508.03680]. Agent Lightning
decouples agent execution from RL training: it wraps any agent
built on LangChain, AutoGen, CrewAI, the OpenAI Agents SDK,
LangGraph, or custom Python with effectively zero code changes.
The framework's `LightningRL` algorithm formalizes agent execution
as a Markov decision process, defines a unified data interface,
and handles credit assignment so that any agent's trajectories
can be decomposed into training transitions — including in
**multi-agent scenarios and dynamic workflows**.

A particularly direct precedent for Aegir's streaming-SQL tagging
target exists in Agent Lightning's documentation: a
**LangGraph-based SQL agent** trained with the VERL RL algorithm
against task rewards. The Aegir generalization is to substitute
*R(O, I)* — which already discriminates schema-and-ontology
quality and is hash-stable across runs — for the SQL-agent reward
and run the same training loop against the Flink-SQL / Spark-SQL
streaming-tagging task. Agent Lightning also enables **selective
optimization** that targets specific sub-agents or steps in a
multi-agent workflow, which fits the swarm's `FrozenSpecialist` +
`SwarmOrchestrator` shape directly.

### Selecting an optimization layer

The three layers compose rather than compete:

| Layer | What it adjusts | When to use |
|---|---|---|
| GRPO (weight-level) | Single-policy weights | Bounded corpus; training compute available in chunks; the long-horizon Signals M4 apparatus |
| GEPA (prompt-level) | Prompts of an LLM-based system | Online adaptation to new domains; multi-agent pipelines; rollout-budget-constrained settings |
| Agent Lightning (agent-level RL) | Agent behavior incl. tool use, routing, multi-step | Multi-agent scenarios with delayed reward; framework-agnostic; streaming-SQL targets |

All three target the same verifier *R(O, I)*. That property — the
verifier is the durable asset, the optimization layers slot in
above it — is the project's methodological commitment: the
membranes are the durable core, and the swarm generalization
slots in above them downstream.

## The forward connection: latent lenses

The freshest concrete role for these modules is the **latent-lens
design direction** banked alongside the
[inverted topic layer](./roadmap/phase_gate_inverted_topic_layer.md)
(design note: `docs/scratch/2026-07-07/152519_latent_lens_design.md`).
There, every latent space is a *lens* over the ontology-grounded
topic registry — content-addressed by the triple
`(registry_sha × encoder_sha × projection_sha)` — and never the
registry itself:

- **`AlignmentProjection` is the module shape for per-modality
  lenses** (ladder stage L4): per-modality encoders (tables and
  schematics first) project into the shared anchor space exactly
  the way the module already projects recurrent states between
  differently-sized agents.
- **`RWKVStateFusion` is the LatentMAS tie-in** (opportunistic,
  after L3): the topic lens becomes a fusable specialist *state* —
  a topic-prior channel fused into the primary model's recurrent
  state alongside domain specialists.

The trust doctrine transfers verbatim: a lens changes *how fast*,
never *what* is verified — the text-space ColBERT/qdrant
adjudication (`src/aegir/ontology/topic_layer.py`) remains the
oracle of record until a differential harness earns co-signing.

## What this chapter does not commit to

- **The swarm is not yet operational.** The modules above exist in
  `src/aegir/swarm/` but are not exercised by any current training
  run; the operational system remains the single-policy
  membrane-gated pipeline.
- **GEPA and Agent Lightning are not integrated with the swarm.**
  GEPA-style reflective evolution runs today on the ontology's
  definitional rigor (`scripts/evolve_rigor.py`); neither
  framework yet drives the swarm modules. Integration follows the
  [Signals Programme](./signals_programme.md)'s milestone gates,
  not a calendar.
- **The order of adoption is provisional.** Whether prompt-level
  optimization (GEPA) or agent-level RL (Agent Lightning) becomes
  operational first depends on which scaling pressure
  (large-corpus breadth vs. streaming-online adaptation) arrives
  first. The [roadmap](./roadmap.md) tracks both.

## References

- LatentMAS — recurrent-state sharing as multi-agent
  communication. arXiv:2511.20639.
- Agrawal, L. A., et al. (2025). *GEPA: Reflective Prompt Evolution
  Can Outperform Reinforcement Learning.* arXiv:2507.19457; ICLR
  2026 Oral. Reference implementation: `dspy.GEPA`.
- Microsoft Research. (2025). *Agent Lightning: Train ANY AI Agents
  with Reinforcement Learning.* arXiv:2508.03680. Documentation
  includes a LangGraph SQL-agent training example.
- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the limits of
  mathematical reasoning in open language models.* (Source of the
  GRPO algorithm.)

**Internal references.**

- [RLVR for ontology generation](./ontology/rlvr.md) — the
  verifier *R(O, I)* that all three optimization layers target;
  the long-horizon Signals-M4 methodological record.
- [Semantic-engine authoritative reference](./ontology/production_state.md)
  — the operational state of the current single-policy semantic
  engine.
- [Roadmap](./roadmap.md) — the milestone structure; the K2.5
  PARL plan (supervised → reward → PARL → swarm RL) folded into
  the Signals milestone ladder, with the swarm modules persisting
  as the infrastructure documented here.
