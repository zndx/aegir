# RLVR for ontology generation

This chapter is the externally-readable description of the
project's reinforcement-learning-with-verifiable-reward (RLVR)
program — paper 1 of the two-paper structure documented in the
[concept brief](./concept_brief.md). The
[semantic-engine authoritative reference](./production_state.md) is
the canonical empirical surface; this chapter is the methodological
counterpart, accessible without the concept brief's research-design
overhead.

The chapter is organized in five parts: the verifiable-reward
setting and why it fits OWL ontology generation; the four-component
deterministic verifier *R(O, I)*; the GRPO training program that
targets it; how the verifier generalizes as the system scales
beyond a single policy; and the paper-2 application that this work
enables downstream.

## The verifiable-reward setting

Reinforcement learning with a verifiable reward (RLVR) is the
training discipline in which a policy is updated against a reward
that can be computed deterministically from the policy's output,
without an LLM judge in the loop. The verifier is a function — not
a model — and its output is hash-stable: identical inputs produce
bit-identical reward values across re-runs. This shape has been
demonstrated on mathematics [DeepSeekMath; Shao et al. 2024;
DeepSeek-R1; DeepSeek-AI 2025] and on code execution
[CodeRL, Reasoning-SQL]; the present project applies it to OWL
ontology composition.

OWL is a particularly natural target for RLVR. The artifact is
graph-structured (classes, properties, axioms with restrictions and
equivalentClass intersections), and a sound-and-complete
description-logic reasoner can verify both *structural* properties
(does the artifact load? are slot fills well-typed?) and *semantic*
properties (does the artifact entail what its templates claim it
entails?). The verifier *R(O, I)* combines those checks with a
corpus-alignment component that measures whether the ontology's
verbalizations are on-topic for a target corpus *I*.

The combination — graph-structured output with a semantically
grounded deterministic verifier targeting an LLM policy under
RLVR — is the contribution claim. Adjacent prior art either emits
flat (head, rel, tail) triples without a reasoner-based verifier
[AutoGraph-R1; Tsang et al. 2026], emits QA reasoning traces rather
than OWL [K2V; Yuan et al. 2026], or uses plain SFT against a
custom regulariser [OLLM; Lo et al. 2024]. The
[concept brief](./concept_brief.md) documents each differentiation
in detail.

## The verifier *R(O, I)*

Let *O* = compose(*C*, σ) denote an OWL ontology composition
produced from the procedural catalog *C* with slot-fill σ. Let *I*
denote a fixed input text corpus. The verifier
*R*: OntologyComposition × Corpus → [0, 1] is defined as

> *R*(*O*, *I*) = *R_A*(*O*) · ( *a* · *R_B*(*O*) + *b* · *R_C*(*O*) + *c* · *R_D*(*O*, *I*) )

with four components. *R_A* is a hard structural gate: it returns 1
iff every (template, σᵢ) pair type-checks against the catalog's
typed slot DSL, and returns 0 otherwise. Because every catalog
template was DeepOnto-validated offline at catalog construction
time [He et al. 2023], *R_A* = 1 implies the materialized OWL also
passes DeepOnto loadability at runtime. *R_B* measures
complex-class density relative to a structural-shuffle null
distribution. *R_C* is a coarse semantic-richness proxy via cached
verbalization length. *R_D* is the Hungarian-optimal cosine
alignment between the composition's verbalizations and a BERTopic
topic model fit to *I* [Grootendorst 2022], normalized against the
same structural-shuffle null.

The aggregation weights `{a, b, c} = {0.50, 0.05, 0.45}` were
locked by a sweep over the unit simplex against a 30-ontology
hand-authored discrimination test set (15 known-good, 15 known-bad).
With those weights, the verifier achieves AUC 0.9956 and mean
*R*-separation 0.336 on the test set. A held-out evaluation set of
50 scenarios (25 good + 25 bad), authored before any policy-side
RL work began, gives separation 0.5129 against the locked verifier
— leakage-free with respect to any policy that subsequently trains
against *R*. The full empirical surface is documented in the
[semantic-engine authoritative reference §4](./production_state.md#4-the-four-component-verifier-ro-i).

The verifier is deterministic, hash-stable, and has no JVM or Java
dependencies in its runtime hot path. DeepOnto is invoked only at
catalog construction time; the runtime verifier reads pre-cached
verbal templates from the JSON catalog. Per-sample scoring on CPU
takes about 0.02 s once the encoder and topic model are loaded,
which means the verifier is not the rate-limiting step in any
practical RL training loop.

## GRPO at the weight level (paper 1)

The current operational training program is a GRPO-trained policy
on Qwen3.5-9B-Base with a LoRA adapter on attention and MLP
projections. The corresponding `SAE-Res-Qwen3.5-9B-Base`
residual-stream adapter is held untouched so that the
interpretability claim about the trained policy's representations
survives weight updates [sparse-autoencoder feature decomposition;
Cunningham et al. 2024]. The training pipeline includes:

1. **Constrained-decode JSON Schema enforcement** via
   lm-format-enforcer, wired through a wrap on `model.generate`
   that survives TRL's `unwrap_model` indirection. Without this
   wrap the schema is built but never reached by the generation
   path, and the policy emits free-form text that *R_A* clamps to
   zero — a failure mode that produced a 1690-step zero-reward run
   before the wrap was added.
2. **Rejection-sampling SFT warm-start.** The Base model emits
   zero-reward outputs cold; the warm-start samples compositions
   under constrained decoding across rotated few-shot variations,
   scores them with the verifier, retains the *R* ≥ 0.3 subset,
   and supervised-fine-tunes the Base on that retained corpus
   before GRPO begins.
3. **Per-iteration verifier scoring** with the locked *R(O, I)*
   and group-relative advantage estimation.

Paper 1's two subordinate claims are *C1 — discrimination* (the
locked verifier discriminates known-good from known-bad ontologies
on the test set; established at AUC 0.9956) and *C2 —
optimizability* (GRPO training of the policy against *R* produces
compositions whose *R*-distribution exceeds prompt-evolved and
human-authored baselines on the held-out 50; **under empirical
test**). The choice of warm-start procedure (Option A:
rejection-sampling SFT — currently running; Option B:
Instruct-paired model + Instruct-paired SAE adapter; Option C:
Self-Distillation Fine-Tuning [SDFT; Shenfeld et al. 2026]) is
explicitly under revision; the
[authoritative reference's empirical-test section](./production_state.md#6-the-current-empirical-test-under-revision)
names what the in-flight run will and will not settle.

## Generalizing the verifier: scaling beyond a single policy

As the system scales to larger metadata landscapes — GitTables
(1M+ tables), the WikiTables corpus, and streaming sources such as
Flink SQL and Spark SQL where schemas appear continuously rather
than as static batches — a single-policy weight-trained approach
faces two pressures. First, the breadth of in-scope concepts
expands past what a single LoRA-fine-tuned policy of fixed capacity
can absorb without forgetting. Second, online adaptation to
newly-arrived schemas in a streaming context calls for an
optimization loop that can react faster than a full GRPO retrain.

The verifier *R(O, I)* is the durable asset across this transition.
Whatever the optimization layer (policy weights, prompts, agent
configurations, multi-agent routing), the same deterministic
verifier supplies the reward signal. Two recent frameworks make
the optimization layers above the weight surface explicit and
externalize them as engineering substrate the project can adopt as
operational pressure makes it useful:

- **GEPA — Reflective Prompt Evolution** [Agrawal et al. 2025;
  arXiv:2507.19457; ICLR 2026 Oral]. GEPA is a genetic-Pareto prompt
  optimizer: given an LLM-based system, it samples trajectories,
  reflects on them in natural language to diagnose failure modes,
  proposes prompt updates, and combines complementary lessons along
  the Pareto frontier of its own attempts. The paper reports that
  reflective prompt evolution can outperform reinforcement learning
  on certain agentic tasks under a fixed budget. For this project,
  the relevance is direct: GEPA's outer loop targets a programmatic
  fitness signal, and *R(O, I)* is one. Substituting *R(O, I)* in
  place of GEPA's example fitness gives prompt-level optimization
  of an ontology-emitting LLM system without weight updates. A
  reference implementation lives in DSPy as `dspy.GEPA`.
- **Agent Lightning — RL for agent systems** [Microsoft Research
  2025; arXiv:2508.03680]. Agent Lightning is a framework that adds
  RL-based training to agents built on LangChain, Microsoft AutoGen,
  the OpenAI Agents SDK, or arbitrary custom code, with effectively
  zero code modification to the agent itself. The framework
  formalizes agent execution as a Markov decision process, defines a
  unified data interface, and introduces a hierarchical RL algorithm
  (LightningRL) with explicit credit assignment so any agent's
  trajectories can be decomposed into training transitions. For this
  project, the relevance is that *R(O, I)* can act as the RL reward
  for any agent built on this substrate — including multi-agent
  configurations where one agent retrieves context (sampled passages
  from a target corpus *I*), another proposes catalog compositions,
  and a third refines slot fills. The
  [agent swarm scaffolding](../agent_swarm.md) in the codebase is
  the project-side complement to this generalization.

The unifying point is methodological: paper 1 establishes that
*R(O, I)* discriminates ontology quality and is optimizable via
weight-level GRPO. Once that's established, the same verifier
becomes the reusable substrate for prompt-level optimization (GEPA)
and agent-level RL (Agent Lightning) as the engineering pressure
from streaming sources and 10⁶-table metadata landscapes makes
those optimization layers necessary. The hardest gate to cross is
verifier validity; the rest is engineering on top of a stable
substrate.

No project work has been committed to GEPA or Agent Lightning
integration yet — that work follows paper 1 and the operational
scale-up to GitTables and streaming Flink / Spark SQL tagging. The
[roadmap](../roadmap.md) names that scale-up as the forcing
function.

## Paper 2 — ontology-grounded byte-level pretraining

Paper 2's claim is downstream of paper 1: verbalizations from
*R*-passing ontologies (produced by paper 1's policy) measurably
improve byte-level pretraining of Aegir's hierarchical sequence
model on stratified held-out evaluation. Paper 2 is contingent on
paper 1's policy producing verifier-passing compositions at corpus
scale; its methodology will be refined after paper 1's results
constrain it. The pretraining track that paper 2 augments is
documented in [Pretraining](../pretraining.md).

## What this chapter does not claim

- **Paper 1's C2 (optimizability) is not yet established.** C1 —
  the verifier discriminates ontology quality on the test set — is
  established at AUC 0.9956. C2 is the experimental claim currently
  under test; the in-flight GRPO run is the first empirical
  attempt.
- **The choice among warm-start procedures (Options A, B, C) is
  not settled.** The current run tests Option A; a comparison
  against Option B is the next-priority experimental step. Any
  claim built on the *specific* warm-start procedure being correct
  is unsupported until at least one direct comparison is run.
- **GEPA and Agent Lightning are not yet integrated.** The scaling
  argument in the section above identifies them as the
  methodological layer the project will reach for when single-
  policy training proves limiting; it is not a description of
  current work.
- **Paper 2's lift is not yet measured.** Whether *R*-passing
  ontology verbalizations measurably improve downstream pretraining
  utility is a separate experimental question with its own
  evaluation surface.

## References

**RLVR core method.**

- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the limits of
  mathematical reasoning in open language models.* (Origin of the
  GRPO algorithm.)
- DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning
  capability in LLMs via reinforcement learning.*

**Optimization layers above policy weights.**

- Agrawal, L. A., et al. (2025). *GEPA: Reflective Prompt Evolution
  Can Outperform Reinforcement Learning.* arXiv:2507.19457; ICLR
  2026 Oral. Reference implementation: `dspy.GEPA`.
- Microsoft Research. (2025). *Agent Lightning: Train ANY AI Agents
  with Reinforcement Learning.* arXiv:2508.03680.
- Shenfeld, I., Damani, M., Hübotter, J., Agrawal, P. (2026).
  *Self-Distillation Enables Continual Learning.* arXiv:2601.19897.
  (On-policy self-distillation as a candidate warm-start.)

**Adjacent OWL / KG / verbalization work.**

- Lo, A., Jiang, A. Q., Li, W., Jamnik, M. (2024). *OLLM: Generating
  ontologies from texts.* NeurIPS 2024.
- Tsang, et al. (2026). *AutoGraph-R1.* arXiv:2510.15339, ICLR 2026
  submission.
- Yuan, et al. (2026). *K2V — Knowledge-to-Verification.* ICLR 2026
  submission.
- He, Y., Chen, J., Antonyrajah, D., Horrocks, I. (2023). *DeepOnto:
  A Python package for ontology engineering with deep learning.*
- Liu, et al. (2025). *OntoTune.* WWW 2025.

**Topic modeling and interpretability.**

- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure.*
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L.
  (2024). *Sparse Autoencoders Find Highly Interpretable Features
  in Language Models.* ICLR 2024.

**Internal references.**

- [Concept brief](./concept_brief.md) — full research design and
  literature review.
- [Semantic-engine authoritative reference](./production_state.md)
  — the empirical surface this chapter's claims rest on.
- [Charter](./charter.md) — the outward contract that the SDG
  ontology serves.
- [Roadmap](../roadmap.md) — the two-paper milestone structure
  this chapter sits within.
- [Agent swarm](../agent_swarm.md) — the project-side scaffolding
  for the multi-agent generalization sketched in the scaling
  section.
