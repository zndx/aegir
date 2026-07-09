# Concept brief — RLVR for ontology generation

### A four-component verifiable reward and GRPO-trained policy for OWL composition

*Draft v0.5 — 2026-05-09 — companion to [Charter](./charter.md) and
[Migration](./migration.md). Supersedes v0.1, v0.2, v0.3, v0.4.*

> **Status (2026-06-29) — research design of the long-horizon Signals
> M4 apparatus.** This brief is the detailed research design for the
> RLVR program (the verifier *R*, the prior-art positioning, and the
> P0–P9 phase structure) — the **Signals M4** apparatus, an
> SAE-instrumented-Qwen local policy GRPO-trained to autonomously
> generate ontology extensions. Its four-component verifier *R(O, I)*
> is now **realized as the deterministic membrane stack** (HermiT/CCO,
> OntoClean, OQuaRE), and the agent-mediated propose/dispose loop is
> building and proving that reward today, in direct service of M4. This
> brief remains the authoritative statement of the *contribution claim*
> and the seven required prior-art differentiations. For the **current**
> state of the program read its two companions, which carry the
> empirical and externally-readable surfaces:
> - the [RLVR chapter](./rlvr.md) — the externally-readable
>   methodological description, with the locked verifier and current
>   policy; and
> - the [semantic-engine authoritative reference](./production_state.md)
>   — the canonical empirical surface (locked weights, AUC, the
>   in-flight GRPO run).
>
> Where this brief's specifics have been overtaken, they are corrected
> inline below; the verifier definition, the prior-art
> differentiations, and the phase/gate methodology are preserved as
> written. Note also that this brief concerns the **RLVR/GRPO research
> track**, which is distinct from the ontology **rigor** program (the
> OQuaRE publish gate, OntoClean membranes, and intermediate-class
> authoring documented in the [Authors Guide](./authors_guide.md)); the
> two share the SDG ontology but are governed separately.

> **Status addendum (2026-07-09).** Two of this brief's substrates
> have since been superseded (their definitions below are preserved
> as written). **R_D's corpus-fitted topic model** (BERTopic/KMeans →
> `T_I.pkl`) was deprecated 2026-07-07; its successors are
> **congruence** (`src/aegir/ontology/congruence.py` — chapter ↔
> input-window concept alignment over the same ColBERT/qdrant
> substrate the harvest classifies against) and the **inverted topic
> layer** (`src/aegir/ontology/topic_layer.py`; phase gate:
> [Inverted Topic Layer](../roadmap/phase_gate_inverted_topic_layer.md)).
> And the catalog *C* is no longer hand-authored: the seed families
> are retired, and `src/aegir/ontology/catalog/catalog.json` is
> accreted by the derive → promote → realize engine
> (`scripts/derive_ontology.py` → `scripts/promote_candidates.py` →
> `scripts/build_realized_ontology.py`).

## Objective

Define a four-component deterministic verifier *R* over OWL ontology
artifacts and the target text corpus *I*, then train an LLM policy
π_θ via formal RLVR (GRPO with *R* as the reward) to produce
ontologies that (a) exceed prompt-evolved and human-authored
baselines on *R*, and (b) admit downstream verification that each
*R*-component carries predictive validity for byte-level pretraining
utility on Aegir's stratified-eval surface.

The contribution is the verifier *R* together with the RL training
loop that targets it. Pretraining lift is a downstream external
validity check; it is the subject of a separate application paper
scoped at the end of this brief.

## Scope split — two papers

To avoid the chain-too-long failure mode of the v0.1 brief, the
research program decomposes into two independently citable papers
that share infrastructure:

| Paper | Headline claim | Phases | Status in this brief |
|---|---|---|---|
| **Paper 1 — RLVR for ontology generation** | A four-component verifier *R* and a GRPO-trained policy π_θ produce ontologies exceeding human and prompt-evolved baselines on *R*; each *R*-component is shown to discriminate quality on a held-out test set. | P0 → P6 | **Primary scope** of this brief. |
| **Paper 2 — Ontology-grounded byte-level pretraining** | Verbalizations from *R*-passing ontologies (produced by paper 1's policy) measurably improve byte-level pretraining of a hierarchical sequence model on Aegir's stratified eval. | P7 → P9 | **Application follow-on**, scoped here at lower resolution. Earned only if paper 1 lands. |

This brief commits primarily to paper 1. Paper 2 is sketched at the
phase level but its methodology is left to be revised after paper 1's
results constrain it.

## Contribution claim (locked)

We define a four-component verifier *R*: OntologyArtifact × Corpus →
[0, 1]. We demonstrate three subordinate claims, all tied to *R*:

- **C1 — Discrimination.** *R* discriminates ontologies of varying
  quality on a held-out test set (*R* on known-good ≫ *R* on
  known-bad with a margin defended by AUC against a labeled set).
- **C2 — Optimizability.** GRPO training of an LLM policy π_θ
  against *R* produces ontologies whose mean *R* exceeds
  prompt-evolved and human-authored baselines at p < 0.05 over a
  held-out generation set.
- **C3 — Component validity.** Each component of *R* (R_A, R_B, R_C,
  R_D) carries predictive validity for downstream pretraining
  utility, established by ablation in paper 2; relative weights
  used in paper 1's aggregation are derived from paper 2's ablation.

The brief commits to C1 + C2 as paper 1's headlines. C3 is paper 2's
primary contribution. The two papers cite each other.

## Load-bearing novelty (sharpened in v0.5)

Per the v1 P0 literature review at
`docs/scratch/2026-05-09/225830_lit_review_v1.md`, each component
recipe of this brief has prior art on adjacent artifacts:

- **RL on graph-structured outputs:** AutoGraph-R1 (knowledge graphs),
  the AMR-RL chain (semantic DAGs), Lehmann & Haase 2012 (symbolic
  RL on EL++ concepts).
- **Deterministic schema validators with RL:** SRL, RL-Struct (JSON
  Schema).
- **Deterministic execution validators with RL:** CodeRL, Reasoning-SQL.
- **Verbalization corpora for KGs:** KELM/TEKGEN (Wikidata ABox →
  retrieval corpus).
- **End-to-end LLM ontology generation:** OLLM (NeurIPS 2024, SFT +
  regulariser, taxonomic backbone).
- **Ontology-guided LLM optimization:** OntoTune (SFT
  self-distillation), K2V (RLVR + KG, but emits QA traces).

What is *not* present in any of these is the combination this brief
proposes:

> The OWL artifact is the only graph-structured output where the
> verifier can run a sound-and-complete reasoner producing both
> structural and semantic verdicts (consistency, entailment,
> class-hierarchy coverage). That combination — graph-structured
> output with a *semantically grounded* deterministic verifier
> targeting an LLM policy under RLVR, plus the verbalization-corpus
> pretraining application — is the gap.

The brief's contribution is the synthesis of the four precursor
recipes (graph-output RL, schema-validator RL, code/SQL execution
RL, verbalization-corpus pretraining) onto an artifact (OWL) where
the verifier acquires DL deductive semantics that none of the
precursors have.

## Adjacent prior art — explicit differentiations

Seven adjacent works must be addressed head-on in this brief and
any paper that emerges from it. Each differentiation is stated
explicitly below.

**KELM / TEKGEN (Agarwal, Ge, Shakeri, Aharoni, NAACL 2021).** The
single closest *verbalization corpus* recipe: verbalize a structured
knowledge source into natural-language sentences, integrate as a
model training corpus, measure downstream effect.

> KELM verbalizes ABox triples from Wikidata into a retrieval
> corpus evaluated on QA. We verbalize TBox axioms from a bespoke
> OWL ontology into a flat byte-level pretraining slice evaluated
> on stratified held-out bits-per-byte. The structural content of
> the verbalized text differs (taxonomic and logical class
> expressions vs. instance-level facts); the integration mechanism
> differs (pretrain mix vs. retrieval); the evaluation isolates
> ontology-specific contribution rather than generic QA lift.

**OLLM (Lo, Jiang, Li, Jamnik, NeurIPS 2024).** Closest *end-to-end
LLM ontology generation* approach. Fine-tunes an LLM with a custom
regulariser that reduces overfitting on high-frequency concepts;
produces taxonomic backbones from scratch.

> OLLM uses plain SFT with a regulariser, not RL; produces
> taxonomic backbones only, not full OWL with restrictions or
> equivalentClass axioms; evaluation is graph-similarity against
> reference ontologies. We train via GRPO against a deterministic
> verifier that scores OWL-reasoner consistency, axiom complexity,
> and topic alignment with a target corpus; the policy emits full
> OWL compositions with restrictions and equivalentClass
> intersections.

**AutoGraph-R1 (Tsang et al., arXiv:2510.15339, ICLR 2026
submission).** Closest *RL-trained graph-emitting policy*. A
GRPO-trained LLM policy emits a knowledge graph from text; reward
is a "Knowledge-Carrying Reward" computed from the graph's
downstream RAG utility, judged extrinsically.

> AutoGraph-R1 emits flat (head, rel, tail) triples — ABox-style
> instance facts — and uses an extrinsic LLM-judge reward via
> downstream QA accuracy on retrieved triples. We emit OWL TBox
> compositions with class axioms and use an *intrinsic*,
> *semantically grounded* deterministic verifier (DL-reasoner
> consistency check + programmatic structural property checks +
> topic-model alignment). Sound-and-complete deductive reasoning
> is unavailable to AutoGraph-R1's flat-triple output by
> construction; OWL's class-axiom expressivity is what makes the
> DL-reasoner verifier shape possible.

**K2V — Knowledge-to-Verification (Yuan et al., ICLR 2026
submission).** Closest *RLVR + KG-derived reward* methodology.
Builds a KG from text and frames KG completion as a verifiable QA
task to derive dense rule-based rewards for LLM reasoning.

> K2V's policy emits QA reasoning traces, not OWL. The verifier
> checks subtask correctness, not ontology loadability / axiom
> density / topic alignment. K2V proves the RLVR-with-KG-derived-
> reward shape works; we apply that shape to OWL generation
> directly, with the verifier targeting structural and semantic
> properties of the artifact rather than QA accuracy on
> downstream tasks.

**OntoTune (Liu et al., WWW 2025).** Iteratively refines an LLM
against an ontology-grounded objective.

> OntoTune iterates an LLM against an ontology-grounded objective
> via SFT, with the reward implicit in does-LLM-already-know-this
> gating. The model emits natural-language answers, not OWL. We
> train an LLM policy via GRPO with an explicit deterministic
> verifier whose output is a continuous reward in [0, 1], and the
> policy emits OWL ontology compositions whose well-formedness is
> guaranteed by the catalog's typed-slot grammar.

**Zaitoun, Sagi, Peleg (AAAI Symposium Series 2024).** Closest
OWL-specific verbalization-derived training data.

> Zaitoun et al. use LLM-assisted verbalization of OWL axioms to
> create text→OWL supervised fine-tuning pairs. We treat the
> verbalizations as a flat self-supervised byte-level corpus
> mixed with general pretraining text, with no instruction-pair
> framing.

**OnT — Language Models as Ontology Encoders (Yang, Chen, He, Gao,
Horrocks, arXiv:2507.14334, 2024).** Closest *TBox-axiom-aware
embedding* approach. Compositional verbalization of OWL class
expressions feeds a pretrained Sentence Transformer, re-trained via
hyperbolic-space hierarchy/role/conjunction losses.

> OnT verbalizes TBox axioms but uses them as auxiliary-objective
> embedding training (hyperbolic loss on hierarchy / role /
> conjunction); the underlying LM is not pretrained on the
> verbalizations. We treat verbalizations as flat next-token
> pretraining bytes mixed with general-purpose corpus.

These seven citations are required in the related-work section of
any paper that emerges from this brief. Secondary methodological
precedents (SRL / RL-Struct, AMR-RL chain, CodeRL / Reasoning-SQL,
DRAGON, Lehmann & Haase 2012, OLLM) are listed in the lit review v1
(`docs/scratch/2026-05-09/225830_lit_review_v1.md`) citations index.

P0 exit gate is **firmly green** as of v1; the chain-of-three claim
survives a depth-of-search expansion across the three highest-risk
axes.

## Verifier *R* — formal definition

The verifier is structured around a **procedurally pre-computed
catalog** *C* (described in the next subsection). DeepOnto runs
*offline* during catalog construction; the runtime verifier does
not call DeepOnto and does not require a JVM. This decision keeps
DeepOnto out of the RL loop's critical path and out of any
pretraining inference path.

Let *O* = compose(C, σ) be an ontology composed by the policy from
catalog templates with slot-fill σ. *I* is a fixed input text
corpus. *R* has four components.

**R_A (Well-formedness).**

```
R_A(O) = 1 if all (template, σ_i) pairs in O type-check against C else 0
```

A composition is well-formed iff every selected template's slot
constraints are satisfied by the chosen fillers (typed term
inventory; types declared per-template at catalog construction
time). R_A is a hard gate; R(O) = 0 if R_A = 0. By construction,
R_A = 1 implies the rendered OWL also passes DeepOnto loadability,
because every catalog template was DeepOnto-validated offline.

**R_B (Complex-class density).**

```
R_B(O) = min(complex_count(O) / τ_B, 1)
```

where `complex_count(O)` is the number of templates in *O* whose
`is_complex` flag is set in *C* (DeepOnto-determined offline by
running `onto.get_asserted_complex_classes()` on the rendered
template). τ_B is the 95th percentile of complex-count from the
structural-shuffle null distribution computed once per catalog.

**R_C (Verbalization quality, as semantic-richness proxy).**

By construction, every template in *C* verbalizes cleanly (offline
gate at catalog construction time), so the binary "does it
verbalize" question is uninformative at runtime. R_C is repurposed
as a continuous semantic-richness proxy:

```
R_C(O) = clip(mean_verbal_length(O) / L_target, 0, 1)
```

where `mean_verbal_length(O)` is the mean character length of the
pre-cached verbalizations of templates in *O*, and L_target is
calibrated from the catalog's distribution (P1 sets this so that
the median template gives R_C ≈ 0.5).

**R_D (Topic alignment with corpus *I*).**

Let *V(O)* be the verbalization corpus of *O*, constructed by
concatenating each template's pre-cached verbalization with the
slot-fillers substituted in. Fit BERTopic on *V(O)* (the topic
model on *I* is fitted once and frozen at P1). Compute the
Hungarian-optimal one-to-one matching between the *V(O)* topics
and the frozen *T_I* topics under cosine similarity in the c-TF-IDF
representation space. R_D is the mean matched cosine similarity,
normalized to [0, 1] against the structural-shuffle null
distribution.

R_D is the only runtime component that recomputes a topic model;
its cost is the dominant per-step verifier cost. Mitigation: the
sentence-embedding backbone is computed once on *V(O)* (small —
typically 100s–1000s of sentences for a single composition) so the
hot path is HDBSCAN clustering on cached embeddings, ~seconds per
ontology on CPU.

**Aggregation.**

```
R(O) = R_A(O) · (a · R_B(O) + b · R_C(O) + c · R_D(O))
```

with a + b + c = 1. Initial weights {a, b, c} are derived from the
P2 verifier-validation phase by maximizing AUC against a labeled
ontology test set. The weights are *not* tuned during P5 RL training
— they are fixed before the policy sees the verifier. This
separation prevents the policy from gaming the weight-discovery
process.

The verifier is implemented as `scripts/aegir-verify`, deterministic,
hash-stable across runs given fixed input. **No JVM, no DeepOnto,
no Java dependencies at runtime** — the catalog encapsulates all
DeepOnto-derived knowledge as flat data.

## Methodology

### Input corpus *I*, pinned

*I* is the held-out subset of v2's mixed-corpus pretrain mix
restricted to **SchemaPile + FinePDFs-lab** (the same choice as
v0.1 of this brief, for the same reasons: reproducible, downstream-
aligned, disjoint from v2 trained-time eval). Size: ~200 MB.

The 200 MB working budget is *not* validated for BERTopic stability
(BERTopic, unlike LDA, is sensitive to corpus size in different ways
— too few documents and HDBSCAN clustering is unstable; too many and
sentence-embedding compute dominates). P1's verifier-implementation
phase includes a corpus-size sensitivity sweep before *I* is locked
for downstream phases.

### Topic model — BERTopic primary, NMF ablation

LDA is **not** the default; project consensus from earlier work is
that LDA's bag-of-words assumption fails on DDL-heavy text where SQL
syntax tokens dominate vocabulary frequency. BERTopic over a sentence-
embedding backbone (default: `all-MiniLM-L6-v2` for speed; switchable
to `gte-large` for headline runs) is the primary choice. NMF over
TF-IDF vectors is the robustness ablation.

For both topic models, the c-TF-IDF representation per topic is
computed from the original corpus tokens; alignment is in this
shared representation space.

### Procedural catalog *C* — offline DeepOnto, runtime lookup

The catalog is the methodological pivot of v0.3. DeepOnto's role is
moved from runtime to catalog-construction time; the runtime
verifier (and any downstream pretraining pipeline that consumes
catalog-rendered ontologies) has no DeepOnto dependency.

**Catalog structure.** *C* is a flat data artifact: a JSON or
SQLite table where each row is a *template* and contains:

| Field | Source | Purpose |
|---|---|---|
| `template_id` | catalog assignment | unique handle |
| `manchester_template` | hand or LLM-authored, pre-validated | OWL Manchester syntax with typed slots, e.g. `Class: {X:Class} SubClassOf: {p:ObjectProperty} some {Y:Class}` |
| `slot_types` | derived from template | typed-slot constraints for the policy's slot-fill |
| `is_complex` | DeepOnto offline | result of `onto.get_asserted_complex_classes()` after rendering with placeholder fillers |
| `verbal_template` | DeepOnto offline | result of `OntologyVerbaliser.verbalise_class_expression()` with slot fillers as variables |
| `mean_verbal_length` | DeepOnto offline | character length used by R_C |
| `bfo_anchor_path` | catalog metadata | `rdfs:subClassOf` chain to BFO upper class |
| `provenance` | catalog metadata | author, date, gate-version |

**Construction (P1 deliverable).**

1. Author or generate ~500 candidate templates spanning the OWL 2
   axiom shape inventory (subclass, equivalent-with-intersection,
   existential, universal, cardinality, owl-thing-anchored, etc.).
2. Spawn one JVM, instantiate each template with placeholder
   fillers, run DeepOnto's `Ontology` loader, `get_asserted_complex_classes()`,
   and `OntologyVerbaliser`. Record `is_complex`, `verbal_template`,
   `mean_verbal_length`.
3. Drop templates that fail to load, fail to verbalize, or whose
   verbalization is shorter than 5 chars.
4. Generate the structural-shuffle null distribution (200
   shuffles) of compositions over *C* and cache the per-shuffle
   `complex_count` and `R_D` values. τ_B and R_D's null calibration
   live in *C*'s metadata.
5. Commit *C* as a versioned artifact. *(As built, the catalog passed
   through seven hand-authored family JSON files
   (`01_foundation` … `07_long_tail`), since retired (2026-07-07);
   the live catalog is the single derived
   `src/aegir/ontology/catalog/catalog.json`, accreted by the
   derive→promote loop — not the `C-v0.1.{json,sqlite}` file this
   brief originally proposed. The null statistics live in
   `null_stats_canonical.json` and the frozen v0.3-era topic model in
   `T_I_canonical.pkl`.)*

After P1, the JVM is gone. The RL loop, verifier, and any v3
pretraining pipeline consume *C* by lookup.

**Implications and trade-offs.**

- **Pro: the policy's outputs are well-formed by construction.**
  Slot-typed composition cannot produce OWL that fails to load; R_A
  becomes a structural type-check rather than a parser exception.
- **Pro: per-step verifier cost drops by ~1–2 orders of magnitude.**
  No JVM init (~5 s amortized over a batch becomes 0). No DeepOnto
  parse per generation. RL training throughput improves
  proportionally; ablation runs become cheaper.
- **Pro: pretraining-pipeline cleanliness.** v3 verbalization slices
  drawn from *C*-compositions inherit no Java dependency; the v3
  data path is pure Python + Rust (HF tokenizers, fla kernels).
- **Con: bounded expressivity.** The policy can only compose what
  *C* contains. Novel axiom shapes the catalog doesn't cover are
  unreachable by the policy. Mitigation: catalog coverage is itself
  a methodological knob; ablations on catalog size establish the
  expressivity / efficiency frontier. Authoring genuinely novel
  axiom shapes remains a human-baseline activity.
- **Con: R_C signal is degraded.** Runtime "does it verbalize" is
  trivially true; we repurpose R_C as a length proxy, which is a
  weaker signal than gaius's pass/fail. The brief acknowledges
  this and tests in P2 whether R_C's reweighted form retains
  discriminative validity.
- **Con: discrimination claim (C1) needs care.** Known-bad
  ontologies for the verifier-validation test set must be
  constructable within the catalog (e.g., compositions of only
  trivial templates). Out-of-catalog "bad" ontologies don't test
  the runtime verifier; they test the catalog-construction step.
  P2 explicitly distinguishes these regimes.

This pivot is the single most important architectural change in
v0.3 vs. v0.2. Subsequent sections assume it.

### Bespoke ontology authorship — human baseline

The project author produces a baseline `aegir-vocab.ttl` against the
same structural commitments specified earlier:

- ≥ 50 named classes, of which ≥ 25 sit at depth ≥ 3 in `rdfs:subClassOf`
- ≥ 15 complex asserted classes (existential / universal /
  cardinality / boolean intersections; ≥ 3 `owl:equivalentClass`
  with non-trivial Manchester-syntax bodies)
- BFO 2020 ancestry on every leaf, mediated through CCO
- `rdfs:label` and `skos:definition` on every term
- All authorship is the project's own; no content lifted from any
  non-public reference set

These thresholds are picked-by-convention for the human baseline
target. They are *not* the brief's structural claim about ontologies
in general; they are a target the human author aims for, against
which the RL-trained policy is compared.

### Verbalization corpus *V(O)* — from catalog lookup

For each composition *O* = compose(*C*, σ):

1. For each (template_id, σ_i) pair in *O*, look up the template's
   `verbal_template` in *C* and substitute the slot fillers σ_i to
   produce the rendered verbalization sentence.
2. Filter: drop empty/degenerate substitutions (slot filler
   produces a 0-length string after substitution).
3. Deduplicate by sentence-embedding cosine similarity > 0.95 (same
   encoder as the topic model, to avoid distributional shift
   between dedup and topic fitting).
4. Two configurations tested in ablation: **plain** (each
   substituted verbalization as one document) and **templated**
   (substituted verbalization paired with its immediate sub/super
   class templates' verbalizations as adjacent sentences, found by
   walking the composition's subClassOf graph).

Plain is the headline configuration; templated is ablation only.
Verbalization corpus size per ontology is bounded by composition
size and is fully predictable from *C*.

### Null distributions — properly constructed

The v0.1 brief described Gate D's null as "shuffling word-topic
assignments." That construction was incoherent — it perturbs the
topic-model fit, not the ontology. v0.2 replaced it with a
structural shuffle of arbitrary OWL. v0.3 specializes the structural
shuffle to *catalog compositions* and lifts the entire computation
into P1a (offline catalog construction), so it does not appear in
the runtime verifier path.

**Null for R_B.** A null composition is generated by drawing
catalog templates uniformly at random (preserving count) and
filling slots with uniformly-sampled fillers from the typed term
inventory. This preserves the structural shape (template-count,
axiom-kind distribution) while destroying any topic-aligned
selection signal. τ_B = 95th percentile of complex_count over 200
null compositions, computed once per catalog version.

**Null for R_D.** Same null composition as R_B, then render the
null verbalization corpus from the cached `verbal_template`s, fit
BERTopic on it, compute alignment against the frozen *T_I*. R_D is
normalized so that null-mean alignment maps to 0 and observed
best-case (human-baseline + 2σ headroom) maps to 1. 200 null
compositions per catalog version; cost is amortized because *T_I*
is fitted once per *I* version and the catalog templates'
verbalizations are pre-cached.

All null statistics are stored as catalog metadata in
`catalog/C-v0.1.{json,sqlite}`. The runtime verifier reads τ_B and
the R_D normalization constants from the catalog; it does not
re-run the null construction.

### RL policy and training loop

**Base policy.** *(Superseded — see the [authoritative
reference](./production_state.md):
the operational policy is now `Qwen3.5-9B-Base` with a held-out
`SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` residual-stream adapter, sized
to the 6×4090 envelope. The 27B design below is the brief's original
target and the rationale for it still holds at the smaller scale.)*
`SAE-Res-Qwen3.5-27B-W80K-L0_100` (instruct
variant), a Qwen 3.5 27B base with a residual-stream sparse
autoencoder of width 80K and average L0 ≈ 100 active features per
token. Two reasons for this choice:

1. **Capacity.** A 27B instruct model handles structured-syntax
   composition (catalog template selection + typed slot-fill) more
   reliably than a 7B model, especially with the controlled output
   space the catalog imposes.
2. **Interpretability dividend.** SAE residual-stream features make
   the policy's internal reasoning inspectable. At P5 we log SAE
   feature activations during generation; at P6 the comparison
   study analyzes which features differentiate gate-passing from
   gate-failing generations. This is a methodological enhancement
   that the brief earns "for free" by selecting an SAE-equipped
   base — vanilla 7B models do not offer this surface.

**Parameter strategy.** LoRA adapters on the base; SAE weights
frozen and read-only (the SAE provides interpretability, not
training signal). Full fine-tune of a 27B base is infeasible on
6×4090; LoRA + sharded weights (FSDP or tensor-parallel) is the
realistic envelope.

**Memory envelope.** 27B × 2 bytes (bf16) = 54 GB weights. Sharded
across 6×4090 (24 GB each, 144 GB aggregate): ~9 GB per GPU for
weights, leaving ~15 GB per card for KV cache, activations, LoRA
optimizer state, and group-size-8 generation buffers. Context
window is constrained for RL training to ~4–8K tokens (ontology
compositions don't need 80K) to keep memory headroom; the SAE's
80K width is a representation-space property, not a context
constraint.

**RL algorithm.** GRPO (Group Relative Policy Optimization, per
DeepSeek-R1 / DeepSeekMath) over groups of 8 samples per prompt
(reduce to 4 if memory pressures during P4 smoke test). Reward is
*R(O_i)* computed from the policy's catalog-composed output.
Critic-free; advantage is group-relative.

**Prompt design.** Each prompt is a `(domain_seed, structural_constraint)`
pair — a short natural-language description of the ontology's
intended scope, plus the structural commitments (class count,
depth, complex-class count) the policy is to satisfy. The policy
emits a structured composition: a sequence of
`(template_id, slot_fillers)` tuples, decoded into rendered OWL by
a deterministic post-processor. Domain seeds are drawn from a
held-out set so paper 1's evaluation is on unseen-during-training
prompt distribution.

**Training budget.** ~1000–3000 GRPO steps, group size 4–8,
~120–200 GPU-hours on 6×4090 with LoRA + tensor-parallel sharding.
The catalog-precompute pivot eliminates DeepOnto's per-step JVM
cost; the new dominant cost is generation throughput on the 27B
policy. P4's smoke test confirms the actual per-step wall clock
before P5 commits.

**Checkpointing.** Per-step *R* mean, per-step max-*R*, and
per-step SAE feature-activation summary statistics logged. Best-*R*
LoRA adapter and final-step LoRA adapter are kept. Both evaluated
separately at the P5 exit gate.

### Comparison study (P6)

Three policies generate ontologies from the same held-out prompt
set:

- **Random LLM (no RL):** Qwen2.5-7B-Instruct without any
  optimization.
- **Prompt-evolved (DSPy/GEPA):** following gaius's approach —
  evolve the prompt against *R* without weight updates.
- **GRPO-trained (this brief):** the policy from P5.

Each generates 100 ontologies on the held-out prompts. Mean *R*,
median *R*, and *R* distribution shape are reported per policy. C2
is evaluated by paired comparison over the same prompts — does the
GRPO policy score higher than each baseline at p < 0.05 under a
Wilcoxon signed-rank test?

The human-authored baseline is one ontology per author-week of
effort; it is reported as a single point on the *R* axis with
methodology-section discussion of why it is or is not exceeded.

## Phase structure with formal gates

Each phase carries an **entry gate** (preconditions) and an **exit
gate** (verification cases that must pass before progressing).
Failure at an exit gate halts forward progress until resolved or the
brief is revised.

### P0 — Literature review and positioning

- **Scope:** Review prior art across (a) ontology learning from text,
  (b) verbalization-augmented language modeling, (c) verifiable
  reward in language models, (d) data curation with verifiable
  signals, (e) topic-model evaluation of ontologies. Produce a
  positioning document at `docs/scratch/YYYY-MM-DD/HHMMSS_lit_review.md`.
- **Entry gate:** brief approved by project lead.
- **Exit gate:** positioning document explicitly states (i) what is
  done in prior art, with citations; (ii) what is open; (iii) the
  specific intersection this brief targets, with a "we are not
  aware of prior work that..." statement supported by the review.
  If novelty does not survive, brief is revised before P1.
- **Estimated effort:** ~2 weeks of focused reading.

### P1 — Catalog *C* construction + runtime verifier

- **Scope:** This is the largest engineering phase. Two sub-deliverables:
  - **P1a — Catalog construction (offline DeepOnto).** Author/generate
    ~500 candidate axiom templates spanning OWL 2 axiom shapes;
    spawn one JVM, run DeepOnto loadability + complex-class +
    verbalizer over each; drop failures; cache results to
    `catalog/C-v0.1.{json,sqlite}`. Compute structural-shuffle null
    distribution (200 shuffles) and cache τ_B, R_D normalization
    statistics.
  - **P1b — Runtime verifier (no JVM).** Implement
    `scripts/aegir-verify` reading from *C*: structural type-check
    for R_A, lookup-based R_B and R_C, BERTopic fit + Hungarian
    alignment for R_D. Implement aggregation. Lock *I* + topic
    model + catalog version.
- **Entry gate:** P0 exit passed.
- **Exit gate:**
  1. *C* contains ≥ 200 surviving templates spanning at least 5
     distinct axiom shapes (subclass, equivalent-with-intersection,
     existential, universal, cardinality).
  2. `aegir-verify` runs end-to-end with no Java/JVM dependency
     active.
  3. `aegir-verify <composition.json>` produces deterministic
     *R* ∈ [0, 1] hash-stable across 3 independent runs.
  4. Per-composition runtime ≤ 1 s on CPU (target informed by RL
     throughput budget).
  5. Corpus-size sweep on *I* shows BERTopic stability (silhouette
     score variance < 0.05).
- **Estimated effort:** ~3 weeks (catalog construction is the
  bottleneck — template authorship + DeepOnto validation pass
  takes 1.5 weeks, runtime verifier ~1 week, corpus sweep + locks
  ~0.5 week).

### P2 — Verifier validation (claim C1)

- **Scope:** Build a labeled test set of ontologies — known-good
  (BFO, OBO Foundry exemplars, hand-authored quality samples),
  known-bad (LLM-generated junk, syntactically valid but
  conceptually empty, structurally truncated). Compute *R* on each.
  Tune aggregation weights {a, b, c} to maximize AUC. Lock weights.
- **Entry gate:** P1 exit passed.
- **Exit gate:**
  1. Labeled test set ≥ 30 ontologies, ≥ 10 known-good, ≥ 10
     known-bad.
  2. AUC of *R* against label ≥ 0.85.
  3. Mean(R | good) − Mean(R | bad) ≥ 0.30.
  4. Weights {a, b, c} locked and committed to the verifier as a
     constant; re-running the verifier reproduces the AUC.
- **Estimated effort:** ~2 weeks (test set construction is the
  bottleneck).
- **Failure mode:** if AUC < 0.85, the verifier does not discriminate
  enough to be useful as an RL reward. Diagnose which component is
  weakest and revise (most likely Gate D is too noisy or Gate B's
  threshold is mis-calibrated). Iterate before P3.

### P3 — Human-authored baseline ontology

- **Scope:** Project author produces `aegir-vocab.ttl` against the
  structural commitments. The author may use the catalog *C* as a
  drafting tool (selecting + slot-filling templates) or compose
  free OWL outside the catalog; the latter is recorded so that the
  comparison study (P6) can fairly compare *catalog-bound* policy
  outputs against *catalog-free* human authorship. Iterate against
  the verifier until R(human-authored) ≥ 0.70 (sanity).
- **Entry gate:** P2 exit passed.
- **Exit gate:**
  1. `aegir-vocab.ttl` parses, satisfies the structural commitments
     mechanically.
  2. R(aegir-vocab.ttl) ≥ 0.70 against locked verifier — note that
     scoring a free-OWL human-authored ontology requires the catalog
     to be expressive enough to *encode* the human author's axiom
     shapes for verifier purposes; mapping from free OWL to catalog
     compositions for scoring is a P3-internal subtask.
  3. Manual review confirms the ontology is genuinely the project's
     own work (per [Charter §Provenance discipline](./charter.md#provenance-discipline)).
- **Estimated effort:** ~4–8 weeks. **The brief acknowledges this is
  the largest creative effort and the hardest to bound.** Two-week
  estimates from v0.1 are dropped.

### P4 — RL infrastructure smoke test

- **Scope:** Stand up the GRPO loop with Qwen2.5-7B + LoRA. Verify
  the loop converges on a trivial reward (R_trivial = "output
  contains the string `Class:`") within 50 steps. Verify GPU
  budget per step matches estimates.
- **Entry gate:** P3 exit passed.
- **Exit gate:**
  1. Trivial-reward training reaches mean R_trivial = 1.0 within
     50 steps.
  2. Per-step wall clock and memory profile within 2× of the
     budget estimate.
  3. Checkpoint save/load round-trips cleanly.
- **Estimated effort:** ~1 week.

### P5 — RL training run (claim C2)

- **Scope:** Train π_θ (LoRA over `SAE-Res-Qwen3.5-27B-W80K-L0_100`)
  via GRPO against locked *R* on the held-out prompt training set.
  Log per-step mean *R*, max *R*, gate-pass rates per component,
  and SAE feature-activation summary statistics. Save best-*R* and
  final-step LoRA adapters.
- **Entry gate:** P4 exit passed.
- **Exit gate:**
  1. Training completes within budget (≤ 200 GPU-hours total on
     6×4090; ~3× v0.2's 72-hour budget to reflect 27B vs. 7B).
  2. Best-*R* checkpoint produces R(π_θ) > R(human-authored
     baseline) on a 50-prompt held-out evaluation set.
  3. Per-component gate-pass rates ≥ 80% on held-out evaluation
     set (i.e., the policy isn't exploiting one component while
     ignoring the others).
  4. SAE feature-activation logs collected; ready for P6 analysis.
- **Failure mode:** if the policy fails to exceed human baseline,
  diagnose (under-trained, reward sparsity, prompt-set distribution
  too narrow, catalog expressivity bound). Re-train or revise
  reward composition.

### P6 — Comparison study (claim C2 finalized) and paper 1 writeup

- **Scope:** Generate 100 ontologies each from random-LLM
  (`SAE-Res-Qwen3.5-27B...` no RL), prompt-evolved
  (DSPy/GEPA over the same model), and GRPO-trained (this brief)
  policies on 100 held-out prompts. Compute *R* per ontology.
  Wilcoxon signed-rank pairwise tests. Effect-size estimation.
  Per-component analysis. SAE feature-attribution analysis: which
  features differentiate gate-passing from gate-failing
  generations? Draft paper 1.
- **Entry gate:** P5 exit passed.
- **Exit gate:**
  1. GRPO mean *R* > prompt-evolved mean *R* at p < 0.05.
  2. GRPO mean *R* > random-LLM mean *R* at p < 0.05.
  3. Per-component analysis isolates *which* gates the GRPO policy
     improves on (informs C3 and paper 2's ablations).
  4. SAE feature-attribution analysis identifies ≥ 5 features whose
     activation differs significantly between high-*R* and low-*R*
     generations (interpretability methodological enhancement).
  5. Paper 1 draft circulated for internal review.
- **Estimated effort:** ~3 weeks.

### P7 — Verbalization corpus + v3 pretrain (paper 2 begins)

- **Scope:** Generate verbalization corpora from (a) human-authored
  baseline, (b) prompt-evolved best, (c) GRPO best. Run v3
  pretraining at the same weight (0.05 of corpus mix) for each
  configuration plus a v2-replication baseline. Each run is the
  full v2 schedule (~10 GPU-hours). Add a new `eval.ontology-recall`
  slice fit on held-out verbalizations.
- **Entry gate:** P6 exit passed.
- **Exit gate:**
  1. Four pretrain runs complete; metrics + stratified eval
     committed.
  2. Comparison table: v2 vs. v3-{human, evolved, GRPO} on the
     full stratified eval surface.
- **Estimated effort:** ~2 weeks (mostly unattended training).

### P8 — Pretraining ablations and component validity (claim C3)

- **Scope:** For the best-performing v3 configuration from P7,
  ablate each verifier component: re-train the policy with R'
  that drops one component at a time, regenerate verbalization
  corpus, re-pretrain. Identify which gates carry predictive
  validity for downstream lift.
- **Entry gate:** P7 exit passed *and* at least one v3 configuration
  shows ≥ 0.10 bpb lift on `eval.ontology-recall` over v2.
  (If no configuration shows lift, paper 2's contribution is a
  bounded negative result; ablations refocus on understanding
  why.)
- **Exit gate:**
  1. Ablation table per gate.
  2. Statistical test on whether each gate's removal causes
     significant degradation.
  3. C3 statement is empirically supported or empirically refuted.
- **Estimated effort:** ~6 weeks (multiple pretrain runs).

### P9 — Paper 2 writeup

- **Scope:** Draft paper 2 covering P7 + P8 results.
- **Entry gate:** P8 exit passed.
- **Exit gate:** paper 2 draft circulated.

## Resource budget summary

| Resource | P0–P6 (paper 1) | P7–P9 (paper 2) | Total |
|---|---|---|---|
| GPU-hours (training) | ~200 (P5 RLVR with 27B policy) | ~50 (4 pretrains) + ~150 (8 ablation pretrains) = ~200 | ~400 |
| GPU-hours (catalog construction, P1a) | ~30 (DeepOnto pass over ~500 templates; CPU-bound in practice) | — | ~30 |
| Wall-clock | ~16–20 weeks | ~10–12 weeks | ~7 months optimistic, ~10 realistic |
| Author-weeks | ~12 (P0 lit review + P3 ontology authorship dominate; P1a catalog authoring adds ~2) | ~5 | ~17 |
| Paper outputs | 1 | 1 | 2 |

The brief's largest single risk is P3 (human-authored ontology). A
clean, gate-passing, BFO-anchored, project-domain ontology of the
required structural shape may consume 6–8 weeks; it can also stall
if the project domain doesn't have a natural ontology shape.
Mitigation: choose the project domain in P0 for ontology
tractability, not just for downstream-task alignment.

The catalog-construction pivot (v0.3) shifted the engineering
profile: P1 grew (~3 weeks vs. v0.2's 2) because catalog authoring
+ DeepOnto pass is now in scope, but P5 became more expensive
(27B vs. 7B base policy, ~200 vs. ~75 GPU-hours) and the v3
pretraining and inference paths are now JVM-free. Net engineering
cost is comparable; runtime production characteristics are
substantially better.

## What this brief does not commit to

- A specific ontology domain. Chosen during P0 against literature
  review and authorial expertise.
- A specific RL algorithm beyond "GRPO-family." If GRPO underperforms
  in P4, PPO or even REINFORCE with baseline are fallbacks.
- A specific policy model beyond `SAE-Res-Qwen3.5-27B-W80K-L0_100`.
  If memory pressure during P4 forces a smaller model, downsizing to
  a 7B SAE-equipped variant (or a non-SAE 7B with the
  interpretability dividend dropped) is permitted with a documented
  rationale. The brief's headline claims do not depend on the SAE
  surface — they depend on GRPO with a deterministic verifier.
- Catalog *C* size beyond "≥ 200 surviving templates spanning ≥ 5
  axiom shapes." Larger catalogs improve expressivity at the cost
  of P1a authorship time.
- Paper-2 success. Paper 2 stands or falls on P7 + P8 results;
  paper 1 stands independently on P6.

## Risks and bounding negative results

- **C1 fails (verifier doesn't discriminate).** Diagnose at P2;
  iterate on aggregation weights or component definitions before
  P3. Worst case: the four-gate framing is insufficient and the
  brief is revised.
- **C2 fails (RL doesn't beat baselines).** Diagnose at P5; revise
  RL infrastructure or reward shaping. If paper 1's headline does
  not land, the lit-review-bounded claim "GRPO with this verifier
  produces comparable but not superior ontologies to prompt
  evolution" is still publishable as a negative result, with
  diagnostic value for the field.
- **C3 fails (no pretraining lift).** Paper 2 reports the bounded
  negative result: a verifier with strong discrimination and an RL
  loop that maximizes it does *not*, on this dataset, produce
  ontologies whose verbalizations measurably help byte-level
  pretraining. This is publishable as a bound on RLVR's reach.
- **Verbalizer brittleness.** DeepOnto's `OntologyVerbaliser` is
  template-driven; the policy may exploit verbalizer-friendly axiom
  patterns at the expense of semantic depth. Mitigation: P2's
  test set includes ontologies with diverse axiom shapes; AUC
  computation surfaces verbalizer exploitation.
- **Topic-model brittleness.** BERTopic on small corpora is unstable
  in HDBSCAN clustering. Mitigation: P1 includes corpus-size
  sensitivity sweep; if instability is intractable, NMF becomes
  primary and BERTopic ablation.
- **Provenance drift.** As the policy is trained against the human
  baseline + held-out prompt distribution, it may converge toward
  generating ontologies that read as derivative of the baseline.
  The Charter's [Provenance discipline](./charter.md#provenance-discipline)
  applies to *all* ontologies entering the project artifact bundle;
  policy outputs that are accepted into `aegir-vocab.ttl` (vs.
  remaining in the policy's evaluation set) go through the same
  PR review.
- **Author-week budget overrun.** P3 dominates; if it stalls, paper 1
  cannot complete. Mitigation: P3's "≥ 0.70" threshold can be relaxed
  if the bottleneck is verifier-strictness rather than
  authorship quality (revisit P2 weights).

## References (committed for P0 lit review)

The P0 lit review is *committed* — these are the starting set, not
exhaustive.

**Verifiable reward in language models:**

- Lambert, N., et al. (2024). *Tülu 3: Pushing frontiers in open
  language model post-training.*
- DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning
  capability in LLMs via reinforcement learning.* (GRPO original.)
- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the limits of
  mathematical reasoning in open language models.* (GRPO algorithm.)

**Ontology engineering with deep learning:**

- He, Y., Chen, J., Antonyrajah, D., Horrocks, I. (2023). *DeepOnto:
  A Python package for ontology engineering with deep learning.*
- Auer, S., et al. (2023). *SciQA: A Scientific Question Answering
  Benchmark for Scholarly Knowledge.* (Ontology-grounded LM
  evaluation.)

**Verbalization and language modeling:**

- Petroni, F., et al. (2019). *Language models as knowledge bases?*
  (LAMA probing.)
- Logan, R., et al. (2019). *Barack's wife Hillary: Using knowledge
  graphs for fact-aware language modeling.* (KGLM.)
- Zhang, Z., et al. (2019). *ERNIE: Enhanced language representation
  with informative entities.*
- Wang, X., et al. (2021). *KEPLER: A unified model for knowledge
  embedding and pre-trained language representation.*

**Data curation with verifiable signals:**

- Albalak, A., et al. (2023). *A survey on data selection for
  language models.*
- Penedo, G., et al. (2024). *FineWeb: Decanting the web for the
  finest text data at scale.*

**Topic modeling:**

- Blei, D., Ng, A., Jordan, M. (2003). *Latent Dirichlet allocation.*
  (Legacy reference; not the headline method.)
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure.* (Headline method.)
- Lee, D., Seung, H. (1999). *Learning the parts of objects by
  non-negative matrix factorization.* (NMF baseline.)

**Ontological foundations:**

- Arp, R., Smith, B., Spear, A. (2015). *Building ontologies with
  Basic Formal Ontology.* MIT Press.
- Common Core Ontologies (CCO). github.com/CommonCoreOntology/CommonCoreOntologies.

**Aegir / sibling project internal:**

- [Charter](./charter.md) — outward contract and provenance
  discipline.
- [Migration](./migration.md) — vocabulary authorship that produces
  the human baseline ontology consumed at P3.
- [Training Regime §10](../training_regime.md#10-v2-mixed-corpus-pretrain-2026-04-27)
  — v2 baseline against which paper 2's lift is measured.
- Sibling project Gaius: `scripts/ontology_denovo_pipeline.py` and
  `scripts/validate_generated_ontology.py` — gates A–C reference
  implementation; gaius's gate D is stub-only, replaced by this
  brief.

## Status

- **v0.1** — superseded; named four gates but framed as "RLVR" while
  describing static validation; mixed two contribution claims; null
  distribution incoherent; LDA as default; 1–2 week ontology
  authorship estimate unrealistic.
- **v0.2** — superseded; locked contribution to verifier *R* + GRPO
  loop; split into two papers; replaced null distribution with
  structural-shuffle null; promoted BERTopic; revised P3 estimate
  to 4–8 weeks; committed literature review as P0. Used Qwen2.5-7B
  as policy and assumed live DeepOnto in the verifier.
- **v0.3** — superseded; introduced procedural catalog and SAE-Res-Qwen
  policy, but did not yet incorporate the P0 lit review's
  adjacent-work differentiations.
- **v0.4** — superseded; added KELM / OntoTune / Zaitoun et al.
  differentiation paragraphs from v0 lit review.
- **v0.5** — this document. Incorporates v1 P0 lit review findings
  (`docs/scratch/2026-05-09/225830_lit_review_v1.md`): three new
  must-cite differentiations (OLLM, AutoGraph-R1, K2V) plus OnT
  for the TBox-embedding axis; a new "Load-bearing novelty"
  section that sharpens the contribution claim to "the OWL
  artifact is the only graph-structured output where the verifier
  can run a sound-and-complete reasoner producing both structural
  and semantic verdicts"; positions the brief as the synthesis of
  four precursor recipes (graph-output RL, schema-validator RL,
  code/SQL execution RL, verbalization-corpus pretraining) onto
  OWL where the verifier acquires DL deductive semantics. P0 exit
  gate firmly green. Two architectural changes vs. v0.2:
  1. **Procedural catalog *C*.** DeepOnto runs offline at catalog
     construction time only. The runtime verifier and the v3
     pretraining/inference paths have no JVM, no DeepOnto, no
     Java dependencies. Per-step verifier cost drops by 1–2
     orders of magnitude. Trade-off: bounded expressivity (policy
     can only compose what *C* contains) and weaker R_C signal.
  2. **Policy upgraded** to `SAE-Res-Qwen3.5-27B-W80K-L0_100`
     (instruct). Capacity for structured-syntax composition; SAE
     residual-stream features add interpretability surface for
     P5/P6 analysis. Memory envelope tighter; GPU-hour budget for
     P5 grows from ~75 to ~200.
- **v0.3 → v1.0** transition: contingent on P0 exit gate (positioning
  doc finalized). Until then this brief is provisional and the
  contribution claim is subject to revision based on lit-review
  findings.

Updates track in `docs/scratch/YYYY-MM-DD/` session notes.
