# Cross-pollination: TAPAS lineage × BERTopic ontology curation × textbook-quality synthesis

**Date:** 2026-05-17
**Status:** v0.3+ direction memo. Out of v0.2 scope. Authoritative
sketch of how the abandoned TAPAS-lineage methodology, the BERTopic
ontology-driven curation pipeline, and Vik Paruchuri's
[textbook_quality](https://github.com/VikParuchuri/textbook_quality)
recipe converge on a single high-leverage synthetic data program.

## The shared shape

Three pipelines that look unrelated on the surface share the same
underlying architecture:

```
  structured input  →  combinatorial synthetic generation  →  deterministic verifier  →  training signal
```

|                          | Structured input          | Synthetic generator              | Verifier                                | Output                          |
|---|---|---|---|---|
| **TAPAS intermediate-pretrain** | Wikipedia tables + text  | Random SQL templates / entity swaps | Column membership check / substring match | Binary True/False labels        |
| **TAPEX execution-trace pretrain** | Wikipedia tables          | Random SQL queries              | SQLite executor                          | (query, table) → executed answer |
| **BERTopic ontology curation (current)** | OWL ontology compositions | NL verbalizations                | BERTopic alignment with T_I reference   | R = R_A · R_M · R_C reward      |
| **textbook_quality (Vik)** | Topic seed             | LLM + web retrieval             | implicit (LLM judgment)                  | Long-form prose corpus          |
| **textbook_quality + ontology (proposed)** | Ontology graph        | LLM + ontology retrieval        | Ontology consistency check               | Textbook-quality + tables + Q&A |

The leverage point is that **all four are interchangeable on the
generator side** — only the verifier changes. The TAPAS lineage spent
five years optimizing the generator side; our ontology gives us a
verifier that's stronger than any of theirs.

## Cross-pollination targets, ranked by leverage

### 1. Ontology-grounded textbook synthesis (highest)

**The new vision.** Don't just generate tables; generate
textbook-quality prose with tables, figures, and Q&A pairs embedded —
all grounded in the ontology. textbook_quality demonstrates the
LLM+retrieval generation pipeline works; we *replace web retrieval
with deterministic ontology retrieval*.

Pipeline shape (parallel to textbook_quality's
topic_generator → topic_augmentor → book_generator):

```
ontology_topic_picker  →  outline_generator  →  chapter_generator  →  embedded_artifact_generator  →  verifier
       │                       │                      │                          │                       │
   (sample N        (LLM-generate         (LLM-generate prose         (LLM-generate             (ontology-
   concepts from   chapter outlines        with retrieval of           tables / figures        consistency
   ontology graph) anchored on each       relevant axioms,             grounded in the         check; reject
                   concept, by depth)     hierarchies, examples        cited axioms)           hallucinations)
                                          from the ontology)
```

What gets generated per "textbook":
- **Prose chapters** explaining ontological concepts in natural language — every named entity is an ontology IRI, every assertion is an axiom from the ontology.
- **Embedded tables** — extension queries over the ontology rendered as tables. e.g., "all subclasses of `bfo:Continuant` along with their direct parents and labels" → an extension table.
- **Figure captions** — text descriptions of class hierarchy diagrams, property graphs, axiom decompositions. Even if we don't render the figures, the captions are training-grade text grounded in real structural facts.
- **Q&A pairs** — TAPEX-style, but the executor is the OWL reasoner (HermiT, Pellet, or ELK). Sample a SPARQL query, execute, format as Q&A.

Why this dominates textbook_quality alone:
- **Zero hallucination by construction.** Every claim is verifier-checkable against the ontology. Hallucinations are not just detectable but auto-rejectable.
- **Combinatorial scale.** A 1000-class ontology with 50 properties admits ~10^5 single-axiom statements; multi-step reasoning expands that into the billions. Pretraining-scale supervision from a single-author ontology.
- **Multi-modal in disguise.** Tables, figures, prose, Q&A — all the input modes a model needs to learn — emerge from one ontology source.

Verifier signals available at each stage:
- Topic sampling: ontology-graph-traversal-cost (no LLM needed).
- Outline: each section title must correspond to an ontology concept; reject otherwise.
- Chapter prose: every named entity must IRI-resolve; every quantitative claim (subclass count, etc.) must execute against the ontology.
- Embedded tables: must be a valid extension query result.
- Q&A: SPARQL-executable; gold answer is the executor output.

### 2. TAPEX-style SPARQL execution traces

The natural transposition of TAPEX onto ontology data: generate
SPARQL queries against the ontology graph, execute deterministically,
train the model to autoregress query+answer.

For the model: it learns to predict structured ontology query
answers given the surrounding context. This is **the** missing piece
for ontology-aware downstream tasks — even more fundamental than the
textbook synthesis above, because it's the building block on which
multi-step reasoning depends.

Concrete shape:
- Sample subset of the ontology (5-10 classes, their properties, a few axioms).
- Sample a SPARQL query template (1-hop: `subClassOf`; 2-hop: transitive closure; aggregation: count of instances; existence: ASK queries).
- Render to natural-language ("How many direct subclasses does X have?") and SPARQL form.
- Execute via HermiT/Pellet → gold answer.
- Train: `<ontology fragment> [SEP] <question> [SEP] <answer>` autoregressive.

Aligns with `train_pretrain.py`'s existing LM head — same scaffold as
Phase 0.5 synth-SQL but with a real executor instead of pattern-match.

### 3. Counterfactual axiom-swap

TAPAS's counterfactual modality, transposed. For each ontology
composition with a verbalization, generate a corrupted version by
swapping one slot (Class / ObjectProperty / Individual) for another
of the same OWL type from elsewhere in the ontology. Train a
discriminator (or use as reward signal) to detect the corruption.

Crucial property: the corrupted statement should be **detectably
wrong by ontology lookup** but **lexically similar** to the original.
The model is forced to learn structural consistency rather than
surface patterns.

Mapping onto BERTopic curation: corrupted verbalizations should
score *much lower* on T_I-alignment than originals. This gives a
free hard-negative pool for contrastive training of the verifier
itself, or as discriminator-warmup data before GRPO.

### 4. TableFormer-style order-invariance augmentation

For ontology compositions, the analog of TableFormer's row/column
permutation: same axiom rendered in different syntactic orderings
should produce equivalent verbalizations.

  `Class: X SubClassOf: p some Y`
  `X SubClassOf: p some Y`
  `X ⊑ ∃p.Y`

All express the same axiom. Train the model to produce equivalent
verbalizations under permutation by augmenting the SFT corpus with
order-permuted forms, and reward verbalization equivalence (via T_V
cosine similarity above some threshold) under permutation.

Marginal but consistent gain on robustness splits — analogous to
TableFormer's 4-6% retention against perturbed tables on TAPAS.

### 5. Intermediate-pretrain as RL warmup curriculum

TAPAS's three-stage pipeline (MLM → intermediate-pretrain →
fine-tune) maps onto our SFT pipeline as:

  SFT-r0 (base)  →  discriminator-warmup-r0.5  →  GRPO-r1

The discriminator-warmup phase trains the model on the
counterfactual binary classification task before introducing the
multi-component R reward. This bootstraps the model's "what looks
right" intuition under dense supervision, before the sparse-reward
GRPO regime has to drive it.

Cheap to implement: reuse the counterfactual generator from §3, train
2-3 epochs with simple CE loss on the corrupted/original binary
target, then resume GRPO from that checkpoint.

### 6. ReasTAP-style multi-step chained-axiom reasoning

Generate compositions that require multi-hop ontology traversal:

  `Given: A ⊑ B, B ⊑ C; conclude: A ⊑ C`
  `Given: a:Person, hasParent(a, b), hasParent(b, c); conclude: hasGrandparent(a, c)`

Train the model on chain-of-reasoning verbalizations: not just the
final axiom, but each intermediate step articulated. This is what
ontologies *uniquely* enable that natural-language pretraining
doesn't — verifiable multi-hop deduction.

Defer until after §2 (SPARQL execution traces) lands; ReasTAP
extends TAPEX, not replaces it.

## v0.3 roadmap stitched together

Putting everything in order:

```
v0.2 (in flight)
  ├── Phase 0:   TAPAS Wikipedia table-text LM pretrain
  ├── Phase 0.5: TAPAS-style synth-SQL + counterfactual on tables
  └── Phase 1:   SOTAB-CTA fine-tune

v0.3 (next)
  ├── Phase 0.7: SPARQL-execution-trace pretrain (§2)
  ├── Phase 0.8: textbook-quality ontology corpus (§1)
  │              [largest single asset; subsumes §3-§6 as sub-modes]
  └── Phase 1':  WTQ + ontology-Q&A fine-tunes

v0.4 (after numbers come in)
  ├── §4 order-invariance augmentation if robustness splits weak
  ├── §5 discriminator-warmup-r0.5 curriculum for GRPO efficiency
  └── §6 multi-step chained-axiom for ontology multi-hop reasoning
```

The textbook-quality ontology corpus (Phase 0.8) is the headline asset.
Everything else feeds into it or follows from it. The corpus has these
properties that no existing artifact has:

1. **Pretraining scale** (potentially 10B+ tokens at full ontology
   exercise).
2. **Zero-hallucination** (deterministic ontology verifier).
3. **Multi-modal in text form** (prose + tables + Q&A + figure
   captions, all consistent).
4. **Domain-specific by construction** (the ontology is the domain).
5. **Re-usable** (publish as an HF dataset; others can train on it).

## Sub-questions to resolve before Phase 0.8 starts

These need design decisions before we can scope the corpus build:

- **Which ontology(s)?** The current Aegir ontology charter implies
  BFO + CCO + domain-specific. For a publishable corpus, would we
  want to base on a public ontology (DBpedia, Wikidata, schema.org)
  for replicability? Or commit to BFO-rooted domain ontologies for
  technical depth?
- **What's the LLM generator?** textbook_quality assumes GPT-3.5/4
  via OpenAI. We have local SFT-rN checkpoints (Qwen3.5-9B based).
  Cost trade-off: GPT-4 produces higher-quality prose per generation
  but locks us into rate limits + cost; local generation is free at
  scale but quality-bounded by SFT-r2's current state. **Likely
  answer: local generation with explicit ontology grounding makes
  GPT-4 less load-bearing than for textbook_quality, because the
  *substance* is guaranteed correct regardless of LLM eloquence.**
- **What's the executor?** HermiT/Pellet/ELK for OWL reasoning;
  RDFLib + SPARQL endpoint for queries. All open-source, all stable.
- **What's the byte-level serialization for the corpus?** Need to
  decide whether to keep tables / figures separable in the parquet
  schema (for selective ablations) or fully serialized as bytes
  (simplest for the existing `train_pretrain.py`).

## Action items for the next compute window

While Phase 0 + 0.5 + 1 ship for v0.2:

- [ ] Write a textbook_quality fork or adapter that targets
      our ontology graph as the retrieval source (replacing Serply/SerpAPI).
- [ ] Wire a HermiT/Pellet executor into the proto-converter
      pipeline (parallel infra to `tapas_proto_to_aegir.py`).
- [ ] Scope a single-ontology proof-of-concept: take one of our
      Manchester-template ontologies, generate ~1MB of textbook prose
      via local LLM + ontology retrieval, manually inspect quality.
- [ ] Pre-register the eval target: WTQ-equivalent for ontology Q&A
      (existing ontology benchmarks: LLMs4OL track, OAEI tasks).

## The bigger picture

Vik Paruchuri's textbook_quality showed that a small team can
generate publication-grade pretraining data with the right pipeline.
The TAPAS lineage showed that specialized table understanding has
deeper untapped methodology than the leaderboard chasers ever
explored. Your BERTopic ontology curation pipeline gives us a
verifier with structural properties neither has access to.

Putting them together: a synthetic data corpus where every byte is
verifiable, every table is consistent with surrounding prose, every
Q&A pair is executor-grounded, every figure caption maps to a real
ontological structure — and the model is byte-level, recurrent, and
trains on a tinybox.

This is the v0.3 thesis. v0.2 finishing strong is the proof that
the underlying H-Net/RWKV substrate can support it.
