# Pretraining

This chapter describes the byte-level pretraining track of the
project. The operational pretraining run is the **v2 mixed-corpus
byte-level pretrain** completed 2026-04-27 — 122k training steps on
a 2 GB mixed corpus (FineWeb-Edu + SQaLe + SchemaPile + FinePDFs-lab),
next-byte-trained at the architecture described in
[Architecture](./architecture.md). The pretrain produced the
backbone that the M2 milestone of [Track 1](./roadmap.md) fine-tunes
for Column Type Annotation; the v2 result is the empirical anchor in
[Training Regime §10](./training_regime.md).

The chapter is organized in two parts: the operational pretraining
track that the v2 run instantiates, and the long-term direction in
which ontology-grounded synthetic data feeds successive pretraining
generations.

## The operational pretraining track

### Why byte-level

Standard pretraining uses subword tokenization (BPE, SentencePiece)
fit once on a pretraining corpus. Subword tokens fragment tabular
data unpredictably — a column value `"$1,234.56"` may tokenize as
five tokens or two depending on the corpus the BPE was fit on, and
the boundary between adjacent columns has no consistent
representation in the token stream. For column annotation, this
fragmentation is structurally harmful: the model has to re-learn
the boundary structure that the original CSV / JSON delimiters
expressed losslessly. Byte-level input avoids the question — every
byte is a primitive — at the cost of longer sequences.

Aegir's hierarchical architecture (H-Net dynamic chunking on top of
RWKV-7 time-mixing) makes byte-level training tractable: the
routing module learns where token-like boundaries should be from
the sequence's content, replacing fixed tokenization with a
content-adaptive operation that propagates through training. See
[Architecture](./architecture.md) for the recursive hierarchy and
[Hierarchical Dynamic Chunking](./architecture/chunking.md) for the
boundary mechanism.

### The v2 mixed-corpus pretrain (2026-04-27)

The v2 corpus is 2 GB of mixed text drawn from four sources:

| Source | Role |
|---|---|
| FineWeb-Edu | Curated educational prose; general language-modeling signal. |
| SQaLe | Natural-language → SQL pairs; structured-query reasoning. |
| SchemaPile | Database schemas with metadata; relational-syntax signal. |
| FinePDFs-lab | LIMS-domain PDF text; in-distribution signal for the metadata-tagging target. |

122k training steps were run at the `small` configuration (56M
parameters), single GPU, ≈10 h wall clock, with a cosine LR schedule
and AdamW. Stratified held-out evaluation against trained-time
matched slices shows the result the M2 milestone depends on:
non-degenerate representations across all four sources, ≈2 bpb
drops on the domain-targeted FinePDFs-lab slice relative to a
randomly-initialized baseline, and no regression on general prose.
The full bits-per-byte table is in [Training Regime
§10](./training_regime.md).

The v2 pretrain is the project's first real backbone. It
established that the architecture converges under byte-level
pretraining on real corpora — a precondition for any fine-tune work
and for any ontology-grounded synthetic regime beyond it. The
fine-tune that closes the M2 loop is described in [Supervised
Bootstrapping](./roadmap/supervised.md); the diagnostic that
motivated the v2 pretrain in the first place is in [Diagnostic Case
Study](./pretraining/diagnostic_case_study.md).

### v3 — multi-GPU step-up

The v3 pretrain is conditional on M2 clearing its liveness gate;
M3 of [Track 1](./roadmap.md) describes the planned step-up to
6 × RTX 4090 multi-GPU training at the next byte-budget bump
(roughly 8 GB at ≈7 h vs. v2's 10 h on 2 GB single-GPU). The target
evaluation thresholds — keep `eval.fineweb-held` ≤ 1.61, push
`eval.finepdfs-lab-held` below 1.78, no regression on SchemaPile or
SQaLe — anchor v3 against the v2 baseline. The v3 corpus mix may
incorporate verifier-passing synthetic slices from Track 2's RLVR
policy once that policy produces compositions at corpus scale; see
the next section.

## The long-term direction — ontology-grounded synthetic data

The byte-level pretraining track exists alongside a coupled research
program that generates structured training data from a deterministic
ontology rather than discovering structure in scraped corpora. The
shape of that program is described in detail in the [semantic-engine
authoritative reference](./ontology/production_state.md) — Figure 3.1
is the closed-loop pipeline diagram. The short version:

1. Verbalize OWL ontology compositions from the SDG procedural
   catalog (540 templates, 522 verbalized) into natural-language
   text via the cached DeepOnto verbalizations. The verbalizations
   are deterministic and reproducible from the locked catalog.
2. Score compositions via the four-component deterministic verifier
   *R(O, I)* (described in the [Ontology chapter](./ontology.md)),
   retaining only verifier-passing compositions.
3. Render synthetic relational tables from catalog compositions
   under known ontological provenance, so each column's source
   entity is known by construction.
4. Mix verbalizations and synthetic tables into a v3-or-later
   pretraining corpus alongside real text.
5. Evaluate the pretrain lift on the Track 1 stratified-eval surface
   to attribute any improvement to the ontology-grounded slice —
   the **paper 2** claim, scoped in [Roadmap](./roadmap.md).

The ontology-grounded synthetic-data direction has converged on the
procedural-catalog approach above. The current iteration of that
approach is the in-flight RLVR policy that produces OWL compositions
in Track 2 of the roadmap. Until that policy produces
verifier-passing compositions at corpus scale, the v3 pretrain works
against the v2 mix or a manually-curated extension of it; the
ontology-grounded slice is added at the point where Track 2's
outputs are operationally available.

### Why this scales

The bottleneck in conventional table annotation is human labeling.
The bottleneck in this synthetic regime is policy throughput — the
RLVR policy must produce verifier-passing compositions, and the
verifier must score them — which is embarrassingly parallel. The
multiplicative structure of the pipeline gives generous headroom:

| Stage | Multiplier | Source |
|---|---|---|
| Catalog templates | 540 | SDG catalog (Batches 1–7) |
| Slot-fill compositions | 10²–10⁴ per template | Combinatorial slot-fill space |
| Verbalization rendering | 1:1 with composition | Cached DeepOnto verbalizations |
| Verifier-passing fraction | constrained by R-threshold | Locked weights `{0.50, 0.05, 0.45}` |

The diversity of the training data is bounded by the SDG ontology's
expressivity — currently 540 templates with explicit cross-context
cousining across LIMS, governance, kernel observability, lineage,
and Dempster-Shafer belief structures. The ontology itself is a
versioned outward contract; growth happens through the Track 2
process documented in the [Ontology chapter](./ontology.md).

### How this connects to Aegir's three target tasks

The two pretraining inputs — real corpora (v2) and ontology-grounded
synthetic slices (v3 and beyond) — both serve the same three
downstream tasks:

- **Column Type Annotation (CTA).** Real-corpus pretraining gives
  general language and tabular-syntax signal; synthetic slices add
  per-column entity types under known provenance.
- **Column Property Annotation (CPA).** Cross-column relationships
  in real corpora are noisy; synthetic slices supply clean
  cross-column relations from the catalog's `sdg:*` property
  declarations.
- **Data Element Discovery.** Cross-table groupings under known
  ontological provenance are the synthetic regime's distinctive
  contribution — real corpora do not supply ground-truth data
  elements at scale.

The first two tasks are addressable from v2 alone. The third
benefits most directly from the synthetic regime and is the
strongest motivator for completing Track 2.

## Sub-pages

[Training Tactics](./pretraining/training_tactics.md) and the
five **Stage**-named sub-pages
([Stage 1: Ontology Extraction](./pretraining/ontology_extraction.md),
[Stage 2: Schema Projection](./pretraining/schema_projection.md),
[Stage 3: Synthetic Data Generation](./pretraining/synthetic_generation.md),
[Stage 4: Training Objective](./pretraining/training_objective.md),
[End-to-End Example](./pretraining/end_to_end_example.md))
describe an earlier exploratory SysMLv2 / ORM pipeline that
preceded the convergence on the procedural-catalog approach above.
Each carries a "Deferred framing" banner pointing at the active
work. They are preserved in the repository for archival continuity.

The [Diagnostic Case Study](./pretraining/diagnostic_case_study.md)
documents the 2026-04-19 SOTAB-CTA representation-collapse incident
that motivated the v2 pretrain in the first place; it is the
chapter's primary historical reference.
