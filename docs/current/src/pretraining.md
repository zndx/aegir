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
incorporate verifier-passing synthetic slices from the
ontology-grounded corpus pipeline once that corpus is available at
the budget v3 needs; see the next section.

## The long-term direction — ontology-grounded synthetic data

The byte-level pretraining track exists alongside a coupled research
program that generates structured training data from a
deterministically-grounded ontology rather than discovering
structure in scraped corpora. The ontology, the rigor program that
governs it, and the closed-loop corpus pipeline are documented in the
[Ontology chapter](./ontology.md); the
[Authors Guide](./ontology/authors_guide.md) is the **canonical**
reference for every metric, gate, and disposal membrane. The short
version of how that program produces pretraining bytes:

1. **Derive and realize the ontology.** `sdg-ontology` is a
   BFO 2020 / CCO-grounded domain ontology, content-derived from
   FinePDFs and realized to a HermiT-validated OWL artifact at
   `corpora/ontology/sdg-ontology.{omn,owl}` (with a consistency
   certificate at `corpora/ontology/HERMIT_CERTIFICATE.md`). Its
   classes are **intermediate-depth subsumers** — the property-bearing
   classes a heterogeneous-but-coherent column belongs to — and they
   *are* the CTA/CPA annotation vocabulary. An **agent-mediated
   propose / dispose feedback loop** drives the ontology: an engine
   proposes axioms; deterministic membranes (parse → HermiT with CCO
   imported as a reasoning authority → OntoClean) dispose and return
   their reason; the agent refines. The seven family catalogs
   (`src/aegir/ontology/catalog/01…07`) are a seed and regression
   baseline; the live driver is the content-first derivation pipeline,
   not a fixed template count.
2. **Generate ontology-grounded chapters.** `scripts/generate_chapter.py`
   synthesizes textbook chapters grounded in the ontology — in the
   current path, content-first from a FinePDFs harvest
   (`--from-harvest`) — calling a generation backend that is either the
   local gRPC engine (`engine/<capability>`, $0) or a weighted
   GLM / Grok mix. Each chapter cites ontology templates, verbalizes
   their axioms into prose, and embeds RI-true relational tables and
   views projected from the DDL spine (`src/aegir/ontology/ddl.py`,
   `realize.py`), so each column's source entity is known by
   construction.
3. **Verify each chapter.** `scripts/verify_chapters.py` runs a
   four-scorer verification loop — `R_topic` (alignment with FinePDFs
   style anchors; dropped for content-first chapters), `R_iri` (cited
   templates' key terms present in prose), `R_density` (markdown-table
   structure), and `R_axiom` (table headers match slot types) — and
   composites them as a geometric mean, accepting at `τ_accept` 0.50.
4. **Mix the accepted chapters into a v3-or-later pretraining corpus**
   alongside real text, and **evaluate the pretrain lift** on the
   Track 1 stratified-eval surface to attribute any improvement to the
   ontology-grounded slice — the **paper 2** claim, scoped in
   [Roadmap](./roadmap.md).

The ontology-grounded corpus is also published as an independent
deliverable: the [SHARE-docs](./roadmap/phase_share_docs.md) browsable
corpus and the `corpora/` submodule (zndx/sdg-corpora). Its publication
is gated on the ontology's OQuaRE quality model — `sync --push` is
refused below GREEN, the hard gate the Authors Guide documents.

### Why this scales

The bottleneck in conventional table annotation is human labeling.
The bottleneck in this synthetic regime is generation-and-verification
throughput — the pipeline must generate ontology-grounded chapters and
the verifier must score them — which is embarrassingly parallel, and
runs at $0 against the local gRPC engine on local GPUs. The diversity
of the training data is bounded by the ontology's expressivity, which
is itself growing: the content-first derivation accretes
FinePDFs-derived intermediate classes rather than enumerating a fixed
catalog. Independent constraints on the regime are tracked as
pre-registered gates in `EVIDENCE.md` — in particular the corpus's
maximum **non-repetitive token yield**, which caps the ontology-grounded
fraction of any large pretraining budget, and the M2 lift that the v3
corpus mix must demonstrate over a no-ontology control.

### How this connects to Aegir's three target tasks

The two pretraining inputs — real corpora (v2) and ontology-grounded
synthetic slices (v3 and beyond) — both serve the same three
downstream tasks:

- **Column Type Annotation (CTA).** Real-corpus pretraining gives
  general language and tabular-syntax signal; synthetic slices add
  per-column entity types under known provenance.
- **Column Property Annotation (CPA).** Cross-column relationships
  in real corpora are noisy; synthetic slices supply clean
  cross-column relations from the ontology's `sdg:*` property
  declarations and the DDL spine's FK edges.
- **Data Element Discovery.** Cross-table groupings under known
  ontological provenance are the synthetic regime's distinctive
  contribution — real corpora do not supply ground-truth data
  elements at scale.

The first two tasks are addressable from v2 alone. The third
benefits most directly from the synthetic regime and is the
strongest motivator for completing the corpus pipeline.

## Sub-pages

The **Training Tactics** note and the five **Stage**-named sub-pages
(*Stage 1: Ontology Extraction*, *Stage 2: Schema Projection*,
*Stage 3: Synthetic Data Generation*, *Stage 4: Training Objective*,
*End-to-End Example*) describe an exploratory SysMLv2 / ORM pipeline —
a long-horizon framing that preceded the convergence on the
ontology-grounded chapter pipeline above. They are preserved in the
repository (`docs/current/src/pretraining/`) as background to the
active work; they are not wired into the rendered book.

The [Diagnostic Case Study](./pretraining/diagnostic_case_study.md)
documents the 2026-04-19 SOTAB-CTA representation-collapse incident
that motivated the v2 pretrain in the first place; it is the
chapter's primary historical reference.
