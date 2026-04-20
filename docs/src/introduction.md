# Introduction

Aegir is a hierarchical sequence model for **semantic column annotation** and **cross-table data element discovery** on relational data. Given one or more tables, Aegir predicts semantic types for individual columns (Column Type Annotation), identifies properties and relationships between columns (Column Property Annotation), and discovers coherent **data elements** -- groups of semantically related columns that span multiple tables in a data warehouse.

## Problem Setting

Enterprise data warehouses contain thousands of tables with columns whose meaning is often opaque: generic names (`col0`, `field_42`), inconsistent conventions across teams, and no machine-readable metadata. Understanding what each column represents -- and which columns across different tables refer to the same real-world concept -- is foundational to data governance, privacy compliance, and integration.

Current approaches to this problem fall into two categories:

**Pattern and heuristic-based methods** identify column types through regex detectors (email, SSN, credit card patterns), name matching, embedding similarity, and gradient-boosted classifiers trained on hand-engineered features. These methods work well for structurally distinct types but struggle with **confusable pairs** -- columns whose value distributions are nearly identical but whose semantic types differ (e.g., advertising IDs vs GUIDs, bank account numbers vs payment card numbers). They also require manual enumeration of data element patterns and cannot generalize to novel relationship types.

**Learned sequence models** (DODUO, RECA, REVEAL) treat the table as a token sequence and classify columns via fine-tuned transformers. REVEAL's key insight is that **context column selection matters**: choosing the right neighboring columns (via MMR diversity sampling) dramatically improves annotation accuracy. However, these models operate on single tables in isolation and use fixed subword tokenizers that fragment tabular data unpredictably.

Aegir bridges these approaches. It is designed to be trained **in situ** alongside evidence-based classification pipelines -- consuming the same serialized table representations, but learning cross-column and cross-table relationships end-to-end rather than relying on manually enumerated patterns. Specifically:

- **Column Type Annotation (CTA)**: Classify individual columns into a semantic taxonomy (e.g., SIGDG ontology categories, Schema.org types, DBpedia classes).
- **Column Property Annotation (CPA)**: Identify properties and relationships between column pairs (e.g., "city is-located-in country").
- **Data Element Discovery**: Identify groups of related columns across tables that constitute coherent real-world entities (e.g., a PaymentCard data element spanning `card_number`, `expiry`, `cardholder` columns across billing, transaction, and customer tables).

The third task -- cross-table data element discovery -- is where the greatest value lies for enterprise governance. Current pipelines discover data elements through keyword-based schema matching and post-classification co-occurrence analysis. A model that learns these relationships from data can generalize beyond enumerated patterns, handle non-English and abbreviated column names, and resolve confusable pairs by leveraging cross-table structural context that no single-table classifier can access.

Target benchmarks:

- **SOTAB** -- Semantic column annotation on Web tables (Schema.org types)
- **GitTables** -- Large-scale column type detection across 1M+ CSV tables from GitHub (100% generic column names -- the hardest regime)
- **WikiTables** -- Column annotation on Wikipedia HTML tables

> **Where we are**: the training regime is under targeted redesign. The
> first direct-supervision attempt on SOTAB produced complete
> representation collapse; a three-phase diagnostic (prediction
> distribution + cluster geometry + MCL inflation sweep, the last
> borrowed from van Dongen's graph-clustering work) confirmed it and
> motivated a pivot to **byte-level pretraining on raw tabular corpora**
> plus **path-prediction fine-tuning** against the ontology hierarchy.
> See [Training Regime](./training_regime.md) and the
> [Diagnostic Case Study](./pretraining/diagnostic_case_study.md) for
> the full reasoning and staged plan.

## Key Innovations

**Byte-level dynamic chunking as learned tokenization.** Rather than using a fixed tokenizer (BPE, SentencePiece), Aegir operates on raw bytes and learns to segment sequences into variable-length chunks via content-dependent boundary prediction. A routing module measures cosine similarity between consecutive hidden states; high dissimilarity triggers a chunk boundary. This makes the "tokenization" fully differentiable and adapted to the data distribution -- critical for tabular data where delimiters, numeric formats, and encodings vary wildly across sources.

**All-RWKV recurrent architecture.** The primary sequence processing blocks use RWKV-7 time mixing with flash-linear-attention Triton kernels. RWKV-7 maintains a constant-size recurrent state matrix of shape `(B, H, head_size, head_size)` regardless of sequence length. This gives O(1) memory per token during inference and, critically, makes the recurrent state a fixed-size object that can be serialized, transmitted, and algebraically combined across agents.

**ROSA suffix automaton for exact pattern retrieval.** The ROSA (RWKV Online Suffix Automaton) module provides lossless infinite-range retrieval by constructing an online suffix automaton over binarized hidden representations. While RWKV-7 learns smooth sequence-level patterns, ROSA can retrieve exact substring matches from arbitrarily far in the past -- enabling precise pattern detection (email formats, card number structures) that complements the learned recurrent state.

**Agent swarm with state fusion for cross-table reasoning.** Multiple specialist agents can process different tables or column families in parallel. Because RWKV recurrent states are fixed-size matrices, they can be fused via attention-weighted combination, learned gating, or projection -- far more efficiently than merging transformer KV caches, which grow linearly with sequence length. This architecture enables cross-table data element discovery: each agent processes a table, and the fused state captures inter-table relationships that no single-table model can learn.

**In-situ training within evidence pipelines.** Aegir is designed to integrate with Dempster-Shafer theory (DST) evidence fusion pipelines as a learned evidence source. Its predictions -- with calibrated confidence -- feed into the same conjunctive combination framework alongside cosine similarity, gradient boosting, pattern detectors, and name matching. The model learns from the pipeline's own bootstrap labels and SAGE-validated features, creating a self-improving loop where Aegir's learned representations replace hand-engineered heuristics as they prove their value.

## Architecture at a Glance

Aegir uses a recursive hierarchy defined by nested layout strings:

```
arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]
```

This reads as: 2 RWKV-7 encoder blocks, then a sub-hierarchy (2 encoder blocks, 4 main blocks, 2 decoder blocks), then 2 RWKV-7 decoder blocks. At each non-innermost stage, dynamic chunking downsamples the sequence before passing it to the next level, and an EMA-based dechunking module reconstructs the full resolution on the way back up.

The block types -- RWKV-7, ROSA, MHA, Mamba-2 -- can be freely mixed within any stage using compact layout strings like `"w4T1r2"`.
