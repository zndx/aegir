# Introduction

Aegir is a hierarchical sequence model for **column type annotation** (CTA) and **column property annotation** (CPA) on relational tables. Given a table with multiple columns, Aegir predicts semantic types (e.g., "city", "date", "currency") or properties for each column, using the full table context -- neighboring columns, headers, and cell values -- as input.

## Problem Setting

Column annotation is a core task in data integration, data lake management, and knowledge graph construction. The input is a serialized table: byte-level token sequences encoding column headers and cell values across multiple columns. The model must classify the target column while attending to its surrounding context.

Target benchmarks:

- **SOTAB** -- Semantic column annotation on Web tables (Schema.org types)
- **GitTables** -- Large-scale column type detection across 1M+ CSV tables from GitHub
- **WikiTables** -- Column annotation on Wikipedia HTML tables

## Key Innovations

**Byte-level dynamic chunking as learned tokenization.** Rather than using a fixed tokenizer (BPE, SentencePiece), Aegir operates on raw bytes and learns to segment sequences into variable-length chunks via content-dependent boundary prediction. A routing module measures cosine similarity between consecutive hidden states; high dissimilarity triggers a chunk boundary. This makes the "tokenization" fully differentiable and adapted to the data distribution.

**All-RWKV recurrent architecture.** The primary sequence processing blocks use RWKV-7 time mixing with flash-linear-attention Triton kernels. RWKV-7 maintains a constant-size recurrent state matrix of shape `(B, H, head_size, head_size)` regardless of sequence length. This gives O(1) memory per token during inference and, critically, makes the recurrent state a fixed-size object that can be serialized, transmitted, and algebraically combined across agents.

**ROSA suffix automaton for exact pattern retrieval.** The ROSA (RWKV Online Suffix Automaton) module provides lossless infinite-range retrieval by constructing an online suffix automaton over binarized hidden representations. While RWKV-7 learns smooth sequence-level patterns, ROSA can retrieve exact substring matches from arbitrarily far in the past, complementing the learned recurrent state.

**Agent swarm with state fusion.** Multiple specialist agents (e.g., one per column type family) can process the same table in parallel. Because RWKV recurrent states are fixed-size matrices, they can be fused via attention-weighted combination, learned gating, or projection -- far more efficiently than merging transformer KV caches, which grow linearly with sequence length.

## Architecture at a Glance

Aegir uses a recursive hierarchy defined by nested layout strings:

```
arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]
```

This reads as: 2 RWKV-7 encoder blocks, then a sub-hierarchy (2 encoder blocks, 4 main blocks, 2 decoder blocks), then 2 RWKV-7 decoder blocks. At each non-innermost stage, dynamic chunking downsamples the sequence before passing it to the next level, and an EMA-based dechunking module reconstructs the full resolution on the way back up.

The block types -- RWKV-7, ROSA, MHA, Mamba-2 -- can be freely mixed within any stage using compact layout strings like `"w4T1r2"`.
