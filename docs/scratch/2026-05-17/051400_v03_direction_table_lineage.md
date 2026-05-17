# V0.3+ direction: mining the TAPAS lineage for H-Net/RWKV alpha

**Date:** 2026-05-17
**Status:** Forward-direction sketch. Explicitly **out of v0.2 scope**.
v0.2 finishes Phase 0 (table-text LM pretrain) + Phase 0.5 (synth-SQL +
counterfactual) + Phase 1 (SOTAB fine-tune). This doc plants the flag
for what comes after.

## Thesis

The TAPAS lineage (TAPAS → TableFormer → TAPEX → OmniTab → ReasTAP)
peaked around 2022 and was effectively abandoned by the field as
general LLMs scaled past it on the same benchmarks. But it wasn't
abandoned because the *ideas* were wrong — it was abandoned because
the general-LLM regime made the specialized inductive biases optional.

**Aegir's regime is different.** Sub-200M parameters, byte-level
tokenization, recurrent backbone (RWKV-7), dynamic chunking (H-Net) —
we're firmly in the regime where scale isn't the lever and biases
matter. The 2020-2022 specialized work was *tuned for this regime*.
Untapped methodology to mine, ranked by transposability + expected
value below.

## Ranked v0.3+ candidates

### 1. **TAPEX-style synthetic SQL execution traces** (highest EV)

TAPEX (Liu et al., ICLR 2022) achieves WTQ 57.5%, WikiSQL 89.5%, SQA
74.5%, TabFact 84.2% — all by pretraining a BART seq2seq on **massive
synthetic SQL execution traces**. The recipe:

1. Sample a real table from a corpus.
2. Sample a random SQL query against that table (WHERE, GROUP BY,
   aggregations, ORDER BY).
3. Execute via SQLite/DuckDB → gold answer (deterministic, free).
4. Train: `<table> [SEP] <SQL> [SEP] <answer>` as next-token LM.

**Transposition to Aegir is trivial.** Our `train_pretrain.py` LM head
already handles this format. We need:
- A SQL grammar generator (similar shape to `synth_table_qa.py` but
  emits real SQL strings instead of NL statements).
- A SQLite/DuckDB executor wrapper.
- A new corpus driver: `scripts/generate_synth_execution_trace.py`.
- ~10M (table, SQL, answer) triples from the TAPAS corpus.

**Expected EV:** TAPEX outperformed TAPAS by 8-15% absolute on every
benchmark using essentially the same backbone. Recipe-level gain is
huge; architecture matters less than the pretraining target.

**v0.3 milestone:** Aegir + Phase 0.7 (TAPEX-style) → WTQ eval, target
~30-40% accuracy (vs TAPEX-large's 57.5%; we're ~5x smaller).

### 2. **TableFormer order-invariant row/column biases** (medium EV)

TableFormer (Yang et al., ACL 2022) extends TAPAS with learnable
relative attention biases that encode row/column structure in a
**task-independent, order-invariant** way. Avoids 4-6% drops to row/
column permutation that plain TAPAS exhibits.

**Transposition to Aegir is non-trivial.** RWKV TimeMix doesn't have
explicit attention biases — it has per-channel time-decay (the `w`
parameter in `chunk_rwkv7`). The structural analog isn't direct.

What we *can* do:
- Add per-cell **role embeddings** that encode `(row_index, col_index)`
  as a learned bias added to byte embeddings at cell-start positions.
- Train the model to be invariant to row permutation by augmenting
  the Phase 0 / 0.5 pretraining data with randomly-permuted-row
  versions of each table.

**Expected EV:** small (1-3% absolute). Most useful for robustness
splits (e.g., SOTAB's `format_heterogeneity` test set) rather than
headline accuracy.

### 3. **OmniTab combined NL + synthetic** (small EV — already mostly captured)

OmniTab (Jiang et al., NAACL 2022) combines TAPEX synthetic + TAPAS
natural-language pretrain. Our planned pipeline already does this:
Phase 0 (TAPAS-style NL pretrain) + Phase 0.5 (NL stub via counter-
factual + synth-NL via synth-SQL) + (future) Phase 0.7 (TAPEX-style).

**Expected EV:** combinatorial gain over each phase alone. Mostly free
once we have Phases 0.5 and 0.7 in place.

### 4. **ReasTAP multi-step reasoning patterns** (deferred)

ReasTAP extends synthetic pretrain to multi-step reasoning (nested
subqueries, multi-hop joins). Builds on TAPEX. Worth pursuing only
*after* TAPEX-style Phase 0.7 lands and we know what the single-step
ceiling looks like.

## Concrete v0.3 roadmap (post-v0.2 ship)

| phase | what | corpus | est. wall (6x 4090) |
|---|---|---|---:|
| 0.7 | TAPEX-style synth execution-trace pretrain | ~10M (table, SQL, answer) | 8-16 h |
| 0.8 | (optional) TableFormer-style row-permute augmentation on Phase 0.5 | same as 0.5, permuted | 4-8 h |
| 1.0 | WTQ fine-tune + eval | wikitablequestions (~22K) | 1-2 h |
| 1.1 | TabFact fine-tune + eval | tab_fact (~118K) | 2-3 h |
| 1.2 | Re-run SOTAB-CTA with stronger backbone | SOTAB | 2-4 h |

Headline target: Aegir-2x (~120M params) hitting **WTQ ≥ 30%** would
be a credible TAPEX-comparable result for an unconventional backbone.
WTQ ≥ 50% would be SOTA-of-the-era for non-transformer models.

## What v0.2 should bake into the v0.3 substrate

A few choices in v0.2 should be made with v0.3 in mind:

- **Cell-boundary sentinel byte** (already validated) is a TableFormer-
  flavored prior that TAPEX-style pretrain also benefits from. Keep on
  by default.
- **TRUE/FALSE label tokens** (added today) generalize: TAPEX-style
  *answer* tokens go through the same LM head. Add reserved IDs 8-11
  for answer-formatting tokens (`[NUM]`, `[CELL]`, `[ROW]`, `[YES]`)
  before Phase 0.7 needs them.
- **train_pretrain.py architecture** (LM head + tied embeddings) is
  exactly what TAPEX uses. No re-architecting needed.

## References

- TableFormer (ACL 2022) — `arxiv:2203.00274`
- TAPEX (ICLR 2022) — `arxiv:2107.07653`, HF `microsoft/tapex-*`
- OmniTab (NAACL 2022) — `arxiv:2207.03637`
- ReasTAP (EMNLP 2022) — `arxiv:2210.12374`
- WTQ — `wikitablequestions` on HuggingFace
- TabFact — `tab_fact` on HuggingFace

The unifying observation: every gain in this lineage came from
**richer synthetic pretraining objectives**, not from architectural
changes. That maps perfectly onto our story — H-Net/RWKV is the
architectural bet, and we get to inherit TAPAS-lineage's data recipes
wholesale.
