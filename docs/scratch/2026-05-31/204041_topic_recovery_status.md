# Topic Recovery Experiment — Status and Next Steps

**Date:** 2026-05-31
**Active branch:** trunk
**Experiment dir:** `build/experiments/topic_recovery_v0/`

## What we set out to test

Can we generate prose via Grok-4.3, seeded by topics drawn from real FinePDFs document-topic mixtures, whose resulting topical structure is recoverable when run through BERTopic? The end goal is an objective feedback loop that gauges generation quality without depending on hand-curated rubrics or human-in-the-loop review — a quality metric a smaller domain-adapted model could be evaluated against.

## What we've actually shipped (this iteration)

### Calibration of the FinePDFs input pool

`scripts/experiment_finepdfs_topic_density.py` + `scripts/experiment_topic_recovery.py --phase calibrate`

- Streamed and deterministically sampled 10,000 FinePDFs documents (seed=4649, matching the audit run)
- Embedded with `sentence-transformers/all-mpnet-base-v2`
- BERTopic with explicit UMAP `random_state=42`: **141 topics, 38.9% noise**
- `approximate_distribution` computed per-doc: persisted as `(10000, 141)` matrix
- **Natural topic density at τ=0.05: mean 4.24 topics/doc, p50=4, p90=7, max 12**
- All artifacts persisted: `topic_centroids.npy`, `topic_distr.npy`, `topic_info.json`, `finepdfs_sample.parquet`

### DeepOnto verbaliser pipeline (working)

`scripts/experiment_verbalise_sdg.py` + `build/experiments/audit_risk_mini.ttl`

- Env fix: `LD_PRELOAD=/nix/store/.../expat-2.7.5/lib/libexpat.so.1` resolves the pyexpat ABI mismatch
- Bootstrap: `aegir.ontology.deeponto_harness.ensure_jvm()` before importing `deeponto.onto`
- Spacy `en_core_web_sm` model downloaded (DeepOnto's POS tagger dependency)
- Hand-crafted 5-complex-concept mini-ontology (`audit_risk_mini.ttl`) produces 5 readable English verbalizations via `verbalise_class_expression`
- Non-triviality check (`get_asserted_complex_classes() > 0`) is the canonical "is this ontology useful?" signal

### Single-passage demonstration

`scripts/experiment_bertopic_my_prose.py`

- Hand-written ~400-word audit charter prose from the 5 verbaliser outputs
- BERTopic identified 4 clusters (Policy+Requirement merged) from 14 sentences
- Sentence-level seed-alignment: 13 of 14 sentences nearest a seed verbalisation with sim ≥ 0.4
- Established that topics ARE identifiable post-hoc in a single passage when seeded with known concepts (recoverability, not narrowness)

### Topic recovery experiment — half_a (247 of planned 500)

`scripts/experiment_topic_recovery.py --phase generate --tag half_a --n-docs 250 --gen-seed 20000`

| Metric | Value |
|---|---:|
| docs generated | 247 (3 of 250 had no topics at τ=0.05, dropped) |
| total prose | 1.42 MB |
| mean topics/prompt | 4.38 (matches FinePDFs's 4.24 calibration) |
| unique input topics seeded | 137 of 141 (97% coverage) |
| response chars (mean) | 5,730 |
| reasoning chars (mean) | 946 (well under 8192 token budget) |
| latency (mean) | 28.8s |
| failures / truncations | 0 |
| cost | ~$1.45 |

### Recovery measurements (the actual finding)

**Cluster-level recovery** via fresh BERTopic on output (volume-bounded):
- Output: **8 clusters, 25.5% noise**
- All 8 clusters mapped to seeded input topics, cosine 0.66–0.90
- **Cluster recovery rate: 5.8% (8/137)** — ceiled by BERTopic's `min_cluster_size=10` against our 247-doc corpus

**Per-doc alignment** via cosine vs input topic centroids (volume-independent):

| Metric | Value |
|---|---:|
| **hit@1** (nearest input topic IS a seeded topic) | **84.6%** (209/247) |
| hit@3 | 96.4% |
| hit@5 | 99.2% |
| mean nearest-topic similarity when hit | 0.692 |
| median rank of seeded topics (of 141) | 5.5 |

Random-chance hit@1 would be ~3% (4.4/141). At 84.6% we're **~28× better than chance**. The pipeline genuinely works at the per-document level.

## Key methodological corrections we made

1. **DeepOnto verbaliser is the core, not a downstream nicety.** Previously I'd hand-paraphrased verbalisations. The real tool produces deterministic, rule-based output. The catalog's `verbal_template` field has held Manchester syntax all along.

2. **A "non-trivial" ontology requires `get_asserted_complex_classes() > 0`.** The SDG vocabulary file (`sdg-vocab.ttl`) is a property vocabulary with 7 named classes and 0 complex class expressions — it is structurally trivial by this measure. Templates need to be *instantiated* into a concrete OWL file for DeepOnto to operate meaningfully on them.

3. **Topics-per-byte (corpus-level) is not a meaningful invariant.** Splitting docs in half doubles the topic count for the same bytes. The per-document density (4.24 at τ=0.05) is the load-bearing calibration because it's a per-document property invariant to corpus slicing.

4. **`max_tokens` ≠ Grok's actual ceiling.** Grok-4.3 has a 1M context window with no separate output cap; the 4K/8K values we'd been using were dspy.LM defaults. Reasoning tokens count against the budget invisibly.

5. **DSPy's `cache=True` default invalidated an early experiment.** Sending the same prompt 80× returned the same cached response 80×. Defeated with `cache=False`. Watch this trap for any constant-prompt experiment.

6. **Sample topics from real document mixtures, not the marginal.** Random combinations of independently-sampled topics aren't a distribution real-world documents exhibit. Sampling one FinePDFs doc's topics-at-τ=0.05 preserves the joint distribution of co-occurring concepts.

7. **BERTopic requires a minimum viable corpus** (~50-100 docs for clean operation, ~150 for stable structure). Below that the tool errors or degenerates. This is analogous to LDA's floor but located differently — UMAP's k-nearest-neighbors requirement + c-TF-IDF's inter-document contrast.

8. **BERTopic's hard assignment vs `approximate_distribution`.** Default returns 1D `topics_` array (single topic per doc). `approximate_distribution` returns soft mixtures via sliding-window aggregation — the LDA-like multi-topic view.

## Open decision

**Should we run half_b (additional 247 docs, ~$1.45, ~1h45m)?**

| Outcome from half_b | Reading |
|---|---|
| Cluster recovery climbs from 5.8% → ~12% | Mechanical doubling; precision will stay high |
| Per-doc hit@1 stays 84-86% | Validates that the 84.6% wasn't a lucky 247-doc subset |
| Per-doc hit@1 drops significantly | Would tell us we got lucky on the seed sample; matters for the publishable claim |

My current recommendation: **run half_b for the validation** but not for the headline number. The pipeline question is answered at the per-doc level. The corpus-level cluster recovery requires ~2,000+ docs to actually saturate (~$6+) — that's a separate decision.

## Next steps

### Immediate (decision required)
- [ ] Run half_b (247 more docs, ~$1.45) — adds validation, marginal recovery improvement
- [ ] Or: stop here and document, accept the per-doc hit@1 as the load-bearing metric

### Near-term (after half_a/half_b decision)
- [ ] **Bridge to ontology stage**: replace bare-FinePDFs-topic seeds with DeepOnto-verbaliser-derived concepts. Compare hit@1 against the FinePDFs-seeded baseline. This is the load-bearing-or-ceremony question for the ontology, framed as a controlled A/B.
- [ ] **Extend `audit_risk_mini.ttl`** from 5 to ~30-50 complex_concepts, validating monotonic-increasing `get_asserted_complex_classes()` at each ratchet
- [ ] **Re-run the loose-grok experiment** with real DeepOnto verbaliser outputs (the original experiment that failed due to (a) hand-paraphrases instead of verbaliser output, (b) DSPy cache bug)

### Medium-term
- [ ] Decide which recovery metric to publish:
  - Per-doc hit@1/hit@5 (volume-independent, what matters for downstream training corpus quality)
  - Cluster recovery (volume-dependent, what matters for the "we recovered N topics from M docs" headline)
- [ ] If publishing cluster recovery: budget for 2K-5K docs to actually saturate
- [ ] Re-evaluate the FinePDFs-topic vs ontology-verbaliser A/B for the v0.3 corpus story

### Pipeline integration (later)
- [ ] Hook this into `train_pretrain.py` to generate the actual training corpus
- [ ] Decide on byte-level vs token-level objective alignment
- [ ] Family complex / R_axiom infrastructure: revisit with real verbalisations (the prior R_axiom calculations were against Manchester syntax disguised as verbalisation; results may shift substantially)

## Artifacts

```
build/experiments/
├── audit_risk_mini.ttl                              # hand-crafted 5-concept mini-ontology
├── finepdfs_topic_density/summary.json              # initial corpus-level calibration
├── finepdfs_topic_density_v2/summary.json           # with approximate_distribution
├── loose_grok_v0/                                   # early (broken) loose-prompting experiment
└── topic_recovery_v0/
    ├── calibration/                                  # 10K FinePDFs, BERTopic, distr
    │   ├── finepdfs_sample.parquet
    │   ├── topic_centroids.npy
    │   ├── topic_distr.npy                          # 10000 × 141 soft mixtures
    │   ├── topic_info.json
    │   ├── topic_ids.json
    │   └── manifest.json
    └── generated/
        ├── pilot/generated.parquet                   # 10-doc validation
        └── half_a/
            ├── generated.parquet                     # 247 docs, 1.42 MB prose
            ├── analysis.json                         # BERTopic recovery output
            └── manifest.json

scripts/
├── experiment_finepdfs_topic_density.py             # corpus-level + soft per-doc density
├── experiment_topic_recovery.py                     # three-phase: calibrate / generate / analyze
├── experiment_verbalise_sdg.py                      # DeepOnto verbaliser probe
├── experiment_bertopic_my_prose.py                  # tiny-corpus BERTopic on hand prose
├── experiment_bertopic_one_doc.py                   # single-doc edge case probe
└── experiment_loose_grok.py                         # bare-ontology prompting (early)
```

All scripts run under `LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python ...`. The DeepOnto verbaliser script additionally needs `LD_PRELOAD=/nix/store/pa6n8nrmgq8jswk2pkrl5qprcls1r0ch-expat-2.7.5/lib/libexpat.so.1`.
