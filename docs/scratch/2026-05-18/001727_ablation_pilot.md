# Ontology load-bearing pilot — 3-arm ablation, 30 chapters total

**Date:** 2026-05-18 00:17 UTC
**Run dirs:**
- `/raid/checkpoints/aegir-artifacts/ablation_v0/72cc17cff35b63f6/` — arm A (full)
- `/raid/checkpoints/aegir-artifacts/ablation_v0/3a81d92e48daaa13/` — arm B (no-ontology)
- `/raid/checkpoints/aegir-artifacts/ablation_v0/7434ac168d97acf7/` — arm C (no-schema)
- `/raid/checkpoints/aegir-artifacts/ablation_v0/topic_correspondence.parquet`

**Cost:** ~$0.30 (30 chapters via xai/grok-4.3, controlled for model)

## Experimental design

The question: before scaling to a $300+ corpus, does the ontology stage genuinely add value, or is it ceremony?

Three arms, same model (Grok 4.3), same 10 (template-set, style-anchor) seed pairs (seed 5000), 10 chapters per arm:

| arm | style anchors | ontology axioms | LIMS schema instruction |
|---|:-:|:-:|:-:|
| A — **full** | ✓ | ✓ | ✓ |
| B — **no-ontology** | ✓ | ✗ | ✓ |
| C — **no-schema** | ✓ | ✓ | ✗ |

B isolates the ontology contribution. C isolates the LIMS-schema-prompt contribution.

## Two findings, both real

### Finding 1: the ontology pulls chapters off the FinePDFs distribution

Topic-correspondence: embed each chapter, find its nearest topic centroid among the 200 audit topics, check the rank of the anchored topic in that ordering.

| arm | mean anchor_rank ↓ | hit@1 ↑ | hit@3 ↑ | hit@10 ↑ | mean anchor_sim ↑ | mean top1_sim |
|---|---:|---:|---:|---:|---:|---:|
| full | 8.00 | 30.0% | 60.0% | 80.0% | 0.218 | 0.369 |
| **no-ontology** | **1.30** | **80.0%** | **100.0%** | **100.0%** | **0.365** | 0.524 |
| no-schema | 13.50 | 20.0% | 30.0% | 60.0% | 0.219 | 0.387 |

No-ontology dramatically wins: chapters land on their anchored FinePDFs topic as nearest-neighbor 80% of the time vs the full pipeline's 30%. Anchor similarity is 67% higher.

**Mechanism:** the cited ontology axioms (Manchester syntax, BFO/CCO upper-ontology vocabulary, slot-typed templates) pull generation toward a specific structural/symbolic register that diverges from the FinePDFs governance-document register the style anchors set. The two signals (anchor-style, axiom-content) pull in different directions, and axioms win.

### Finding 2: without the ontology, structural grounding collapses

Verifier (R_topic, R_iri, R_density, R_axiom; composite geometric mean; τ_accept=0.50):

| arm | accepted | borderline | rejected | dominant failure mode |
|---|:-:|:-:|:-:|---|
| full | **8/10** | 2/10 | 0/10 | low_topic in 8 |
| **no-ontology** | 0/10 | 1/10 | **9/10** | **low_axiom in 10, low_iri in 4** |
| no-schema | 9/10 | 1/10 | 0/10 | low_topic in 10 |

Removing the ontology causes:
- R_axiom (table-header slot-vocabulary check) → 0
- R_iri (cited template keywords in prose) → drops from 1.0 to ~0.6
- R_density (table presence) — unchanged (the schema prompt is still on)
- R_topic — unchanged

No-ontology chapters do produce tables, but the tables have no ontological semantics — they're just relational structure with arbitrary domain content.

## The tension is real and load-bearing

This isn't "the ontology is useless" or "the ontology is essential." It's:

**The ontology trades topic-distribution alignment for structural grounding.**

The full pipeline produces chapters that:
- Cite specific OWL axioms and slot types in prose
- Use ontology slot vocabulary in table headers
- Cross-reference Manchester-syntax axioms (Axiom 1, Axiom 2, ...)
- BUT drift in topic-space toward axiom-grounded register, away from natural FinePDFs distribution

The no-ontology arm produces chapters that:
- Stay in the FinePDFs distribution (governance documentation register)
- Produce relational tables grounded in LIMS-domain content
- BUT have no axiom-traceability — cell values are arbitrary plausible-instances, not derivable

## What this means for v0.3 scaling

We cannot answer "scale or not" from this data alone. The answer depends on what we're optimizing the corpus to teach the byte-LM:

- If the goal is **"learn the FinePDFs byte distribution"** → no-ontology is structurally better data. The ontology is hurting us.
- If the goal is **"learn axiom-grounded table reasoning"** → no-ontology is missing the entire signal. The full pipeline is the only option.
- If the goal is **both** — and this is the actual v0.3 thesis — we have a tension to resolve at the data-mix level, not the prompt level.

The next experiment must be **downstream training**: do we get a measurably better table-aware byte-LM from the ontology-grounded substrate than from the no-ontology one? If yes, ontology is load-bearing for our thesis even though it hurts topic-correspondence. If no, ontology is ceremony and we ship the no-ontology corpus.

## Note on the audit / verifier methodology

R_topic in the verifier still shows ~0.22 across full and no-schema arms, consistent with the known length-asymmetry issue (#114 task). But the topic-correspondence analyzer (which uses chapter-vs-topic-centroid cosine directly, not chapter-vs-anchor) shows clear signal: 0.218 (full) vs 0.365 (no-ontology). The same measurement methodology produces a 67% gap — methodology consistency holds; the difference between arms is real, not artifact.

## Concrete recommendation

Do the Aegir-tiny indicative training comparison **tonight**, unattended:

1. Generate ~500 chapters per arm (~$2-3 total, ~30 min wall on Cerebras for arms with schema-rich Grok mix; longer for pure-Grok).
2. Train Aegir-tiny on each (~13M params, 1-2h on 6 GPUs).
3. Eval all variants on SOTAB-CTA tiny held-out.
4. Compare macro F1. Margin >5% favors the structurally-grounded substrate; margin <2% suggests the topic-correspondence advantage of no-ontology dominates.

Either result is publishable. Either tells us how to ship v0.3.

## Alternative interpretation worth raising

There's a third possibility I want to flag explicitly:

The full-pipeline chapters look subjectively excellent (we read one in detail earlier — LIMS-style cross-joinable tables with axiom citations). The topic-correspondence "drift" might not be drift at all — it might be **enrichment**: the chapters become MORE specific than any single FinePDFs topic because they fuse ontology-grounded content INTO the topic's natural register. Per the audit, our ontology has zero topics covered at τ_high=0.55 against FinePDFs — meaning the ontology lives in a region of embedding space slightly orthogonal to the FinePDFs corpus, but adjacent to it. Generating in that adjacent region produces content that's *related but not identical* to any single FinePDFs topic.

If this interpretation is right, the chapters are higher-value than no-ontology even from a corpus-density-per-byte perspective: they're carrying signal the FinePDFs corpus alone doesn't carry. The byte-LM should benefit from that.

The Aegir-tiny training settles which interpretation is correct.
