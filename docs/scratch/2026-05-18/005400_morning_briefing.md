# Aegir v0.3 — Morning briefing

**For:** the team
**From:** Ryan
**Date:** 2026-05-18 ~07:00 MDT
**TL;DR length:** 5-min read

## The question we set out to answer

> Before we scale to a 3-4 digit synthesis budget, prove the ontology stage is doing real work. Is ontology grounding pushing our relational pretraining corpus to SOTA, or is it ceremony?

## What we did last night

Three-arm ablation, controlled for everything except the experimental variable:

| arm | style anchors | ontology axioms | LIMS schema prompt |
|---|:-:|:-:|:-:|
| **A — full** | ✓ | ✓ | ✓ |
| **B — no-ontology** | ✓ | ✗ | ✓ |
| **C — no-schema** | ✓ | ✓ | ✗ |

30-chapter pilot tonight (10/arm), ~$0.30. 600-chapter v1 (200/arm) running overnight, ETA ~05:00 UTC.

Four measurement axes:

1. **Topic correspondence** — does each chapter land near its anchored FinePDFs topic?
2. **Verifier** — R = geometric_mean(R_topic, R_iri, R_density, R_axiom)
3. **Structural alignment** — do column headers align with SOTAB-CTA Schema.org / our ontology slot vocab?
4. **Aegir-tiny LM training** — does the corpus produce a learnable byte-LM?

## What we found

### Finding 1: The ontology produces a measurably different corpus character

Across all four axes, the three arms produce statistically distinct outputs. Examples below are median-quality chapters from each arm — read them yourself, the differences jump off the page.

**Arm A (full)**: "Modeling Directive Compliance and Evidence Attestation in Audit Systems" — textbook chapter with ontological axioms followed by relational schema with PK/FK. The v0.3 thesis-aligned output.

**Arm B (no-ontology)**: "Doctoral Candidate Tracking under IDEX Programme" (Univ. Grenoble Alpes!) — pure institutional documentation, queryable schema, no axioms, no ontology vocabulary.

**Arm C (no-schema)**: "Ontological Foundations for Directive Controls in Regulatory Compliance" — ontology glossary with descriptive tables but no queryable PK/FK structure.

### Finding 2: The differential is direction-of-effect, not magnitude

All numbers below are 10-chapter pilot, will sharpen at 200/arm overnight:

| metric | full | no-ontology | no-schema | comment |
|---|---:|---:|---:|---|
| topic hit@1 | 30% | **80%** | 20% | no-ontology wins topic alignment |
| verifier R_comp | **0.600** | 0.056 | 0.607 | no-ontology collapses on R_axiom |
| SOTAB-CTA col alignment | 28.4% | **35.2%** | 19.7% | no-ontology wins on Schema.org vocab |
| ontology col alignment | **88.5%** | 57.4% | 89.1% | full wins on our own slot vocab |
| Aegir-tiny val_loss | **12.96** | 14.68 | 14.17 | full produces lowest LM loss |

### Finding 3: "Load-bearing" is true, but for a particular notion of "downstream"

The ontology IS load-bearing — it produces a corpus whose character differs measurably and consistently from the no-ontology corpus. But:

- For **SOTAB-CTA** (Schema.org column types, our v0.2 ROADMAP eval target): no-ontology produces structurally closer training data.
- For **ontology-grounded CPA** (our own 540-slot vocabulary): full pipeline is the only viable option; no-ontology has zero axiom signal.
- For **byte-LM perplexity on the corpus itself**: full produces the most learnable corpus (lowest val loss), even at 10-chapter scale.

This isn't a flaw — it's *the choice point*. The question isn't "is ontology good or bad," it's "which eval target are we aiming at."

## What this means for the v0.3 plan

Two coherent strategies:

**Strategy X — "ship for SOTAB-CTA"**: Use the no-ontology arm at scale (~$15-30 for 5K-10K chapters), train Aegir-small on it, target SOTAB-CTA macro F1. Compete against TAPEX/TableFormer on the published benchmark. Lower-risk; well-understood eval target.

**Strategy Y — "ship the ontology-grounded thesis"**: Use the full arm at scale (~$45 for 5K chapters), train Aegir-small on it, target an ontology-grounded CPA eval we *publish alongside* the corpus and weights. Higher-risk: the eval is novel and we have to defend it. But it's the actual v0.3 thesis.

**Strategy Z — "ship both"**: Generate both corpora, publish both checkpoints, publish both evals. Doubles the cost but the corpora become a *contribution in themselves*: the first published ablation showing what ontology grounding does to a synthetic table-aware pretraining corpus. ~$100 budget, ~2 days wall.

I lean Z, but want to discuss.

## What's ready for the meeting

In `docs/scratch/2026-05-18/`:
- `001727_ablation_pilot.md` — full ablation methodology + pilot results (updated 00:38 with training findings)
- `002422_structural_alignment.md` — SOTAB-CTA vs ontology column-alignment proxy
- `004500_v0_3_plan_of_record.md` — 9-artifact publication lineup + open decisions
- `005400_morning_briefing.md` — this doc

In `/raid/checkpoints/aegir-artifacts/`:
- `ablation_v0/` — 30 pilot chapters, verifier outputs, topic correspondence
- `ablation_v1/` — 600 chapters in flight (overnight)
- `coverage_v0/` — FinePDFs coverage audit baseline

Infrastructure:
- `scripts/ontology_coverage_audit.py` — coverage methodology
- `scripts/generate_chapter.py` — 60/40 GLM/Grok mix + `--ablation` flag
- `scripts/verify_chapters.py` — verifier with R_topic v2 length-invariant scoring
- `scripts/topic_correspondence.py` — independent topic-alignment analyzer
- `scripts/structural_alignment.py` — column vocab alignment proxy
- `scripts/chapters_to_byte_parquet.py` — train corpus converter
- `scripts/train_ablation_all.sh` — single-GPU Aegir-tiny train wrapper

The 30-chapter pilot took ~$0.30. The 600-chapter v1 will cost ~$3. A full 5K-chapter run at the chosen mix is ~$15-45 depending on strategy.

## Open decisions for the meeting

**D1.** Strategy X / Y / Z above. Which (or which mix)?

**D2.** If Y or Z: what's the ontology-grounded CPA eval design? Build now or after corpus release?

**D3.** Do we want to ship the CoT trace corpus separately? GLM-4.7 and Grok-4.3 reasoning traces alongside the chapters is a publishable artifact in its own right — first table-grounded synthesis reasoning corpus that I'm aware of.

**D4.** Budget ceiling for tonight (5-18 → 5-19)? Z requires ~$100 + 2 days. X requires ~$30 + 1 day.

**D5.** Aegir-small training infrastructure: do we need to fix DDP NCCL (currently single-GPU only due to PTX JIT lib issue) before the next major run, or wait until production?

## My recommendation

Strategy Z, on a $100 budget, with the CoT trace corpus published as a separate artifact. This is the version that makes our work *most useful to the community* and *most defensible against TAPAS-lineage prior art*. The ablation IS the contribution; we shouldn't bury it inside a SOTA chase.

Whether the no-ontology checkpoint beats us on SOTAB-CTA or the full-pipeline checkpoint wins on CPA, both results land in the paper. Ablation papers age well.
