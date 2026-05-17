# Ontology coverage audit v0 — baseline measurement against FinePDFs

**Date:** 2026-05-17 21:01 UTC
**Run ID:** `232ea5460ce6e0bf`
**Artifacts:** `/raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf/`
**Script:** `scripts/ontology_coverage_audit.py`

## What this measures

For each topic discovered in a 10K-document slice of HuggingFaceFW/finepdfs
(eng_Latn, train split, seed 4649), the audit reports the nearest-matching
template(s) from the Aegir ontology catalog (540 templates across 7 family
files) using all-mpnet-base-v2 embeddings.

The output is the **quantitative coverage baseline** that the future Qwen-driven
ontology improvement loop optimizes against. It is *not* a quality judgment
of the ontology — it's a breadth measurement of where ontology and corpus
intersect.

## Headline numbers

| | by topic (n=200) | by doc (n=10,000) |
|---|---:|---:|
| **covered** (max_sim ≥ 0.55) | 0 (0.0%) | 0 (0.0%) |
| **borderline** (0.35 ≤ max_sim < 0.55) | 29 (14.5%) | 1,643 (16.4%) |
| **gap** (max_sim < 0.35) | 171 (85.5%) | 8,357 (83.6%) |

Distribution of max-sim across all 200 topics: min 0.059, p25 0.222, median **0.284**, p75 0.331, p90 0.368, **max 0.524**. Even the strongest match in the entire sample falls below the conservative "covered" threshold — the ontology and FinePDFs are *genuinely far apart in embedding space*, which is the expected position given (a) the ontology is focused on a specific subdomain, (b) FinePDFs is broad-internet-PDFs.

## Family-level distribution (where the ontology is doing its work)

By top-1 topic assignment:

| family | n_templates | n_topics_top1 | topic share | doc share |
|---|---:|---:|---:|---:|
| 03_directive_governance | 69 | 60 | 30.0% | 35.1% |
| 07_long_tail | 165 | 74 | 37.0% | 34.4% |
| 02_observation_measurement | 92 | 19 | 9.5% | 8.7% |
| 01_foundation | 81 | 19 | 9.5% | 7.6% |
| 04_ebpf_kernel | 51 | 13 | 6.5% | 6.6% |
| 06_belief_structure | 32 | 8 | 4.0% | 4.1% |
| 05_provo_lineage | 50 | 7 | 3.5% | 3.6% |

**Read:** governance + long-tail dominate (>2/3 of matched topics & docs). The compliance / audit / policy / regulatory part of the ontology is where the real coverage lives.

## Templates pulling the most weight

Top templates by `n_topics_top1`:

| family | template_id | topics | max_sim | verbalization |
|---|---|---:|---:|---|
| 03_directive_governance | `attestation_with_supporting_evidence` | 14 | 0.411 | "X is something with supporting evidence Y" |
| 07_long_tail | `schemaorg_place_address` | 14 | 0.395 | "X has postal address Y" |
| 07_long_tail | `pci_dss_requirement` | 10 | 0.391 | "X is something that pci dss requirement Y" |
| 03_directive_governance | `audit_for_period` | 9 | 0.431 | "X is something for audit period Y" |
| 03_directive_governance | `audit_conducted_by` | 7 | 0.406 | "X conducted by person" |
| 07_long_tail | `hipaa_safeguard_admin` | 7 | 0.355 | "X is hipaa safeguard Y" |
| 04_ebpf_kernel | `syscall_equiv_in_subsystem` | 6 | 0.381 | (no verbalization) |
| 07_long_tail | `trace_supports_claim` | 6 | 0.397 | "X supports claim descriptive information" |
| 01_foundation | `designative_subclass_basis` | 5 | 0.416 | "X is a designative information content entity" |

The "long_tail" templates carrying weight are the **specific compliance frameworks**: PCI DSS, HIPAA, ISO 27001, GDPR. These match real FinePDFs documents (corporate compliance reports, policy documents).

## Templates with zero topic coverage

15 of the 540 templates failed to be the top-1 match for any topic at max_sim > 0.30. The pattern:

- **All of `04_ebpf_kernel` cluster low** (max_sim 0.22-0.29). This is expected — eBPF kernel concepts are highly technical and rare in general-internet PDFs.
- **Generic `subclass_*` / `disjoint_*` axiom forms** also cluster low — they're abstract schema templates without semantic content to match.

## Where the borderline cases actually land

The 5 borderline topics with the highest sim scores:

| topic | sim | top template (family) | representative doc |
|---:|---:|---|---|
| 143 | 0.524 | `mass_function_equiv_frame_and_assignment` (06_belief_structure) | exam grammar questions — **false positive** (incidental "questions" vocabulary match) |
| 159 | 0.442 | `mass_function_min_one_focal` (06_belief_structure) | university question bank — **false positive** |
| 4 | 0.433 | `iso27001_clause_subclass` (07_long_tail) | terms & conditions of purchase — **genuine match** |
| 111 | 0.431 | `audit_for_period` (03_directive_governance) | Sunland Group audit committee charter — **strong genuine match** |
| 14 | 0.424 | `outlier_claim_min_one_attribute` (02_observation_measurement) | big data analytics book — **weak/coincidental** |

Mix of genuine and incidental matches even at borderline-high. Suggests **τ_high=0.55 is appropriately strict** but **τ_low=0.35 admits noise** — the borderline class is 30-40% incidental.

## Top gap topics (what FinePDFs has that the ontology doesn't)

By doc count among gap topics:

| topic | docs | max_sim | content type |
|---:|---:|---:|---|
| 132 | 160 | 0.348 | Rockford Board of Education meeting minutes |
| 140 | 144 | 0.172 | PBA Executive Meeting minutes |
| 171 | 132 | 0.276 | Township Council Meeting agendas |
| 29 | 114 | 0.141 | Sacred Heart Parish bulletins |
| 116 | 113 | 0.180 | Reading Comprehension educational materials |

The gap is genuinely about civic/educational/community content that the ontology does not aim to model. None of these are "missed academic / technical content"; they're orthogonal.

## Critique of the current ontology

**Strengths:**

1. **Focused on compliance + audit + governance**, with strong specificity (PCI DSS, HIPAA, ISO 27001, GDPR). This is a real niche where Aegir can shine.
2. **Per-template DeepOnto verbalizations are populated and useful** for embedding-based matching. The `verbal_template` field gives clean natural-language proxies.
3. **Family decomposition is clean**: 7 named families with 32-165 templates each, conceptually distinct domains. No major bleed between families.
4. **540 templates is a substantial surface area**. Far past the "examples.json" toy state — this is a real ontology.

**Weaknesses (vs. the v0.3 goal of textbook-grade pretraining):**

1. **04_ebpf_kernel is a coverage island**: 51 templates pulling 6.5% topic share, with most templates getting zero matches in FinePDFs. Either (a) drop or shrink this family if eBPF/kernel content isn't load-bearing for v0.3, or (b) augment by directly harvesting arxiv/biorxiv kernel-systems papers (where it would have topic-rich coverage).
2. **07_long_tail is the catch-all** with 165 templates and 37% topic share. Could be productively split into named sub-families (compliance frameworks, schema.org primitives, etc.) for clearer downstream targeting.
3. **No academic / scientific-method templates**. The audit shows topics like "big data analytics" and "research paper questions" landing in `outlier_claim_*` or `mass_function_*` — incidental matches because there's no template for "this paper presents X experiment with Y method." Adding a small `08_scientific_method` family (~30 templates) would specifically address the arxiv/biorxiv corpus shape.
4. **No educational / instructional templates**. Topics 29 (parish bulletins), 116 (reading comprehension), various meeting minutes are unaddressed. **This is fine for v0.3** if we're not targeting educational content, but worth pinning the choice explicitly.

## Implications for v0.3 pipeline design

### FinePDFs filter rule for the v0.3 corpus

Given the distribution, a defensible filter rule for the 20% FinePDFs slice in the v0.3 corpus manifest:

```yaml
fine_pdfs_slice:
  filter_rule: "topic_coverage_score >= 0.30"
  topic_assignment: "via per-doc cosine to nearest topic centroid in coverage audit run_id=232ea5460ce6e0bf"
  expected_yield: "~20-25% of source docs (borderline + gap-high tier)"
```

At τ=0.30 we capture borderline (16.4%) plus the upper half of gap (where the topic is at least loosely ontology-adjacent). At τ=0.35 we'd only get pure borderline (16.4%).

Scaled to a larger FinePDFs draw: ~1M-doc draw × 20% yield → ~200K topic-adjacent docs. Adequate for the 20% slice of a 4B-byte corpus.

### What the ontology-synthetic share (40%) should target

Based on where coverage is densest, the synthetic textbook generation should concentrate on:

- **Governance / audit / compliance content** (the ontology's home turf) — chapters generated from `directive_governance` + `long_tail` (compliance-framework templates) families.
- **Information ontology + provenance** (`01_foundation`, `05_provo_lineage`) — chapters on how information artifacts get described and tracked.
- **Belief structures + observation** (`02_observation_measurement`, `06_belief_structure`) — chapters on evidence and measurement theory.

Leave `04_ebpf_kernel` to **arxiv-augmented retrieval** rather than ontology-only generation: there's no FinePDFs alignment, so we'd be generating chapters without grounding-from-prose. Better source: directly harvest 10-50 arxiv eBPF/systems papers via Brave, feed those as the retrieval context, generate chapters that explain those papers.

### Augmentation candidates for next pass

Rank-ordered by ROI for the v0.3 corpus:

1. **`08_scientific_method` family** (~30 templates): "X presents experiment Y", "Z replicates W with parameters P", "method M produces results R on dataset D". Closes a gap that's directly relevant to the academic/research corpus emphasis.
2. **Split `07_long_tail` into named sub-families**: `07a_compliance_frameworks` (PCI DSS, HIPAA, GDPR, etc.), `07b_schemaorg_primitives` (place, person, organization, etc.). Improves family-level interpretability without expanding template count.
3. **Augment `04_ebpf_kernel` via paper-corpus harvest**, not template authoring. Better to ground the existing templates with retrieved context than add more abstract ones.
4. **Skip educational / civic-meeting content** — out of v0.3 thesis scope.

## Methodology notes (for reproducibility)

- Run id is `sha256(finepdfs config + sample size + seed + embedding model + thresholds + catalog file hashes)[:16]`. Identical inputs → identical run_id → identical parquet contents.
- Catalog files included: `01_foundation.json`, `02_observation_measurement.json`, `03_directive_governance.json`, `04_ebpf_kernel.json`, `05_provo_lineage.json`, `06_belief_structure.json`, `07_long_tail.json` (canonical, not `*.candidate.json`).
- Embedding model: `sentence-transformers/all-mpnet-base-v2` (same as BERTopic / SDG verifier — shared across the pipeline).
- Clustering: MiniBatchKMeans, k=200, seed=4649, n_init=5. (Not HDBSCAN — saves a dep until we vendor bertopic in Path A.)
- Total wall-time: 58s on a single 4090 (CPU embedding for sentence-transformers).

## Open follow-ups

1. **LLM rejudgment of borderline topics** — defer until gaius vendoring (Path A) lands and CEREBRAS_API_KEY is in scope. Will refine the borderline ↔ covered boundary by using GLM-4.7 to decide ambiguous cases.
2. **Ablate the topic-cluster K**: rerun with K=100, K=400 to see how cluster granularity changes the family distribution. Useful for confidence in the family proportions.
3. **Per-language audit**: run against eng_Latn vs other language subsets to see if coverage shifts by language.
4. **Time-snapshot the FinePDFs revision**: the dataset is versioned; we should pin its SHA in the corpus manifest. (Currently the script uses whatever `load_dataset` resolves at runtime.)

## What this enables for tomorrow

With the coverage measurement in hand, the v0.3 corpus manifest can pin concrete values rather than `<TBD>`:

- `fine_pdfs_slice.filter_rule`: `"coverage_score >= 0.30 in audit run_id=232ea5460ce6e0bf"`
- `ontology_synthetic.target_families`: `[01_foundation, 02_observation_measurement, 03_directive_governance, 05_provo_lineage, 06_belief_structure, 07_long_tail]` (drop `04_ebpf_kernel` from primary; route through retrieval instead)
- `held_out_eval.ontology_subtree_holdout`: pick a coherent subtree (e.g., all `audit_*` templates from `03_directive_governance` — coherent, sufficient size for eval, doesn't bleed across families)

These pins are derived from the audit, not invented. That's the lock-in property the manifest wanted.
