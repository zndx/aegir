---
language:
- en
license: cc-by-4.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- feature-extraction
tags:
- ontology
- structured-generation
- rejection-sampling
- bertopic
- topic-alignment
- self-distillation
- rlvr
- aegir
pretty_name: "SDG BERTopic Input-Intermediate-Output Correspondence"
---

# SDG BERTopic Input-Intermediate-Output Correspondence (v0.1)

**Status:** v0.1, peer-review preview. Schema and methodology may evolve.
**Curator:** [@zndx](https://huggingface.co/zndx)
**Project:** Aegir — `https://github.com/<...>` (link pending public release)

A research dataset capturing the **full computational trace** from
constrained-decode-generated ontology compositions through BERTopic
intermediate state to verifier scoring. Built for reproducibility,
verifier-methodology validation, and as a paired data resource for
upcoming Aegir model checkpoints.

## What's in it

`correspondence.parquet` — 1,405 rows, ~12.4 MB. Each row captures
the complete **input → intermediate → output** trace for one
rejection-sampled ontology composition:

| Field | Type | Description |
|---|---|---|
| `sample_id` | string | `v2_NNNNN` or `v3_NNNNN` |
| `policy` | string | `base` (665 rows) or `sft-r1` (740 rows) |
| `corpus_version` | string | `v2` (base policy) or `v3` (SFT-r1 policy) |
| `prompt_variation_idx` | int32 | 0–7; index into the 8-element prompt-template rotation |
| `prompt_text` | large string | Full chat-template input (system + few-shot + user) |
| `raw_completion` | large string | Policy's exact JSON output, pre-parsing |
| `composition_template_ids` | list\<string\> | Parsed ordered list of `template_id`s |
| `composition_slot_fillers_json` | large string | JSON-encoded list of slot-filler dicts (one per entry) |
| `n_entries` | int32 | Length of the composition |
| `verbalizations` | list\<string\> | Manchester-syntax verbalisations (one per entry) |
| `n_verbalizations` | int32 | Number of valid verbalisations rendered |
| `t_v_centroids_flat` | list\<float32\> | KMeans centroids on verbalisation embeddings (flattened) |
| `t_v_centroids_shape` | list\<int32\> | `[k_v, 384]` to reshape `t_v_centroids_flat` |
| `t_v_k` | int32 | Actual cluster count (≤ requested k=12; truncated when n_verbalizations < 12) |
| `alignment_per_t_i_centroid` | list\<float32\> | Per-T_I-centroid max-cosine score; shape `(100,)` |
| `raw_alignment` | float32 | Verifier's pre-normalisation R_D score |
| `normalized_alignment` | float32 | Verifier's R_D after null-stats normalisation |
| `r`, `r_a`, `r_b`, `r_c`, `r_d` | float32 | Verifier components and aggregate |
| `r_stored` | float32 | Original R value at rejection-sampling time (sanity check) |

**Drift check:** for all 1,405 rows, the re-computed `r` equals the
stored value to within tolerance 1e-3 — the dataset is internally
consistent and re-derivable from the raw composition + catalog.

### Side files in the dataset repo

- `T_I_centroids.npy` — `(100, 384)` float32. The **input-corpus
  reference clustering** all rows align against. Computed once from
  the pinned input corpus (SchemaPile + FinePDFs-lab held-out
  partition).
- `T_I_metadata.json` — encoder name, n_docs, KMeans config.
- `verifier_meta.json` — C1-locked weights, thresholds (`τ_B`,
  `L_target`), null-distribution stats.
- `catalog_meta.json` — full list of template IDs in the catalog
  used to score these compositions (`v0.7.0-combined`).
- `extraction_stats.json` — aggregate stats from the extraction run.

## How it was produced

### Stage 1 — corpus generation (rejection sampling)

Two rejection-sampling rounds against the locked verifier, using
xgrammar-based constrained decoding to guarantee schema-valid JSON
output:

| Corpus | Policy | n kept | Acceptance | R mean | Unique templates |
|---|---|---:|---:|---:|---:|
| v2 | Qwen3.5-9B-Base (cold) | 665 | 94.5 % | 0.522 | 171 / 540 |
| v3 | Qwen3.5-9B-Base + SFT-r1 LoRA | 740 | 96.4 % | 0.513 | 88 / 540 |

Constrained-decoding details: `xgrammar 0.2.0` with the full
540-branch discriminated-union schema over catalog `template_id`
values. With this constraint, every output is structurally valid
JSON with an in-catalog `template_id`, so R_A = 1 on every kept
sample. Threshold for inclusion: `R ≥ 0.3`. Generation parameters:
`temperature=1.0`, `top_p=0.95`, `max_new_tokens=640`, sampling from
8 rotated prompt variations (different few-shot examples + target
composition sizes 3 / 6 / 9).

### Stage 2 — BERTopic correspondence extraction

For each kept composition, this dataset captures:

1. The verbalisation of each composition entry via the catalog's
   per-template `verbal_template` (slot fillers substituted by
   regex).
2. A fresh KMeans clustering (`T_V`) over the verbalisation
   embeddings, using the same encoder
   (`sentence-transformers/all-MiniLM-L6-v2`) and random state
   (`4649`) the verifier uses internally.
3. Per-row alignment vectors against the pre-cached **T_I** topic
   model (k=100 centroids fit on 10,000 documents from the pinned
   input corpus).
4. The full four-component verifier score
   (R_A, R_B, R_C, R_D, aggregate R) recomputed deterministically.

The KMeans + alignment computation is deterministic given
`(verbalizations, encoder, random_state)`. The pipeline runs in
~50 s on CPU for 1,405 rows.

## What this enables

- **Reproduce the verifier exactly** from raw composition input
  using only the side files (`T_I_centroids.npy` + catalog + weights).
- **Build alternative scoring functions** over the same intermediate
  state — e.g. different alignment metrics, alternative weightings,
  per-centroid weighting schemes.
- **Inspect the policy's topic distribution** in T_I space — every
  row carries its T_V centroids and their alignment with each of the
  100 reference T_I centroids.
- **Compare iterations** — `policy in {base, sft-r1}` lets you
  observe how SFT on a 665-sample self-distilled corpus shifts the
  resulting policy's topic concentration.
- **Train downstream models** that take a scenario and emit a
  composition; the dataset is paired and verifier-scored end-to-end.

## Findings from this corpus (selected)

A held-out 50-scenario evaluation on each policy produced the
following cross-stage comparison:

| Stage | overall mean R | good-scenario mean | bad-scenario mean | R_A pass rate | AUC (good vs bad) |
|---|---:|---:|---:|---:|---:|
| Base | 0.208 | 0.205 | 0.210 | 0.55 | 0.478 |
| SFT-r1 (665 samples × 2 epochs) | **0.289** | 0.311 | 0.268 | 0.68 | **0.590** |
| SFT-r2 (740 samples × 2 epochs) | 0.318 | 0.309 | 0.327 | 0.76 | 0.475 |

A clean positive result for one round of self-distilled SFT (+39 %
overall, symmetric improvement on good vs bad scenarios, plus AUC
discrimination); a clean **negative result on naive iteration** —
round 2 raises the floor on bad scenarios but doesn't move the
ceiling on good ones, and AUC discrimination regresses to random.
Round-2 corpus has only **88 unique catalog templates vs round-1's
171**, evidence of mode collapse.

These findings are intentionally surfaced in the dataset because they
shape what's defensible to claim about the corpus and motivate the
**diversity-preserving extensions** future rounds will explore (mix-in
of round-1 samples during round-2 SFT, anti-clustering reward
penalties, higher round-2 sampling temperature).

## Limitations

- **Domain coverage**: the corpus is over-represented in
  `sdg:` / `cco:` (data warehouse governance & observability)
  namespaces, reflecting the input corpus composition.
- **Template coverage**: 183 of 540 catalog templates appear; the
  remainder were unreachable from the 8 prompt variations × policies
  used here.
- **Verbalisation gaps**: some catalog templates have empty
  `verbal_template` fields; those entries are dropped from
  `verbalizations` (hence `n_verbalizations ≤ n_entries`).
- **Single base model**: all generations come from
  `Qwen/Qwen3.5-9B-Base`. Cross-model generalisation is not assessed.
- **Verifier weights are C1-locked** (`a=0.50, b=0.05, c=0.45`); the
  dataset values reflect that specific aggregation. Component scores
  are provided so alternative aggregations can be applied.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("zndx/sdg-bertopic-correspondence-v0.1", split="train")
print(ds[0]["composition_template_ids"])
print(ds[0]["t_v_centroids_shape"])  # e.g. [3, 384]

# Reconstruct T_V centroids:
import numpy as np
row = ds[0]
shape = row["t_v_centroids_shape"]
t_v = np.array(row["t_v_centroids_flat"], dtype=np.float32).reshape(shape)

# Load the reference T_I centroids (download from this repo's files):
t_i = np.load("T_I_centroids.npy")  # (100, 384)

# Pairwise cosine sim (both L2-normalised already):
similarities = t_v @ t_i.T  # (k_v, 100)
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{sdg-bertopic-correspondence-v01,
  title  = {SDG BERTopic Input-Intermediate-Output Correspondence (v0.1)},
  author = {Hill, Ryan and contributors},
  year   = {2026},
  url    = {https://huggingface.co/datasets/zndx/sdg-bertopic-correspondence-v0.1}
}
```

## Related artifacts

- [`zndx/sdg-sft-r1`](https://huggingface.co/zndx/sdg-sft-r1) — LoRA
  adapter that produced the v3 portion of this corpus.
- [`zndx/sdg-sft-r2`](https://huggingface.co/zndx/sdg-sft-r2) — second-
  round LoRA, demonstrating the diminishing-returns finding above.

## Changelog

- **v0.1** (2026-05-17) — initial release. 1,405 rows from two
  self-distillation rounds against Qwen3.5-9B-Base + the C1-locked
  SDG ontology verifier.
