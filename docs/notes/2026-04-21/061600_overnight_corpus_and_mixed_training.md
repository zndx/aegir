# Overnight: corpus items 2–5 + mixed-corpus pretraining kickoff

*2026-04-21 06:16 MDT — autonomous execution per user authorization of 2026-04-20*

## What completed while you slept

### Corpus items 2–5 (all ✅)

| Item | Output | Size | Notes |
|------|--------|------|-------|
| 2 — SQL dialects | `/raid/datasets/aegir-corpus-v1/code/sql/` | 101 MB, 5 files | Stack v2 gated → pivoted to 5 non-gated HF sources (NSText2SQL, gretel, Clinton, know_sql, spider). |
| 3 — Apache docs | `/raid/datasets/aegir-corpus-v1/code/docs/` | 24.6 MB, 11 files | Shallow-clone + docs/ extract. Trino failed (no docs path); hive empty. |
| 4 — Ontology prose | `/raid/datasets/aegir-corpus-v1/ontology-prose/` | 1.29 MB, 4 files | 4,976 class definitions rendered from Schema.org (JSON-LD), DBpedia (RDF-XML), BFO (RDF-XML), CCO (Turtle). |
| 5 — Synthetic tables | `/raid/datasets/aegir-corpus-v1/synthetic/` | 117 MB | 4,547 templated Schema.org tables with labels sidecar. 83 seed classes. |

All four land in the manifest alongside the existing 26 GB FineWeb-Edu slice.

### Manifest composed

`scripts/compose_corpus_manifest.py` → `/raid/datasets/aegir-corpus-v1/MIXTURE.json`.

```
prose.fineweb-edu    w=0.550  files=8  26471.56 MB
code.sql             w=0.100  files=5    101.24 MB
code.docs            w=0.050  files=11    24.62 MB
ontology.raw         w=0.030  files=4      5.42 MB
ontology.prose       w=0.120  files=4      1.29 MB
synthetic.tables     w=0.100  files=1    117.21 MB
annotations          w=0.050  files=2     25.04 MB
Total: 7 slices, 26.12 GB
```

Weights were set to target 40% domain-specific exposure without overwhelming the prose backbone. Actual byte coverage per epoch is of course weight-weighted, not proportional to disk.

### Mixed-corpus training running

`scripts/pretrain_mixed.py` launched at 06:15 MDT, PID 505469, GPU 0.

- L12 D768 flat `["w12"]` (same config as the 1.774 bits/byte baseline)
- BlinkDL recipe: lr 6e-4→6e-5, warmup 10, betas (0.9, 0.99), adam_eps 1e-18, wd 0.001
- **2 GB byte budget** (~2× the FineWeb run) → 122,070 planned steps at bs=16, seq=512, grad_accum=2
- Workers=2; weighted IterableDataset samples one slice per sequence
- Output: `outputs/mixed/20260421T061556Z/`, logging every 50 steps to `metrics.jsonl`

Expected wall-clock: ~10–12 h on single 4090 (prior run hit ~1.4 steps/s). If things crash overnight I'll catch it in the morning log.

## What to expect when you check in

**Primary signal**: final bits/byte on the held-out mixed loader vs. the FineWeb-only 1.774 baseline. Because the dominant slice is still FineWeb, drift on the prose side should be small; the domain slices add vocabulary coverage (SQL keywords, Apache project jargon, ontology class names, synthetic table rows) that the FineWeb-only model never saw.

**Secondary**: whether the model gracefully handles bytes that don't appear in FineWeb (e.g. `\x03` doc separators from SQL shards, structured table rows from item 5). The loss curve per slice is not logged in this first run — the next run can stratify.

## Items deferred to next waking hours

1. **Stack v2 dedup access request** — cleaner SQL slice once gated. No-op until credentials granted.
2. **FinePDFs-Edu subset** — prose depth. Good deferred target if we hit prose plateaus.
3. **Stratified held-out eval** — need separate loaders per slice to measure `bits/byte[sql]`, `bits/byte[docs]`, etc. Straightforward extension of the eval harness.
4. **Multi-GPU pretrain** — current single-GPU utilisation at 95%. DDP port is mechanical; overnight run fits in the one-GPU envelope.

## Files added this session (autonomous)

- `scripts/download_corpus_item2.py` (+ Stack v2 fallback path to 6 non-gated SQL sources)
- `scripts/download_corpus_item3.py` (Apache docs shallow-clone pipeline)
- `scripts/gen_corpus_item4.py` (ontology → prose renderer, multi-format)
- `scripts/gen_corpus_item5.py` (Schema.org templated table generator)
- `scripts/compose_corpus_manifest.py` (weighted manifest builder)
- `scripts/pretrain_mixed.py` (BlinkDL recipe over manifest slices)

All committed to trunk prior to kickoff (last pre-run SHA: `79942ab`). Any new artifacts from the mixed run are in `outputs/mixed/` and not committed.

## Morning action menu

Pick one, depending on what the overnight run shows:

- **If curve converged cleanly below 1.774**: domain mix helps; lock the recipe, extend to 10 GB budget, add held-out stratification, then swap to multi-GPU for a "proper" 20B byte run on the HGX system.
- **If curve stalled or diverged**: inspect per-slice contributions. Likely suspect is the raw ontology slice (OWLs are structurally anomalous bytes) — drop `ontology.raw` and re-run.
- **If final ≈ 1.774**: mix is benign but doesn't help prose — pivot to downstream probes (CTA accuracy on GitTables slice) to see if domain vocabulary translates to task metric gains.
