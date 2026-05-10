# v2 corpus mix kickoff: SQaLe + SchemaPile + FinePDFs-lab

*2026-04-23 23:24 MDT*

## What changed from v1

| Slice | v1 weight / source | v2 weight / source |
|---|---|---|
| prose.fineweb-edu | 0.55 / 26 GB on disk | **0.35** / same disk |
| prose.finepdfs-lab | — | **0.10** / streaming filter on `HuggingFaceFW/finepdfs` (in-progress, 1.0 GB at kickoff time, growing toward 20 GB cap) |
| code.schemapile | — | **0.10** / `trl-lab/schemapile` rendered as CREATE TABLE + sample VALUES (61 MB, 22 989 schemas) |
| code.sql | 0.10 / 100 MB v1 fallbacks | **0.05** / same (kept) |
| code.docs | 0.05 / 24 MB Apache | unchanged |
| ontology.raw | 0.03 / 5 MB | unchanged |
| ontology.prose | 0.12 / 1.3 MB | **0.10** |
| synthetic.tables | 0.10 / 117 MB | **0.07** |
| sqale.nl-sql | — | **0.10** / `trl-lab/SQaLe-text-to-SQL-dataset` rendered as schema + `-- Q:` + SQL (14.6 GB, 486k rows) |
| annotations | 0.05 / 25 MB | **dropped from train** — repurposed as `eval.spider` (held-out only; SQaLe was generated from these) |

## Notable pivots from the original plan

- **OpenCoder RefineCode → SchemaPile**. RefineCode-code-corpus-meta is metadata only (file paths into Stack v2, which is gated). The published `opc-fineweb-code-corpus` is *prose mentioning code* from CommonCrawl, not language-tagged code. SchemaPile is the actual upstream of SQaLe and gives us 22 989 real-world DDLs at the same quality bar SQaLe inherits.
- **SQaLe size 14.6 GB rendered, not 3.6 GB raw**. The Hugging Face card lists ~3.6 GB on-disk but rendered to schema-prefixed text format with `\x03` separators it's ~4× larger. Disk usage fine (241 GB free on /raid).
- **FinePDFs-lab eval slice empty at v2 kickoff**. Streaming filter writes the train shard until the last 512 MB, then writes eval shard. At 1 GB collected and growing, eval file is still 0 bytes. Manifest composer skipped that eval slice — so v2 has 4 of 5 held-out slices wired. FinePDFs eval will land in v3.

## Stratified eval — new in v2

`pretrain_mixed.py` now runs per-slice held-out eval every 5 000 optimizer steps (`--eval-every 5000`). Each slice gets 16 MB sampled (`--eval-bytes 16_000_000`) under `model.eval()` + `torch.no_grad()`, mean nats/byte → bits/byte logged to `metrics_eval.jsonl`.

Smoke test (50 MB budget, eval-every 200) at step 200 produced:

| slice | bits/byte (fresh model) |
|---|---|
| eval.fineweb-held | 2.95 |
| eval.schemapile-held | 2.96 |
| eval.sqale-held | 2.53 |
| eval.spider | 4.49 |

These are baseline reads of the *random-init* model after just 200 steps — the trajectory matters, not the values.

## v1 → v2 success criteria

- `bits/byte[fineweb-held]` ≤ 1.78 (no prose regression vs FineWeb-only baseline 1.774)
- `bits/byte[sqale-held]` substantially below `bits/byte[fineweb-held]` (proves NL+DDL+SQL learning, not distribution)
- `bits/byte[schemapile-held]` < `bits/byte[fineweb-held]` (proves DDL structure transfer)
- `bits/byte[spider]` should *also* drop substantially without ever being trained on — proves SQaLe coverage transfers to its source distribution

## Run state at kickoff

- v2 PID 2187438, GPU 0, output `outputs/mixed-v2/20260426T232240Z/`
- 122 070 planned steps at bs=16 seq=512 grad_accum=2 (same envelope as v1)
- Step 100: loss 2.11 — same trajectory as v1 (which hit 2.15 at step 100)
- ETA: ~10–12 h on single GPU
- Background: `download_corpus_item7_finepdfs.py` continues filling the 20 GB cap; will eventually populate the 512 MB eval shard for v3

## Files added / modified this session

- `scripts/download_corpus_item6_sqale.py` (NEW)
- `scripts/download_corpus_item7_finepdfs.py` (NEW, in-progress run)
- `scripts/download_corpus_item8_schemapile.py` (NEW; was named `_opencoder_sql.py` until OpenCoder pivot)
- `scripts/compose_corpus_manifest.py` (MODIFIED — adds v2 SLICES + EVAL_SLICES, dual-manifest output)
- `scripts/pretrain_mixed.py` (MODIFIED — `_eval_slice`, `run_eval`, `--eval-manifest`/`--eval-every`/`--eval-bytes`)

## Next morning

1. Read `metrics_eval.jsonl` for stratified bits/byte per slice across the run.
2. Compare to v1's 1.202 headline: did headline drop, *and* did per-slice eval show real domain learning?
3. If FinePDFs download completed, recompose manifest to pick up the eval slice for v3 reference.
4. If headline stalled, inspect per-slice losses for the offending slice — most likely candidates are `code.schemapile` (rendered DDL is structurally weird) or `ontology.raw` (held over from v1, raw OWL bytes).
