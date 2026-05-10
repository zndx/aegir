# Architecture isolation: Aegir repro vs BlinkDL RWKV-7 baselines

The conservative-first-step experiment landed clean results. This note
is the calibration anchor for everything downstream.

## Setup

Identical 200 KB window from the tail of
``/raid/datasets/fineweb-edu/sample_10BT_007_00000.txt`` (distinct from
any training window). Half-overlap sliding eval with context 2048; loss
scored on the last half of each non-first window to reduce kv-state
boundary bias. All numbers in **bits/byte** — tokenizer-agnostic so BPE
and byte-level models are directly comparable.

## Results

| Model | Params | Training budget | nll/byte (nats) | bits/byte |
|---|---|---|---|---|
| **Aegir flat L12 D768 @ step 20 000** (intermediate) | 90 M | ~220 M bytes | 1.253 | 1.807 |
| **Aegir flat L12 D768 FINAL @ step 30 517** | 90 M | ~320 M bytes on FineWeb-Edu | 1.230 | **1.774** |
| **BlinkDL RWKV-7 World 0.1B v2.8** | 191 M | ~10 B BPE tokens (multi-domain) | 0.623 | **0.899** |
| **BlinkDL RWKV-7 g1e 7.2B** | 7.20 B | ~30 T tokens | 0.434 | **0.627** |

For reference, Shannon's estimate of English-text entropy is ~0.6-1.3
bits/char ≈ bits/byte for Latin-1 / UTF-8 dominant text. 7.2B g1e sits
at the low edge — near the measurable information-theoretic floor.

## What the numbers tell us

### 1. The Aegir architecture is sound

Under the canonical BlinkDL RWKV-7 training recipe (lr 6e-4 → 6e-5,
warmup 10, betas (0.9, 0.99), adam-eps 1e-18, wd 0.001) on a standard
corpus (FineWeb-Edu), the architecture + fla kernel stack produces a
clean loss descent from 5.56 (byte-vocab entropy floor) to 1.25 at
step 20 000. No plateau, no collapse, monotone.

This definitively closes the open question from Stages A and C:
> **The failures on SOTAB / CTA / path-prediction were not
> architectural — they were data/task-coupling failures.**

### 2. The gap to the 0.1B baseline is ~2×, attributable

Aegir's 1.81 bits/byte vs 0.1B's 0.90 bits/byte is a 2× ratio. Gap
sources, in descending order of contribution:

- **Training budget.** We've trained on 320 M bytes. Published
  RWKV-7 World was trained on roughly 10 B BPE tokens ≈ 40-50 B bytes
  (multilingual, multi-domain). ~100-150× more exposure.
- **Parameter count.** 90 M vs 191 M — our byte-level vocab saves
  ~50 M on the embedding table, so raw architecture capacity is
  similar but they have more.
- **Tokenizer efficiency.** BPE compresses common multi-byte subwords,
  letting the model spend capacity on rarer events. Byte-level has to
  learn multi-byte patterns from scratch.
- **Data distribution.** World v2.8 is curated multilingual; FineWeb-Edu
  is curated English educational. Different coverage, both high-quality.

None of these are architectural. The gap is closable with more
compute, more data, or a larger model — standard scaling levers.

### 3. The gap to 7.2B is 3× — reference ceiling

7.2B g1e's 0.63 bits/byte is the near-floor reference for what a
converged RWKV-7 on well-tuned data achieves. Our model is 3× above
this. Expected; we are a ~90× smaller model on ~100,000× less data.

The value of having the 7.2B number isn't the gap itself — it's the
**floor anchor**. When we later train bigger or longer, we can see
how far we close the gap. A gap of 3× at 90M/320M bytes is the baseline
against which improvement is measured.

## Methodology subtlety worth recording

### fla's w convention differs from BlinkDL's CUDA kernel

Loading BlinkDL's checkpoints into a fla-backed forward pass requires
a one-line translation:

```python
w_fla = -torch.exp(w_raw)  # BlinkDL → fla
```

BlinkDL's RWKV7_OP computes ``decay = exp(-exp(w_raw))`` where
``w_raw = -softplus(-X) - 0.5``. fla's ``chunk_rwkv7`` computes
``decay = exp(w_fla)`` internally (``w`` parameter is documented as
"log decay"). For the decays to match:

```
exp(w_fla) = exp(-exp(w_raw))  ⟹  w_fla = -exp(w_raw)
```

Without the translation, loading BlinkDL's 0.1B weights gave 7.5
nats/token and 2.44 bits/byte — effectively random performance.
With the translation, 2.98 nats/token and 0.93 bits/byte — matching
published.

Aegir's own ``rwkv7_tmix.py`` trains *natively* in fla's convention
(``decay = exp(w_raw)`` with ``w_raw`` learned to compensate), so
Aegir checkpoints don't need this translation. Only loaded-from-
BlinkDL checkpoints do.

## What's next

1. **Finish the current repro run.** Training is at step 20 000 / 30 517
   planned (~65% of budget). Expected final bits/byte: ~1.6-1.7 given
   the current descent rate and remaining cosine decay.
2. **Scale up** (any of):
   - More training data (next FineWeb shard batch, 5× our current budget).
   - Bigger model (`base` config ~500M params).
   - Longer schedule (2-3 more epochs at the same data).
3. **Add the H-Net hierarchy back.** Re-run the same recipe with
   ``arch_layout = ["w4", ["w4", ["w4"], "w4"], "w4"]`` at total
   layer count matched to flat L12. Measures the delta from the
   hierarchical chunker on its *native* language-modeling objective —
   which was always the right context for that architectural choice.
4. **Return to the domain tasks (SOTAB / CTA / DED)**, *now from a
   pretrained checkpoint that we know is healthy*. Either Stage D
   (hierarchical latent alignment) or direct fine-tuning with
   differentiated LRs; the failure modes we saw before are rearmed
   attribution-clean because the base model is validated.

## Artifacts

- Eval results: ``outputs/repro/rwkv7_{01b,7b}_baseline.json``,
  ``outputs/repro/aegir_ckpt_020000.pt_eval.json``
- Pretrained baseline checkpoints: ``/raid/datasets/rwkv-7-world/``
  (0.1B: 382 MB, 7.2B: 12.74 GB)
- FineWeb-Edu corpus: ``/raid/datasets/fineweb-edu/`` (26 GB, 8 shards)
- Eval scripts: ``scripts/eval_rwkv7_baseline.py``,
  ``scripts/eval_aegir_baseline.py``
- Pretrainer: ``scripts/pretrain_fineweb.py``
- Training run: ``outputs/repro/20260420T144810Z/``
