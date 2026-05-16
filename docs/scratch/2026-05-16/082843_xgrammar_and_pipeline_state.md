# 2026-05-16 — xgrammar adoption + pipeline state

## Headline

xgrammar 0.2.0 replaces lm-format-enforcer as the default constrained-
decode backend. Combined with re-enabling the full 540-branch schema
(now affordable), corpus-extraction throughput improved **~14×** over
the lmfe-lite path from the May 11/12 runs.

Concrete numbers (smoke comparison, 32 samples, single 4090):

| Backend  | Schema | Wall    | Kept | nz_frac | reward_mean |
|----------|--------|---------|-----:|--------:|------------:|
| lmfe     | lite   | 10.4 m  |   7  |   0.27  |       0.126 |
| xgrammar | lite   |  2.3 m  |   5  |   0.34  |       0.087 |
| xgrammar | full   |  2.5 m  |  23  | **1.00**|       0.513 |

Full schema with xgrammar is the production setting. R_A hallucinations
are eliminated (every parseable composition has a valid template_id),
so the verifier-acceptance rate goes from ~25% to ~94%.

## Why xgrammar is faster

lmfe walks the entire 248K Qwen vocab on every token decision to
intersect with the parser state; that's O(V) per token. xgrammar
pre-compiles a token-trie / pushdown automaton at schema-build time,
making per-token enforcement O(1). At our vocab/schema scale that's
a 5-15× per-token speedup.

The schema-compile cost also dropped: 0.7s for full 540-branch (vs
lmfe's ~7s for either lite or full).

## xgrammar 0.2.0 gotchas

Two real ones, both fixed in `src/aegir/rl/decoding.py`:

1. **Vocab-size mismatch**. xgrammar wants the *model's* LM-head vocab
   size, not the *tokenizer's*. Qwen3.5-9B-Base: `tokenizer.vocab_size=
   248044` but `model.config.vocab_size=248320`. The LM head can sample
   tokens in `[248044, 248320)`; a grammar built on the smaller range
   raises `AssertionError: accept_token returned False` on those.

2. **tvm-ffi strict typing on `accept_token`**. xgrammar 0.2.0's
   `contrib.hf.LogitsProcessor.__call__` does `sampled_token =
   input_ids[i][-1]` and hands the 0-d tensor straight to
   `GrammarMatcher.accept_token`. The tvm-ffi binding rejects this
   with `TypeError: Expected int but got ffi.Tensor`. Patched with a
   subclass that coerces to Python int.

The `make_xgrammar_logits_processor_factory` helper returns a zero-arg
callable — required because the matcher state is single-use across
`generate()` calls.

## xgrammar-2 (the paper)

[arxiv 2601.04426](https://arxiv.org/abs/2601.04426). Three new ideas
vs 0.2.0:

1. **TagDispatch** — runtime grammar switching triggered by emitted
   tags. Lets a model fluidly transition between free-form thinking
   and structured tool calls.
2. **Cross-Grammar Cache** — reuses substructures across schemas.
3. **Earley-based adaptive token mask cache** + JIT compression.

Headline: 6× faster compilation. For our pipeline that saves ~1.5s
per run — not material. The structural feature (`TagDispatch`) is
interesting for future multi-stage agentic loops, but for the current
"prompt → constrained composition → score" pipeline nothing in
xgrammar-2 changes our story. **0.2.0 stays the production backend.**
When/if xgrammar-2 ships as code, we revisit.

## Pipeline state (end of day)

Five durable changes landed today:

1. `aegir.rl.decoding` gained `make_xgrammar_logits_processor_factory`
   (commit `4ba0324`).
2. `scripts/p5_rejection_sample.py` gained `--decode-backend
   xgrammar|lmfe` flag, default xgrammar (`4ba0324`).
3. `scripts/p5_train.py` gained the same flag in its `[6b/9]` hook
   (`53004f9`).
4. `scripts/p5_sft.py` auto-converts FSDP `.distcp` → PEFT
   `adapter_model.safetensors` as a `[6/6]` teardown step on rank 0
   (`bf23550`). Eliminates the manual conversion script.
5. `scripts/p5_eval.py` is new (`7907e60`). Loads a policy + adapter,
   evaluates on the 50-scenario held-out set under xgrammar-full,
   writes a per-checkpoint results JSON.

## Corpus state

- v1 (May 11/12, lmfe-lite): 75 samples, R mean 0.455 — too small to
  differentiate SFT vs cold-start.
- **v2 (May 16, xgrammar-full): 665 samples**, R mean 0.522, median
  0.526, max 0.981. 171 unique catalog template_ids covered (32% of
  540). 38 min wall-clock.

v2 is at `/raid/checkpoints/p5/sft-corpus/rejection_samples_v2.jsonl`.

## SFT state

`p5-sft-r1` trained on the v2 corpus:
- 2 epochs, 84 grad steps, 69.6 min on FSDP×2
- Final train_loss 0.216, token_acc 95.2%, entropy 0.194
- Adapter at `/raid/checkpoints/p5-sft-r1/adapter_model.safetensors`
  (256 LoRA tensors)

## GRPO state (this evening)

Launched `p5-warm-r1` (warm-start from sft-r1, 300 steps, xgrammar,
num_gen=4, per_device=2). Also launched `p5-cold-r1` in parallel —
killed after ~1h because two-process CPU contention pushed step times
to 18+ min on warm and 26+ on cold.

Surprising finding: **post-kill, warm-r1's step time barely changed
(~17.6 min vs ~18.3 min)**. The bottleneck is the *intrinsic per-step
verifier CPU cost* (BERTopic + sentence-transformers + KMeans × 16
completions), not GPU or CPU-contention. Two parallel GRPO runs is too
much for this box's verifier path; one run is fine but slow.

This means the 8-hour evening window can only afford ~25-30 GRPO steps
solo. Phases 3-6 (post-GRPO rejection-r2 → SFT-r2 → GRPO-r2 → LR
ablation) of this evening's plan are infeasible at the current pace.

First reward metric at warm-r1 step 5: **0.2591** — the same plateau
value as the May 12 cold-start. Conjecture: 9× more SFT samples
(665 vs 75) still doesn't shift the starting point because GRPO updates
from a tiny LoRA delta land in the same basin.

## What this implies for next session

Two open questions blocked by today's verifier-throughput floor:

1. **Does the 0.26 plateau actually move with more steps?** Need 100+
   GRPO steps at LR=1e-5 to see policy drift, per the May 12 finding.
   At 18 min/step that's 30+ hours per run.

2. **Does SFT bigger than 665 / 2 epochs differentiate?** Possible the
   665-sample corpus is still too small. Could push to 2000+ samples
   in a longer rejection-sampling round (xgrammar makes this feasible
   in a few hours).

**Mitigation idea: parallelise the verifier.** Run the 16 per-step
`verify()` calls in a multiprocessing pool. Each `verify()` is
independent — embarrassingly parallel. With 4-8 workers, per-step
verifier cost could drop 4-8×, putting us at 3-5 min/step — back in
budget for 300-step runs.

That's the highest-leverage engineering improvement for next session.
The model is GPU-idle for ~70% of each step while verify() runs serial
on CPU.

## Held-out evaluation now ready

`scripts/p5_eval.py` evaluates a policy on the 50-scenario held-out
set (25 good + 25 bad scenarios authored before P5 began).
Cross-checkpoint comparison metric: **AUC of (good vs bad) mean_R**.

When the warm-r1 GRPO run completes (or is stopped), we'll have
three comparable checkpoints:

- Base (no adapter)
- SFT-r1 (sft on 665-sample corpus, no GRPO)
- warm-r1 (sft-r1 + N steps of GRPO)

This is the comparison the user wanted from the start. Eval per
checkpoint takes ~30-60 min on single GPU.
