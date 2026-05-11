# Rejection-sampling SFT → GRPO bootstrap pipeline

## Why this exists

The 24-hour `just p5-train` run on 2026-05-11 produced zero reward with
zero variance for 1690 steps. Two compounding causes:

1. **Constrained decoding was never connected to the generation path.**
   `composition_json_schema()` was built and printed in the launch
   summary, but `GRPOConfig.generation_kwargs` doesn't accept a
   `prefix_allowed_tokens_fn`, so TRL's `model.generate(...)` ran free.
   Fix: monkey-patch `model.generate` after `load_policy` to inject the
   constraint into every call (the patch survives FSDP wrap because
   `accelerator.unwrap_model` returns the same Python object).

2. **Qwen3.5-9B-Base doesn't follow prompts.** Even with constrained
   decoding forcing schema-valid JSON, the Base model picks
   semantically random `template_id` / `slot_fillers` from the
   540-branch catalog. Verifier reward is near-zero almost everywhere,
   so GRPO has no gradient signal to climb.

Switching to `Qwen3.5-9B-Instruct` would solve (2) but invalidate the
`SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` SAE adapter, which is trained on
Base's residual activations. Instruct's residual stream has drifted;
the SAE wouldn't reconstruct, and the interpretability dividend
(the SAE-Res-Qwen branding) goes away.

The standard fix is **SFT warm-start** — but generic instruction-tuning
data (Vicuna, Tülu 3, UltraChat) is the wrong augmentation for a narrow
schema-emission task. The right answer is task-specific SFT data.

## The pipeline

```
                   ┌─────────────────────────────────────────────┐
                   │ Qwen3.5-9B-Base                            │
                   │ (no adapters, no SFT)                      │
                   └───────────────┬─────────────────────────────┘
                                   │
                  ┌────────────────▼─────────────────────────┐
                  │ scripts/p5_rejection_sample.py          │
                  │ N_iters × 8 prompt variations × K       │
                  │ completions each, constrained decode,   │
                  │ score with verify(), keep R ≥ τ         │
                  └────────────────┬─────────────────────────┘
                                   │
                              rejection_samples.jsonl
                                   │
                  ┌────────────────▼─────────────────────────┐
                  │ scripts/p5_sft.py                       │
                  │ 1-epoch LoRA SFT on the high-reward     │
                  │ samples; completion-only loss; FSDP×2   │
                  └────────────────┬─────────────────────────┘
                                   │
                                p5-sft/
                                (LoRA adapter)
                                   │
                  ┌────────────────▼─────────────────────────┐
                  │ scripts/p5_train.py                     │
                  │   --init-checkpoint /raid/.../p5-sft    │
                  │ GRPO from warm-started policy           │
                  └─────────────────────────────────────────┘
```

The corpus is **self-distilled** — no external strong model required.
We rely on the Base model to occasionally hit non-zero `R` under
constrained decoding (the conjecture this run is testing); even a 1-5%
hit rate gives us a usable SFT corpus.

## Why this is paper-worthy

If self-distillation works, this is a clean two-step bootstrap with no
external API dependency:

- the locked verifier is sufficient signal,
- constrained decoding is sufficient prior structure,
- random-sampling-with-rejection extracts enough high-reward seeds
  to pull GRPO out of the zero-reward basin.

The contribution claim sharpens to: "the verifier + constrained-decode
+ self-distilled rejection corpus is sufficient to teach a base LM a
schema-constrained generation task without any external instruction
tuning or human demonstrations."

If rejection sampling **fails** to produce ≥30-50 high-reward samples,
we fall back to an external bootstrap (Claude / GPT-4 / Qwen3.5-72B-Instruct
via API), and the paper still tells the same story — just with an
external warm-start step.

## Operational notes

- **Constrained decoding is CPU-bound** at our schema size (540
  branches × 248K vocab). Observed: both ranks pinned at 100% CPU,
  GPUs idle at 0%, during generation steps. lm-format-enforcer's
  per-token `prefix_allowed_tokens_fn` call walks the schema state
  machine and intersects with the vocab. For paper-grade throughput
  we may need to swap to `xgrammar` or a vLLM-side grammar.
- **lmfe shim**: newer `transformers` releases moved
  `PreTrainedTokenizerBase` out of `tokenization_utils`; lmfe 0.11.3
  still imports the old path. `make_lmfe_prefix_allowed_tokens_fn`
  shims it before the import.
- **PEFT-LoRA + FSDP + bf16**: kept the same dtype-alignment forward
  pre-hook on every `nn.Linear` from `p5_train` in the SFT script.
  Without it, `RuntimeError: expected mat1 and mat2 to have the same
  dtype, but got: float != c10::BFloat16` recurs at LoRA forward.

## Recipes

```
just p5-rejection-sample --n-iters 50 --k-per-iter 16
just p5-sft --corpus /raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl
just p5-train --init-checkpoint /raid/checkpoints/p5-sft
```
