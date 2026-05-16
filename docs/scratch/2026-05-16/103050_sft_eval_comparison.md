# 2026-05-16 — SFT-r1 vs base, held-out eval comparison

## Headline

The 665-sample 2-epoch SFT (`p5-sft-r1`) improves overall mean reward on
the held-out 50-scenario set by **+39%** vs base model. This is the
signal we couldn't see in the GRPO per-step `reward_fn/mean` metric
(which stays clamped at the 0.2591 plateau for 13 steps regardless of
SFT or no SFT).

## Numbers

Held-out set: `tests/p5_held_out_50/labels.json` — 50 scenarios
(25 "good" + 25 "bad"), 4 compositions per scenario, scored under the
locked verifier (C1 weights {a=0.50, b=0.05, c=0.45}) with xgrammar-full
constrained decoding.

| metric              |   base |  sft-r1 |        Δ |
|---------------------|-------:|--------:|---------:|
| overall_mean_r      | 0.2077 | 0.2893  | **+39.3%** |
| overall_median_r    | 0.1269 | 0.3269  | **+157.6%** |
| overall_std_r       | 0.2227 | 0.2296  |    +3.1% |
| good_mean_r         | 0.2054 | 0.3107  | **+51.2%** |
| bad_mean_r          | 0.2101 | 0.2679  |   +27.5% |
| auc_good_vs_bad     | 0.4784 | 0.5904  |   +23.4% |
| r_a_pass_rate       | 0.5500 | 0.6800  | **+23.6%** |

**Median jumped from 0.13 to 0.33 (+158%)** — SFT raised the *floor*
much more than the ceiling. The base model had many scenarios where it
scored 0 (parse failures, hallucinated template_ids before the schema
constraint kicks in fully). SFT-r1 produces fewer of those degenerate
cases.

**R_A pass rate from 55% → 68%** — even with constrained decoding
forcing schema-valid JSON on both runs, base hits more max-token
truncations and parse failures. SFT learned to emit shorter, well-formed
compositions.

## Why the GRPO plateau hid this signal

The GRPO per-step reward (`reward_fn/mean = 0.2591`) is the *mean across
8-16 in-distribution rejection-sample-style prompts*. Both base and
SFT-r1 hit the same plateau on those because:

1. The prompts in the GRPO dataset are very similar to the prompts in
   the SFT corpus (both come from `PromptConfig.user_template` with
   varied few-shot seeds).
2. The model+adapter is constrained to schema-valid JSON.
3. Under the bimodal verifier reward structure (R_A=0 invalid /
   R_A=1+R_B+R_D valid), most outputs land at R_A=1+modest-R_D ≈ 0.26.

So the GRPO step metric stays flat. But the held-out set uses
**different prompt framing** ("Compose an ontology that captures this
scenario: ...") and **different scenarios** (HIPAA, eBPF, schema
evolution, etc.). The base model breaks down on this OOD shift; SFT-r1
generalises substantially better.

## Practical implication

For our paper claim, the SFT-only result is already publishable:
**a 665-sample self-distilled SFT corpus, generated from base + xgrammar
+ verifier, improves out-of-distribution policy quality by ~40%**. We
don't *need* the GRPO step to make a contribution; the bootstrap
mechanism alone is the demonstration.

GRPO might still help (compounding gains), but its value-add must be
measured on the held-out set, not on the in-distribution GRPO step
metric.

## Round-2 in flight

Launched `p5_rejection_sample.py --init-checkpoint /raid/.../p5-sft-r1`
to generate a round-2 corpus using SFT-r1 as the rollout policy.
Expected: higher acceptance, broader template coverage, higher mean
reward in the kept samples than the v2 corpus (665 from base).

Next step (post-user-return): SFT round 2 on that corpus, then eval
again to measure if iterated SFT compounds.

## Verifier-CPU floor remains the throughput limiter for GRPO

Per-step GRPO time stuck at ~18 min on this box because each step does
16 verify() calls serially (BERTopic + sentence-transformers + KMeans
on CPU). Killing the parallel cold-r1 run didn't materially help.

**Next-session engineering bet:** parallelise `verify()` across the
16-per-step completions via a `multiprocessing.Pool`. Each call is
embarrassingly parallel; with 4-8 workers the per-step verifier cost
could drop 4-8×, putting 300-step GRPO runs comfortably in budget.

## Artifacts

- `/raid/checkpoints/p5/eval_base_held_out_50.json`
- `/raid/checkpoints/p5-sft-r1/eval_held_out_50.json`
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_v2.jsonl` (665, base)
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_v3.jsonl` (in flight, SFT-r1)
- `/raid/checkpoints/p5-sft-r1/adapter_model.safetensors` (256 LoRA tensors)

## Today's durable code changes

- `4ba0324` rl: add xgrammar constrained-decode backend — 14× corpus throughput
- `53004f9` p5_train: --decode-backend xgrammar|lmfe in [6b/9] hook
- `bf23550` p5_sft: auto-convert FSDP .distcp → PEFT adapter at end of train()
- `7907e60` scripts: p5_eval.py — held-out evaluation driver
- `fcd1dfc` docs: 2026-05-16 — xgrammar adoption + evening pipeline state
- `14576f8` p5_rejection_sample: --init-checkpoint for iterated self-distillation
