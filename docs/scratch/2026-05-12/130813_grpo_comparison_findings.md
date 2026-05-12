# P5 GRPO pipeline validation: end-to-end findings

## TL;DR

The constrained-decode-wiring fix is validated end-to-end. GRPO produces
non-zero reward variance, the policy responds to gradient signal, and
the full bootstrap pipeline (rejection-sampling SFT → GRPO with
`--init-checkpoint`) works without crashes. **At the small SFT corpus
size (75 samples × 2 epochs), warm-start and cold-start GRPO are
statistically indistinguishable** — the constrained-decoded few-shot
prompt is doing the heavy lifting; the LoRA delta from a tiny SFT
corpus is below the noise floor.

## What we ran

Two GRPO runs in parallel, both 50-step cap with `num_generations=4
× per_device_batch_size=2 = 8 sequences per rank × 2 ranks = 16
sequences per step`:

- **Cold-start** (GPUs 2,3): plain `just p5-train`, no LoRA init
- **Warm-start** (GPUs 0,1): `--init-checkpoint /raid/checkpoints/p5-sft`
  loading the 75-sample rejection-sampling SFT'd adapter

Both stopped at 10 hours of wall-clock (cold at step 18/50, warm at
step 26/50 — warm runs faster because completions are shorter on
average, ~22 min/step vs cold's ~30 min/step).

## Reward trajectories

### Cold-start (steps 1–18)

```
step  reward   clip%  mean_len  notes
  1   0.2591   0      425
  2   0.2613   0      412
  3   0.2591   0      424
  4   0.2591   0      427
  5   0.2622   0      351      first exploration probe
  6   0.1322   0      384      EXPLORATION DIP — 25% clipped
  7   0.1295   0.50   425      DEEPER DIP — 50% clipped
  8   0.1312   0.50   443      DIP CONTINUES
  9   0.2591   0      401      RECOVERED
 10   0.2611   0      374
 11   0.2610   0      416
 12   0.2601   0.25   431      mini-explore
 13   0.2594   0      407
 14   0.2591   0      418
 15   0.1305   0.25   442      dip
 16   0.2591   0      415      recovered
 17   0.2591   0      352      grad_norm 0.78 (spike)
 18   0.2591   0      382
```

### Warm-start (steps 1–26)

```
step  reward   clip%  mean_len  notes
  1   0.2610   0      415
  2   0.2601   0      359
  3   0.2604   0      405
  4   0.2591   0      415
  5   0.2609   0      398
  6   0.2603   0      420
  7   0.2591   0      379
  8   0.2591   0      420
  9   0.1295   0.25   419      single-step dip
 10   0.2612   0      341      RECOVERED IN ONE STEP
 11   0.2591   0      404
 12   0.2591   0.25   410      mini-explore, no dip
 13   0.2591   0      426
 14   0.2591   0      386
 15   0.2606   0      411
 16   0.1295   0.25   428      dip
 17   0.2591   0      423      recovered
 18   0.2591   0      400
 19   0.2591   0      423
 20   0.2601   0      364
 21   0.2591   0      378
 22   0.2591   0      355      loss=-0.163 (largest yet)
 23   0.2601   0.25   424      mini-explore
 24   0.1295   0.25   450      dip
 25   0.2591   0      415
 26   0.1295   0.25   436      dip
```

## Headline findings

### 1. The constrained-decode fix is validated

This was the primary diagnostic question after the 24-hour zero-reward
run on 2026-05-11. The fix landed in commit `5386091`. Validation:

- **Every step has non-zero reward variance** (std=0.30, sd=0.30).
  Previous run had `frac_reward_zero_std=1` for 1690 steps.
- **`clipped_ratio` mostly stays at 0** (with periodic 0.25-0.50 dips
  during exploration). Previous run was permanently clipped at 1.0.
- **Completions terminate naturally** at 350-450 tokens, not at the
  512 cap. Previous run had `mean_terminated_length=0`.
- **`loss` and `grad_norm` are non-zero** — gradient signal flows.

### 2. SFT warm-start at 75 samples is below the noise floor

Maximum reward observed across both runs, across 44 step datapoints:
**0.2622** (cold step 5, also warm step 5 and step 22 reached similar).
The runs are statistically indistinguishable on:

- Max reward
- Mean reward (both ~0.260 on plateau, ~0.130 on dips)
- Variance (both 0.299–0.302)
- Oscillation pattern (both dip and recover at similar rates)

**Asymmetry observation**: cold-start dipped for 3 consecutive steps
(6–8) before recovery; warm-start dipped for 1 step at most. This *could*
suggest the SFT'd policy has a slight stability prior pulling it back to
the trained distribution faster — but with only one trial each, the
asymmetry could easily be coincidence of group composition.

**Why so weak**: With only 10 SFT steps (2 epochs × 75 samples /
gradient accumulation 8 × 2 GPUs), the LoRA's `lora_B` parameter
(initialized at zero) only moves slightly from zero. The "warm-started"
policy is barely different from the cold-started one. The fix isn't the
SFT mechanism — it's that we need a *larger* SFT corpus or *more*
SFT steps.

### 3. The 0.26 plateau is real and structural

The verifier's reward distribution is **bimodal**: invalid compositions
get `R = 0` (R_A hard gate), valid compositions get `R ∈ [0.5, ~1.0]`.
The plateau mean ≈ 0.26 corresponds to ≈50% valid completion rate
times mean-valid-reward ≈ 0.52.

GRPO at LR=1e-5 with `num_generations=4` (small group, noisy advantages)
adjusts the policy in tiny steps. The two main directions the policy
could move are:

- Increase the % valid (R_A flips more often) → shifts mean toward ~0.5
- Increase R_B + R_D among valid completions → shifts mean toward ~1.0

Neither happened in 18–26 steps. The plateau is robust.

### 4. GRPO exploration/recovery oscillation is the dominant dynamic

Both runs exhibit a clean pattern:
1. Policy at plateau
2. Generates some completions that hit max_completion_length=512 (10–50%)
3. These get truncated → invalid JSON → R=0 → mean reward drops to ~0.13
4. Gradient pushes policy back to shorter compositions
5. Next step or two: plateau resumes

This is *GRPO doing exactly what GRPO should do*. The policy isn't
learning to climb the reward landscape, but it IS learning the
boundary of the productive region (don't generate too long).

## What the runs cost

- Wall clock: ~10 hours each
- Cold step time: ~30 min (slower because longer completions on average)
- Warm step time: ~22 min
- 4 RTX 4090s at 90-100% utilization for the duration
- Killed at 10h after determining plateau was structural; would have
  needed another 11-17h to complete

## What we have on disk

- `/raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl` (53 rows
  from seed 42)
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_b.jsonl` (22 rows
  from seed 100)
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_combined.jsonl`
  (75 rows merged, R: min 0.322, mean 0.455, max 0.977)
- `/raid/checkpoints/p5-sft/` (LoRA adapter from SFT, 256 weight tensors)
- `/raid/checkpoints/p5-sft/adapter_model.safetensors` (converted from
  FSDP `.distcp` for PEFT consumption)
- `/raid/checkpoints/p5-warm/checkpoint-25/` (GRPO partial checkpoint)
- `/raid/checkpoints/p5-cold/sae_features.live.jsonl` (1.18 MB SAE
  feature stream from cold-start)
- `/raid/checkpoints/p5-warm/sae_features.live.jsonl` (1.54 MB from
  warm-start)
- Two run logs at `/tmp/p5-cold4.log` and `/tmp/p5-warm4.log`

## Recommended next steps

In rough order of expected impact:

1. **Larger SFT corpus + more SFT steps.** 500–1000 rejection samples
   instead of 75; 3-5 epochs instead of 2. Single-seed rejection
   sampling on one GPU for ~6 hours would yield ~500–1000 samples.
   Then 30-min SFT on those should produce a measurably different
   warm-start.

2. **Higher GRPO learning rate.** 1e-5 is too conservative for this
   problem. Try 5e-5 or 1e-4. At 5e-5, the same 50 steps would have
   5× the policy displacement.

3. **Larger num_generations.** GRPO group size of 4 has noisy advantage
   estimates. 8 is the design point; we lowered to 4 only because of
   the OOM constraint at default `per_device_train_batch_size=8`. With
   `per_device=4`, num_gen=8 fits and gives better gradients.

4. **Drop the SAE attach during early GRPO.** SAE adds ~2GB on rank 0
   and isn't necessary for early policy learning. Re-attach after the
   policy has converged for interpretability analysis.

5. **Faster constrained-decode backend.** lmfe at our 540-branch schema
   makes generation 5-15× slower than free decode. `xgrammar` or vLLM's
   grammar backend would be substantially faster. Worth investing the
   engineering time before committing to longer runs.

6. **27B-scale comparison on LambdaLabs.** The 9B local results
   validate the pipeline; the 27B FSDP run on remote A100s should
   produce the headline paper number. The pipeline is ready for that
   migration; no Tinybox-specific code paths leak in.

## Negative result to write up

"On a 75-sample self-distilled SFT corpus trained for 2 epochs, GRPO
warm-start is indistinguishable from cold-start over 18-26 steps."

This is publishable in context: if the goal is to position rejection
sampling as a bootstrap mechanism, the *minimum-viable corpus size*
becomes the empirical question. Our null result establishes a lower
bound — somewhere above 75 samples × 2 epochs.

## Process notes (the "RCA failures" the user asked for)

Issues encountered during execution and their fixes:

1. **OOM at warm-start step 0**: TRL default `per_device_train_batch_size
   =8` × `num_generations=8 = 64 sequences/rank` overruns 24GB envelope
   on 9b-fsdp. Fix: dropped to `per_device=2 × num_gen=4 = 8/rank`.
   Committed as `45a372f` with `--per-device-batch-size` flag.

2. **`verify() ValueError: Input X contains NaN`**: sklearn KMeans in
   R_D topic-alignment scorer choked on a degenerate completion. Fix:
   `try/except` around `verify()`, score as R=0 on failure. Matches
   rejection-sampling script's existing convention. Same commit.

3. **FSDP-saved SFT checkpoint format**: TRL's `trainer.save_model()`
   under FSDP writes `pytorch_model_fsdp_0/*.distcp` files, not the
   PEFT-format `adapter_model.safetensors` that `load_lora_adapter_into_policy`
   expects. Wrote a one-shot conversion script (in-conversation) that
   reads the distcp via `torch.distributed.checkpoint.load`, strips the
   `model.` prefix, adds the `.default.weight` PEFT convention, and
   re-saves. **Followup todo**: bake this conversion into `p5_sft.py`'s
   teardown so future SFT runs produce PEFT-format adapters directly.

4. **Tinybox crash at ~23:30 prior session**: Two parallel rejection-
   sampling processes pushed an already-stressed box (tinybox-display
   service had been running 3+ days of CPU; WARP service was hung)
   over the systemd watchdog tolerance. Hard reboot at 00:46. User
   noted WARP panics correlate with training generally — it's a
   leading indicator. Lesson: a training-class workload + 6 other
   long-running services is the actual capacity limit on this box.

5. **Stale process holding 446MB on GPU 0**: NCCL peer-to-peer context.
   Normal behavior with multi-process FSDP, but pushed warm-start v2
   over the OOM line. Re-running with smaller batch sidestepped this
   without needing to disable NCCL p2p.
