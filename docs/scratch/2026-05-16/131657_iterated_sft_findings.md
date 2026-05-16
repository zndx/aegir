# 2026-05-16 — Iterated self-distillation: base → SFT-r1 → SFT-r2

## Headline

Two rounds of self-distilled SFT, evaluated head-to-head on the 50-scenario
held-out set:

| Stage     | Overall mean R | Good mean | Bad mean | AUC   | R_A pass |
|-----------|---------------:|----------:|---------:|------:|---------:|
| Base      |          0.208 |     0.205 |    0.210 | 0.478 |     0.55 |
| SFT-r1    |          0.289 |     0.311 |    0.268 | 0.590 |     0.68 |
| SFT-r2    |     **0.318**  |     0.309 | **0.327**| 0.475 | **0.76** |

The **first** round of SFT delivered a clean +39% overall improvement,
driven by symmetric gains across good (+51%) and bad (+28%) scenarios
plus a meaningful jump in AUC (good-vs-bad discrimination: 0.478 →
0.590).

The **second** round added +10% on top of that — but **the gain came
entirely from bad scenarios** (+22%), with good scenarios essentially
tied (−0.4%) and AUC regressing back to random (0.475). R_A pass rate
kept climbing (0.55 → 0.68 → 0.76).

## Interpretation

Iterated SFT on a self-distilled corpus shows **diminishing,
asymmetric returns**: the second round raises the "floor" (bad/hard
scenarios become more reliable) but doesn't move the "ceiling"
(good/easy scenarios were already near the model's competence). The
side-effect is **loss of scenario discrimination** — SFT-r2 produces
similar-quality output regardless of input context. Mode collapse,
empirically.

This shows up in the round-2 corpus stats directly:

| Corpus  | n samples | mean R | unique template_ids | acceptance |
|---------|----------:|-------:|--------------------:|-----------:|
| v2 (base policy)    | 665 | 0.522 | **171/540 (32%)** | 94.5% |
| v3 (SFT-r1 policy)  | 740 | 0.513 |  **88/540 (16%)** | 96.4% |

Round-2 has **half the template diversity** of round-1 even though
it's 11% larger. The SFT-r1 model converges to a narrower set of
preferred templates, and the rejection sampler then trains SFT-r2 on
that narrower distribution. The compounding gain that iteration was
supposed to deliver gets traded for specialization.

## Implications for the contribution claim

The May-12 paper-grade claim was: "self-distilled SFT alone improves
out-of-distribution policy quality by ~40%, no GRPO needed." That
claim survives intact and is now more clearly demonstrated with the
larger (665-sample) corpus.

The May-16 expansion of that claim — "iterated self-distillation
compounds gains" — is **falsified** in this naive form. We need to
add a diversity-preserving mechanism between rounds:

1. **Bigger round-1 corpus** (5000+ samples) so SFT-r1 sees more of
   the template space before iterating.
2. **Higher temperature** during round-2 rejection sampling (forces
   exploration outside SFT-r1's preferred modes).
3. **Anti-clustering penalty** in the verifier: subtract a small
   amount of reward when generated template_ids cluster too tightly
   on previously-sampled templates.
4. **Mix-in** during SFT round 2: blend the SFT-r1 corpus *with* the
   SFT-r2 corpus (don't discard the broader-coverage round-1 data).

Option 4 is the cleanest minimal change for next session.

## What still hasn't moved

GRPO from either SFT-r1 or SFT-r2 stays at the **0.2591 plateau** on
the in-distribution step metric. That plateau is structural (bimodal
reward distribution × small advantage-normalized updates × LR=1e-5).
GRPO IS the test we couldn't run today at scale — the verifier-CPU
floor capped us at 18 min/step / ~20 steps in 8 hours.

**Next-session priority (re-confirmed):** parallelize the per-step
`verify()` via `multiprocessing.Pool`. Drops step time ~4-8×, makes
300-step GRPO runs feasible, lets us measure whether GRPO from a
SFT-warmed start can finally move the held-out mean past 0.318.

## Artifacts

- `/raid/checkpoints/p5/eval_base_held_out_50.json`
- `/raid/checkpoints/p5-sft-r1/eval_held_out_50.json`
- `/raid/checkpoints/p5-sft-r2/eval_held_out_50.json`
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_v2.jsonl` (665, base policy)
- `/raid/checkpoints/p5/sft-corpus/rejection_samples_v3.jsonl` (740, SFT-r1 policy)
- `/raid/checkpoints/p5-sft-r1/adapter_model.safetensors`
- `/raid/checkpoints/p5-sft-r2/adapter_model.safetensors`

## Today's durable code changes

- `4ba0324` rl: add xgrammar constrained-decode backend — 14× corpus throughput
- `53004f9` p5_train: --decode-backend xgrammar|lmfe in [6b/9] hook
- `bf23550` p5_sft: auto-convert FSDP .distcp → PEFT adapter at end of train()
- `7907e60` scripts: p5_eval.py — held-out evaluation driver
- `fcd1dfc` docs: 2026-05-16 — xgrammar adoption + evening pipeline state
- `14576f8` p5_rejection_sample: --init-checkpoint for iterated self-distillation
- `bc71dc9` docs: SFT-r1 vs base — held-out eval shows +39% mean R
- `009e37c` p5_sft: barrier + destroy_process_group before FSDP→PEFT conversion

## What the user will see when they wake up

Three policies, three evals, ~14× faster constrained decoding, two
new scripts (p5_eval, p5_rejection_sample with adapter loading), and
a clean cross-round comparison demonstrating both the positive (SFT
works, lots) and negative (naive iteration plateaus, mode collapses)
findings.
