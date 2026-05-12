# Phase 2: Reward Modeling

> **Deferred work.** This page describes a multi-agent
> reward-modeling plan tied to the earlier K2.5 PARL roadmap, which
> has been superseded by the two-track structure in
> [Roadmap](../roadmap.md). The project's active reward modeling is
> the four-component deterministic verifier *R(O, I)* described in
> the [semantic-engine authoritative reference](../ontology/production_state.md);
> the `r_perf` / `r_parallel` / `r_finish` design below is preserved
> for archival continuity and is not currently being implemented.

Design and validate the three reward components that will drive PARL training in Phase 3. The goal is to ensure each reward signal produces meaningful, non-degenerate gradients before combining them into the full PARL objective.

## Reward Components

### `r_perf` -- Performance Reward

The primary quality signal. Computed as F1 accuracy on held-out annotation tasks:

```
r_perf = macro_F1(predicted_labels, ground_truth)
```

For CTA, this is the macro-averaged F1 over all 91 semantic type classes. For CPA, it is the macro F1 over property classes.

This reward is straightforward to compute and directly measures what we care about. The challenge is that F1 is non-differentiable, so it must be used as an RL reward signal rather than a supervised loss (which uses cross-entropy as a differentiable proxy).

### `r_parallel` -- Load Balancing and Specialist Utilization

Adapted from H-Net's `lb_loss`, this reward encourages efficient use of specialists:

```
r_parallel = -alpha * CV(activation_counts) + beta * utilization_rate
```

where:
- `CV(activation_counts)` is the coefficient of variation of specialist activation counts across a batch. Penalizes routing that always sends to the same specialist.
- `utilization_rate` is the fraction of specialists activated at least once in a batch. Rewards using the full specialist pool.
- `alpha`, `beta` are tunable coefficients.

A degenerate router that always activates all specialists or never activates any will score poorly on this component. The reward is maximized when specialists are activated selectively and roughly equally.

### `r_finish` -- Completion Quality

Ensures that the swarm produces complete, non-degenerate outputs:

```
r_finish = coverage_score - degenerate_penalty
```

where:
- `coverage_score` measures the fraction of columns in a table that receive an annotation. A table with 10 columns where only 7 are annotated scores 0.7.
- `degenerate_penalty` fires when the router exhibits trivial strategies: always-on (activating all specialists for every input), always-off (never activating specialists), or constant routing (same activation pattern regardless of input).

## Combined Reward

The three components are combined with annealing weights:

```
r = lambda_1 * r_parallel + lambda_2 * r_finish + r_perf
```

Note that `r_perf` has no lambda coefficient -- it always contributes at full strength. The auxiliary rewards are scaled to be comparable in magnitude to `r_perf` and then weighted down.

## Lambda Annealing Schedule

Following K2.5's approach, the auxiliary reward weights change over training:

| Training Progress | `lambda_1` (parallel) | `lambda_2` (finish) | Rationale |
|-------------------|-----------------------|---------------------|-----------|
| 0-30% | 0.3 | 0.1 | Encourage specialist exploration early |
| 30-70% | 0.1 | 0.3 | Shift focus to complete annotations |
| 70-100% | 0.05 | 0.05 | Let accuracy dominate for fine-tuning |

The annealing ensures that early training explores the specialist activation space (high `lambda_1`), then stabilizes routing toward complete outputs (high `lambda_2`), and finally optimizes purely for annotation accuracy.

## Validation Protocol

Before proceeding to Phase 3, each reward component must pass these checks:

1. **Non-zero gradient flow.** The reward signal must produce non-trivial policy gradients through the router. Verified by checking that `grad(router.weight)` is non-zero after a reward update.

2. **Correct polarity.** Higher quality outputs must produce higher rewards. Verified by comparing rewards on hand-crafted good vs. bad annotation examples.

3. **Independence.** Each component must capture a distinct failure mode. Verified by constructing examples where one component fires but others do not:
   - High `r_perf`, low `r_parallel`: accurate but always uses the same specialist
   - High `r_parallel`, low `r_finish`: well-balanced routing but incomplete annotations
   - High `r_finish`, low `r_perf`: complete annotations but wrong types

4. **Scale compatibility.** All three components should produce values in a comparable range (roughly [0, 1]) to avoid one signal dominating before lambda annealing can take effect.
