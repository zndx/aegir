# Phase 3: PARL Training

> **Deferred work.** This page describes a multi-agent training plan
> (PARL on frozen specialists) that has been deferred indefinitely.
> The active training pipeline is the single-policy GRPO loop on
> Qwen3.5-9B described in the
> [semantic-engine authoritative reference](../ontology/production_state.md).
> The PARL plan below remains in the repository for archival
> continuity; see [Roadmap](../roadmap.md) for the active two-track
> work.

Train the `SwarmOrchestrator` using Parallel Agent Reinforcement Learning, following the K2.5 framework (arXiv:2602.02276). The primary model learns to route inputs to frozen specialists and fuse their recurrent states, optimized via token-level clipping RL.

## Setup

### Primary Model

A fresh Aegir model initialized from the Phase 1 checkpoint. All parameters are trainable. The primary model learns to:
- Process the input table and produce annotations
- Decide which specialists to activate via the `SpecialistRouter`
- Integrate specialist states through `RWKVStateFusion`

### Frozen Specialists

One or more Phase 1 checkpoints frozen with `requires_grad_(False)`. Each specialist is wrapped in a `FrozenSpecialist` that:
- Runs forward passes with `torch.no_grad()`
- Extracts recurrent states from its RWKV layers
- Optionally applies `AlignmentProjection` if its architecture differs from the primary

Initially, Phase 3 uses a single specialist (the best Phase 1 checkpoint). Additional specialists with different training data or hyperparameters are added incrementally.

## Token-Level Clipping RL

K2.5 uses a variant of PPO where the clipping objective is applied at the **token level** rather than the trajectory level. For each token position `t`:

```
L_t = min(
    rho_t * A_t,
    clip(rho_t, 1-eps, 1+eps) * A_t
)
```

where:
- `rho_t = pi_new(a_t | s_t) / pi_old(a_t | s_t)` is the per-token importance ratio
- `A_t` is the advantage estimate at position `t`
- `eps` is the clipping range (starts at 0.2, narrows to 0.1 over training)

Token-level clipping provides finer-grained credit assignment than trajectory-level clipping. For column annotation, this means the router receives distinct gradient signal for each column boundary token, each type-indicative value, and each structural separator.

### Routing as Action Space

The "action" at each routing decision point is the specialist activation vector:

```
a = sigmoid(W_router @ h)   # continuous in [0, 1]^num_specialists
```

The policy `pi(a | s)` is parameterized by the router weights. The RL objective encourages the router to activate specialists when they improve annotation quality and deactivate them when they don't.

## Critical-Steps Optimization

Rather than minimizing total FLOPs or wall-clock time, PARL optimizes the **critical path** -- the longest sequential dependency chain in the computation.

```
critical_path = max(
    primary_forward_time,
    max(specialist_forward_times for activated specialists)
)
```

Specialist forward passes run in parallel (they are independent). The critical path is therefore the maximum of the primary and any single specialist, not the sum. This means:

- Activating additional specialists that run in parallel is **free** in critical-path terms
- The optimizer penalizes only sequential dependencies (e.g., if the primary must wait for specialist state before proceeding)
- This naturally encourages parallel specialist activation over sequential reasoning in the primary

## Training Loop

```
for each batch:
    1. Run primary model through first layer to get routing_hidden
    2. Compute specialist activation scores
    3. Run activated specialists (parallel, no_grad)
    4. Fuse specialist states into primary's recurrent state
    5. Complete primary forward pass
    6. Compute r_perf from annotation accuracy
    7. Compute r_parallel from activation statistics
    8. Compute r_finish from annotation completeness
    9. Combine rewards with annealed lambdas
   10. Compute token-level PPO loss and update primary + router + fusion
```

## Budget-Limited vs Standard Scaling

PARL training alternates between two modes:

**Budget-limited phase:** The router has a hard cap on the number of specialists it can activate per batch. This encourages selective, high-value routing decisions. The cap starts low (1 specialist) and gradually increases.

**Standard scaling phase:** No activation cap. The router is free to activate as many specialists as it wants, paying only the `r_parallel` penalty for inefficient routing. This phase tests whether the router has learned meaningful selectivity.

The alternation prevents the router from converging to a trivial "activate everything" strategy during standard scaling while still allowing it to learn from unrestricted experimentation.

## Success Criteria

Phase 3 is complete when:

1. The primary model with specialist fusion exceeds the standalone Phase 1 baseline by a meaningful margin (target: +2-5 F1 points on SOTAB-CTA hard split)
2. The router activates specialists selectively (not all-on or all-off) and the activation pattern varies with input content
3. The lambda annealing schedule produces smooth training curves without reward collapse
