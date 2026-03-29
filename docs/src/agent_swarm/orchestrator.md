# K2.5 PARL Orchestrator

The `SwarmOrchestrator` coordinates a trainable primary Aegir model with multiple frozen specialist agents, following the Parallel Agent Reinforcement Learning (PARL) pattern from Kimi K2.5 (arXiv:2602.02276). Implementation is in `src/aegir/swarm/orchestrator.py`.

## Architecture

```
                    +-------------------+
                    | SwarmOrchestrator |
                    +-------------------+
                            |
             +--------------+--------------+
             |              |              |
     SpecialistRouter   Primary      FrozenSpecialists
     (sigmoid gates)    (trainable)   (frozen params)
             |              |              |
             |              |    +---------+---------+
             |              |    |         |         |
             |              |  Spec_0   Spec_1   Spec_N
             |              |    |         |         |
             +--> activation -->  state fusion  <----+
                  mask           (RWKVStateFusion)
```

The primary model is the only component whose parameters are updated during PARL training. Specialists are frozen checkpoints that contribute their recurrent states when activated by the router.

## SpecialistRouter

The router decides which specialists to activate for a given input. It maps the primary agent's hidden representation to per-specialist activation scores:

```python
scores = sigmoid(W_router @ hidden_states)   # (B, num_specialists)
activation_mask = scores > threshold          # default threshold = 0.5
```

Sigmoid gating (rather than softmax) allows zero, one, or multiple specialists to be activated simultaneously. This is critical for the column annotation task where a table may require expertise from several domain specialists, or none at all.

## PARL Reward Structure

The combined reward follows K2.5's formulation:

```
r = lambda_1 * r_parallel + lambda_2 * r_finish + r_perf
```

### Reward Components

**`r_perf`** -- Performance reward. F1 accuracy on the annotation task (CTA or CPA). This is the primary signal that drives annotation quality.

**`r_parallel`** -- Parallelism and load balancing reward. Encourages efficient specialist utilization: activate specialists when they help, avoid activating them when they don't. Adapted from H-Net's `lb_loss` which penalizes unbalanced routing across experts.

**`r_finish`** -- Completion quality reward. All columns in a table must be annotated, and the router must not degenerate into always-on or always-off patterns. Penalizes incomplete annotations and trivial routing strategies.

### Lambda Annealing Schedule

Following K2.5, the lambda weights anneal over training:

| Phase | `lambda_1` (parallel) | `lambda_2` (finish) | Rationale |
|-------|----------------------|---------------------|-----------|
| Early | 0.3 | 0.1 | Encourage exploration of specialist activation |
| Mid | 0.1 | 0.3 | Shift focus to completion quality |
| Late | 0.05 | 0.05 | Let `r_perf` dominate for final accuracy |

The initial values (`lambda_parallel=0.3`, `lambda_finish=0.1`) are set in the orchestrator constructor. Annealing is managed by the training loop.

## Token-Level Clipping RL

K2.5 uses a variant of PPO with token-level clipping rather than trajectory-level. This provides finer-grained credit assignment:

- Each token's routing decision gets its own clipped surrogate objective
- Critical tokens (column boundaries, type-indicative values) receive higher weight
- The clipping range narrows over training to stabilize converged policies

## Critical-Steps Optimization

Rather than minimizing total computation, the orchestrator minimizes the **critical path** -- the longest chain of sequential dependencies. Specialist activations that can run in parallel do not increase the critical path even if they increase total FLOPs. This encourages the router to prefer parallel specialist activation over sequential reasoning in the primary model when both achieve similar accuracy.

## Forward Pass

```python
orchestrator = SwarmOrchestrator(
    primary_model=primary,
    specialists=[spec_cta, spec_cpa, spec_geo],
    fusion=RWKVStateFusion(num_heads=8, head_size=64, num_agents=3),
    d_model=512,
    activation_threshold=0.5,
)

result = orchestrator(
    input_ids=tokens,
    mask=mask,
    routing_hidden=pooled_hidden,  # from primary's first layer
)

# result["output"]           -- primary model output
# result["specialist_outputs"] -- list of activated specialist results
# result["activation_mask"]  -- (B, num_specialists) boolean mask
```

When `routing_hidden` is `None`, specialist activation is skipped entirely and only the primary model runs. This allows the same orchestrator to be used in both supervised pre-training (no specialists) and PARL training (with specialists).
