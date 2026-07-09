# Phase 4: Agent Swarm RL

> **Deferred work (status refreshed 2026-07-09).** This page describes
> a multi-specialist swarm RL plan that has been deferred indefinitely;
> the four-phase K2.5 PARL roadmap folded into the
> [Signals Programme](../signals_programme.md) milestone ladder. The
> agent-swarm modules in `src/aegir/swarm/` remain in the codebase as
> infrastructure scaffolding, but no operational training currently
> uses them. See [Roadmap](../roadmap.md) for the active two-track
> work and [Agent Swarm](../agent_swarm.md) for the module-level
> scope statement.

Scale from a single specialist to a full multi-specialist swarm with dynamic spawning, wide/deep search patterns, and adaptive specialist allocation based on table complexity.

## Search Patterns

### Wide Search -- Parallel Column Analysis

Process multiple columns simultaneously by routing them to different specialists:

```
Table: [col_A, col_B, col_C, col_D, col_E]

Specialist 0 (geographic): col_A, col_C
Specialist 1 (temporal):   col_B
Specialist 2 (numeric):    col_D, col_E
Primary:                   all columns (final fusion)
```

Each specialist processes its assigned columns in parallel. The primary model receives fused states from all specialists and makes the final annotation decision. Wide search scales annotation throughput linearly with the number of specialists, bounded by the critical path of the slowest specialist.

### Deep Search -- Hierarchical Type Reasoning

For ambiguous columns, chain multiple specialists in sequence to progressively refine the type prediction:

```
Column: "Springfield" (city? state? person name?)

Step 1: Specialist 0 (general) --> geographic entity (0.6) | person name (0.3)
Step 2: Specialist 3 (geographic) --> city (0.7) | administrative region (0.2)
Step 3: Primary --> city (final, high confidence)
```

Deep search trades latency for accuracy on hard cases. The orchestrator learns when to invoke additional reasoning steps by monitoring the confidence of intermediate predictions.

### Combined Wide-Deep

For complex tables, the orchestrator can combine both patterns: wide search across easy columns (one specialist pass each) and deep search on ambiguous columns (multiple specialist passes). The PARL reward structure naturally encourages this: `r_parallel` rewards wide parallelism, `r_perf` rewards deep accuracy, and critical-steps optimization keeps the overall latency bounded.

## Dynamic Specialist Spawning

Rather than a fixed specialist pool, Phase 4 introduces dynamic spawning based on table complexity signals:

```
complexity = f(num_columns, label_entropy, column_diversity)

if complexity < threshold_low:
    activate 0-1 specialists (primary handles it alone)
elif complexity < threshold_high:
    activate 2-3 specialists (wide search)
else:
    activate N specialists + enable deep search
```

The complexity estimator is a lightweight head on the primary model's first-layer output. It learns to predict how much specialist assistance a given table requires.

### Specialist Pool Management

- **Warm pool:** Pre-loaded specialists kept on GPU memory, ready for immediate activation.
- **Cold pool:** Specialists on CPU/disk, loaded on demand for rare table types.
- **Spawn budget:** Maximum number of active specialists at any time, set by available GPU memory.

## Expected Scaling Characteristics

### State Fusion Cost

With `N` specialists, each contributing `L` layers of state with shape `(B, H, K, V)`:

```
Fusion FLOPs per layer:
  weighted_sum:    O(N * H * K * V)     -- linear in N
  gated:           O(N * H * K * V)     -- linear in N
  concat_project:  O(N^2 * H * K^2 * V^2)  -- quadratic in N (due to projection weight size)
```

For the `weighted_sum` mode (recommended for swarms), fusion cost grows linearly with specialist count and is negligible compared to the specialist forward passes themselves.

### Throughput Scaling

| Specialists | Expected Throughput | Expected Accuracy | Notes |
|-------------|--------------------|--------------------|-------|
| 0 (primary only) | 1.0x baseline | Phase 1 F1 | No overhead |
| 1 | ~0.9x (routing overhead) | +2-5 F1 | Phase 3 result |
| 3 | ~0.85x | +5-10 F1 | Wide search on typical tables |
| 8+ | ~0.7x | +8-15 F1 | Wide+deep on complex tables |

Throughput decreases reflect routing overhead and state fusion cost. The critical-path optimization means that parallel specialists do not compound latency, so throughput degradation is sublinear in the number of specialists.

### Memory Scaling

Each frozen specialist consumes GPU memory for its parameters but no optimizer state (frozen). The primary model requires both parameters and optimizer state.

```
Memory per specialist: ~model_params * sizeof(dtype)
Memory for primary:    ~3x model_params * sizeof(dtype)  (params + grad + optimizer)
Memory for fusion:     negligible (O(H * K * V) parameters)
```

With bf16 and a 120M-parameter base model, each specialist costs ~240MB. A 6x RTX 4090 setup (144GB total) can support approximately 8-10 specialists alongside the primary model and optimizer state.

## Success Criteria

Phase 4 is complete when:

1. The swarm demonstrates measurable accuracy gains from adding specialists beyond the Phase 3 single-specialist result
2. Wide search provides throughput-proportional accuracy gains on easy tables
3. Deep search provides accuracy gains on the hardest SOTAB-CTA classes (those below 0.5 F1 in Phase 1)
4. Dynamic spawning correctly allocates more specialists to complex tables and fewer to simple ones
