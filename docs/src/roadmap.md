# Roadmap: K2.5 RL Post-Training

This section outlines the four-phase plan for training Aegir from a supervised baseline through full multi-agent reinforcement learning with PARL orchestration.

## Overview

The training follows a progressive complexity increase, where each phase builds on the previous one's checkpoints and infrastructure:

```
Phase 1              Phase 2              Phase 3              Phase 4
Supervised     -->   Reward         -->   PARL           -->   Agent
Bootstrapping        Modeling             Training             Swarm RL

Train base           Design reward        Train orchestrator   Scale to
Aegir on CTA/CPA    components and       with frozen           multi-specialist
benchmarks           validate signals     specialists           swarms
```

## Phases

### [Phase 1: Supervised Bootstrapping](./roadmap/supervised.md)

Train the base Aegir model on column annotation benchmarks (CTA, CPA) with byte-level input and dynamic chunking. Establish baseline F1 scores and validate the hierarchical architecture on real table data.

### [Phase 2: Reward Modeling](./roadmap/reward.md)

Design and validate the three reward components (`r_perf`, `r_parallel`, `r_finish`) that will drive PARL training. Calibrate lambda weights and verify that the reward signal produces meaningful gradients.

### [Phase 3: PARL Training](./roadmap/parl.md)

Freeze the best Phase 1 checkpoint as a specialist and train a new primary model with the PARL orchestrator. Use token-level clipping RL with critical-steps optimization.

### [Phase 4: Agent Swarm RL](./roadmap/swarm_rl.md)

Scale from a single specialist to a full swarm with dynamic specialist spawning. Implement wide search (parallel column analysis) and deep search (hierarchical type reasoning) patterns.

## Design Principles

1. **Each phase produces a usable checkpoint.** Even Phase 1 yields a competitive standalone column annotation model.

2. **Frozen specialists are never modified.** PARL training only updates the primary model and the routing/fusion modules. This prevents catastrophic forgetting in specialists and simplifies the training loop.

3. **Reward components are validated independently.** Phase 2 exists specifically to ensure that `r_parallel` and `r_finish` produce meaningful gradients before combining them with `r_perf` in Phase 3.

4. **Complexity is additive, not multiplicative.** Each phase adds exactly one new dimension of complexity (multi-task --> reward signals --> RL policy --> multi-agent), making failures easy to diagnose.
