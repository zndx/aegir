# Batch 6 (BeliefStructure) + P5 scaffolding

*2026-05-10 — closes the optional Q1 BeliefStructure commitment
and lands the model-side P5 scaffolding in parallel, per the
recommended-parallel path from the post-Batch-5 readiness
assessment.*

## Batch 6 summary

| Metric | Result |
|---|---|
| Candidate templates authored | 32 |
| Loaded into DeepOnto without error | 32 (100%) |
| Verbalized round-trip | 30 (94%) |
| `is_complex` flagged | 26 (81%) |
| Axiom-family coverage | F1, F2, F3, F4, F5, F6, F7 (all 7) |
| New `sdg:*` properties declared in TTL | 17 |
| Reused existing `sdg:*` properties | 2 (`governedBy`, `observedAt`) |
| Wall-clock harness time | 45 s |

### Branch coverage

| Branch | Templates | Notes |
|---|---|---|
| `cco:DescriptiveICE / sdg:MassFunction` | 8 | DST mass function over a frame of discernment; F1, F2, F3, F4, F5 |
| `cco:DescriptiveICE / sdg:BeliefInterval` | 8 | [bel, pl] credal-state bracket; F1, F2, F3, F4, F5 |
| `cco:DescriptiveICE / sdg:Evidence` | 7 | Supports/refutes claim; F1, F2, F6 (independent ∪ correlated) |
| `cco:DescriptiveICE / sdg:Claim` | 7 | Subject of belief structure; F1, F2, F4, F7 (`not Disconfirmed`) |
| `cco:DescriptiveICE` cross | 2 | `evidence_observed_by_process`, `claim_governed_by_directive` |

### Failures

Two templates fail DeepOnto verbalization:

| Template | Error |
|---|---|
| `mass_function_equiv_frame_and_assignment` | `EquivalentTo` complex restriction |
| `belief_interval_equiv_bel_and_pl` | `EquivalentTo` complex restriction |

Same DeepOnto verbaliser limitation as in Batches 4 and 5 — both
load cleanly, both retain `is_complex=False` after the harness
sets it, neither gets a manufactured fallback (per RL-utility
guidance).

### Why DST in the SDG ontology

Atelier's existing DST module computes mass functions and belief
intervals over column-tag hypotheses. The Batch 6 vocabulary
gives the SDG ontology shared structural vocabulary so the
agent-swarm state-fusion layer (`src/aegir/swarm/state_fusion.py`)
can consume Atelier's belief structures using the same property
names without translation. This is cross-context cousining at
the explicit-uncertainty layer: the same `sdg:Evidence` /
`sdg:Claim` / `sdg:BeliefInterval` triplet covers
- LIMS quality-tier evidence (Atelier sample tagging),
- audit findings supporting/refuting compliance claims (Batch 3),
- lineage-edge plausibility from PROV-O records (Batch 5),
- column-tag claims at confidence levels in the CTA/CPA pipeline.

## Combined catalog state (post-Batch-6)

| Metric | Value |
|---|---|
| Total templates | **375** |
| Verbalized | **357 (95%)** |
| `is_complex` | 320 (85%) |
| Schema errors | 0 |
| `sdg:*` properties in TTL | 192 |

Per-batch:

| Batch | Templates | Verbalized | Complex |
|---|---|---|---|
| 1 — Foundation | 81 | 73 (90%) | 68 |
| 2 — Observation + Measurement | 92 | 91 (99%) | 79 |
| 3 — Directive + Governance | 69 | 66 (96%) | 58 |
| 4 — eBPF kernel granularity | 51 | 48 (94%) | 44 |
| 5 — PROV-O / OpenLineage | 50 | 49 (98%) | 45 |
| 6 — BeliefStructure | 32 | 30 (94%) | 26 |

## Null distribution refresh

| Metric | Post-B5 (343) | Post-B6 (375) | Δ |
|---|---|---|---|
| `null_mean` | 0.41710 | 0.41750 | +0.00040 |
| `null_std` | 0.00900 | 0.00867 | −0.00033 |
| `null_p5` | 0.40203 | 0.40329 | +0.00126 |
| `null_p95` | 0.43205 | 0.43357 | +0.00152 |

All shifts within sampling noise. Locked weights `{0.50, 0.05, 0.45}`
remain in effect.

## P4 smoke test (post-Batch-6)

| Profile | Mean R |
|---|---|
| A_complex_slot_rich | 0.5253 |
| B_mixed | 0.5256 |
| C_trivial_basic | 0.0185 |
| D_stub_filler | 0.4718 |

Discrimination spread A − C: **0.5068**. Determinism + non-degenerate-std
+ reward-variance + quality-discrimination gates all hold. P4 smoke
test still ✓ PASS.

## P5 scaffolding landed

New package `src/aegir/rl/` wires the engineering between the
catalog, the locked verifier, and a future GRPO/RLVR training
run. Heavy deps (`transformers`, `peft`, `outlines`/`lmformatenforcer`,
`torch` SAE hook) load lazily so the smoke test runs without
those packages installed.

| New file | Purpose |
|---|---|
| `src/aegir/rl/__init__.py` | Package docstring + module index |
| `src/aegir/rl/policy.py` | `PolicyConfig` + `load_policy(cfg)`. SAE-Res-Qwen3.5-27B + LoRA. `leave_sae_untouched=True` keeps the residual SAE bottleneck out of the LoRA target list. |
| `src/aegir/rl/decoding.py` | `composition_json_schema(catalog)` builds a discriminated-union JSON Schema covering all 375 catalog templates. `outlines` default backend, `lm-format-enforcer` also wired. |
| `src/aegir/rl/sae_logging.py` | `SAELogger` forward-hook on the residual SAE; records top-k feature activations at configurable token cadence. |
| `src/aegir/rl/grpo_loop.py` | `init_grpo_state` / `grpo_iteration` / `compute_group_advantages` / `hot_reload_catalog`. Catalog hot-loaded per call. |
| `src/aegir/rl/eval.py` | `EvalConfig` / `evaluate(...)` / `_auc(...)`. C1 test set + 50-ontology held-out set (latter authored when P5 begins). |
| `scripts/p5_scaffold_smoke.py` | End-to-end wiring smoke test. |

### P5 scaffold smoke-test results

```
catalog: 375 templates (version=0.6.0-combined)
hot-reload no-op detected unchanged: True

policy-load dry run:
  base_model_id: anthropic/SAE-Res-Qwen3.5-27B-W80K-L0_100
  dtype: bfloat16
  tensor_parallel_size: 6
  lora_rank: 16, alpha: 32
  lora_targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  leave_sae_untouched: True
  expected_trainable_pct_estimate: ~0.3-0.5% of base

constrained-decode schema:
  backend: outlines
  n_template_branches: 375
  schema_size_chars: 138K (within typical LogitsProcessor budgets)

single-composition score (locked verifier):
  R=0.3245  R_A=1.00  R_B=0.60  R_C=0.491  R_D=0.000

3 GRPO iterations against stub policy:
  iter 0-2: R mean=0.4886, std=0.0750, R_A pass=100%, invalid=0
  elapsed: 0.01s (3.3ms/iteration on CPU; verifier dominates)

P5 scaffold smoke test ✓ PASS
  catalog hot-reload:        ✓
  policy-load dry run:       ✓
  decoding-schema build:     ✓
  verifier scoring:          ✓
  GRPO iteration end-to-end: ✓
```

### Pending engineering before actual P5 training begins

| Item | Estimate |
|---|---|
| `pyproject.toml`: add `transformers`, `peft`, `trl`, `outlines` (or `lm-format-enforcer`) | 30 min |
| 27B base load + tensor-parallel sharding across 6× RTX 4090 | ~1 day |
| Composition-format prompt template + few-shot context | ~1 day |
| Held-out 50-ontology set authoring (`tests/p5_held_out_50/`) | ~1 day |
| Wire stub `rollout_fn` to real model + `policy_step_fn` to GRPO gradient | ~1 day |
| Run + monitor 200 GPU-hour training | ~33 wall-clock hours |

The verifier-side dependencies are all in place. The remaining
work is purely model-side — no further catalog or verifier
changes are required before P5 begins.

## Files touched this session

| Status | Path |
|---|---|
| NEW | `src/aegir/ontology/catalog/06_belief_structure.candidate.json` (32 templates) |
| NEW | `src/aegir/ontology/catalog/06_belief_structure.json` (DeepOnto-populated) |
| NEW | `src/aegir/rl/__init__.py` |
| NEW | `src/aegir/rl/policy.py` |
| NEW | `src/aegir/rl/decoding.py` |
| NEW | `src/aegir/rl/sae_logging.py` |
| NEW | `src/aegir/rl/grpo_loop.py` |
| NEW | `src/aegir/rl/eval.py` |
| NEW | `scripts/p5_scaffold_smoke.py` |
| edited | `src/aegir/ontology/sdg-vocab.ttl` (+17 properties under "Batch 6" header) |
| edited | `src/aegir/ontology/catalog/combined.json` (now 375 templates) |
| edited | `src/aegir/ontology/null_stats.json` (refit against combined) |
| edited | `docs/current/ontology/charter.md` (P1 readiness, catalog-state, **new P5 scaffolding** sections) |
| NEW | `docs/scratch/2026-05-10/051635_batch6_p5_scaffold.md` (this file) |

## Path forward

P1 stays cleared with substantial margin (357 vs 200 floor).
The verifier and reward-signal layer continue hash-stable.

- **Batch 7 — long tail + benchmark gap-filling** (~150 templates)
  closes the brief's ~520-template target. Mechanical authoring;
  ~6 wall-clock hours of harness time. The verifier hot-loads
  Batch 7 templates per GRPO iteration — no Batch-7 dependency
  on P5 timing.
- **P5 — actual GRPO training** with SAE-Res-Qwen3.5-27B-W80K-L0_100
  against the locked verifier. Scaffolding done; only
  pyproject + checkpoint-load + held-out set + 200 GPU-hours
  remain. Can begin in parallel with Batch 7.
- **Parallel hygiene track** — v2 → SOTAB head fine-tune.
  Independent.

## Book-narrative integration update

The post-Batch-6 catalog state should be reflected in the project
book under Chapter 3 (semantic engine: aegir + atelier) and
Appendix A ("What is real today"):

- **Chapter 3 — closed-loop training description.** Cite **375
  templates / 357 verbalized / 95% / P1 cleared / closed-loop
  ontology→reward pipeline operational + P5 scaffolding landed**.
  The closed-loop description should mention the verifier
  signal flowing from procedurally-generated catalog
  verbalizations (no DeepOnto in the RL hot path) through
  locked weights `{0.50, 0.05, 0.45}` into GRPO advantages,
  with a 0.13 s/group budget on CPU and the model-side P5
  scaffolding in `src/aegir/rl/` ready to wire to the real
  27B policy.
- **Appendix A — "What is real today".** Call out the
  cross-context cousining as load-bearing evidence:
  - Trace + LabRun share `sdg:ObservationProcess`.
  - SQL CHECK / SysMLv2 Constraint / eBPF security policy
    co-anchored at `cco:DirectiveICE`.
  - `sdg:Allocation` + `sdg:Transformation` share `bfo:Process`.
  - `sdg:LineageEdge` sits at `cco:DescriptiveICE`
    alongside `sdg:ColumnPolicy` (B3) and the new DST
    primitives `sdg:BeliefInterval` / `sdg:Claim` /
    `sdg:Evidence` / `sdg:MassFunction` (B6).
  This last point is the explicit-uncertainty hook: the
  same `sdg:Evidence` → `sdg:Claim` → `sdg:BeliefInterval`
  triplet covers LIMS quality-tier evidence (Atelier sample
  tagging), compliance attestation (Batch 3 audits), lineage
  plausibility (Batch 5), and column-tag claims at confidence
  levels (CTA/CPA). The Atelier ↔ Aegir state-fusion layer
  consumes this vocabulary directly.

The book itself is outside this repo; this note is for the
Atelier-side draft to lift.
