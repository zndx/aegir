# Production state of Ægir's semantic engine — 2026-05-10

This document is the canonical lift-ready extract for the
Signals 360 book manuscript (Chapter 3 — semantic engine,
Appendix A — what is real today). It supersedes the per-closeout
integration notes in `docs/scratch/2026-05-10/*` for the
narrative-extraction purpose; the closeouts remain as historical
session records.

The facts here are verifiable against the repository state at
the date above. Each claim is anchored to a specific file or
output that can be re-derived.

## Chapter 3 — Closed-loop semantic engine

The semantic engine compiles a **bespoke OWL ontology** — the
**Signals Data Governance (SDG)** ontology — into a procedural
catalog that drives reinforcement learning with verifiable
rewards. SDG is grounded in BFO 2020 + the Common Core
Ontology (CCO), with five top-level branches:

- **Artifact** (CCO) — material things, datasets, programs.
- **DesignativeICE** (CCO) — names, identifiers, designators.
- **DescriptiveICE** (CCO) — measurements, claims, lineage.
- **DirectiveICE** (CCO; alias of `cco:ont00000965` "Prescriptive
  ICE") — requirements, controls, policies, constraints.
- **Process** (BFO 2020 `bfo:0000015`) — observation,
  derivation, governance.

The five branches are *cross-cousined*: every domain context
contributes templates to multiple branches under shared
parents, so the ontology grounds claims across LIMS sample
tagging, MBSE/SysMLv2 system design, database metadata
governance, kernel-trace observability, and PROV-O / OpenLineage
lineage simultaneously. The cousining is load-bearing
architectural invariant — it forces the ontology to express
real cross-context concepts rather than discipline-specific
aliases of the same idea.

### The catalog

The catalog is a JSON list of **540 Manchester-syntax templates**
(`src/aegir/ontology/catalog/combined.json`), authored across
seven batches over April–May 2026:

| Batch | Branch focus | Templates | Verbalized |
|---|---|---|---|
| 1 — Foundation | five-branch top-level + cross-cousining | 81 | 73 (90%) |
| 2 — Observation + Measurement | LabRun, Trace, eBPFEvent (process), Reading | 92 | 91 (99%) |
| 3 — Directive + Governance | HipaaRule, ColumnPolicy, Audit, AttestationActivity | 69 | 66 (96%) |
| 4 — eBPF kernel granularity | eBPFProgram, KernelHook, Syscall, Map | 51 | 48 (94%) |
| 5 — PROV-O / OpenLineage | Transformation, LineageEdge, Allocation, ProvenanceAgent | 50 | 49 (98%) |
| 6 — BeliefStructure (DST) | MassFunction, BeliefInterval, Evidence, Claim | 32 | 30 (94%) |
| 7 — Long tail / benchmark gap | SOTAB schema.org, CTA/CPA, NIST/ISO/SOC2/GDPR/HIPAA/PCI compliance, DST-ops, schema-evolution, OpenTelemetry, eBPF helpers, cross-branch density | 165 | 165 (100%) |
| **Combined** | | **540** | **522 (97%)** |

Each template is a parameterised OWL axiom (Manchester syntax)
with named slots in a typed mini-DSL: `{X:Class}`, `{p:ObjectProperty}`,
`{Y:Individual}`, etc. A composition is a list of `(template_id,
slot_fillers)` pairs that, when materialised, forms a small
ontology fragment.

The catalog is paired with **314 declared `sdg:*` object
properties** in `src/aegir/ontology/sdg-vocab.ttl`, every one
with `rdfs:label` + `skos:definition`. The TTL is closed:
every term referenced in any template is declared.

### The verifier — R(O, I)

Compositions are scored end-to-end by a four-component verifier
*R(O, I)* implemented at `aegir.ontology.verifier.verify(...)`:

- **R_A** — structural type-check (slot-fillers vs declared
  slot_types). Hard gate — `R = R_A · (a R_B + b R_C + c R_D)`.
- **R_B** — complex-template density. The fraction of
  composition templates flagged `is_complex=True` (axiom carries
  a non-trivial slot-bearing restriction beyond a basic
  subclass), saturating at τ_B.
- **R_C** — semantic-richness proxy via mean DeepOnto-cached
  verbal length against an L_target.
- **R_D** — Hungarian-optimal cosine alignment between *T_V*
  (the composition's verbalization corpus) and *T_I* (a held-out
  SchemaPile + FinePDFs-lab corpus subset, ~10K sentences,
  k=100 topics, sentence-transformers/all-MiniLM-L6-v2 encoder),
  normalised against a 200-sample structural-shuffle null
  distribution.

The aggregation weights `{a, b, c} = {0.50, 0.05, 0.45}` were
locked from the **C1 sweep** (`tests/ontology_test_set/`,
30 hand-authored ontologies, 15 good + 15 bad): the locked
weights yield **AUC 0.9956** and a mean separation of
0.336 between good and bad ontologies. The sweep is
re-runnable via `just c1-validate`; results at
`tests/ontology_test_set/c1_results.json`.

The verifier is **hash-stable**: identical inputs produce
identical R values across re-runs. Determinism is gated in
the P4 smoke test (`scripts/p5_smoke_test.py` — yes the script
was named P4 first; the P5 scaffold smoke is `p5_scaffold_smoke.py`).

### Verifier signal flow

The reward signal flows from **procedurally-generated catalog
verbalizations** into GRPO advantages:

```
composition (template_id + slot_fillers per entry)
   ↓ aegir.ontology.verifier.verify
R_A · (0.50 · R_B + 0.05 · R_C + 0.45 · R_D)
   ↓ aegir.rl.grpo_loop.compute_group_advantages
group-relative z-normalized advantages
   ↓ policy_step_fn (trl.GRPOTrainer.step when wired)
LoRA adapter gradient update on attention + MLP projections
   (SAE residual bottleneck untouched per leave_sae_untouched=True)
```

Crucially, **DeepOnto is offline-only**. The runtime verifier
runs procedurally on cached verbal_template strings; no JVM,
no Java dependency in the RL hot path. DeepOnto is invoked
only at catalog-build time via `scripts/build_catalog.py`,
which writes `verbal_template` and `mean_verbal_length` into
the JSON catalog and exits.

End-to-end timing: 0.13 s per 8-sample GRPO group on CPU
once the encoder and *T_I* are loaded, of which 0.02 s is
per-sample verifier scoring. With a 27B real policy, the
policy's forward + sampling pass will be the rate-limiting
step, not the verifier.

### Cross-context cousining — concrete instances

The cross-cousining can be inspected directly in the catalog
(`grep -E "branch" src/aegir/ontology/catalog/combined.json`):

- `bfo:Process` is shared by `sdg:LabRun` (LIMS), `sdg:Trace`
  (database/MBSE), `sdg:eBPFEvent` (kernel observability),
  `sdg:Transformation` + `sdg:Allocation` (lineage), and
  `sdg:Audit` + `sdg:AttestationActivity` (governance). Any
  composition that pulls observation, lineage, and audit
  templates is grounded at the same BFO upper class.
- `cco:DescriptiveICE` is shared by `sdg:Reading` (LIMS
  measurement), `sdg:ColumnPolicy` (database governance),
  `sdg:LineageEdge` (PROV-O), and the DST primitives
  `sdg:Evidence` / `sdg:Claim` / `sdg:BeliefInterval` /
  `sdg:MassFunction`. The same `sdg:Evidence` → `sdg:Claim` →
  `sdg:BeliefInterval` triplet covers LIMS quality-tier
  evidence, audit findings supporting/refuting compliance
  claims, lineage-edge plausibility from PROV-O records, and
  column-tag claims at confidence levels.
- `cco:DirectiveICE` (alias of `cco:ont00000965`) is shared by
  `sdg:HipaaRule`, `sdg:ColumnPolicy`, `sdg:Sql Constraint`,
  `sdg:SysMLv2 Constraint`, and the eBPF security-policy
  templates from Batch 4. SQL `CHECK` clauses and SysMLv2
  `Constraint` blocks land at the same upper class.
- `cco:DesignativeICE` is shared by `sdg:Identifier` (database
  primary keys), `sdg:Syscall` (kernel-API surface), and the
  Batch 7 schema.org alignment properties (Person.email,
  Place.geoCoordinates, etc.). Syscalls and DB identifiers are
  cousins, not separate disciplines.

### Atelier ↔ Aegir state-fusion via DST

The Batch 6 BeliefStructure primitives (`sdg:MassFunction`,
`sdg:BeliefInterval`, `sdg:Evidence`, `sdg:Claim`) provide
shared structural vocabulary so the agent-swarm state-fusion
layer (`src/aegir/swarm/state_fusion.py`) can consume Atelier's
belief structures using the same property names without
translation. Cross-context cousining at the explicit-uncertainty
layer.

## Appendix A — What is real today

As of 2026-05-10, the following components are operational and
verifiable in the Aegir repository:

### Ontology + verifier (P0–P2 complete)

- ✓ **540 catalog templates** at `src/aegir/ontology/catalog/combined.json`,
  composed across 7 batches and validated by `just check-ontology-schema`
  with 0 errors.
- ✓ **522 templates verbalized** (97%) by the offline DeepOnto harness
  (`scripts/build_catalog.py`).
- ✓ **485 templates flagged is_complex** (90%) — the long tail and
  Batch 7 cross-branch density push complexity high.
- ✓ **314 sdg:* properties** declared in `src/aegir/ontology/sdg-vocab.ttl`
  with `rdfs:label` + `skos:definition` for each.
- ✓ **Locked verifier weights `{0.50, 0.05, 0.45}`** from the C1
  AUC 0.9956 / separation 0.336 sweep.
- ✓ **Null distribution** at `src/aegir/ontology/null_stats.json`:
  `null_mean = 0.41372`, `null_p95 = 0.43012`, `n_samples = 200`,
  refit against the 540-template combined catalog.
- ✓ **Held-out 50 evaluation set** at `tests/p5_held_out_50/labels.json`
  (25 good + 25 bad scenarios, separation 0.5129 against locked
  verifier). Authored before P5 begins to keep it leakage-free.

### Reward-signal infrastructure (P3–P4 complete)

- ✓ **Verifier core** at `src/aegir/ontology/verifier.py` —
  importable, hash-stable, no JVM in the hot path.
- ✓ **Topic alignment** at `src/aegir/ontology/topic_alignment.py` —
  sentence-transformers + KMeans + Hungarian, runs without the
  broken sentence-transformers→datasets→pyarrow path by loading
  the encoder via `transformers` directly.
- ✓ **DeepOnto harness** at `src/aegir/ontology/deeponto_harness.py`
  with the verbalization fidelity fix (slot-marker count + length
  ranking, suppression of redundant subsumption candidates when
  the complex form succeeds).
- ✓ **P4 smoke test** (`scripts/p5_smoke_test.py`) — re-runs
  identical rewards + advantages across calls; profile A
  (slot-rich complex) consistently outscores profile C
  (trivial basic) by ≥ 0.5 R-units.

### P5 scaffolding (model-side wiring)

- ✓ **`src/aegir/rl/` package** — 7 modules:
  - `policy.py` — split load: base model `Qwen/Qwen3.5-27B`
    (tokenizer + weights) + SAE adapter repo
    `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100` (32 `layer{i}.sae.pt`
    files, downloaded via `download_sae_adapters`). LoRA on
    base attention + MLP projections;
    `leave_sae_untouched=True`.
  - `decoding.py` — discriminated-union JSON Schema for 540
    catalog branches (~198 KB), outlines + lm-format-enforcer
    backends.
  - `sae_logging.py` — forward-hook on residual SAE for top-k
    feature recording at configurable token cadence.
  - `grpo_loop.py` — `init_grpo_state` / `grpo_iteration` /
    `compute_group_advantages` / `hot_reload_catalog`. The
    catalog is hot-loaded per call so future catalog
    extensions integrate without restart.
  - `eval.py` — `evaluate(...)` against the C1 test set + the
    held-out 50, with internal `_auc(...)` (no sklearn dep).
  - `prompt.py` — system + 3-row deterministic few-shot + user
    instruction template.
  - `__init__.py` — package docstring + module index.
- ✓ **RL stack in main deps** — `transformers>=4.46`,
  `peft>=0.13`, `trl>=0.12`, `outlines>=0.1`, `accelerate>=1.1`
  in `pyproject.toml [project] dependencies`. Activate with
  the usual `uv sync` (no extras required).
- ✓ **`scripts/p5_scaffold_smoke.py`** — end-to-end wiring
  smoke test (catalog hot-reload, policy-load dry run, schema
  build, verifier scoring, 3-iteration GRPO loop with stub
  policy). Passes in ~5 s on CPU. Verifier is rate-limiting,
  not the loop.

### What is *not yet* operational

The remaining work is purely model-side execution, not catalog
or verifier work:

- ✓ **`uv sync`** — main deps now include the full RL stack
  (`transformers`, `peft`, `trl`, `outlines`, `accelerate`).
  Run `just sync` for `uv sync` + automatic patched-wheel
  restore from `build/wheels/`.
- ✓ **`scripts/p5_train.py` launcher** — wires
  `PolicyConfig` → `load_policy(cfg)`, `RunMetadata` +
  `SidecarCallback` for checkpointing, `reward_fn` closure
  over the locked verifier, prompt dataset (n_prompts × num_generations),
  `trl.GRPOTrainer`. Supports `--dry-run` (pre-flight summary,
  no GPU work) and `--resume` (strict-by-default drift check;
  `--no-strict-resume` to override). Reachable via
  `just p5-train --dry-run` / `just p5-train`.
- ✓ **Checkpointing** — HF `Trainer` covers model + LoRA +
  AdamW + LR scheduler + RNG + AMP scaler; `aegir.rl.checkpointing`
  adds catalog-version pin, locked-weight hash, null-stats
  hash, run id, SAE log spill, GRPO metrics JSONL,
  `verify_resume_metadata(strict=True)` policy guard.
- ✗ **27B base load + FSDP sharding across 6× RTX 4090** —
  pending hardware availability verification + ~54 GB bf16
  weights download. **TP=6 is not viable** (Qwen3.5-27B's
  GQA config has `num_kv_heads=4`, `intermediate_size=17408`,
  `hidden_size=5120`, none of which divide by 6). Legal TP
  sizes are {1, 2, 4} — only TP=4 fits 24 GB GPUs and wastes
  2 of 6. **FSDP** shards by parameter rather than by attention
  head, gives ~9 GB base weights per GPU, full hardware
  utilization. `just p5-train` routes through `accelerate
  launch --use_fsdp --num_processes 6 --fsdp_sharding_strategy
  FULL_SHARD --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
  --mixed_precision bf16`.

### Disk routing + budget (system drive is space-constrained)

The system drive `/` had **15 GB free** at last audit; **no P5
artifact may land there**. All large artifacts route to `/raid`:

| Destination | Path | Mechanism |
|---|---|---|
| HF model + dataset cache | `/raid/cache/huggingface` | `HF_HOME` env (set in `devenv.nix`) + the `~/.cache/huggingface → /raid/cache/huggingface` symlink |
| P5 checkpoints + LoRA + logs | `/raid/checkpoints/p5/` | `CheckpointConfig.output_dir` default; launcher `--output-dir` default |
| Benchmark datasets | `/raid/datasets/` | existing convention (Justfile `get-sotab` / `get-gittables`) |

`/raid` has **713 GB free** (post-cleanup, 2026-05-10). Budget for P5:

| Item | Size | Notes |
|---|---|---|
| Qwen3.5-27B base weights | ~54 GB | bf16 sharded safetensors. |
| SAE adapter (full set, all 64 layers) | ~205 GB fp32 | Fits comfortably with the post-cleanup headroom. |
| LoRA adapter checkpoints (rolling 3) | < 1 GB | Trivial. |
| Optimizer state (LoRA AdamW) | ~few GB | Trivial. |
| GRPO metrics JSONL + SAE feature logs | MB-scale | Append-only. |
| **Total P5 footprint** | **~260 GB** | leaves ~450 GB free on /raid after launch. |

**SAE attachment strategy** (when wired): full per-layer download
via `download_sae_adapters(cfg, num_layers=64)` is now viable
without pruning. The `download_sae_adapters` helper remains
best-effort per-layer (missing layers warn-and-skip rather than
fail), so a partial subset is still fine if memory rather than
disk becomes the constraint at hook-attachment time.

The launcher prints disk-space readouts at every invocation so
the operator sees free space before launching.
- ✗ **200 GPU-hour GRPO training run** — pending; user-initiated
  via `just p5-train` (no flags). Hard-to-reverse compute
  commitment; the launcher's `--dry-run` is the recommended
  pre-flight before launch.
- ✗ **Held-out 50 evaluation post-training** — runs after the
  training run completes via `aegir.rl.eval.evaluate(...)`.

## Repository invariants

The following invariants are enforced and verifiable:

- `just check-ontology-schema` exits 0 (1624 templates × 7 catalog
  files validate).
- `just c1-validate` regenerates the C1 sweep AUC 0.9956 against
  the committed locked weights.
- `just p4-smoke` re-runs and passes the determinism +
  non-degenerate-std + reward-variance + Profile-A-beats-Profile-C
  gates.
- `LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_scaffold_smoke.py`
  passes the catalog hot-reload + policy-load dry run +
  decoding-schema build + verifier scoring + 3-iteration GRPO
  loop gates.

These four commands together verify the full closed-loop
ontology → synthetic data → RL corpus pipeline is operational
end-to-end at the verifier and reward-signal layer.

## Lift instructions for the book draft

When the Atelier-side book draft incorporates this content:

- **Chapter 3 (semantic engine: aegir + atelier)** — lift the
  "Closed-loop semantic engine" section above. The five-branch
  diagram, locked weights, R(O, I) formula, and the
  cross-cousining instance list are all stable as of 2026-05-10
  and grounded in the repo. The reward-signal flow diagram
  reflects what `aegir.rl.grpo_loop.grpo_iteration` actually
  computes today (with stub `rollout_fn` / `policy_step_fn`,
  but the pipe is real and tested).
- **Appendix A (what is real today)** — lift the "What is real
  today" / "What is not yet operational" sections directly.
  The ✓/✗ status flags map to concrete repo state and command
  output that the book reviewer can verify.

When P5 training actually runs and produces held-out evaluation
results, append a "P5 training run + held-out result" section
to this document; that is the final operational milestone for
the semantic engine narrative.
