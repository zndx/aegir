# Leaderboard → Convergence Observatory (enhancement ideas)

**Status: IDEAS — not built.** Captured 2026-06-18 (RH) once the FinePDFs→ontology→DDL→
corpus-with-embedded-views pipeline materialized and the viz layer became **live HoloViews/Panel/
Datashader served by a bokeh server behind the gateway proxy** (UI-U5; commits 8c42bbd / 6ac9491).
The leaderboard's only prior design intent was *"kinda like W&B but with HoloViews."* Now there's a
reason to aim higher.

## Where it is today
One row per training run; clicking a run opens a drawer with live HoloViews curves (loss / F1 /
per-stage chunking boundary), rendered by `aegir.viz.runs_app` over the bokeh server, embedded via
`<PanelView>`. Air-gapped, no npm `@bokeh/bokehjs`.

**Built — the Training section + Sweeps (v1).** The lineup left-nav now has a **TRAINING** group below
the lenses; its first entry, **Sweeps**, opens a live HoloViews **parallel-coordinates** panel
(`aegir.viz.sweeps_app`) over the runs — each run a line across `model_size · num_params · lr · epochs
· macro/micro F1 · val_loss`, colored by best macro-F1, reading live from `outputs/runs`. The Landing
"Runs" card is now "Sweeps" → `/lineup?open=training/sweeps`. HoloViews-native by directive (no
canonical-PCP widget) — the door is open to the *superior* version (`datashade` for many runs,
`hv.link_selections` axis brushing). The TRAINING group is extensible: the ideas below become further
entries (each a `kind:"training"` note with a `viz_app` frontmatter + a bokeh-server app).

## The reframe
The pipeline now emits **coupled Data Products** (ontology · relational/DDL footprint · corpus with
embedded views) *and* model runs, and the convergence loop couples them (proxy signals → model eval;
see [[aegir-convergence-loop]]). With a live viz layer, Atlas provenance, and the lineup all in place,
the leaderboard should grow from "training curves" into the **convergence observatory**: the join of
**model-runs × data-products × Signals gates**. It's the natural surface to answer "is the ontology
load-bearing, and is the model elucidating the relational structure we built?"

## Enhancement ideas (roughly prioritized)

1. **Run ↔ data-product lineage (highest leverage).** Each run records the corpus snapshot
   (`sdg_corpus_v0_3/<hash>`), ontology catalog (`ae7dbee`), and coverage run it trained on. Surface
   them as a "provenance" tab that cross-links into the **lineup** (the collections/chord it consumed)
   and **Atlas** (the `RE_GROUNDS_TO` loop-closure subgraph). Closes the run↔data-product loop visibly
   — the same live-viz embed the chord uses. See [[atlas_age_provenance_graph]], [[lineup_kb_projection]].

2. **Signals gate panels.** Per run, show the M1/M2/M3 criteria status (M1 H-Net isolation; M2 3-arm ×
   α×β instrument validity; final gate: matches RWKV-7 on general non-degeneracy AND beats the
   no-ontology ablation on relational + DE-elucidation, CI-clean). A pass/fail strip turns the
   leaderboard into the **gate dashboard**, not just curves. See [[signals_programme]].

3. **Ablation-arm comparison.** The corpus carries `full / no-ontology / no-schema` arms — overlay
   their curves (HoloViews overlay / HoloMap by arm) for the same data to read the ontology's
   load-bearing-ness directly, instead of eyeballing separate runs.

4. **Eval-instrument-aware panels** (per [[ontology-cpa-eval-methodology]]). The instrument is the
   binding constraint, so plot **sample-efficiency curves** (perf vs #examples) not full-data points,
   **PR metrics** not ROC-AUC, with **bootstrap/permutation CIs**, plus control-task / held-out-type /
   MDL-probing panels. This is where "W&B-like" stops being enough.

5. **DE-elucidation / CPA progress (the north-star).** Track the model's data-element-elucidation
   (CPA) over runs and tie it to the lineup **Schema-chord densification** (already framed as the
   term↔table many-to-many *progress metric*). The leaderboard becomes where "is the model learning
   the relational structure we built?" is answered.

6. **Datashader at scale.** Per-step loss/grad traces over millions of steps, and a **coverage
   heatmap** of the relational footprint (which tables/views a run's training data exercised) →
   server-side Datashader rasterization. Aligns with the tier-one compute posture
   ([[compute_posture_tier_one]]); the live stack already supports it.

7. **Interactive Panel widgets** (now unblocked by the live server — the static path couldn't).
   Run grouping/tags, metric pickers, cross-run config diff, run notes — Panel widgets / Tabulator.

8. **Corpus-quality panels beside model metrics.** Surface the convergence proxies (coverage-close
   R1, topic-recovery, family-complex) next to model metrics, since the loop is one coupled system —
   one observatory for both products.

## Dependencies / sequencing
- Most of these need **runs to carry data-product provenance** in `metadata.json` (corpus/ontology/
  coverage hashes + arm) — a small `RunArtifacts.start` addition; do that first (cheap, unblocks #1–#5).
- #2/#5 depend on the Signals gates + the discriminating relational eval instrument (M2/M3 tasks #42/#43).
- #6 needs `datashader`/`dask` added (deliberate `uv pip install`; the live viz layer already fits).
- Keep the **live HoloViews/PanelView** path — these are all panes on the same bokeh server, embedded
  the same way; no new viz transport needed.

## RL training (GRPO/RLVR) — the observatory is the reward instrument
P5 already runs **GRPO/RLVR** (`src/aegir/rl/`): the policy generates ontology compositions, the
**deterministic verifier composite R** is the reward (`parallel_verify` → `verifier`), z-score group
advantages, no critic. **GRPO is the right fit and PPO is not the question:** a value network is pure
overhead (memory + a second model to tune) when the reward is a cheap, parallelized, deterministic
grader — PPO earns its keep only with *learned/noisy* reward models or dense per-token credit, neither
of which applies here. The live questions are not PPO; they are **(a) reward-variance collapse** (if R
saturates or the structural gate `R_A` zeroes a whole group, advantages vanish → "0 reward variance
forever") and **(b) GRPO refinements** (length-bias debias, `advantage_normalization` choice — already
parameterized). Both are *observability + reward-design* problems → this observatory.

The earlier ideas are not left behind — RL makes them central, and most are nearly free because
`GRPOMetrics` already logs them to a `metrics_jsonl`:
- **Reward dynamics** — `rewards_mean ± rewards_std` band + min/max: the headline RL panel, and the
  **reward-variance band IS the GRPO health monitor** (the collapse canary). `advantage_mean/std` =
  signal strength. Already logged → a panel reading the GRPO `metrics_jsonl` (like `runs_app`).
- **Reward-component decomposition** — `R_A·(0.50·R_B + 0.05·R_C + 0.45·R_D)` over training (a stacked
  / small-multiples / PCP view). Small add: have `parallel_verify` log the sub-scores, not just `R`.
  This is idea #4 (eval-instrument-aware) for RL.
- **Verifier-pass-rate gates** — `R_A` structural-gate %, HermiT-consistency %, coverage-close % =
  idea #2 (Signals gate panels), as the RL pass-rate dashboard.
- **SAE feature stream** — already live (`/api/p5/sae/stream`); an RL-interpretability panel,
  `datashade`-d over GRPO steps (idea #6 at scale).
- **Sweeps PCP** (built) → the **GRPO hyperparameter tuner**: `group_size · kl_coefficient ·
  advantage_normalization · lr` × outcomes (final reward, pass-rate). This is the canonical RL-sweep use.
- **Unify the two run surfaces**: `/api/runs` (supervised CTA/CPA) + `/api/p5/runs` (GRPO/RLVR) both
  flow into the lineup observatory (a Training ▸ Reward entry beside Sweeps). They are separate today.
- **GRPO-vs-PPO, if ever litigated**, is an *ablation-arm* comparison (idea #3) — but the reward shape
  says GRPO; spend the cycles on reward granularity + curriculum, watched via the variance band.

**Real-world use:** a verifiable-reward RL observatory (reward dynamics + verifier-pass-rate gates +
data-product lineage + reproducible provenance, with the catalog hot-reload closing the loop — edit the
ontology, the reward changes on the next rollout) is genuinely product-grade RLVR experiment
management, differentiated from W&B by the verifiable-reward + ontology-grounded-lineage semantics.
