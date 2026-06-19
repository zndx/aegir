# Provenance — Verifiable Tasks & Lineage (the through-line)

**Status: DESIGN / reframe — not built.** Captured 2026-06-19 (RH). Reframes the top-line **"Tasks"**
card (today an unlinked `Statistic`, originally conceived as an Atropos-style RL-task surface) into
**Provenance: Verifiable Tasks & Lineage** — the spine the whole pipeline already has.

## Why the pivot (what Atropos told us)
[NousResearch/Atropos](https://github.com/NousResearch/Atropos) is a clean RL-environments gym: an
`BaseEnv` bundles *rollout generation + scoring + dataset*, runs as a **microservice** pushing
`ScoredDataGroup`s (trajectories + scores + metadata) to a **trainer-agnostic Trajectory API**
(`run-api`), with **verifiable/rule-based rewards** front and center (GSM8K exact-match, tool-calling,
code-exec). It nails the *RL-task* half — and the tell is what it lacks: **no formal versioning/
provenance system** (provenance is "implicit in server state" + JSONL lineage). That gap is exactly
Aegir's asset. Atropos's "task" = a verifiable environment; our insight — *a sequence of events and
gates over versionable intermediate artifacts* — is what Atropos doesn't model and Atlas only
half-models. Their union is the differentiator, so the card should name it.

## The data model
> A **provenance DAG**: nodes are **versioned artifacts** (FinePDFs ground → ontology catalog *vN* →
> DDL spine → corpus snapshot → model checkpoint → GRPO/eval run); edges are **verifiable events** —
> a derivation that *passed a gate* / *earned a reward* / *lifted a downstream eval*. Each edge carries
> its verdict.

This single structure subsumes the three things the card was straddling:
- **RL Tasks (Atropos-style)** = one edge kind: a rollout scored by the verifier R → a policy/
  checkpoint. The GRPO loop + `parallel_verify` already is this.
- **Enterprise lineage (Atlas)** = the DAG itself. Atlas is *already the provenance store* (OpenLineage
  datasets/jobs/runs + the `RE_GROUNDS_TO` loop-closure edge — see [[atlas_age_provenance_graph]]). So
  **Provenance = the Atlas lineage graph + a verification overlay**.
- **Gates** = the edge verdicts (Signals M1/M2/M3, HermiT consistency, realization-as-CPA, coverage-R1,
  the TBD downstream RWKV evals) — the "verifiable" in Verifiable Tasks & Lineage.

It also subsumes the observatory's **run↔data-product lineage** (idea #1 in
[leaderboard_observatory.md](./leaderboard_observatory.md)) — that was a slice; Provenance is its
substrate. And it makes the **convergence loop** legible as a chain, not a vibe:
`ontology vN —(R↑, HermiT✓)→ corpus —(byte/byte↓)→ model` (cf. [[aegir-convergence-loop]]).

## Atlas integration
Atlas (OpenLineage on AGE) holds the lineage; Provenance adds the **verification overlay** on the
edges (R-pass, HermiT-consistent, coverage-R1, downstream-eval-lift) and the **artifact versions**
(catalog versions, the lineup archive snapshots, corpus hashes, checkpoints). The Provenance panel
sources the Atlas graph and renders it with gate verdicts — the integration RH sensed. Direction:
emit the RL/eval gate events as OpenLineage facets on the existing run/dataset nodes.

## Adopt-vs-keep Atropos (orthogonal to the pivot)
Provenance *wraps* whichever RL harness — keep `grpo_loop` (our verifier R / HermiT / reasoner is a
richer reward than exact-match), but Atropos's **microservice + Trajectory-API decoupling** is a good
pattern to borrow if we grow to *many* verifiable tasks (DE-elucidation, CPA, downstream RWKV evals as
separate environments feeding one trajectory queue). Borrow the shape, not necessarily the code.

## The card / panel (v1 sketch)
"Tasks" (unlinked stub) → **Provenance** → a lineup panel rendering the artifact-lineage-with-gates DAG
(HoloViews graph via the bokeh-server `PanelView` — GraphRenderer renders correctly there, unlike the
npm `@bokeh/bokehjs` build), sourced from Atlas. Unifies the lenses (artifacts) + Sweeps/Reward (runs)
+ the gates into one verifiable-lineage view. Likely a new lineup nav group (PROVENANCE) or a top-level
view; `kind:"provenance"` note + a `provenance_app` bokeh server, same pattern as the Training panels.

## Dependencies / sequencing
- Cheapest first slice: render the **existing Atlas lineage subgraph** (ground → ontology → corpus →
  run) as a HoloViews graph panel — proves the surface before adding the verification overlay.
- The verification overlay needs gate verdicts as data: the RLVR reward (have it), HermiT/coverage
  (have them), downstream RWKV evals (TBD — the observatory's downstream-coupling metrics feed here).
- Artifact versions: catalog versions + corpus hashes + lineup archive snapshots already exist; wire
  them as node versions (the `RunArtifacts.start` provenance stamp — also the observatory unblocker).
