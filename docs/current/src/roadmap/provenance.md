# Provenance — Verifiable Tasks & Lineage (the through-line)

**Status: v1 BUILT — ILLUSTRATIVE, not definitive.** The cheapest first slice (live Atlas lineage DAG,
no verification overlay yet). Captured 2026-06-19 (RH); v1 landed same day. Reframes the top-line
**"Tasks"** card (an unlinked `Statistic`, originally conceived as an Atropos-style RL-task surface) into
**Provenance: Verifiable Tasks & Lineage** — the spine the whole pipeline already has. The card now links
to `/lineup?open=training/provenance`; the panel is a **Training ▸ Provenance** sibling of Sweeps and
Reward (`src/aegir/viz/provenance_app.py`), rendering the type-level convergence chain read live from the
`aegir_hx` Atlas graph, with **tap-a-node → open its data-product lens** navigation. Next: the
**verification overlay** (per-edge gate verdicts) and instance-level drill-in (§Maturity, §Dependencies).

## Maturity: illustrative, NOT definitive
The v1 panel — both the DAG and the panels its node-clicks open — is a **legibility sketch**. It proves
the surface (live Atlas graph → HoloViews DAG → tap-to-navigate at the narrow lens width), but every
modelling choice in it is provisional scaffolding. **Do not build heavily on the current shapes.** The
axes we expect to iterate (RH, 2026-06-19):

- **Granularity — type-level → instance-level.** Today each node is an artifact *type* (8 of them) and
  edges are aggregated counts: a "shape of the pipeline" cartoon, not the real lineage. The definitive
  view is the **versioned artifacts themselves** (catalog *vN*, corpus-snapshot hash, checkpoint, a
  specific GRPO/eval run) with their actual derivation edges — likely expand/collapse between the two.
- **Artifact set & layout — curated whitelist → topology-derived.** The `STAGE` map (which types, which
  left→right column) is a hand-assigned constant that forces a clean multipartite layout. Real lineage
  is not strictly layered (a Run touches Template/Topic/Chapter/Dataset across "stages"; the
  `RE_GROUNDS_TO` loop-closure edge is a genuine cycle). Derive the node set + layout from the graph and
  the v0.3/Signals artifact taxonomy; render the loop honestly instead of flattening it.
- **Node→panel routing — coarse type→whole-lens → contextual drill-in.** Tapping a node opens its
  data-product *lens in full* (the `TARGET` map), ignoring which instance was tapped — and Run/Job →
  `lens/content` is frankly a stretch (Run/Job are training/orchestration; their real home is a
  run-detail panel, or Sweeps/Reward scoped to that run, or a provenance instance note). Definitive: the
  tapped artifact's identity seeds a panel *about that artifact* (this template, this run, this dataset).
- **The verification overlay is absent — the "Verifiable" half is unbuilt.** Edges are plain
  derivations; the point of *Verifiable* Tasks & Lineage is per-edge gate verdicts (R-pass ·
  HermiT-consistent · coverage-R1 · downstream-eval-lift) encoded on the graph. That is the increment
  that turns the cartoon into the thesis artifact (§Dependencies).

So: **current implementation = illustrative.** Definitive = instance-level, topology-derived,
context-routed, verification-overlaid. Treat the v1 maps (`STAGE` / `TARGET`) and the type-level framing
as scaffolding to be replaced, not as settled design.

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

## The card / panel (as built)
"Tasks" (unlinked stub) → **Provenance** → a lineup panel rendering the artifact-lineage DAG
(HoloViews graph via the bokeh-server `PanelView` — GraphRenderer renders correctly there, unlike the
npm `@bokeh/bokehjs` build), sourced from Atlas. Unifies the lenses (artifacts) + Sweeps/Reward (runs)
into one lineage view. **Landed as a Training ▸ Provenance sibling** (not its own nav group): a
`kind:"training"` note carrying `frontmatter.viz_app="provenance_app"`, served by the `provenance_app`
bokeh app — the exact pattern as the Sweeps/Reward panels. The verification overlay (per-edge gate
verdicts) is the next increment on top of this surface.

## Dependencies / sequencing
- ✅ **DONE (v1)** — Cheapest first slice: render the **existing Atlas lineage subgraph** (ground →
  ontology → corpus → run) as a HoloViews graph panel — proves the surface before the verification
  overlay. Live in `provenance_app.py`: type-level meta-graph (Family/Topic → Template → Chapter →
  Column/Dataset → Job/Run), `networkx.multipartite_layout` → `hv.Graph` directed, degrades gracefully
  when Atlas is down. The 9 type-edges render (Run→Template 794, Run→Topic 397, Dataset→Column 386, …).
- The verification overlay needs gate verdicts as data: the RLVR reward (have it), HermiT/coverage
  (have them), downstream RWKV evals (TBD — the observatory's downstream-coupling metrics feed here).
- Artifact versions: catalog versions + corpus hashes + lineup archive snapshots already exist; wire
  them as node versions (the `RunArtifacts.start` provenance stamp — also the observatory unblocker).
