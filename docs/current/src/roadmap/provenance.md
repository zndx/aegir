# Provenance — Verifiable Tasks & Lineage (the through-line)

**Status: BUILT (instance-level ego-graph) — ILLUSTRATIVE, not definitive.** Captured 2026-06-19 (RH);
the first slice (a live type-level Atlas DAG) landed same day, and was **superseded 2026-06-24 by an
instance-level ReactFlow ego-graph** (#97, commit `0657569`). Reframes the top-line **"Tasks"** card (an
unlinked `Statistic`, originally conceived as an Atropos-style RL-task surface) into **Provenance:
Verifiable Tasks & Lineage** — the spine the whole pipeline already has. The card links to
`/lineup?open=training/provenance`; the panel is a **Training ▸ Provenance** sibling of Sweeps and Reward.
The **verification overlay** (per-edge gate verdicts) is still absent — that is the increment that turns
the navigable lineage walk into the thesis artifact (§Maturity, §Dependencies).

## What is built (v1.5 — the instance-level ego-graph)
The panel renders a node's **first-order neighbourhood** as a ReactFlow graph
(`ui/src/components/ProvenanceGraph.tsx`, `@xyflow/react` 12), read live from the `aegir_hx` Atlas / Apache
AGE graph via the gateway endpoint **`/api/provenance/ego?focal=<vid>`** (`src/aegir/gateway/app.py`).
Outgoing/derived neighbours sit right, incoming/sources left, edges are labelled by relationship type;
**clicking a neighbour opens *its* ego-graph in a new panel** — the lineage is walked node-by-node in
panel-trail fashion, not shown as one static type-level DAG. With no `focal`, the endpoint seeds an anchor
(a `Dataset` / `Run` / `Chapter`); each node uses the AGE internal `id()` as identity, the neighbourhood is
capped at 40 (with "more exist" surfaced), and the Atlas-core `vertex` label and `__rdbms_*` table
internals are excluded. The panel degrades gracefully to an empty state when the graph is down. A
`provenance/<vid>` synthetic note (gateway `kb_note`) lets a clicked node resolve as a panel; the lineup
note is `kind:"provenance"` carrying an `ego_focal` seed (`src/aegir/lineup/build.py`).

This replaced the original v1 — a **type-level** HoloViews/bokeh DAG (`src/aegir/viz/provenance_app.py`:
the convergence chain `Family/Topic → Template → Chapter → Column/Dataset → Job/Run` via
`networkx.multipartite_layout` → `hv.Graph`). That bokeh app is left in place but **no longer embedded**;
the live path is the React ReactFlow component (the `@xyflow/react` GraphRenderer renders correctly client-
side, where the npm `@bokeh/bokehjs` build of an `hv.Graph` did not — the reason for the move).

## Maturity: illustrative, NOT definitive
The current panel is a **legibility sketch**. It proves the surface (live Atlas graph → ReactFlow ego-graph
→ tap-to-walk at the narrow lens width), but several modelling choices remain provisional scaffolding. **Do
not build heavily on the current shapes.** The axes (RH, 2026-06-19), with their status updated to the
ego-graph:

- **Granularity — type-level → instance-level. ✅ DONE.** The ego-graph nodes are now the **versioned
  artifacts themselves** (a specific `Chapter`, `Run`, `Dataset`, `Template`, …, by AGE node id), not
  artifact *types* with aggregated counts. The earlier "shape of the pipeline" cartoon is superseded by the
  real, walkable lineage neighbourhood.
- **Node→panel routing — coarse type→whole-lens → contextual drill-in. PARTIALLY DONE.** Clicking a node
  now opens *that node's own* ego-graph (its identity seeds the next panel), rather than the type's whole
  lens. The remaining gap is a richer **artifact-detail** panel — e.g. a `Run`/`Job` should reach a
  run-detail view (or Sweeps/Reward scoped to that run), not just its lineage neighbourhood.
- **Artifact set & layout — curated whitelist → topology-derived. STILL OPEN.** The node set is now derived
  from the live graph (the focal's actual neighbours), but the global lineage is still not laid out from
  topology — there is no whole-graph multipartite/loop-aware view, and the `RE_GROUNDS_TO` loop-closure
  edge is a genuine cycle that a single ego-hop does not render as a loop. A topology-derived overview
  (expand/collapse between the walk and a whole-graph layout) remains future work.
- **The verification overlay is absent — the "Verifiable" half is unbuilt. STILL OPEN.** Edges are plain
  derivations; the point of *Verifiable* Tasks & Lineage is per-edge gate verdicts (R-pass ·
  HermiT-consistent · coverage-R1 · downstream-eval-lift) encoded on the graph. That is the increment that
  turns the navigable lineage into the thesis artifact (§Dependencies).

So: **current implementation = illustrative, instance-level navigation without verdicts.** Definitive =
topology-derived overview + contextual artifact-detail panels + a verification overlay. Treat the present
node/edge shapes as scaffolding to be extended, not as settled design.

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
sources the Atlas graph live (`/api/provenance/ego`) and walks it node-by-node — the integration RH
sensed. Direction: emit the RL/eval gate events as OpenLineage facets on the existing run/dataset nodes,
then render those facets as the per-edge verdict overlay.

## Adopt-vs-keep Atropos (orthogonal to the pivot)
Provenance *wraps* whichever RL harness — keep `grpo_loop` (our verifier R / HermiT / reasoner is a
richer reward than exact-match), but Atropos's **microservice + Trajectory-API decoupling** is a good
pattern to borrow if we grow to *many* verifiable tasks (DE-elucidation, CPA, downstream RWKV evals as
separate environments feeding one trajectory queue). Borrow the shape, not necessarily the code.

## The card / panel (as built)
"Tasks" (unlinked stub) → **Provenance** → a lineup panel rendering a node's **instance-level ego-graph**
(ReactFlow / `@xyflow/react`), sourced live from the `aegir_hx` Atlas graph via `/api/provenance/ego`.
Unifies the lenses (artifacts) + Sweeps/Reward (runs) into one navigable lineage. **Landed as a Training ▸
Provenance sibling** (not its own nav group): a `kind:"provenance"` note carrying an `ego_focal` seed.
Unlike the bokeh Sweeps/Reward panels (which mount a `viz_app` via `PanelView`), Provenance is a native
React component, the move that fixed the client-side graph-render path. The verification overlay (per-edge
gate verdicts) is the next increment on top of this surface.

## Dependencies / sequencing
- ✅ **DONE (v1)** — Cheapest first slice: rendered the **existing Atlas lineage subgraph** as a type-level
  HoloViews graph panel (`provenance_app.py`: `Family/Topic → Template → Chapter → Column/Dataset →
  Job/Run` via `networkx.multipartite_layout` → `hv.Graph`), proving the surface. **Superseded** by the
  ego-graph below.
- ✅ **DONE (v1.5)** — Instance-level **ReactFlow ego-graph** (`ProvenanceGraph.tsx` + `/api/provenance/ego`)
  replacing the static type-level DAG: a node's 1-hop neighbourhood, click-to-walk in panel-trail fashion,
  graceful empty state when Atlas is down (#97, commit `0657569`).
- The verification overlay needs gate verdicts as data: the RLVR reward (have it), HermiT/coverage
  (have them), downstream RWKV evals (TBD — the observatory's downstream-coupling metrics feed here).
- A topology-derived **whole-graph overview** (loop-aware layout, expand/collapse against the ego-walk).
- Artifact versions: catalog versions + corpus hashes + lineup archive snapshots already exist; wire
  them as node versions (the `RunArtifacts.start` provenance stamp — also the observatory unblocker).
