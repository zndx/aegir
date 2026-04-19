# UI advance + concept brief v3

Second burst of today's session, running in parallel with the Phase-1
SOTAB Schema.org CTA training run kicked off on GPU 1.

## What shipped

### Concept brief v3 (`build/draft-concept-brief_v3.md`)

Iteration after user review of v2 and Mergekit:

- **Phased plan** now committed explicitly. Phase 1 Aegir-only →
  Phase 1.5 Mergekit specialist fusion → Phase 2 (conditional)
  Aegir+Nano latent alignment.
- **Tokensurgeon spike** named as Phase 2 Step 0. If Nano's BPE
  embedding matrix can be transplanted onto Aegir's byte vocabulary
  via `mergekit-tokensurgeon`, the dual-tokenizer problem dissolves
  and the span-alignment + W_a ceremony is replaced by plain
  distillation loss on matched byte positions.
- **De-risker (1000-pair ridge R²)** explicitly becomes Step 1 only
  if tokensurgeon fails.
- **Falsifiable claims per phase.** No phase is a prerequisite for
  publishing the previous phase's result; each phase is an
  independently-citable measurement.

### Tokensurgeon architecture check (desk review, no code)

Reviewed `arcee-ai/mergekit`'s `docs/tokensurgeon.md`:

- Tokensurgeon is architecture-agnostic on the tokenizer axis — operates
  on the embedding matrix + LM head only, indifferent to internal
  Mamba/Transformer topology.
- `--byte-match` / `--prefix-match` flags suggest byte-level vocabulary
  is a supported path.
- Uses OMP (Orthogonal Matching Pursuit) by default to approximate
  missing-token embeddings as sparse combinations of *shared* tokens.
- **Risk the v3 brief does not yet name**: the shared vocabulary
  between Aegir (256 bytes + specials) and Nano (BPE ~128-256k) is
  small — mostly single-byte ASCII tokens if Nano's BPE emits them at
  all. OMP with k=64 over a shared set of ~256 anchors is a thin
  reconstruction basis. The transplant will *run* but the resulting
  model's internal activations were trained on BPE and may not
  gracefully consume bytes.

  → The spike protocol should therefore check *semantic quality* (held-out
  perplexity on raw-byte sequences) not merely "does it compile,"
  before deciding whether to skip the v2 span-alignment path.

### UI advance

Everything mirrors the Atelier shell where it makes sense, so the
cross-UI merge planned for next week has less surface to reconcile.

**Landing** — four Ant stat cards across the top (Service / Runs /
Tasks / Terms), reading `/api/stats`. Layout matches Atelier's
4-card top row exactly — same grid gutter, same column breakpoints.

**Classifications** — rewritten around Apache Atlas vernacular. The
Aegir task registry is exposed as an Atlas-compatible catalog (root
`AegirTask`, per-task children with `num_labels` and per-entity
attributes). Tree view + task table on the left; latest-run aggregate
metrics on the right (reused from prior M1 scope). Corpus-level sync
and editable classifications land in M2.

**Ontologies** — three-tab layout:
1. **Map your vocabulary** — CSV/TSV drag-drop upload (or paste),
   posts to `/api/ontology/subsume`, renders ranked ontology-parent
   suggestions with confidence + full BFO path.
2. **ICE/BFO skeleton** — the common training ontology nodes
   (BFO upper, CCO mid, ICE:DataElement as the domain anchor).
3. **DBpedia** — 120-type gt-signals vocab retained from prior M1.

### Gateway endpoints (new)

- `GET /api/stats` — consolidated Landing payload.
- `GET /api/classifications/catalog` — Atlas-shape tree from task registry.
- `GET /api/ontology/vocabulary` — BFO/CCO/ICE skeleton.
- `POST /api/ontology/subsume` — stub predictor with stable response
  shape. Started as always-`ICE:DataElement`-at-0.05; iterated into a
  small keyword heuristic (`customer|patient|user → CCO:Agent`;
  `order|payment|visit → BFO:Process`; `role|title → BFO:Role`;
  `invoice|document|id|number → ICE:DataElement`) so the upload flow
  produces *meaningful* suggestions on plausible vocab. M2 BERTSubs
  predictor drops in without touching the response shape.

Existing endpoints unchanged.

### BDD coverage

Four new tier-0 scenarios in `features/gateway/endpoints.feature`:

- `stats` returns the 4-card Landing payload (runs/service/tasks).
- `classifications/catalog` has the Atlas `AegirTask` root + child
  attributes.
- `ontology/vocabulary` contains `ICE:DataElement`.
- `ontology/subsume` accepts 3 terms, returns 3 suggestions, predictor
  identifies as a stub.

Also added reusable step helpers: nested-field equality, greater-than,
contains-node checks, POST-with-N-terms. `just bdd-0` green: 12
features, 38 scenarios, 0 failed.

### UI-convergence posture

All three pages now use Ant patterns (`Card`, `Tabs`, `Table`,
`Upload.Dragger`, `Tree`, `Statistic`) that appear in Atelier's
shell. Next week's UI merge should be straightforward:

- Page registration in `App.tsx` — both projects use
  react-router-dom v7 identically.
- Layout / Header — Atelier's is richer (dataset switcher, agent
  status); Aegir can adopt the full Layout when we lift it.
- Stats card shape is already byte-compatible on both sides (4 cards,
  grid breakpoints match, status colors match).

## What's running

- **GPU 1**: Phase-1 SOTAB Schema.org CTA, small model, 3 epochs.
  Log at `/tmp/phase1_sotab.log`. 15 min elapsed at time of writing;
  still in MMR embedding phase (27k columns × MPNet encode). First
  epoch will validate the MMR cache pays off.

## What's next (natural continuation)

1. Let SOTAB finish; confirm MMR cache hit on rerun.
2. Launch SOTAB-DBpedia and GitTables-DBpedia baselines on free GPUs
   once SOTAB-Schemaorg epoch 1 lands (indicates dataset init path
   is healthy).
3. Phase 1.5 Mergekit fusion experiment once Phase 1 checkpoints exist.
4. Tokensurgeon spike — half-day effort. Schedule for the Phase 1 →
   1.5 transition; the outcome (not the timeline) decides whether
   Phase 2 begins.

## Commits this burst

- `b0b5d74..` UI stats cards + Classifications + Ontologies + gateway
  endpoints + concept brief v3.
- `a59ba90` Ontology subsumption stub heuristic (keyword-based,
  still a stub but meaningful for demos).
- This note.

Tier-0 BDD still green throughout. UI `pnpm build` clean
(4 MB Leaderboards chunk flagged; acceptable for M1, code-split
candidate for M2).
