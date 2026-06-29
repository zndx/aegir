# Agent-Mediated Refinement Loop (adopted 2026-06-24)

A layered verification cascade that turns single-shot chapter generation (a *truncated creative process*)
into an iterative, membrane-gated refinement loop. The ACP-wrapped agent is the **proposer only**; every gate
is a deterministic membrane effector the agent cannot run or bypass. Extends the meta-harness
([[meta_harness_boundary]]), the HermiT membrane (`reasoning_gates.py`), the DDL spine, and the corpus
pipeline.

## Why
Audit (`docs/scratch/2026-06-23/audit_chapter_quality_methodology.md`) found the corpus structurally rich but
semantically hollow: concept-salad assembly, ~9% placeholder cells, an L1 mix that puts `ASHRAE 62.1` in a
medical-imaging column and makes the model rationalize it, and no depth/refinement. These are symptoms of a
single LLM pass with no review and no ground-truth re-entry. The fix is not six patches — it is an agent that
proposes, is *gated by the membrane*, and re-enters with a typed critique. The patches become the agent's
toolbelt; the membrane stays the oracle.

## Substrate mapping
- **Orchestrator** — the meta-harness FSM (`src/aegir/meta_harness/fsm_rete.py`): states
  `PROPOSE → VALUE_GATE → RI_GATE → PROSE_GATE → COMMIT`, with a `CRITIQUE` loop-back. RETE rules fire one
  effector per cycle; the `Agenda` orders the cascade; the append-only trace is the provenance.
- **Proposer (PROPOSE effector)** — hermes-agent `AIAgent` (`/home/rch/local/src/oss/hermes-agent/run_agent.py:437`),
  driven as a library, pointed at the local vLLM OpenAI endpoint. Proposes candidate values + scaffold edits
  + prose into `build/dev/scratch/`. Scaffold tools (`scaffold/synth_column`, `scaffold/rebuild_table`,
  `scaffold/draft_prose`) registered via `registry.register`; handlers call our ontology-realization code.
- **Membrane (gate effectors — deterministic, OUR code, run by the FSM not the agent):**
  - **VALUE_GATE** — value-level HermiT: extend `reasoning_gates.render_batch` with a *value-ontology
    fragment* (entity-value pools as class-instances + domain axioms) and classify once. Checks: class
    membership, property cardinality, **disjointness** (kills ASHRAE-in-imaging), value-range. → admission
    set + unsat/equivalence trace; rejects return a structured delta.
  - **RI_GATE** — admitted values → transient in-memory relational view (DDL spine loader): FK/RI assertions,
    `CREATE VIEW` Data-Element predicates, aggregate coherence (column values within hypernym, row-count
    bounds). → typed critique on failure.
  - **PROSE_GATE** — only after relational gates pass: structural isomorphism (prose entities ↔ view keys),
    semantic entailment (embedding / exact-mention), length distribution vs FinePDFs samples + truncation
    boundary diagnostics. → causal critique + re-invoke with the verified spine as immutable context.
  - **COMMIT** — full-cascade success: `scratch → current` corpus JSONL + chapter artifacts (sdg-corpora RC).
    Metrics sidecar: HermiT admission rate, RI violations, prose↔table entailment, length distribution,
    iteration count.

## Invariant
The agent **proposes**; the membrane **disposes**. Gates are FSM effectors external to hermes; the agent can
request scaffold tools and draft prose but cannot run HermiT/RI/correspondence or commit. Refinement therefore
cannot add confident new errors that survive verification — it can only converge toward an admissible artifact
or exhaust its iteration budget.

## Key decisions / load-bearing pieces
- **Proposer transport:** drive `AIAgent` as a library for inc-0 (fastest; invariant holds because gates are
  external). The full ACP-wire form — meta-harness as ACP *client* exposing scaffold tools to hermes over ACP
  — is the production refinement (note, not inc-0).
- **Model endpoint:** hermes → `http://127.0.0.1:8100/v1` (the engine's vLLM, OpenAI-compat) for inc-0; a thin
  OpenAI-compat proxy over the gRPC engine for production (preserves strict layering [[capability_grpc_engine]]).
- **The value ontology is the critical new authored artifact** — the membrane's value-disjointness/range
  axioms over the entity-value pools. Bootstrappable by the existing LLM-deriver + HermiT-admission machinery
  (the value axioms are themselves membrane-gated). VALUE_GATE is only as sharp as this fragment.
- **`scratch` is the proposal staging** (sibling to `current`/`archive`) — `scratch → current` on commit.

## Increment plan
- **inc-0 — close the loop on ONE chapter, measure the delta.** Take ch0 (the salad/placeholder chapter);
  PROPOSE (re-synth placeholder columns, flag the salad) → VALUE/RI/PROSE gates → measure placeholder rate,
  value-coherence, RI, prose-correspondence, length vs the single-shot baseline. Proves the loop *improves* a
  chapter. Build order: (a) value-HermiT prototype (aegir env, highest value); (b) hermes smoke against the
  vLLM; (c) the FSM skeleton wiring the two.
- **inc-1 — value ontology + VALUE_GATE** at corpus scope (bootstrap the value axioms, gate the pools).
- **inc-2 — full cascade** (RI views + prose correspondence) as meta-harness effectors; dual-register output.
- **inc-3 — skills accumulation + scale** (hermes skills library; calibrate-strong-then-distill-local; the
  ACP-wire transport; reproducibility via cached transcripts/seeds).

## Risks
- Agency must extend to the **scaffold** (re-select templates, re-synth columns), not just prose — else it
  polishes the salad. The scaffold tools enforce this.
- Cost/throughput (~10× single-shot) — free solar GPUs make it time not money; cap iterations; calibrate with
  a strong agent then run local.
- Reproducibility — cache agent transcripts + seeds so the corpus stays regenerable-from-truth.
- Value-ontology coverage — the membrane is bounded by the authored axioms; start with the high-frequency
  disjointness violations the audit found.
