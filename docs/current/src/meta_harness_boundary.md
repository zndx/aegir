# Agent-Mediated Meta-Harness — Signals & Boundaries spine

Status: design of record (2026-06-16). Implements the reactive pivot of `just mediate`
(the agent-mediated reference-ontology builder) as a **RETE/FSM control spine** within
explicit boundary conditions (Holland, *Signals and Boundaries*). This is a **spine, not a
feature** — built clean and complete up front, because a cogent fact/rule/agenda/FSM
architecture cannot be retrofitted out of a procedural loop.

## 1. The Holland mapping (what the spine actually is)

This is not decoration; it names the structure we established empirically.

| Holland S&B | Aegir meta-harness |
|---|---|
| **Boundary / membrane** | the **contract** — the conjunctive gate suite that admits a construct as "inside the ontology": DeepOnto verbalizes + asserts a complex class · Polyglot validates the DDL · R1 topic-specificity · E6-A topic-survival · BFO-anchor · novelty · schema-realism |
| **Signals** | the **gate verdicts** — the dense per-construct/topic scorecard the engine reasons over |
| **Internal model / tags** | the **objectives** (pre-registered, scored) the agent navigates by |
| **Bounded adaptive agent** | the **FSM** — a signal-responsive navigator that "tacks and jibes" toward a contract-passing construct, never sailing straight into the contract (the one-shot generator proved you can't) |
| **Emergence (many agents)** | the **scale end-state** — concurrent multi-topic mediation as supervised actors; this is where the full CAS lives, and where a true RETE discrimination network earns its keep |

The membrane is the load-bearing asset; the **producer is swappable** (one-shot LLM → agent →
hand). The harness is the producer that navigates the membrane by signal.

## 2. The spine API (stable; this is the part you cannot retrofit)

```
Fact(kind, payload, id, rev)                 # unit of working memory
WorkingMemory                                # assert / update / retract / query; global revision counter
Rule(name, salience, when(signals,ctx)->bool, then(ctx)->Effect)   # production rule
Agenda                                       # conflict set → resolve (salience ↓, specificity ↓, recency ↓) → fire one/cycle; log the full set + choice
Objective(name, score(signals)->float, satisfied(signals)->bool)   # scored, pre-registered
Effector  (protocol, EXTERNAL to the spine):
    mint(topic, feedback) -> construct       # the LLM agent (stochastic)
    gate(construct, topic) -> signals        # the deterministic contract (DeepOnto, polyglot, R1, …)
Trace                                        # append-only JSON: every fact change, rule firing, conflict
                                             #   resolution, state transition, effector call
```

**Matching is a swappable internal, not part of the spine.** `Rule.when` *is* the alpha-test
layer. The first implementation evaluates rules by direct scan (correct, O(rules) per cycle —
trivial at our scale). A full RETE discrimination network (alpha/beta nodes, join sharing,
token propagation) drops in **behind this unchanged API** when working memory grows to many
concurrent constructs × topics. Swapping the matcher is licensed *because* the spine is clean;
it is **not** retrofitting the spine.

## 3. The FSM (5 states, rule-driven transitions)

```
OBSERVE_CONTRACT  →  gate current construct (or seed) → assert signal facts
SELECT_OBJECTIVE  →  run the rule cycle → highest-salience rule sets the `objective` fact
ACT_AND_SIGNAL    →  invoke Effector.mint for the objective (with feedback from signals)
EVALUATE_FEEDBACK →  Effector.gate the new construct → update signal facts
TERMINATE_OR_ITERATE → a rule fires success (promote) or give-up (max iters); else → OBSERVE
```

The FSM is the outer loop; **rules** decide the objective (SELECT) and termination (TERMINATE).
Signal facts are asserted in OBSERVE/EVALUATE. Every state is itself a fact, so rules match on
`(state, signals)`. Determinism lives here.

## 4. The signal vector (dense facts the engine reasons over)

Per (topic, construct): `r1_on`, `r1_shuffled`, `r1_ci_low` (on−shuffled bootstrap CI low),
`deeponto_ok`, `deeponto_complex` (asserted complex class), `polyglot_ok`, `novelty` (max cos
vs seed∪admitted), `schema_entropy` (per-anchor column-set entropy vs SchemaPile),
`n_cols`, `iterations`, `state`.

## 5. Seed rules (8–12, grounded in the committed gates; salience in brackets)

1. `no_construct` [100] → objective `draft_initial`
2. `deeponto_fail` [90] → `fix_verbalizability` (membrane floor: unparseable ⇒ no cognition)
3. `not_complex` [88] → `fix_nontriviality`
4. `polyglot_fail` [80] → `fix_ddl` (views need cogent relational schema + values)
5. `r1_not_specific` [70] (`r1_ci_low <= 0`) → `enrich_domain_terms` (the binding constraint)
6. `novelty_low` [60] → `diversify` (not a seed duplicate)
7. `schema_canned` [50] (`schema_entropy < floor`) → `de_can`
8. `r1_improving` [40] (r1_on rose but not CI-clean, iters<max) → `refine`
9. `contract_satisfied` [120] (all gates pass ∧ `r1_ci_low > 0`) → `promote` (terminate-success)
10. `budget_exhausted` [110] (`iterations >= max`) → `give_up` (terminate, logged)

Higher-salience termination/floor rules dominate; the agenda logs the full conflict set each cycle.

## 6. Objectives (pre-registered, composable, scored)

`draft_initial` · `enrich_domain_terms` · `fix_verbalizability` · `fix_nontriviality` · `fix_ddl`
· `diversify` · `de_can` · `refine` · `promote` · `give_up`. Each scores its attainability from the
current signals; the rules select; the effector executes. Objectives are the *only* extension
point for new behaviour — add an objective + a rule, never a procedural branch.

## 7. Determinism & reproducibility (precise claim)

- **Control plane is deterministic:** given the same signal facts, the same rules fire in the same
  order (salience → specificity → recency); every firing/transition/conflict-resolution is logged.
- **Generation is stochastic but gated:** the LLM effector samples; we do *not* pretend otherwise.
  Reproducibility = log every exchange (we already capture `reasoning_content` to `raw.exchange`) +
  seeds. The membrane (contract) is deterministic, so a stochastic mint is always judged identically.

## 8. Logging substrate (aegir's, not assumed Atelier infra)

JSON sidecars under `evidence/meta_harness/<run>/`: `trace.jsonl` (event stream), `scorecard.json`
(final per-topic contract verdicts), `candidates.json` (admitted constructs). Exchanges → the
existing `raw.exchange` Iceberg table. No "leaderboard gateway / BDD suite" is assumed — those are
not in aegir.

## 9. Boundaries respected (the anti-patterns)

- **No micro-orchestration:** nothing calls `just mediate --topics X` per tool; the FSM invokes
  effectors as *actions chosen by rules from signals*. Sequence is emergent, not scripted.
- **No promotion without the full membrane:** a construct is admitted only when the *conjunctive*
  contract passes — this is the Goodhart guard (you cannot term-stuff R1 without also clearing
  DeepOnto-non-triviality, schema-realism/de-canning, novelty, survival). Optimize on a topic's
  terms; validate R1 on held-out topics.
- **Architecture must move the metric:** the spine earns its complexity only by producing better
  contract-passing constructs more reliably than a dumb loop. Every increment chains to an R1 /
  contract delta or it does not ship.

## 10. Inaugural success criterion (the metric gate on the spine itself)

Run the spine on the gap topics the one-shot generator failed (on-topic R1 ≈ 0.007, Δ+0.003 CI
touching 0). **The spine is a success iff it drives on-topic R1 to CI-clean-positive on ≥1 of
those topics, with every rule activation and state transition logged.** "It ran an iteration" is
not success; moving the metric is.

## Increment ladder

- **inc-0 (this design + engine core):** clean spine — facts/WM/rules/agenda/FSM/objectives/trace,
  naive matcher, stub effector, deterministic end-to-end self-test.
- **inc-1:** wire the real effectors (LLM `mint`; `gate` = `probe_template` + `render_ddl`/polyglot
  + `coverage_r1` + `schema_complexity`); run the inaugural R1 capability proof.
- **inc-2+:** reactivity (message-driven, back-pressure, supervision) + concurrent multi-topic
  actors (the Holland emergence / scale end-state); adopt the RETE discrimination network behind
  the matcher API when working-memory growth justifies it; evolve toward / merge with `swarm/`.
