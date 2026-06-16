# End-to-end pipeline + Meta-Harness + reasoner activation — spec

Status: design of record (2026-06-16). Organizes the next phase. Supersedes the control-plane framing
in `meta_harness_boundary.md` (the RETE/FSM spine is demoted to **harness H₀**, see §2). Two reframes
drive it: (a) Meta-Harness (Lee et al. 2026, `build/resources/2603.28052.pdf`) — optimize the *harness*
(the program wrapping a frozen executor) via an outer-loop coding-agent proposer over a filesystem of
candidates; (b) **the ontology IS the dynamic computational framework** — a reasoner-backed (HermiT)
artifact we evolve in situ from domain inputs. Discipline (the lesson that produced this spec): adopt the
**form**, not the language — every new piece must ground out in concrete, measured computation.

## 0. The end-to-end flow (the complete workflow — keep this in view)

A filesystem-DAG. Each stage writes a dir with `manifest.json` + a `run_id` that hashes its inputs →
content-addressed lineage (no orchestration engine required; see §5).

```
S0 INPUT      finepdfs-lab corpus  (/raid/datasets/aegir-corpus-v1/finepdfs-lab/)
S1 COVERAGE   ontology_coverage_audit.py → coverage_v1/<run>/ {topic_coverage.parquet,
              topic_centroids.npy, manifest}   [FinePDFs → topics → gap/borderline/covered]
S2 EVOLVE ◄── THE META-HARNESS. mediate.py (spine + ACP/Grok mint + ContractGate[+reasoner])
   │          → evidence/meta_harness/<run>/ {trace, scorecard, candidates.candidate.json}
   │          [gap topic → mint construct → gate (DeepOnto+polyglot+R1+novelty+schema+CONSISTENCY)
   │           → promote]. THIS stage is what the OUTER LOOP (§2) optimizes.
S2' REVIEW    charter: editorial review → promote .candidate → catalog family files (human-in-loop)
S3 DDL/SKOS   ddl.py (template_to_table→render_ddl→validate_ddl/polyglot) + build_skos_vocab.py
              → DDL spine + 548-concept SKOS + Atlas rdbms_* projection
S4 CORPUS     generate_chapter.py (ontology+DDL → chapters + verifiable JSON + reasoning traces)
              → chapters.parquet + raw.exchange
S5 VERIFY     verify_chapters.py → raw.chapter_verification
S6 RELEASE    build_atelier_release.py (columns/vocabulary/reference blind benchmark) + HF/GitHub
S7 PRETRAIN   train_pretrain.py → byte model
S8 EVAL       eval_cells_cta / eval_edge_probe / REALIZATION-CPA (§3c) → column/relational skill
```

Feedback edges (the loops): S2's grown ontology → re-run S1 (coverage-close); S5/S8 scores → the
OUTER LOOP reward (§2); S1 gaps → S2 targets. The convergence loop = S1→S2→S3→S4→(S7→S8)→back.

## 1. The two frozen executors (the parallel)

| Meta-Harness (paper) | Our pipeline |
|---|---|
| frozen LLM `M` | frozen **Grok** (the minting model) AND frozen **HermiT** (the reasoner) |
| evolved harness `H` (a program) | the **generation harness** (S2) AND the **ontology `O`** (a reasoner-executed program) |
| outer-loop coding-agent proposer | the Meta-Harness loop (§2) |
| reward = task accuracy | R1/coverage-close + consistency + realization-accuracy (Pareto vs cost) |

We optimize TWO artifacts against TWO frozen executors: the harness (around Grok) and the ontology
(around HermiT). The harness *grows* the ontology; the reasoner makes the ontology *executable*.

## 2. Meta-Harness outer loop (the FORM)

- **A harness `H`** = a single-file program: `run(topic, gate) -> (construct, signals)` — builds the
  mint prompt (contract + topic salient terms + exemplars), calls Grok (frozen), gates, iterates.
  **H₀** = the current inc-1 harness, refactored to one clean program (the RETE/FSM ceremony pruned to
  a minimal loop; let the proposer re-introduce structure only if it earns reward).
- **Candidate filesystem `D`** (the feedback channel): `candidates/{NNN}/{harness.py, traces/, scores.json}`.
  Full, uncompressed — NOT the scalar signal vector (the anti-pattern the paper beats).
- **Proposer `P`** = a coding agent (Claude Code/Opus, or Grok-as-coder) + a *minimal skill* (where to
  write harnesses, how to `grep`/`cat` prior code+traces, what it may edit). It diagnoses from raw
  traces and rewrites the harness (local edit → full rewrite).
- **Eval / reward** = run `H` on a SEARCH SET of gap topics → batch on-vs-shuffled R1 / coverage-close
  + cost (Grok tokens, iters) → Pareto frontier. Proposer never sees the HELD-OUT topic set.
- **Loop** (Algorithm 1): evaluate initial {H₀,…} → for N iters: P reads `D`, proposes k harnesses,
  interface-validate + evaluate + log → return Pareto frontier; final eval on held-out.

## 3. HermiT reasoner activation (make the ontology executable — the FORM, not the word)

Today HermiT is **dormant**: we verbalize + read `get_asserted_complex_classes` (syntactic), never
reason. Activation = three concrete, measured computations:

- **(a) Consistency gate [beachhead].** After a construct passes the syntactic gates, HermiT
  consistency-checks the **cumulative** ontology (seed ∪ admitted ∪ candidate). New `ContractGate`
  signal `consistent`; reject if it makes `O` inconsistent. A deductive check nothing syntactic can do —
  it's what keeps the in-situ-evolving ontology a *coherent* computation. Measure: rejections-for-
  inconsistency; `O` provably consistent as it grows.
- **(b) Inferred hierarchy (classification).** Coverage/structure read HermiT's *inferred* subsumption
  closure, not the asserted SubClassOf chains.
- **(c) Realization-as-CPA [re-homes G-rel].** Map the corpus's verifiable-JSON rows → an OWL ABox →
  HermiT **realize** → the column/entity types & relations *computed by the reasoner*. CPA/CTA becomes
  inference, not a tiny-model probe (which floored → G-rel descoped). Eval = realization accuracy vs the
  held-out `reference.parquet`. The relational computation relocates to the reasoner; the model becomes
  a fast amortization of it, not the thing that must learn it.
- **Caveats (real):** OWL profile — generated complex-class constructs push expressivity; keep near OWL 2
  EL for PTIME ELK where possible, or accept HermiT's DL cost; the ABox bridge (DDL/JSON → assertions) is
  a genuine new pipeline piece. Reasoner invocation via DeepOnto's `OntologyReasoner`/OWL-API (HermiT
  bundled, currently uncalled).

## 4. How they compose

The outer loop (§2) optimizes the harness that grows the ontology; the reasoner (§3) makes the ontology
self-consistent and executable; the end-to-end DAG (§0) is where both live. One sentence: **a coding
agent evolves the program that grows a reasoner-backed, domain-adaptive ontology, judged by what the
reasoner and the corpus compute.**

## 5. Orchestration stance (the Airflow question)

- **Now:** the **filesystem-DAG** (§0) + thin drivers (`just` recipes + small Python runners) +
  `manifest.json`/`run_id` content-addressed lineage. This carries the whole-workflow understanding
  (legibility) without runtime complexity, and matches the Meta-Harness grain (filesystem + agent, not a
  DAG engine). The candidate filesystem `D` (§2) is the same substrate.
- **NOT Airflow now:** it's a runtime orchestrator for stable/recurrent/scheduled flows; ours is in flux,
  and Airflow's scheduler/DB/webserver ceremony would ossify a flow we're still discovering — and
  over-orchestrate the part the proposer should navigate.
- **Later (convergence-loop maturity):** a *lightweight* orchestrator — Metaflow (the Gaius precedent) or
  OpenLineage→Atlas (the project's existing provenance direction) — when S1→S2→…→S8 runs recurrently and
  lineage/scheduling pays off. `components/` (cldr/signals) holds Airflow if we ever need it; default no.

## 6. Increment ladder

- **inc-2a (beachhead):** HermiT **consistency gate** in `ContractGate` (`consistent` signal over the
  cumulative ontology) + a seed rule. Smallest real reasoner computation; immediately makes `O` coherent.
- **inc-2b:** **H₀-clean** — refactor the spine+mint+gate into a single-file harness program with a
  `run(topic, gate)` interface; stand up `candidates/{NNN}/` + interface validation.
- **inc-2c:** the **Meta-Harness outer loop** — proposer + minimal skill + search/eval/Pareto over the
  candidate filesystem; reward = batch R1/coverage-close vs cost on the search set.
- **inc-2d:** **realization-as-CPA** — the ABox bridge + HermiT realize + the symbolic-CPA eval vs the
  held-out reference (G-rel re-homed).

## 7. Reward / decision rules (measurement, so this stays form not language)

- Harness search reward: batch on-vs-shuffled **R1 / coverage-close** on the search set, Pareto vs Grok
  cost; a discovered harness must beat H₀'s frontier on held-out topics to be adopted.
- Reasoner: consistency-gate must reject ≥1 genuinely-inconsistent construct (instrument validity) and
  keep `O` consistent as it grows; realization-CPA valid iff accuracy > control on the v0.3 backbone-free
  symbolic path, CI-clean vs the held-out reference.
- Every increment chains to one of these numbers or it does not ship (the standing rule).

## Verification
1. Control plane unchanged where reused; H₀ run reproduces inc-1 (t124 R1 ≈0.39, promote).
2. inc-2a: consistency gate rejects a hand-crafted inconsistent construct; passes the t124 construct; `O`
   stays consistent across a batch. 3. inc-2c: a discovered harness beats H₀ on held-out coverage-close.
4. inc-2d: realization-CPA selectivity CI-clean vs reference. Artifacts under `evidence/` per stage.
