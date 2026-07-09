# Skills Library & Closed Generate→Re-ground→Refine Engine — Specification

**Status:** v0.1 — accepted; DOF 2 & 4 resolved · DOF 1 & 3 deferred (see §3) · **Date:** 2026-06-05

> **Status (2026-07-09) — dated record; partially realized, partially
> superseded.** The skills package landed
> (`src/aegir/ontology/skills/` — S2/S3/S5/S6/S7 plus the
> per-modality gates in `base.py`), and the closed
> generate→verify→refine shape lives on in the metaflow pipeline
> (`src/aegir/flows/sdg_corpora_flow.py`, `just metaflow`). Two
> substrates named below are superseded: the **BERTopic re-grounding
> spine** (the cached FinePDFs model, `topic_recovery`, `T_I.pkl`)
> was deprecated 2026-07-07 — the loop-closure verifier's successors
> are **congruence** (`src/aegir/ontology/congruence.py`) and the
> **inverted topic layer** (`src/aegir/ontology/topic_layer.py`;
> phase gate:
> [Inverted Topic Layer](../roadmap/phase_gate_inverted_topic_layer.md))
> — and the **family complex** (S6's admissibility gate; §2.1 step 7)
> is retired: cross-entity links are the deriver's to earn, never
> name-match-wired. The catalog is now fully derived
> (`src/aegir/ontology/catalog/catalog.json`; the hand-authored seed
> families are retired). Read the spec as the 2026-06-05 design; do
> not build new consumers on its BERTopic surfaces.

## 0. The spine: FinePDFs as immutable fixed point, and the re-grounding invariant

The pipeline is a **closed, verifiable loop** with FinePDFs as the fixed-point
ground at both ends:

```
FinePDFs ──induce (coverage audit / BERTopic)──► ontology
   ▲                                          + skills library
   │                                                 │
   │                                                 ▼
   │                                    generated corpus (prose ⨉ relational data)
   └──────── re-grounds verifiably (BERTopic topic-recovery) ◄────┘
```

**Re-grounding invariant (the contract every skill and build must satisfy).**
Every `CorpusUnit` carries provenance to (a) ≥1 ontology element (the *what*,
induced from FinePDFs) and (b) ≥1 FinePDFs `SourceSpan` (the *ground*), such that:

- **Fidelity** — the unit's topic distribution re-grounds to its seeding FinePDFs
  topics under the cached BERTopic model: `topic_recovery ≥ τ_topic`.
- **Structure** — any relational data type-checks against the ontology
  slot-types: `r_axiom ≥ τ_axiom`.
- **No drift** — every asserted claim resolves to a `SourceSpan` or ontology
  axiom: `claim_grounding ≥ τ_ground`.

A skill or corpus build is *admissible only if it preserves this invariant.* This
is what converts validation from an external step into an intrinsic consistency
condition.

### Core schemas (referenced by all skill signatures)

| Type | Fields |
|---|---|
| `TopicVec` | FinePDFs BERTopic distribution (cached model; the re-grounding coordinate) |
| `SourceSpan` | `{doc_id, char_range, text, topic_vec}` — a FinePDFs evidence span |
| `OntologyRef` | `{template_id \| class_iri, slot_types, bfo_anchor, verbal_template}` (from the catalog) |
| `Claim` | `{text, grounded_to: SourceSpan \| Axiom \| null, status}` |
| `Provenance` | `{ontology_refs[], source_span_ids[], skill_id@ver, target_topic_vec, claims[]}` |
| `CorpusUnit` | `{kind: prose\|table\|diagram\|example, content_md, schema?, provenance}` where `schema` (tables) = `{columns:[{name, slot_type}], fk_edges:[(col→col)]}` |

`schema.columns[].slot_type` **is the deterministic CTA/CPA/DED label** — ground
truth for the downstream model, type-checked against `OntologyRef.slot_types`.

---

## 1. Skills Library Definition

A **skill** is a typed, versioned generative competency: `skill@semver(grounded
inputs) → CorpusUnit(s) + provenance`, contractually preserving the re-grounding
invariant. The library *replaces the template catalog as the generation driver*:
the ontology supplies the *what*, skills supply the *how*, FinePDFs verifies *that
it held*.

### 1.1 First-class skills

**S1 · `verbalize-axiom`**
- **(a) Signature:** `verbalize_axiom(ref: OntologyRef, fillers, evidence: SourceSpan[]) → CorpusUnit(prose)`
- **(b) I/O schema:** in = one axiom (`ref.verbal_template` + slot fillers) + evidence spans whose `topic_vec` defines the target; out = prose unit; provenance.claims each grounded to a span/axiom.
- **(c) Strategy:** seed with the DeepOnto `verbal_template` as a faithful scaffold; LLM elaborates into prose **constrained to assert only what the evidence spans support** (every sentence → a `Claim` with `grounded_to`). Prompt skeleton: *"Using only the facts in {evidence}, explain {verbalized axiom}. Cite each claim to a source. Match the register of {evidence}."*

**S2 · `synthesize-relational-table-with-cross-FKs`**  *(the column-annotation-bearing core)*
- **(a) Signature:** `synth_relational_table(refs: OntologyRef[], fillers, evidence: SourceSpan[]) → CorpusUnit(table)`
- **(b) I/O schema:** out.content_md = markdown table(s); **out.schema = `{columns:[{name, slot_type}], fk_edges}`** — the CTA/CPA/DED labels. Cell values drawn from / consistent with `evidence`.
- **(c) Strategy:** map ontology entities→tables, slots→columns (slot_type = label), object-properties→FKs; populate cells from evidence values. Two hard post-conditions: **type-check** (headers vs `slot_types` → `r_axiom`) and **value re-grounding** (cell distribution re-grounds to `evidence.topic_vec`). Prompt skeleton: *"Construct relational tables instantiating {refs} with realistic values grounded in {evidence}; emit columns with their ontology slot-types and explicit foreign keys."*

**S3 · `interleave-diagram`**
- **(a) Signature:** `interleave_diagram(refs: OntologyRef[], local_structure) → CorpusUnit(diagram)`
- **(b) I/O schema:** out = mermaid/d2 of ER / taxonomy / dataflow over the same refs; must be **cross-consistent** with any S2 table (edges ↔ FKs) and S1 prose.
- **(c) Strategy:** render the ontology subgraph (BFO anchors + object-property edges); constrain node/edge set to the refs already used in the unit so the diagram cannot introduce ungrounded entities.

**S4 · `worked-example`**  *(the generality driver — Thesis 2)*
- **(a) Signature:** `worked_example(ref: OntologyRef, evidence: SourceSpan[]) → CorpusUnit(prose+data)`
- **(b) I/O schema:** a concrete instantiated case — a populated record, a query+result, or a short reasoning trace — over real values from `evidence`.
- **(c) Strategy:** instantiate the abstract axiom on concrete grounded data and *show the reasoning*; this is where transferable skill (not template recall) is taught. Prompt skeleton: *"Walk through a concrete instance of {ref} using {evidence}; show each inference step."*

**S5 · `ground-claim-to-source`**  *(the fidelity enforcer; runs inline + as a pass)*
- **(a) Signature:** `ground_claim(claim: Claim, evidence: SourceSpan[]) → GroundedClaim | REJECT`
- **(b) I/O schema:** attaches `grounded_to` (span/axiom) or rejects the claim; aggregate → `claim_grounding` rate for the unit.
- **(c) Strategy:** retrieval + entailment check of each claim against evidence/axioms; unsupported claims are dropped or trigger regeneration. This is the skill that makes the corpus *verifiable* rather than asserted.

**S6 · `cross-reference` (link-concepts)**
- **(a) Signature:** `cross_reference(ref: OntologyRef, fc: FamilyComplex) → CorpusUnit(prose links)`
- **(b) I/O schema:** prose connecting the unit's concept to allowable neighbors; the cited family-set must be an **allowed simplex** (`fc.is_allowed`, else `fc.best_face`).
- **(c) Strategy:** use the family complex to weave only co-coherent multi-family links (never a measured puncture), producing the relational richness without incoherent cross-family claims.

**S7 · `topic-anchor` (re-grounding conditioner)**
- **(a) Signature:** `topic_anchor(unit: CorpusUnit, target: TopicVec, evidence: SourceSpan[]) → CorpusUnit'`
- **(b) I/O schema:** rewrites/condition the unit so its embedded topic distribution moves toward `target` (the seeding FinePDFs topics).
- **(c) Strategy:** the skill that directly closes the loop on `topic_recovery` — style/lexis/emphasis transfer toward the FinePDFs target without altering grounded claims or schema. Applied last, re-verified by the engine.

*(Candidate extensions: `define-term`, `summarize-section`, `counterexample`,
`pose-and-answer` — each admitted only via §1.3.)*

### 1.2 Input/output grounding summary
Every skill takes `OntologyRef` (+ `FamilyComplex` where relevant) and FinePDFs
`SourceSpan[]`, and emits `CorpusUnit` with full `Provenance`. No skill may emit a
`Claim` without a `grounded_to`, nor a table column without a `slot_type`. This is
the static guarantee behind the re-grounding invariant.

### 1.3 Versioning & extension without breaking the invariant
- **Identity:** every skill is `skill_id@semver`; each `CorpusUnit.provenance`
  records the exact versions used → reproducible, traceable builds.
- **Pinned skill-set per build:** a corpus build pins a skill-set manifest
  (`{skill_id@ver}`), so any corpus is reproducible and its fidelity is attributable.
- **Skill admission test (the gate):** a new/changed skill version joins the
  library **only if**, on a frozen calibration sample of FinePDFs seeds, the units
  it produces satisfy `topic_recovery ≥ τ_topic ∧ r_axiom ≥ τ_axiom ∧
  claim_grounding ≥ τ_ground`. The invariant is thus enforced *at admission*, not
  hoped for at runtime. Append/version-only; deprecations are explicit.

---

## 2. Closed Generate → Re-ground → Refine Engine

### 2.1 End-to-end control flow (one generation episode)
1. **Seed from FinePDFs** — sample a FinePDFs document **from a non-holdout
   (training) BERTopic cluster** (§2.5); take its topic mixture (`target: TopicVec`)
   and its `SourceSpan[]` as the re-grounding anchor.
2. **Select ontology content** — via the coverage audit, choose `OntologyRef[]`
   matched to `target`, family-diverse, gated by the family complex
   (`is_allowed` / `best_face`).
3. **Compose skills (generation plan):** `S1 verbalize-axiom → S2 synth-relational-table → S3 interleave-diagram → S4 worked-example → S6 cross-reference`, with **S5 ground-claim-to-source as an inline guard** on every unit, then **S7 topic-anchor** toward `target`.
4. **Assemble** the units into a chapter with merged `Provenance`.
5. **Re-ground (VERIFY)** — §2.2.
6. **Score & route** — compute composite `F`; if `F ≥ τ_F` accept into the corpus,
   else fire the matching **refinement trigger** (§2.4) and regenerate.
7. **Update family complex** — record the cited simplex as allowable (F ≥ floor) or
   a puncture (measured-below-floor), per the current `build_family_complex` role.

### 2.2 Re-grounding step (the loop-closure verifier)
- **Primary — BERTopic topic-recovery (CANONICAL, DOF 2 resolved):** embed the
  generated chapter, run the cached FinePDFs BERTopic model
  (`approximate_distribution`), score per-doc recovery against the seeding `target`
  topics → **`topic_recovery` (hit@k / cosine)**. This is the **single** re-grounding
  metric, used identically in-loop and at the downstream generality check; the
  verifier's prior `R_D` (MiniLM→KMeans→Hungarian to `T_I.pkl`) is **retired from
  the in-loop check** and retained for offline diagnostics only — so every iteration
  is scored by the exact embedding+clustering pipeline used downstream. *Its earlier
  30% (full arm) vs 80% (no-ontology) is the explicit gap to close — not a trade-off.*
- **Auxiliary consistency checks:**
  - `r_axiom` — table headers/columns type-check vs ontology `slot_types`.
  - `claim_grounding` — fraction of claims with a valid `grounded_to`.
  - `cross_modal` — diagram edges ↔ table FKs ↔ prose claims agree.
  - `r_iri` — cited templates present in prose.
- **Composite fidelity:** `F = w_t·topic_recovery + w_a·r_axiom + w_g·claim_grounding + w_x·cross_modal` (weights are an open DOF — §3; to be *calibrated against the downstream signal*, not asserted — per the eval-methodology survey).

### 2.3 Quantitative success criteria (loop termination)
A build is **converged/hardened** when, over the FinePDFs topic distribution on a
held-out calibration sample, all hold and are stable across K iterations:

| Criterion | Symbol | Initial target |
|---|---|---|
| Re-grounding | `topic_recovery ≥ τ_topic` | **≥ 0.80** (parity with no-ontology, *while keeping structure*) |
| Structure retained | `r_axiom ≥ τ_axiom` | ≥ 0.45 (family-complex floor) |
| No hallucination | `claim_grounding ≥ τ_ground` | ≥ 0.95 |
| Cross-modal consistency | `cross_modal ≥ τ_x` | ≥ 0.90 |
| Composite, stable | `F ≥ τ_F`, Δ over K iters < ε | τ_F, K, ε TBD (§3) |
| Coverage | accepted-fraction across topic bins ≥ ρ | ρ TBD |

The headline objective is **τ_topic ≥ 0.80 with r_axiom ≥ 0.45 simultaneously** —
i.e., close the 30%→80% re-grounding gap *without* surrendering structure.
**Adopted 2026-06-05 as the loop-termination criterion.** The conjunction is hard —
no weighted trade-off — since any relaxation re-introduces the drift the loop exists
to eliminate.

### 2.4 The two operational levers
- **Lever A — Ontology refinement (incl. family complex).** Fired when
  `r_axiom`/structural or coverage checks fail. Actions: fix/extend templates &
  slot-types; re-induce from FinePDFs coverage gaps (audit); update the family
  complex (promote simplices that co-generate ≥ floor; record punctures that fail).
  Mechanizable later by the GRPO policy optimizing ontology selection against `F`.
- **Lever B — Corpus hardening.** Fired to move from "passes on a sample" to
  "passes across the full FinePDFs distribution at volume." Actions: scale &
  diversify accepted units across topics/families/skill-compositions; dedup &
  balance; enforce thresholds distribution-wide; confirm stable `F`. The output is
  the dataset whose model exhibits generality.

A failing `topic_recovery` routes to **skills** (S1/S7 — the *how* drifted) and/or
ontology selection; a failing `r_axiom` routes to **Lever A** (ontology); a failing
`claim_grounding` routes to **S5** / unit rejection.

### 2.5 Held-out FinePDFs generality check (downstream validation)
- **Holdout (DOF 4 resolved — topic-cluster-level):** partition FinePDFs at the
  **BERTopic cluster level** (the same model used for re-grounding), **before any
  generation**; held-out clusters seed/ground *no* unit, and generation seeds
  (§2.1.1) are drawn **only from non-holdout/training clusters**. This enforces
  cross-*region* generality (not merely unseen documents) and supplies both the
  final generality check and intermediate calibration runs.
- **Train** the model (DED + CTA/CPA heads) on the hardened corpus.
- **Evaluate on tasks derived from the held-out slice:**
  - **Data-element discovery (the end model):** cluster/embed columns extracted
    from held-out FinePDFs-derived relational structures into data elements;
    measure against held-out ground-truth groupings.
  - **CTA/CPA:** column → slot-type on held-out-derived tables, with the
    SOTA-grounded protocol — **PR-based mAP/LRAP + precision@k** (not ROC-AUC),
    **sample-efficiency curves**, **bootstrap/permutation CIs**, vs **random-init**
    and a **matched-token non-grounded** corpus arm.
  - **Generality = transfer to held-out FinePDFs-grounded structure**, not recall
    of trained topics/templates.
- This is Track C, correctly scoped to held-out FinePDFs (the loop's own ground) —
  no external benchmark, no template-recognition artifact.

---

## 3. Open degrees of freedom (for joint review)
1. **Composite `F` weights** `(w_t, w_a, w_g, w_x)` — **DEFERRED to the first
   calibration run**: set from the downstream DED/CTA/CPA signal, not asserted (cf.
   the verifier's P2 sweep, which over-weighted topic at 0.45 among other terms).
2. **Thresholds** `τ_topic, τ_axiom, τ_ground, τ_x, τ_F, K, ε, ρ` — **DEFERRED** to
   calibration; the headline pair `τ_topic ≥ 0.80 ∧ r_axiom ≥ 0.45` is **adopted now**
   (§2.3); the rest set vs. a held-out sample.
3. **Topic-recovery estimator** — **RESOLVED 2026-06-05: BERTopic-recovery is the
   single canonical re-grounding metric, in-loop + downstream; `R_D`
   (MiniLM→KMeans→Hungarian to `T_I.pkl`) retired from the loop, offline-diagnostics
   only.** MDL/codelength remains an optional offline lens.
4. **Skill-composition policy** — **DEFERRED to the P5 RL run**: fixed sequence
   (§2.1) vs. ontology-driven vs. learned (GRPO), decided on evidence.
5. **Grounding granularity** — per-claim vs. per-unit `SourceSpan` attachment.
6. **Engine mode** — single-pass generate-then-verify vs. per-unit iterative repair.
7. **Refinement actuation** — human / agent / GRPO-policy for Levers A & B (the P5
   run is the RL form).
8. **Holdout protocol** — **RESOLVED 2026-06-05: topic-cluster-level FinePDFs
   holdout, partitioned before generation (§2.5)** — the stronger, cross-region test.
