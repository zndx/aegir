# EVIDENCE.md — claims ledger

Evidence-led development discipline (adopted 2026-06-12). Two standing rules:

1. **Pre-registration**: each experiment's hypothesis, instrument, decision rule, and CI method are
   written HERE before the experiment runs. Results update the status with a link to the artifact.
2. **No scaled spend without a green gate**: scaled generation, pretraining runs, RL training, and
   releases queue behind the experiment that gates them.

Statuses: `SUPPORTED` / `REFUTED` / `UNTESTED` / `PARTIAL`. Artifacts live under
`/raid/checkpoints/aegir-artifacts/evidence/<id>/`.

## Ledger

| id | claim | gates (what may not proceed until green) | status |
|----|-------|------------------------------------------|--------|
| E1 | The chapter-verifier composite ranks chapters by downstream pretraining value | proxy-gated corpus filtering, generator admit thresholds, GRPO reward, E5 scale-up | **PARTIAL** — coarse/cross-model signal real; within-model NOT CI-clean → re-derive before fine-grained gating |
| E2 | *(instrument)* the edge-probe validly measures relational/structural skill | E3, the v0.4 headline eval | **UNBUILT** |
| E3 | The DDL-injected (load-bearing) corpus beats no-schema on relational skill | Phase 3 (FK/views), Phase 4 scale-up, v0.4 release | **UNTESTED** |
| E4 | The blind column benchmark is non-trivial; an independent (Atelier) baseline exists | any "Aegir lift" claim | **UNTESTED** |
| E5 | The generator produces semantically genuine, novel, coverage-closing ontology | scaled generation, catalog promotion | **PARTIAL** (shallow gates only) |

## Already-supported claims (for completeness; artifacts on /raid)

- **Synthetic-corpus byte pretraining imparts transferable column skill on real tables**: cells-only
  GitTables-DBpedia CTA, frozen-backbone linear probe — 0.66–0.70 acc vs 0.12 random-init; Hewitt-Liang
  selectivity 0.15→0.56; non-overlapping bootstrap CIs at every N. (`cells_cta_v0/result_full_cells.json`)
  *Caveat:* per-column CTA did NOT separate full/no-ontology/no-schema arms — flat CTA is surface-solvable;
  the ontology claim moved to the relational axis (hence E2/E3).
- **The corpus does not collapse at 2K-chapter scale**: deterministic collapse-trend (distinct-n, gzip,
  intra-bin NN-cosine) flat across all bins; 0 near-dups. (`corpus_quality_full.json`)
- **The typed spine closes the schema width/type gap toward real**: columns/table 2.75→~8.45 (SchemaPile
  median 5, mean 6.6); data-columns 0→4.7 with real type mix. *Caveat:* per-anchor attributes are uniform
  ("canned") — a learnable shortcut; de-canning is measured under E5.

---

## E1 — proxy calibration: does verifier composite track downstream value?

**Claim.** The per-chapter verification composite (`verify_chapters.py`: geometric mean of R_topic,
R_iri, R_density, R_axiom) ranks chapters by their downstream pretraining value.

**Design.** Score all v0.3 chapters (runs `d7646714bdd5e16f` + `811408b392859708`, n=2,235). Quartile-split
by composite at matched byte budgets per slice. Pretrain tiny (13.5M) per quartile — identical steps, seed,
and token budget. Probe each with `eval_cells_cta.py` (frozen backbone, mean-pool, N=1999) + selectivity.

**Decision rule (pre-registered).** SUPPORTED iff acc(Q4) − acc(Q1) > 0 with 95% bootstrap CI excluding 0
AND Spearman rank-correlation across the four quartiles > 0. REFUTED otherwise. If refuted: the proxies are
decoration — re-derive proxy weights against a downstream signal before ANY proxy-gated scale-up (generator
thresholds, GRPO reward, corpus filtering).

**Confounds to report (not post-hoc excuses):** per-quartile model mix (GLM/Grok), family mix, chapter
length. If Q4/Q1 differ grossly in model mix, run a model-stratified secondary split before concluding.

**Cost.** Verification pass + 4 tiny pretrains; GPU-only.

**RESULT (primary, 2026-06-12 — artifacts `/raid/.../evidence/e1/`).** Q1 0.639 [0.606,0.670] →
Q2 0.671 → Q3 0.671 → Q4 **0.717 [0.685,0.746]** (N=1999 probe acc). Q4−Q1 = **+7.7 pts,
CIs non-overlapping**; Spearman ρ=0.95; same separation at N=512; selectivity rises 0.506→0.586.
**Both pre-registered criteria met.** Confound: quartile model-mix shifts (Q1 43% Grok → Q4 96% GLM)
→ the mandated GLM-only stratified replication is running (`evidence/e1_glm/`, matched ~5.35MB slices,
within-GLM R spread 0.144→0.699). Status flips to SUPPORTED iff the stratified split reproduces the
ordering; else the proxy is confounded with model identity and gets re-derived.

**RESULT (stratified, 2026-06-12 — `evidence/e1_glm/`).** GLM-only: Q1 0.686 [0.654,0.717] → Q2 0.707
→ Q3 0.701 → Q4 0.708 [0.677,0.739]. Q4−Q1 = **+2.3 pts, CIs OVERLAP**; ρ=0.80 (Q2>Q3 inversion);
selectivity 0.548→0.590. **Within-model criterion NOT met** — ~⅔ of the primary effect was the
model-mix confound (Grok chapters are both lower-scored and worse training data; GLM-only Q1 0.639→
0.686). **Verdict: PARTIAL.** The composite is validated as a coarse data-quality signal (and the
mixed-corpus result stands for whole-corpus filtering), but it cannot CI-cleanly discriminate quality
*within* a model at this n and compressed within-model R-range. **Consequence (per pre-registration):
no fine-grained proxy-gated filtering/reward until the proxy is re-derived against the canonical
ground (T_I_canonical/coverage_v1) — the re-grounded R_topic/R_axiom are the natural candidates,
re-testable with this exact harness (e1_split --model-filter + matched pretrains + probes).**

## E6 — instrument: pipeline topic-coherence trace + lexical preservation

**Claim.** The ontology is on the *causal* path from input FinePDFs to output chapters — the topic
thread runs THROUGH constructs/DDL/views, not around them via style anchors (the ablation's known
bypass: no-ontology hit 80% topic-anchor recovery at R_axiom 0.009).

**Channel A — topical transfer.** One frozen space (the canonical ground: mpnet, k=200). Each stage's
NL shadow is *assigned* (never refit): S0 input doc → S1 instantiated verbal_template → S2
view-verbalization (NEW object: constituent verbal_templates composed along the join path) → S3
chapter prose. Linkage = stored keys (generated_from_topic, template_ids, t_<id>, view→bases).
Per-stage transfer = topic-signature similarity; "non-trivial" = beats a shuffled-pairing null.
**Decision rule:** S1→S3 transfer beats the null CI-clean ⇒ ontology on-path; S0→S3 passing while
S1→S3 fails ⇒ the bypass, quantified + localized. The per-stage scores are the candidate
**re-derived within-model proxy** E1 demands.

**Channel B — lexical preservation (RH).** Fraction of DeepOnto-verbalization terms reflected in
entity NAMES — measured bidirectionally (recall = expressiveness; precision = name tokens grounded
in the verbalization, i.e. no un-grounded lexicon) against a cross-pairing null, with **table-columns
and view-columns scored separately** to verify preservation through DDL into the text-embedded view
artifacts. Current expected score ≈ 0 (slot columns are x/y; views absent) — **the zero is the
baseline that drives the implementation and then proves it.** Names must track SchemaPile
length/entropy (no term-stuffing). Consequence: semantic names re-open the header channel ⇒ the
benchmark forks into declared *with-names* and *cells-only* (hardened, headline) tracks.

**BASELINE (2026-06-12, `scripts/e6_lexical_preservation.py`).** SEED: table-name recall **0.315**
(null 0.013 — template ids are verbalization-derived), slot-columns **0.000** (x/y), views **absent
(0)**. GENERATED: **0.000 across the board — because generated constructs have EMPTY verbal_template
(DeepOnto never ran on them)**: the DeepOnto gate is not just verification, it *produces* the bridge
object E6 requires. First-run finding; drives the gate implementation.

## E2 — instrument: the edge-probe (relational/structural skill)

**Built from** the verifiable JSON (populated tables + slot-typed schema + spine FK edges):
(a) **column→SKOS code** classification, cells-only (values; anonymized ids) against the held-out reference;
(b) **FK-pair validity** — classify whether (col_A, col_B) stand in a sanctioned FK relation; positives =
spine FK edges realized in chapters; **hard negatives = type-compatible column pairs not in the FK graph**.
Frozen-backbone linear probes, Hewitt-Liang control tasks, bootstrap CIs.

**Validity rule (pre-registered).** The instrument is valid iff, on a known-good backbone (the v0.3
full-arm pretrain), probe accuracy beats its control task (selectivity > 0) with a CI-clean margin on both
(a) and (b). An invalid instrument blocks E3 — fix the instrument, don't reinterpret the task.

## E3 — the load-bearing claim: full (axioms+DDL) vs no-schema

**Claim.** Pretraining on the DDL-injected corpus yields more relational skill than the no-schema
(axioms-without-DDL) contrast.

**Design.** Generate ~300 docs/arm (full vs no-schema) on the **current typed spine**, same seeds/topics/mix.
Matched-token tiny pretrains. Evaluate with E2 (primary) + cells-CTA (secondary). Bootstrap CIs + paired
permutation on the same eval rows.

**Decision rule (pre-registered).** SUPPORTED iff full − no-schema > 0 with 95% CI excluding 0 on E2(b)
(FK validity) or E2(a) (SKOS code). If flat: the load-bearing premise fails at tiny scale — diagnose
(capacity / instrument / recipe) before any Phase 3/4 spend. **Cost.** ~$15 API + GPU.

## E4 — independent baseline + benchmark non-triviality (Atelier)

**Design.** RH runs Atelier's DST fusion (blind: `columns.parquet` + `vocabulary.parquet` from the v0.3
release); we score predictions against the held-out `reference.parquet`.

**Decision rule (pre-registered).** The benchmark is meaningful iff Atelier's macro-F1 sits well above the
trivial floors (majority-class, random) and below ceiling (≈100% would indicate leakage). The resulting
number is **the baseline Aegir must beat**; per-family breakdown becomes the targeting diagnostic.

## E5 — generator genuineness: deep gates + coverage-close + de-canning

**Gates to wire (currently shallow):** DeepOnto load+verbalize (semantic validity); target-topic alignment
(cosine of the construct's verbalization embedding vs the TARGET topic centroid, τ set from the seed
templates' own alignment distribution); novelty (max cosine vs seed ∪ already-admitted < τ_nov).

**Metrics.** Primary: **coverage-close** — re-run the coverage audit with seed+generated catalog and count
gap→covered/borderline flips. Admit-rate is diagnostic only (expected to DROP when deep gates land — that is
healthy). De-canning: per-anchor column-set entropy vs SchemaPile's (uniform stamped attributes ≈ 0 entropy).

**Decision rule (pre-registered).** A scaled generation run is justified iff a 20-gap-topic pilot with all
deep gates active closes ≥25% of its target topics to borderline-or-better AND the novelty gate rejects > 0
candidates (i.e., it is actually binding). Catalog promotion remains review-before-promote per the charter.
