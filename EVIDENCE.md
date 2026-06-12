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
| E1 | The chapter-verifier composite ranks chapters by downstream pretraining value | proxy-gated corpus filtering, generator admit thresholds, GRPO reward, E5 scale-up | **UNTESTED** |
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
