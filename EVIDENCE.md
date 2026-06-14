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
| E1 | The chapter-verifier composite ranks chapters by downstream pretraining value | proxy-gated corpus filtering, generator admit thresholds, GRPO reward, E5 scale-up | **PARTIAL** — valid cross-model / whole-corpus; within-model gating set aside (S1→S3 proxy refuted: no spread; composite spread doesn't translate). Lever = model-selection + whole-corpus filter |
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

**RE-DERIVATION — $0 proxy screen (2026-06-14, `scripts/e1_proxy_screen.py`).** E6-A produced a fresh
within-model proxy candidate (the S1→S3 transfer). Before re-pretraining, screened its within-model
DISCRIMINATIVE SPREAD against the composite (a ranker needs spread to define meaningful quartiles).
**S1→S3 is nearly constant within a model** — GLM cv=**0.033** (p90-p10=0.055), Grok cv=0.034 — vs the
composite's cv=0.50. This is intrinsic, not a bug: S1→S3 measures ontology↔prose correspondence and
*every* full-arm chapter was generated from the ontology, so all score high+similar. **S1→S3 is an
on-path detector (E6-A), NOT a within-model quality ranker — REFUTED for that role.** Re-splitting
quartiles by it would yield near-identical chapters; the screen avoids a foregone-null GPU run.
Meanwhile the composite HAS within-model spread (cv 0.50) yet E1 already showed that spread does not
translate to downstream value within a model. **Net: no available per-chapter proxy CI-cleanly ranks
within-model pretraining value; the validated, actionable levers are MODEL SELECTION (prefer GLM —
S1→S3 0.962 vs Grok 0.927, and Grok leans on input not ontology, cohering with E1) and WHOLE-CORPUS
composite filtering. E1 stays PARTIAL; within-model fine-grained gating is set aside (not merely
deferred) unless a future proxy clears this spread+translation bar.** Artifact: `evidence/e6a/per_chapter.json`.

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

**RESULT — Channel A (2026-06-14, `scripts/e6a_trace.py`; n=2205 of 2235 v0.3 chapters; S2 not
built — views materially absent in v0.3).** Signature = cosine to the 200 canonical centroids;
transfer = cosine of signatures; null = shuffled-pairing (genre-matched), 25 perms; bootstrap CIs.
**S1→S3 beats the null CI-clean in EVERY arm** — ALL Δ=+0.023 [+0.022,+0.024]; GLM Δ=+0.023; Grok
Δ=+0.024 [+0.022,+0.026]. **No bypass:** S1→S3 (obs 0.954) far exceeds S0→S3 (obs 0.873) — the
ontology shadow predicts the prose *more* than the input doc does (the inverse of the pre-registered
risk). S0→S1 Δ=+0.023 and S0→S3 Δ=+0.016 also CI-clean. **Verdict: on-path correspondence SUPPORTED,
pure-bypass REFUTED.** Effect sizes are small in absolute terms (homogeneous corpus → nulls 0.82–0.93)
but the genre-matched null isolates Δ as the topic-specific component, and CIs are tight at n=2205.
**Model diagnostic:** GLM tracks the ontology (S1→S3 0.962 ≫ S0→S3 0.863); Grok leans on the input
(S0→S3 0.904, highest arm) — coheres with E1 (Grok lower-scored, worse training data). *Scope:* this
is correlational evidence the ontology content is on-path beyond chance/genre; the strict load-bearing
/ causal test remains E3 (ablation). The S1→S3 per-chapter transfer is now a concrete within-model
proxy candidate for the E1 re-derivation. Artifact: `evidence/e6a/trace.json`.

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

**SECOND FINDING (same day, gate wired).** With the DeepOnto gate live (9/10 generated constructs
verbalize; 1 honest reject), E6-B exposes that the produced verbalizations are **semantically
vacuous** — "{X} is something that {hasParticipant} {Y}" (content terms ≈ {"something"}; 35-61
chars). Root cause: the verbalizer's axiom-shape selection renders the restriction conjunct and
drops the named BFO/CCO anchor conjunct (auto-declared labels exist but the anchor entity is
skipped). Affects R_C, R_D, the chapter prompt, and both E6 channels from one root. **Fix target:**
`deeponto_harness._verbalize_for_template` shape ranking / anchor-conjunct retention; success =
generated verbalizations carry the anchor noun ("is a process that…") and E6-B generated
table-name recall moves off 0.

**RESOLVED (2026-06-14, `deeponto_harness.py`).** Anchor restoration: the named superclass conjunct
is explicitly asserted in the axiom and has a resolvable label, so `_named_superclass_label` +
`_restore_anchor_noun` replace DeepOnto's vacuous implicit subject ("is something that …") with the
asserted anchor noun ("is an artifact that …") — faithful, not manufactured. Validated through the
real `gate_deeponto` path on an 11-construct generation ($0.05, coverage_v1 gaps): every generated
verbalization now carries the anchor noun, and **E6-B generated table-name recall 0.000 → 0.470**
(null 0.045; SEED 0.315). Two further defects the re-measure surfaced and fixed in the same pass:
(i) **article agreement** ("is a artifact" → "is an artifact", subsumption-fallback path);
(ii) **de-camelCased marker leak** — DeepOnto renders a property marker `ZZisEditionOfZZ` as
`ZZis edition ofZZ`, which the old `ZZ(\w+?)ZZ` regex could not map back; `MARKER_LOOSE_RE` +
`_norm_marker` + slot-aware `_markers_to_slots` recover the canonical slot name (raw-ZZ leaks 3→0).
Also: **cold-start warmup** in `ensure_jvm` (the first probe after JVM start intermittently returned
empty because the verbaliser's NLP stack loads mid-call — would spuriously fail the gate's first
construct). **Remaining E6-B floor:** slot/table-*column* recall is 0.000 for seed AND generated
(column names are bare slot identifiers, not verbalization content) — the next lexical-preservation
target (semantic column naming + views), tracked separately from this fix.

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
