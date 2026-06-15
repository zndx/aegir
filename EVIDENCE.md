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
| E2 | *(instrument)* the edge-probe validly measures relational/structural skill | E3, the v0.4 headline eval | **BUILT; VALID-BUT-NOT-DISCRIMINATING** — selectivity>0 CI-clean on known-good backbone, random floor=0; but E2(b) is byte-value-overlap (arm-invariant) → does not isolate schema skill |
| E3 | The DDL-injected (load-bearing) corpus beats no-schema on relational skill | ~~Phase 3 / v0.4~~ — DESCOPED | **DESCOPED from the gate (2026-06-15)** — not demonstrable at 13.5M (frozen=byte-overlap; fine-tune=capacity floor). Corpus-as-deliverable instead. Harness retained; revisit at larger capacity |
| E4 | The blind column benchmark is non-trivial; an independent (Atelier) baseline exists | any "Aegir lift" claim | **UNTESTED** |
| E5 | The generator produces semantically genuine, novel, coverage-closing ontology | scaled generation, catalog promotion | **RED** — deep gates wired; 20-gap pilot closed 0% (≪25%); novelty binds. Coverage-close metric is register-confounded → reformulate before scale |

## Next-phase gate (pre-registered 2026-06-15; RE-SCOPED 2026-06-15 → corpus-as-deliverable)

**History:** originally DUAL (G-rel + G-cov). After two convergent diagnostics showed the load-bearing
relational claim is NOT demonstrable at 13.5M (frozen E2 = byte-overlap arm-invariant; fine-tuning E3 =
capacity floor — see E2/E3 §§), RH RE-SCOPED: **the ontology/DDL value is the CORPUS, demonstrated where it
is measurable; the tiny-model relational claim is shelved (revisit at larger capacity — the E3 harness is
built and ready).** Before v0.4, BOTH:

- **G-cov (register-fair coverage-close) [PRIMARY]:** the reformulated metric R1 (domain-term alignment;
  §1 of the sketch) clears a PASS BAR — NOT mere non-regression. R1 must FIRST be validated as an instrument
  (on-topic construct scores high, off-topic low, CI-clean) before it can gate; the bar is set from the
  seed catalog's own R1 distribution to its best-covered topics. Anti-gaming via structural gates + de-canning.
- **Corpus-quality:** no-collapse (SUPPORTED — collapse-trend flat), SchemaPile realism (SUPPORTED — width
  2.75→8.45), attribution-clean held-out reference (leak-free), reasoning traces captured. Re-confirm on the
  v0.4 corpus.

- **G-rel — DESCOPED from the gate (2026-06-15).** Not claimed at tiny scale. The fine-tuning-delta harness
  (train.py + e3_finetune_delta.py) is retained; re-run it once a larger backbone exists. E2 remains a
  frozen-rep diagnostic.

Preconditions ($0, run first): **R0 debloat** (embed verbalization-only, not formal Manchester+metadata —
tests the register hypothesis) and **referential-integrity check** (do FK cell values overlap referenced
PKs? — E2(b)-cells-only hinges on it). Full design: `docs/scratch/2026-06-15/185155_phase_gate_*.md`.

**PRECONDITION RESULTS (2026-06-15, `scripts/check_register_and_ri.py`).** Both PASS:
- **R0 register hypothesis CONFIRMED** — verbalization-only embedding lifts per-topic max template↔centroid
  sim +0.023 and recovers **borderline 37 → 55 (+18 topics)** vs the full Manchester+metadata text (0
  covered either way). Register was a real confound in E5's coverage-close → the register-fair direction is
  validated. (Not the whole story: gaps remain, so R1 term-level + generator domain-grounding still matter.)
- **Referential integrity STRONG** — 386 multi-table chapters, 796 FK-shaped columns, value-overlap
  mean **0.996**, median 1.000, 784/796 exactly 1.0. **E2(b)-cells-only is viable** (no generator fix needed).

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

**RESULT (2026-06-15, `scripts/eval_edge_probe.py`; eval on held-out v0.3 JSON tables, cells-only,
chapter-split, 3 seeds, Hewitt-Liang control + bootstrap CIs).** Known-good backbone = `ablation_v1/full`
(the cells-CTA 0.66–0.70 one). **Passes the letter of the rule:** E2(a) role/SKOS acc 0.744 vs ctrl 0.361,
sel **0.383** (min 0.254); E2(b) FK-validity acc 0.985 vs ctrl 0.507, sel **0.478** (min 0.437), PR-AUC
0.988. **Learning-sensitive:** RANDOM-INIT floor = **selectivity 0.000 both heads** (E2(b) PR-AUC 0.266 ≈
base rate) — the signal is learned by pretraining, not present at init. **BUT NOT DISCRIMINATING (the
caveat that matters):** all three pretrained arms score identically (see E3), so E2(b) measures generic
**byte-value-overlap** (`req_*`↔`req_*` vocabulary similarity that any byte-LM learns), not schema-specific
relational skill. Cell-level FK validity *is* value-overlap, so a pooled-rep probe can't isolate the DDL
contribution. **Instrument needs hardening** to target schema-specific skill (e.g. E2(a) on the TYPED
spine, where value→type inference is non-trivial and schema injection could plausibly help) before it can
gate G-rel. Artifacts: `evidence/e2/edge_probe{,_random,_no_schema,_no_ontology}.json`.

## E3 — the load-bearing claim: full (axioms+DDL) vs no-schema

**Claim.** Pretraining on the DDL-injected corpus yields more relational skill than the no-schema
(axioms-without-DDL) contrast.

**Design.** Generate ~300 docs/arm (full vs no-schema) on the **current typed spine**, same seeds/topics/mix.
Matched-token tiny pretrains. Evaluate with E2 (primary) + cells-CTA (secondary). Bootstrap CIs + paired
permutation on the same eval rows.

**Decision rule (pre-registered).** SUPPORTED iff full − no-schema > 0 with 95% CI excluding 0 on E2(b)
(FK validity) or E2(a) (SKOS code). If flat: the load-bearing premise fails at tiny scale — diagnose
(capacity / instrument / recipe) before any Phase 3/4 spend. **Cost.** ~$15 API + GPU.

**RESULT — fine-tuning delta (2026-06-15, `scripts/e3_finetune_delta.py` + train.py harness).** Per RH's
redefinition, fine-tuned the `full` arm end-to-end on SOTAB-DBpedia CPA (single-label CE). **It FLOORS at
tiny scale:** val macro-F1 **≈ 0.002** at both (1024 train / cap 1500) and (2048 train / cap 4000 / 12.7k
available); val loss RISES from epoch 1; **train micro-F1 caps ~0.13** — the 13.5M byte model can't fit
even the train relations well (capacity/representation wall, not mere overfit). Since the BEST-case arm
(full) floors, no_schema floors too → **the delta is uninformative; the sweep was not run** (a floored
task can't discriminate arms). **Convergent verdict with the frozen probe: G-rel (load-bearing relational
skill) is NOT demonstrable at 13.5M** — neither frozen (byte-overlap, arm-invariant) nor fine-tuned
(capacity floor on a real relational benchmark). The schema contribution, if real, needs larger capacity,
an easier/in-distribution learnable relational task, or it lives in the corpus itself (not tiny-model
skill). **Decision pending (RH): re-scope G-rel / scale up / easier task.** Harness committed (9139b58);
artifacts `/tmp/e3_{smoke,diag}_full.json`.

**RESULT — first pass on EXISTING arms (2026-06-15, $0, no new generation).** The 2026-06-05 ablation_v1
checkpoints (full / no_schema / no_ontology, matched-token tiny pretrains) already exist, so E2 was run on
all three over held-out v0.3 tables. **FLAT — no separation:** E2(b) FK PR-AUC 0.988 / 0.980 / 0.987;
selectivity 0.478 / 0.463 / 0.486; E2(a) sel 0.383 / 0.375 / 0.396 — **no_ontology (the weakest arm) ties
or leads**, i.e. pure noise. **Load-bearing claim NOT supported on this axis.** Per the pre-registered
decision rule, diagnose before spend: **(1) instrument** — E2's probes measure byte-value-overlap / value
morphology, learned equally by all arms (the dominant cause); **(2) corpus** — these arms predate the typed
spine (thin id/label/fk schema), so they don't test the current thesis; **(3) capacity** — 13.5M may floor
the schema signal. ⇒ G-rel is NOT green. The real test needs a HARDER instrument (typed-spine E2(a) where
value→type is non-trivial) AND typed-spine ablation arms; do NOT spend on Phase 3/4 / v0.4 on this evidence.

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

**RESULT (2026-06-14, `scripts/generate_ontology.py` deep gates + `coverage_v1/043d7dcc185245c8`).**
All three deep gates wired: DeepOnto verbalize (semantic) → target-topic (cosine of the construct's
`template_embedding_text` vs the audit's PERSISTED true centroid ≥ τ_low 0.35 — the audit's own
borderline floor; centroids now saved by `ontology_coverage_audit.py`) → novelty (max cosine vs seed ∪
admitted < 0.93). Pilot on the 20 worst-covered gaps (GLM): **0/91 admitted; coverage-close = 0/20 =
0% (≪ 25% — FAILS).** Novelty gate binds (17 rejects > 0 — the half that PASSES). DeepOnto rejected 20,
target-topic 62, structural 9. **Verdict: scaled generation NOT justified; E5 stays RED.**

**Two findings (the pilot's value).** (1) *Generator:* on hard gaps GLM emits GENERIC constructs
(`article_has_title`, `author_of_article`, generic processes) that DUPLICATE seed structure (hence the
17 novelty rejects) and don't capture the gap's domain. (2) *Metric/register limit (deeper):* the
single best construct anywhere reached **0.239** vs the 0.35 floor; even domain-attempts
(`malaria_research_activity` 0.239) fall short. Cause: `template_embedding_text` (formal Manchester +
"family:"/"slots:" metadata) and a natural-prose document centroid are different REGISTERS, so
template↔topic cosine is capped low for prose-domain topics. The 37 borderline-reachable topics are the
register-matched formal/governance ones; the gaps are mostly prose domains (herbal medicine, nursing,
education handbooks). **Implication: many coverage-audit "gaps" are register artifacts, not genuine
ontology-coverage holes — so coverage-close via more templates is the WRONG generator objective for
those gaps** (coheres with the plan-of-record: the ontology trades FinePDFs distribution alignment).
Before any generator scale-up, the coverage-close metric needs a register-fair reformulation (e.g.,
align the construct's INSTANTIATED prose, not its formal text; or target structural/CPA coverage, not
topic-distribution coverage). Artifacts: `evidence/e5/pilot.candidate.json`, `evidence/e5/pilot output`.

**Gate hardening (2026-06-15).** `gate_deeponto` now also requires a **non-trivial** construct — an
asserted COMPLEX class (`onto.get_asserted_complex_classes()` non-empty: restriction / intersection /
cardinality), not a bare atomic subsumption. A trivial `X SubClassOf: cco:Artifact` verbalizes ("X is an
artifact") but carries no relational structure; now rejected, so generated constructs are genuinely
non-trivial AND DeepOnto-verbalized (Manchester required to verbalize). Verified: atomic
(`subclass_to_artifact`, `subclass_basic`) → REJECT; restriction (`artifact_with_existential`) → PASS.
Does not change the E5 verdict (coverage-close/target-topic was the binding constraint, downstream of this).
