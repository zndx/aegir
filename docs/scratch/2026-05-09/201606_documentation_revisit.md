# Documentation revisit — scope expansion + v2 baseline integration

*2026-05-09 — first session back after the Apr 27 reboot.*

## Why this session existed

Two things changed since the docs were last touched:

1. **v2 mixed-corpus pretrain landed (2026-04-27).** 122k steps, 2 GB
   mixed corpus, single GPU, ~10 h. Stratified held-out eval shows
   non-degenerate representations and ~2 bpb drops on domain-targeted
   slices, with general prose held flat. This is the project's first
   real backbone — the empirical anchor that subsequent work hangs off.
2. **Ontology + synth ownership consolidates here.** The sibling
   project (Atelier) has formally surrendered ownership of
   `atelier-vocab.ttl` and the ~316-generator synth library; Ægir
   becomes the canonical home, with Atelier moving to consumer-of-
   trained-artifacts status. The sibling's documentation describes its
   own consumption needs but is treated here as input, not
   specification.

Neither change had been reflected in the published mdbook.

## What was stale

- `docs/current/roadmap.md` was entirely a far-future K2.5 RL plan. Reader
  saw no path from "today" to "RL post-training."
- `docs/current/roadmap/supervised.md` predated the Apr 19 representation
  collapse; it treated from-random direct-CTA as Phase 1.
- `docs/current/training_regime.md` ended at §9's Apr 20 verdicts on
  Stage A/B. Stage B was the single-slice GitTables probe; the v2
  9-slice mixed-corpus pretrain that followed was not represented.
- `docs/current/pretraining.md` reads as a description of current state but
  describes an aspirational ontology→synth→table pipeline that doesn't
  exist yet. v2 used real corpora.
- `docs/current/introduction.md`'s "Where we are" callout still framed the
  SOTAB collapse as the open wound; v2 has since converged.
- README's M2 line ("ontology editor with Postgres write paths,
  per-class F1 bars") didn't acknowledge the ontology + synth migration
  scope absorbed from Atelier.
- No ontology section anywhere in mdbook. No `src/aegir/ontology/` or
  `src/aegir/synth/` directories.

## What changed in this pass

| File | Change |
|---|---|
| `docs/current/ontology.md` | NEW — section landing page, scope summary table |
| `docs/current/ontology/charter.md` | NEW — outward contract (`vocab_label_map.json` + checkpoints), design constraints, empirical gate (≥ 3 MCL clusters / ≥ 0.10 macro F1 / ≥ 10 distinct predicted labels), source tree layout, CI obligations, semver rules |
| `docs/current/ontology/migration.md` | NEW — phased migration plan, named cutover criterion (v1.0.0 release tag = sibling-project freezes its TTL) |
| `docs/current/SUMMARY.md` | Added ontology section to navigation |
| `docs/current/roadmap.md` | Rewritten as two-track roadmap: near-term M0–M4 milestones + far-future K2.5 RL post-training. v2 baseline named as M2 anchor; M2 scope expanded to include ontology + synth migration |
| `docs/current/roadmap/supervised.md` | Reframed: Phase 1 = fine-tune from v2 checkpoint, not from random. Liveness gate documented. Model-size table updated to verified params |
| `docs/current/training_regime.md` | §1–§9 preserved verbatim (historical narrative). New §10 (v2 mixed-corpus pretrain), §11 (v2→SOTAB head fine-tune gate), §12 (v3 implications now grounded), §13 (further reading) |
| `docs/current/pretraining.md` | Status callout at the top making the "current vs aspirational" distinction loud. Body unchanged — it is the long-term destination, not a description of what produced `final.pt` |
| `docs/current/introduction.md` | "Where we are" callout updated: SOTAB collapse is no longer the open wound; v2 healthy backbone is the current state; M2 head fine-tune is the next gate |
| `README.md` | Bumped to v0.3.0; Training row mentions v2 pretrain headline; M2 expanded with ontology migration and liveness gate; M3/M4 split out (vocab expansion + multi-GPU vs. competitive F1) |

## What deliberately did not change

- Stages 1–4 of `pretraining.md` (ontology extraction, schema
  projection, synthetic generation, training objective) — they are the
  destination, and no v3 work has displaced their description.
- Agent swarm, architecture, RWKV-7/8, ROSA, block-types pages — all
  current and accurate.
- The K2.5 RL roadmap subpages — moved under "far-future" framing in
  the parent but kept verbatim. They are aspirational but not stale.
- The Apr 19 diagnostic case study — historical document, still
  correct, motivates the v2 pretrain that followed.

## What did not happen, on purpose

- **No code changes.** The ontology / synth scaffolding directories
  named in the charter (`src/aegir/ontology/`, `src/aegir/synth/`,
  `scripts/build_vocab_label_map.py`) do not yet exist. Phase 0 of the
  migration is creating them; that's the next session's work.
- **No vocabulary edits.** Atelier's `atelier-vocab.ttl` is unchanged
  in its own repo; nothing has been copied into Ægir yet. The Phase 1
  copy-not-move step happens after the scaffolding is in.
- **No tier-A vocabulary expansion plan adopted from Atelier.** The
  charter explicitly notes that the externally-suggested tier
  breakdown is directional input, not constraint. Ægir scopes its own
  vocab expansion against its own empirical priorities, post-M2 gate.
- **No M2 timeline committed.** The empirical gate is real; if v2 →
  SOTAB head fine-tune surfaces architectural issues, M3 waits. We
  commit to order, not to dates.

## Next actions (M2 path)

1. Create `src/aegir/ontology/` and `src/aegir/synth/` package
   skeletons. Phase 0 of the [migration](../../src/ontology/migration.md).
2. Copy `atelier-vocab.ttl` into the new home; author SPARQL totality
   query; build deterministic `vocab_label_map.json` script.
3. Fix `_LABEL_DIMS["sotab"] = 91 → 82` in
   `src/aegir/data/table_dataset.py`. Add `label_to_iri()` resolver.
4. Run v2 → SOTAB head fine-tune from
   `outputs/mixed-v2/20260426T232240Z/final.pt`. Measure against the
   liveness gate.
5. If green: tag `vocab_label_map.json` v1.0.0; document the cutover
   in a session note; M3 vocabulary expansion begins on Ægir's
   schedule.
6. If red: stop vocabulary work; debug the architecture-supervised
   interaction. The collapse mode is well-understood; the v2 backbone
   is healthy in the unsupervised regime; the diagnostic toolkit
   (MCL geometry audit, prediction distribution, embedding pairwise
   distance) is already in place.

## Verification

`mdbook build docs/` ran clean. Ontology section is wired into the
generated HTML; cross-links from `roadmap.md`, `supervised.md`,
`training_regime.md` resolve to the new pages. No broken refs.

## Provenance discipline (added late session)

The original migration draft proposed a wholesale TTL copy at the
import boundary. That framing was replaced before any code changed.
The migration plan now describes
`src/aegir/ontology/aegir-vocab.ttl` as **authored fresh**: BFO/CCO
upper structure, the Schema.org and DBpedia terms that map to the
benchmark label sets we train against, and bespoke
`aegir:`-namespace entities the project decides to introduce on its
own terms. External TTL working sets are reference material the
author may consult, not source-of-truth inputs.

Discipline lives in PR review and in the structural shape of the
TTL itself. The mechanical CI checks
([Charter §Mechanical checks](../../src/ontology/charter.md#mechanical-checks))
cover structural integrity — TTL parse, label/definition presence,
BFO ancestry for `aegir:` terms, label-map consistency, SPARQL
totality. Whether a candidate term reads as the project's own
engineering work or as material lifted from elsewhere is an
authorship judgment, made in code review the way every other
authorship judgment is.

A scripted "verify provenance" gate was considered and rejected: any
positive-grounding script would either block legitimate bespoke
novelty (since by construction novel `aegir:` terms aren't in any
public reference set) or rubber-stamp around its own checks.
