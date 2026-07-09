# Personas

*This document names the people the Aegir system is built for and
the workflows each one needs to complete. It is the gate on the
project's BDD scenarios — every feature file in `features/` should
identify the persona(s) it serves, and no feature file is
considered done until its scenarios describe a workflow recognizable
to that persona.*

## Why persona-first

A scenario that exercises a system component (does the WAL line
have the right fields? does the API return 200?) is a wire test.
A scenario that exercises a *user doing something* is a workflow
test. The two are not interchangeable. Wire tests catch
regressions in plumbing; workflow tests describe the surface the
system commits to producing. The Aegir feature suite as of the
last project audit consisted almost entirely of wire tests —
useful as a regression net, silent on whether anyone could
actually use the system to accomplish anything.

This document defines four personas. Each one has a concrete
role, a specific surface they interact with, a workflow they
need to be able to complete, and a definition of "success."
Feature files derive from personas: when a scenario is written,
it should be possible to point at a persona section below and
say "this is the workflow that scenario covers."

Three of the personas are *practitioners* who iterate on the
Aegir-grounded data-governance loop in production. The fourth
is an *external reviewer* whose own published work introduced
primitives Aegir uses or extends. The first three want to *get
their answer*; the fourth wants to know whether the answer is
*defensible*. The two framings are complementary; the system
needs to serve both.

The personas are intentionally archetypal — not modeled on
specific individuals at any organization. Where individuals
exist that resemble these archetypes (and they do, in this
project's actual reviewer pool), they remain unnamed by design.

## Persona 1 — Data-governance director

**Role.** Owns the data-governance program at an enterprise.
Reports to the CDO (or equivalent). Final approval on what tags
get applied to production tables, what compliance constraints
get enforced, and what vocabulary the organization formally
commits to in its Business Glossary. Accountable upward for
governance posture; accountable downward for the catalog.

**Surface.** The Aegir UI's leaderboard, run-detail, and
coverage views; the lineup (current / scratch / archive roots —
the released kasten in *current* is what she signs off on); the
published Atlas Business Glossary; the SDG catalog
(`catalog.json`, rendered with per-template provenance —
pattern, tier, grounds_ddl, domain, source span); compliance
and provenance reports lifted from lineage.

**Daily workflow.** Reviews the coverage report from the latest
training run (what fraction of the in-scope warehouse can be
tagged with the current catalog, where are the systematic
gaps) and the topic-layer census from the latest input window
(which ontology-grounded topics the harvested corpus actually
hit, where aligned mass concentrates, what stayed unassigned).
Examines failure clusters that the steward (Persona 2)
escalates from day-to-day operations. Approves or revises
proposed catalog deltas — new templates, term additions, term
revisions; each delta arrives already disposed by the
promotion membranes, so her review is the human gate on top of
the machine gates, never a substitute for them. Verifies that
approved terms land in the Atlas Business Glossary with their
SKOS hierarchies and isA edges to the SDG classification
types. Signs off on production tag deployments.

**Definition of success.** She can articulate to her CDO, in
language the CDO understands, what the system tags well, what
it tags badly, and what is being done about the latter. She
has a defensible audit trail for every term that appears in
the published Business Glossary: who proposed it, what evidence
motivated the proposal, what catalog templates use it, and
which production tags depend on it.

**What must never happen.** A tag of unknown provenance
appearing on a production table. A template addition that
bypasses her review. A coverage gap that nobody investigates
within an iteration. Drift between the published Business
Glossary and the operational SDG catalog.

## Persona 2 — Data steward

**Role.** Day-to-day operator of the data-governance pipeline.
Works for the director. Lives in the system in the way
production-database administrators live in their databases:
this is where the real work gets done.

**Surface.** The Aegir UI's run-detail pages (CTA results,
failure cases, per-table breakdowns); the lineage trace view
(cta_failure_trace, sample_provenance workflows); the lineup's
item-note trail (Scratch → Content → Items × Topics — the
walkable item → topic → term lineage); the sampling tools
(sampling_strategy); the corpus-gap proposal tool
(corpus_gap_proposal); the SDG catalog with template-minimality
guidance.

**Daily workflow.** Receives an escalation from a downstream
consumer: *column X on table Y is tagged wrong, here's why we
think so.* Opens the run-detail page, locates the failure case
in the held-out evaluation or production run. Initiates a
**CTA failure trace**: which chapters did the model see that
involved the concepts at issue? Which catalog templates did
those chapters cite? Which input-window items aligned to those
concepts in the topic layer — with what margin, against which
competitor — each association pinned to the registry state that
adjudicated it? Identifies a probable cause — a missing
template that should distinguish the two concepts, an
undersampled corpus region that left the model without a clear
contrastive signal (the topic's aligned mass is thin, or the
mass sits unassigned), or an anchor surface that fails to
discriminate the sibling pair. Proposes a fix: a **corpus
addition** (advance the harvest window under a stated strategy
change — the aperture is the strategy's lens, and the change
rides a new `strategy_id`), a **lexicon iteration** (re-author
the term's anchor surface through the annotation membranes,
positive voice only), or a **template revision** (under
template-minimality discipline — propose composition of
existing templates before authoring new ones). Submits the
proposal to the director for approval. Once approved, runs the
iteration: `just metaflow` over the new window, fine-tune the
warm-start, evaluate against the held-out set, compare
congruence and the topic-layer census to the prior run's.
Reports back: did the fix close the failure?

**Definition of success.** They can take any failure case and
produce a defensible explanation plus a defensible fix
proposal, with the artifacts (lineage trace, sampling rules,
proposed catalog delta) attached and the proposal traceable
through the director's review back to a production tag change.
They can defend the template-minimality discipline: every new
template they propose is accompanied by an argument for why the
existing catalog could not express the distinction.

**What must never happen.** A failure case with no traceable
provenance (the lineage substrate exists to prevent this). A
retrain that doesn't move the needle and the steward can't
tell why. Catalog or corpus changes that drift outside the
locked-artifacts discipline. A fix that closes the originating
failure but introduces a worse one elsewhere with no early
warning.

## Persona 3 — Embedded ML engineer

**Role.** Works at a downstream consumer of Aegir. Owns the
production pipeline that integrates Aegir's output —
CTA/CPA tags, the SAE feature dictionary, the trained
checkpoint — into the consumer's broader system (DST evidence
fusion, enterprise classification pipelines, governance
tooling). Their stability is downstream of Aegir's stability;
their reproducibility is downstream of Aegir's reproducibility.

**Surface.** The Aegir gateway API (`/api/leaderboard`,
run-detail endpoints, plot endpoints); the locked-artifacts
table (catalog version, weights hash, null-statistics hash —
pipeline runs additionally pin the Merkle `strategy_id` and
the topic-registry collection sha); the SAE feature dictionary
when it is stable enough to cite; run-metadata sidecars (what
catalog + weights produced this checkpoint); lineage events
emitted to Atlas (canonical) and Marquez (OL push for
compatibility).

**Iteration workflow.** Pulls a new Aegir checkpoint into
their integration test harness. Checks the locked-artifacts
hashes against their pinned baseline. If hashes have changed,
runs their downstream tests to see if behavior changed. If new
SAE features have been surfaced, evaluates whether they are
stable enough across runs to cite from the consumer's own
code. Reports back to the Aegir team when a regression is
attributable to an Aegir change: *your run X shifted SAE
feature 4087's mean activation on banking-PII columns by 35%,
which broke our downstream classifier's confidence
calibration; here is the diff.*

**Definition of success.** They can pin to a known Aegir state
identified by the four-hash quadruple
(catalog_version, locked_weights_hash, null_stats_hash,
run_id), detect when that state has changed, and reproduce any
cited result from the hashes alone — without coordinating with
the Aegir team. The gateway responses they depend on are
versioned; the SAE features they cite are stable across
checkpoint refresh.

**What must never happen.** Aegir making a "small change" that
silently shifts SAE feature semantics. Locked hashes that fail
to capture some relevant aspect of state (catalog drift behind
a stable hash). Gateway responses that change shape without
either a major-version bump or a backwards-compatibility
shim.

## Persona 4 — Peer methods reviewer

**Role.** External researcher whose own published work
introduced primitives that Aegir uses or extends — sequence
modeling architectures (RWKV variants, H-Net dynamic chunking,
xLSTM, related families), reinforcement-learning methodology
(GRPO and successors, the broader RLVR program), sparse-
autoencoder interpretability methods, or adjacent methodology.
Reviewing the project as a collaborator-critic, not as a
gatekeeper. The label "peer" is deliberate: it positions the
project at the same methodological table as the reviewer, with
the implicit invitation to give the feedback that would be
given to a peer's work — not the deference owed to a senior
authority's work.

**Surface.** The [Ontology Authors Guide](./ontology/authors_guide.md)
(the canonical reference for every ontology metric, gate, and
membrane); the authoritative reference in
[production_state.md](./ontology/production_state.md) and the
concept brief in [concept_brief.md](./ontology/concept_brief.md)
for the long-horizon RLVR sub-track (the Signals M4 apparatus);
the realized ontology artifact
(`corpora/ontology/sdg-ontology.{omn,owl}`) and its
`HERMIT_CERTIFICATE.md`; `EVIDENCE.md` (the pre-registered claims
ledger); the dated phase-gate records under `roadmap/` (the
governance/DDL spine; the inverted topic layer, with its
pre-registered τ\*); the locked C1 test set, held-out 50, and
null-statistics snapshot; the verification gates (catalog schema
check, the OQuaRE publish gate, the parse / HermiT / OntoClean
disposal membranes, C1 AUC regeneration, verifier determinism,
end-to-end scaffold); the repository itself (clone and re-run).

**Review workflow.** Reads the Authors Guide and the authoritative
reference. Identifies the load-bearing claims. On the **ontology
rigor program** (the primary, shipped track): that the realized SDG
ontology clears the pre-registered objectives in `EVIDENCE.md` —
**OQ-Rigor** (`definitional_completeness ≥ 0.45` ∧
`realizable_machinery > 0`) and **OQ-Structure** (`bfo_grounded ≥
0.95` ∧ `def_annotation_coverage ≥ 0.90` ∧ `ar > 0` ∧
`oquare_aggregate ≥ 3.5`), with zero unsatisfiable classes under
HermiT; and that the disposal membranes (parse → HermiT/CCO →
OntoClean) are un-fakeable. On the **long-horizon RLVR sub-track**
(the Signals M4 apparatus): verifier discrimination on C1 (AUC
0.9956, mean *R*-separation 0.336), held-out 50 separation
(0.5129), and the policy claim that GRPO can produce compositions
whose *R*-distribution exceeds prompt-evolved and human-authored
baselines. Picks one claim and tries to reproduce it from the
repository alone, with no email to the authors. If reproduction
works, asks the second-order questions. For the rigor program:
**Do the metrics regenerate** — does
`scripts/ontology_metrology.py` on the realized `.owl` produce
the reported numbers, and does `scripts/ontology_oquare.py` return
GREEN against the certificate? **Are the gates actually un-fakeable**
— does an injected contradiction or anti-rigid-over-rigid
subsumption get rejected with a reason? **Do the pre-registrations
bind** — does the topic layer's τ\* regenerate from the
shuffled-window null at the registered seed, with the method
registered before the number (report-not-tune)? For the RLVR
sub-track:
**Are baselines visible** — what does the *R*-distribution look like
under a prompt-evolved policy? under a random-sampling policy? under
no constraint at all? **Is each verifier component validated
independently** — could *R_B* be removed without losing meaningful
discrimination? what does *R_D* alone discriminate? **Is the
experimental design honestly bounded** — are the claims stated at
the confidence level the data supports? **Does the warm-start choice
survive comparison** — is Option A's rejection-sampling + SFT
compared head-to-head against Option B's Instruct-paired baseline,
or against Option C's on-policy SDFT alternative, under matched
conditions?

**Definition of success.** They can reproduce any claim from
the repository alone. They can swap one verifier component
out and re-run the C1 sweep without rebuilding the catalog.
The project's claims are bounded honestly — no overreach
beyond what the experiments support — and any limitation
they raise is either addressed in a follow-up or named in
§ 8 (Limitations) of the authoritative reference. They
leave a collaborative critique that the team can act on,
not a binary thumbs-up/thumbs-down.

**What must never happen.** A claim with no traceable
experiment behind it. An ablation that "would" work but
hasn't been run, presented as if it had. Baselines that
aren't visible in the doc or the repo. Reproducibility that
requires emailing the authors. The team treating the review
as a gatekeeping audit rather than collaborative critique
(which is on the project to invite, not on the reviewer to
volunteer).

## Workflow features → personas

The seven user-facing workflow features and the three
reviewer-facing methodology features, mapped to the personas
they primarily serve. Where a feature serves multiple personas,
the **primary** owner is bolded; secondary owners cite the
feature when their workflow touches it.

| Feature | Director | Steward | ML eng. | Reviewer |
|---|---|---|---|---|
| `cta_failure_trace` | reads clusters | **primary** | regression check | provenance audit |
| `vocabulary_coverage` | **primary** | informs proposals | pins consumer scope | claim audit |
| `vocabulary_subsumption` | **primary** | proposes terms | — | structural-claim audit |
| `sample_provenance` | — | **primary** | citation pins | reproducibility audit |
| `sampling_strategy` | — | **primary** | — | reproducibility audit |
| `sae_vocabulary_alignment` | term-promotion | — | **primary** | interpretability audit |
| `corpus_gap_proposal` | approves | **primary** | — | proposal-logic audit |
| `reproducibility` | — | — | regression baseline | **primary** |
| `ablation_surface` | — | — | — | **primary** |
| `baseline_comparison` | reads summary | — | — | **primary** |

A feature that finds no row in the persona column it claims to
serve is misplaced; either the persona is wrong or the feature
is actually a wire test in disguise.

## Conventions for feature files

When writing a `.feature` file under `features/governance/`
(or any sibling cluster derived from these personas), the
Background section should reference the relevant persona by
role rather than by the formal label:

```gherkin
Feature: CTA failure trace
  As a data steward investigating a misclassification on a
  production table, I need to follow the lineage chain from
  the failing column back through compositions, catalog
  templates, and input-pool documents — so that I can
  produce a defensible explanation and a fix proposal in
  the same iteration.
```

Plainer than `As the Data Steward (Persona 2)`. The formal
label belongs in this document; the feature files reference
the workflow.

For reviewer-facing features, the same convention applies:

```gherkin
Feature: C1 sweep reproducibility
  As an external methods reviewer auditing the verifier's
  discrimination claim, I need to regenerate the C1 sweep
  AUC from the committed test set and locked weights, on a
  fresh clone of the repository, with no coordination with
  the project team.
```

## Cross-references

- [Ontology Authors Guide](./ontology/authors_guide.md) — the
  canonical reference for the ontology's metric suite, the OQuaRE
  publish gate, and the disposal membranes; the primary surface the
  reviewer persona audits against.
- [Aegir's semantic engine — authoritative reference](./ontology/production_state.md)
  — the external/advisory-facing description that the
  reviewer-facing scenarios audit against, and that the
  practitioner scenarios produce evidence for.
- [Concept brief — RLVR for ontology generation](./ontology/concept_brief.md)
  — the research design for the long-horizon RLVR sub-track that locks
  the experimental claims the reviewer persona audits.
- [Ontology Charter](./ontology/charter.md) — the outward
  contract that the director persona owns operationally.

---

*This document gates the feature files in `features/governance/`
and any subsequent feature cluster that claims to test a user
workflow. Adding a persona, retiring one, or revising a
persona's definition of success is a load-bearing change and
should be reviewed alongside the affected feature files.*
