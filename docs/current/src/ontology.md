# Ontology

Aegir is the canonical owner of the bespoke BFO 2020 / CCO-grounded
ontology used by the metadata-tagging stack — the **Signals Data
Governance (SDG)** ontology — and of the ontology-grounded
synthetic-data pipeline that produces in-distribution pretraining
bytes against it. The ontology, its rigor program, the realized OWL
artifact published outward, and the topic layer that measures the
corpus against it all live in this chapter. The chapter covers what
the ontology *is* now, how its classes function as the annotation
vocabulary, the derive → promote → realize → publish chain that grows
and ships it, the quantitative rigor metrics and the formal publish
gate every extension must clear, and the disposal membranes that
enforce rigor rather than assert it.

The ontology conditions everything downstream — it is the
**annotation vocabulary for Column Type / Column Property Annotation
(CTA/CPA)** over wide relational tables. Its classes are not leaf
terms but **intermediate-depth subsumers**: the property-bearing
classes a heterogeneous-but-coherent column belongs to. Defining
those classes well *is* building the annotation vocabulary, and the
gates exist to keep every term a coherent, grounded annotation target.
Putting the ontology next to the model is the only arrangement where
these decisions stay coherent.

## What the ontology is now

The ontology is **content-first and fully derived**. Hand-authored
axiom families were an early scaffold and are retired; live axioms
are derived from FinePDFs passages and admitted through membranes
(parse → HermiT → OntoClean), with provenance (`pattern` / `tier` /
`grounds_ddl` / `domain` / `source_span`) that downstream stages
consume. Discovery is via `schema.catalog_files()`, never globs.
Axiom patterns come from the library in
`src/aegir/ontology/patterns.py` (`fhir_minimum` / `owl2_core` /
`odp` / `sysmlv2`).

The published artifact is **HermiT-certified OWL** at
`corpora/ontology/sdg-ontology.{omn,owl}` (`HERMIT_CERTIFICATE.md`).
SKOS is an entailed annotation face of that OWL (every concept's
defining class exists in the ontology). SHACL is the closed-world
constraint view (`shapes.ttl`): `sh:targetClass` names a realized
class; restrictions become cardinality and filler constraints.
Loadable SQL is a **relational projection** of the same certified
OWL, not a parallel schema. `scripts/check_triad_entailment.py`
measures **OWL ⊨ SKOS ⊨ SHACL** mechanically. Snapshot sizes
(classes, concepts, tables) belong to a given commit, not to the
method.

## The derive → promote → realize → publish chain

```d2
direction: down

passages: FinePDFs passages\n(aperture-filtered, content-hashed)
derive: DERIVE\nscripts/derive_ontology.py\npattern-bound primitives
candidate: catalog.candidate.json\n(staging)
promote: PROMOTE\nscripts/promote_candidates.py\nparse → HermiT → provenance threading
catalog: live ontology\n(membrane-admitted axioms)
realize: REALIZE\nscripts/build_realized_ontology.py\nBFO 2020 + CCO, HermiT certificate
publish: PUBLISH\naegir.lineup.sync\nOQuaRE hard gate → corpora/ontology/

passages -> derive
derive -> candidate
candidate -> promote
promote -> catalog: admitted
promote -> derive: reason (refine) {
  style.stroke-dash: 3
}
catalog -> realize
realize -> publish
```

- **Derive** — the engine (`instruct` on the federation's Qwen3.8-27B over gRPC,
  `src/aegir/engine/`) reads aperture-filtered FinePDFs passages and
  derives axiom-pattern-bound primitives
  (`scripts/derive_ontology.py`), staged in `catalog.candidate.json`.
- **Promote** — `scripts/promote_candidates.py` is membrane-gated
  admission into `catalog.json`: re-parse (DeepOnto), HermiT
  consistency against BFO/CCO, and the provenance threading that
  carries the derivation signals across the boundary (severing
  `grounds_ddl` here once left the relational projection ungrounded — do
  not re-sever). `combined.json` is rebuilt, never edited.
- **Realize** — `scripts/build_realized_ontology.py` instantiates
  templates into concrete OWL over BFO 2020 + CCO (imported as a
  reasoning authority), HermiT signs the certificate, and
  unsatisfiability emits **justification signals**
  (`build/realize_signals.json`) that the `scripts/reauthor_unsat.py`
  agent loop consumes.
- **Publish** — `aegir.lineup.sync` pushes the ontology Data Product
  to the `corpora` submodule (zndx/sdg-corpora), hard-gated by the
  OQuaRE quality gate (`sync._gate()`; no `--push` below GREEN).

## Membranes with reasons — the doctrine

Everywhere in this chapter the same shape recurs: an agent
**proposes**, a deterministic membrane **disposes and returns the
reason**, and the agent responds and refines — a closed loop, never
one-shot propose→drop. Axioms face parse → HermiT (with CCO's
disjointness axioms as the reasoning authority) → OntoClean; authored
retrieval surfaces face the annotation membranes **M1–M8**
(`src/aegir/ontology/annotation_membrane.py`), including the M8
positive-voice rule (definition-by-negation is an embedding
anti-pattern). Rigor is *enforced, not asserted*, and the two
strongest axiom membranes (HermiT and OntoClean) are un-fakeable. The
[Authors Guide](./ontology/authors_guide.md) is the canonical
reference for every metric, band, gate, and membrane.

## The topic layer — the retrieval / measurement face

The same OWL classes are also the corpus's **topic registry**:
topics ≡ ontology-grounded concept anchors in qdrant (`sdg_topics`;
`src/aegir/ontology/topic_layer.py`). FinePDFs items
(anchor-proportional passage windows) map to **one topic or none** by
hierarchical-margin ColBERT MaxSim under a pre-registered,
null-calibrated unambiguity gate; every association is
content-addressed to the collection state that adjudicated it. The
ambiguous mass is the error signal and the **lexicon is the
parameter** — unaligned input indicts the topics, never the input
(the inverted-LDA direction). Registry changes are themselves gated
(the M7 basin gate, `topic_layer.basin_calibration`). On the output
side, **congruence** (`src/aegir/ontology/congruence.py`) classifies
generated chapters against the same ColBERT substrate the harvest
classified inputs with — the BERTopic-era R_D reborn. The
[phase gate](./roadmap/phase_gate_inverted_topic_layer.md) is the
authoritative record; the BERTopic-era instruments
(`topic_alignment.py`, `T_I.pkl`, `build_topic_model.py`) are
deprecated, kept for v0.3 reproducibility only.

## Scope summary

| Concern | Owner | Notes |
|---|---|---|
| SDG ontology IRIs + BFO/CCO grounding | **Ægir** | `src/aegir/ontology/catalog/catalog.json` → realized `corpora/ontology/sdg-ontology.{omn,owl}` |
| Content-first derivation (FinePDFs → OWL) | **Ægir** | `scripts/derive_ontology.py`, `scripts/promote_candidates.py`, `src/aegir/ontology/patterns.py` |
| Grounding-anchor retrieval (CCO + FHIR + accretive) | **Ægir** | `scripts/grounding_anchors.py` |
| Rigor metrology + OQuaRE publish gate | **Ægir** | `scripts/ontology_metrology.py`, `scripts/ontology_oquare.py`, `aegir.lineup.sync._gate` |
| Disposal membranes (parse / HermiT / OntoClean / M1–M8) | **Ægir** | `scripts/build_realized_ontology.py`, `src/aegir/ontology/ontoclean.py`, `src/aegir/ontology/annotation_membrane.py` |
| Topic layer + congruence (retrieval/measurement) | **Ægir** | `src/aegir/ontology/topic_layer.py`, `src/aegir/ontology/congruence.py` |
| Ontology-grounded corpus + relational projection | **Ægir** | `src/aegir/flows/sdg_corpora_flow.py` (`just metaflow`), `scripts/realize_sdg.py`, kvasir DDL/SHACL |
| CTA / CPA dataset loaders | **Ægir** | `src/aegir/data/table_dataset.py` |
| Model training + evaluation | **Ægir** | `train.py`, `train_pretrain.py`, `AegirForColumnAnnotation` |
| **Consumer-side use of the above** | downstream projects | Outside Ægir's design constraints |

A separate sibling project (Atelier) consumes Ægir-produced artifacts
as an independent pretraining-efficacy gate. Atelier's own docs
describe what *it* needs from this contract, but those docs are
advisory input here, not specification.

## Sub-pages

- [Authors Guide — metrics & quality gates](./ontology/authors_guide.md)
  — **canonical**: the full quantitative metric suite (IOF rigor
  dimensions, OntoQA/OQuaRE structural metrics, OntoClean proxies,
  topic-layer instruments), the OQuaRE publish gate with its `[1,5]`
  bands and floors, the disposal membranes, and the pre-registered
  OQ-Rigor / OQ-Structure objectives, with the exact formulas the
  tooling enforces
- [Charter](./ontology/charter.md) — Ægir's internal direction-setter
  for the ontology scope: provenance discipline, the committed
  BFO/CCO branch structure, and external-standard anchors
- [Migration](./ontology/migration.md) — authoring history for the
  initial bespoke vocabulary (dated record)
- [Concept brief — RLVR for ontology generation](./ontology/concept_brief.md)
  — the design of the long-horizon **Signals M4** apparatus: a
  four-component verifiable reward *R(O, I)* over OWL artifacts and a
  GRPO-trained, SAE-instrumented local policy targeting it. That reward
  is now **realized as the deterministic membrane stack** (HermiT/CCO,
  OntoClean, OQuaRE) that the agent-mediated propose/dispose loop —
  documented in the Authors Guide — is building and proving today
- [Semantic engine — authoritative reference](./ontology/production_state.md)
  — the operational-state description of the SDG ontology, the rigor
  program, and the closed-loop synthetic-data pipeline
- [RLVR for ontology generation](./ontology/rlvr.md) — the
  externally-readable methodological chapter for the long-horizon M4
  apparatus: the verifier *R(O, I)*, now realized as the membrane stack,
  and the SAE-instrumented-Qwen policy that GRPO trains against it to
  autonomously generate ontology extensions
- [Skills Library & Generate→Re-ground→Refine Engine](./ontology/skills_loop_spec.md)
  — the dated v0.1 specification of the skills package
  (`src/aegir/ontology/skills/`) and the closed refine loop whose
  shape now lives in the metaflow pipeline
