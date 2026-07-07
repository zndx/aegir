# Ontology

Aegir is the canonical owner of the bespoke BFO 2020 / CCO-grounded
ontology used by the metadata-tagging stack — the **Signals Data
Governance (SDG)** ontology — and of the ontology-grounded
synthetic-data pipeline that produces in-distribution pretraining
bytes against it. The ontology, its rigor program, and the realized
OWL artifact published outward all live in this chapter. The chapter
covers what the ontology *is* now, how its classes function as the
annotation vocabulary, the quantitative rigor metrics and the formal
publish gate every extension must clear, and the disposal membranes
that enforce rigor rather than assert it.

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

`sdg-ontology` is **content-derived from FinePDFs** (qdrant/ColBERT
MaxSim domain filtering over a SKOS index) and **realized to a
HermiT-validated OWL artifact** at
`corpora/ontology/sdg-ontology.{omn,owl}`, with a consistency
certificate at `corpora/ontology/HERMIT_CERTIFICATE.md`. The seven
family catalogs (`src/aegir/ontology/catalog/01…07`) are a **seed and
regression baseline**; FinePDFs-derived intermediate classes accrete
in `catalog.json`, and the live driver is the content-first
derivation pipeline (`scripts/derive_ontology.py`,
`scripts/define_intermediate_classes.py`), not a fixed template count.

The architecture is an **agent-mediated propose / dispose feedback
loop**: an engine proposes axioms; a stack of deterministic membranes
(parse → HermiT with CCO imported as a reasoning authority →
OntoClean) disposes and returns the reason; the agent responds and
refines. Rigor is *enforced, not asserted*. The
[Authors Guide](./ontology/authors_guide.md) is the canonical
reference for every metric, band, gate, and membrane.

## Scope summary

| Concern | Owner | Notes |
|---|---|---|
| SDG ontology IRIs + BFO/CCO grounding | **Ægir** | `src/aegir/ontology/catalog/*.json` → realized `corpora/ontology/sdg-ontology.{omn,owl}` |
| Content-first derivation (FinePDFs → classes) | **Ægir** | `scripts/derive_ontology.py`, `scripts/define_intermediate_classes.py` |
| Grounding-anchor retrieval (CCO + FHIR + accretive) | **Ægir** | `scripts/grounding_anchors.py` |
| Rigor metrology + OQuaRE publish gate | **Ægir** | `scripts/ontology_metrology.py`, `scripts/ontology_oquare.py` |
| Disposal membranes (parse / HermiT / OntoClean) | **Ægir** | `scripts/build_realized_ontology.py`, `src/aegir/ontology/ontoclean.py` |
| Ontology-grounded synthetic corpus + DDL spine | **Ægir** | `scripts/generate_chapter.py`, `scripts/verify_chapters.py`, `src/aegir/ontology/ddl.py`, `realize.py` |
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
  dimensions, OntoQA/OQuaRE structural metrics, OntoClean proxies),
  the OQuaRE publish gate with its `[1,5]` bands and floors, the
  disposal membranes, and the pre-registered OQ-Rigor / OQ-Structure
  objectives, with the exact formulas the tooling enforces
- [Charter](./ontology/charter.md) — Ægir's internal direction-setter
  for the ontology scope: provenance discipline, the committed
  BFO/CCO branch structure, and external-standard anchors
- [Migration](./ontology/migration.md) — authoring history for the
  initial bespoke vocabulary
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
