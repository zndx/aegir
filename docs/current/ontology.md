# Ontology

> **Scope (2026-05-09).** Ægir is the canonical owner of the bespoke
> BFO/CCO-grounded vocabulary used by the metadata-tagging stack, and of
> the synthetic data generators that produce in-distribution training
> bytes against that vocabulary. Vocabulary, generators, and the
> `vocab_label_map.json` artifact published outward all live here. This
> section sets the contract, the directory layout, and the empirical
> gate that has to clear before any vocabulary expansion work begins.

The label space conditions everything downstream — what the model is
asked to predict, what synthetic data the pretraining corpus can include,
which benchmark labels the system claims coverage of, and which BFO
ancestry chains a leaderboard prediction can emit. Putting it next to
the model is the only arrangement where these decisions stay coherent.

## Scope summary

| Concern | Owner | Notes |
|---|---|---|
| Vocabulary IRIs + BFO/CCO grounding | **Ægir** | `src/aegir/ontology/aegir-vocab.ttl` (target home) |
| Synth value generators | **Ægir** | `src/aegir/synth/` (target home) |
| Benchmark label → IRI lookup | **Ægir** | Build artifact: `vocab_label_map.json` |
| SOTAB / GitTables / WikiTables bundle download | **Ægir** | Already wired (`scripts/download_sotab.py`, etc.) |
| CTA / CPA dataset loaders | **Ægir** | `src/aegir/data/table_dataset.py` |
| BFO-grounded prediction emission | **Ægir** | Leaderboard gateway response payload |
| Model training + evaluation | **Ægir** | `train.py`, `AegirForColumnAnnotation` |
| Trained checkpoint distribution | **Ægir** | Existing `outputs/runs/` artifact discipline |
| **Consumer-side use of the above** | downstream projects | Outside Ægir's design constraints |

A separate sibling project (Atelier) consumes Ægir-produced checkpoints
as Dempster-Shafer evidence sources. Atelier's own docs describe what
*it* needs from this contract, but those docs are advisory input here,
not specification. Where Atelier proposes a tiered vocabulary expansion
plan or a particular layout, Ægir is free to take, leave, or reorganize
those suggestions on its own roadmap.

## Sub-pages

- [Charter](./ontology/charter.md) — the contract Ægir publishes
  outward and the design constraints that follow from it
- [Migration](./ontology/migration.md) — authoring the initial
  bespoke vocabulary
- [Concept brief — RLVR for ontology generation](./ontology/concept_brief.md)
  — a four-component verifiable reward *R* over OWL ontology
  artifacts and a GRPO-trained LLM policy that targets it. Two-paper
  scope: paper 1 establishes verifier discrimination + RLVR
  optimizability; paper 2 (follow-on) tests downstream pretraining
  utility
