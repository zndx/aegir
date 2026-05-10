# Pretraining Documentation Complete

## What Was Done

Created 7 new documentation pages for the ontology-grounded pretraining methodology:

### New Files
- `docs/current/pretraining.md` -- Overview with master pipeline D2 diagram, novelty table, scaling argument
- `docs/current/pretraining/ontology_extraction.md` -- Stage 1: Text → BFO ontology extraction
- `docs/current/pretraining/schema_projection.md` -- Stage 2: Ontology → SysMLv2 → data objects → relational schema
- `docs/current/pretraining/synthetic_generation.md` -- Stage 3: Schema population + diversity + confusable type injection
- `docs/current/pretraining/training_objective.md` -- Stage 4: Entity recovery task formulation with multi-task loss
- `docs/current/pretraining/end_to_end_example.md` -- Complete walkthrough: hospital ED domain from text → validated training example
- `docs/current/SUMMARY.md` -- Updated with new section between Architecture and Agent Swarm

### D2 Diagrams
- Master pipeline diagram (5 color-coded stages, vertical flow)
- Ontology extraction pipeline (text → LLM → output → validation)
- Schema projection three-column transformation (ontology → SysMLv2 → code)
- SysMLv2 block diagram for hospital ED example
- Population pipeline (schema + generators → database → serialization)
- Training task formulation (input → model → predictions → loss)
- Training loop (batch → forward → loss → update)
- Validation round-trip (predicted elements ↔ source entities)
- End-to-end hero diagram (9 steps)

### D2 Fixes
- `classes` → `class_list` (reserved keyword in D2)
- `relations` → `relation_list` (reserved keyword in D2)
- `label` → `attach` (reserved keyword in D2)

### Key Decisions
- Used hospital emergency department domain for end-to-end example (universally understood)
- Showed denormalization of VitalSigns/AcuityLevel into Encounter table as schema variation example
- Included confusable type injection table in synthetic_generation.md
- MathJax for all formal definitions (ontology tuple, projection function, loss functions)
- Color coding: pink → purple → blue → orange → green matching pipeline stages
