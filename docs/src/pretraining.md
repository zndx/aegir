# Pretraining: Ontology-Grounded Synthetic Data

How do you train a structured information model at LLM scale when labeled relational data is scarce and expensive?

Conventional approaches to semantic column annotation rely on manually labeled benchmark datasets -- SOTAB, GitTables, WikiTables -- that are costly to create, domain-limited, and rarely capture the cross-table relationships needed for data element discovery. Self-supervised pretraining on raw tables (as in DODUO and TURL) learns useful representations, but the "ground truth" for column semantics remains noisy or absent.

Aegir takes a fundamentally different approach: **we generate the training data from first principles, so the ground truth is always known by construction.**

## The Core Insight

We invert the usual pipeline. Instead of finding tables and labeling them, we:

1. Start from the highest-quality curated text available
2. Extract formal ontological structure using LLMs
3. Project that structure into relational database schemas
4. Populate schemas with realistic synthetic data
5. Train the model to recover the ontological entities from the raw table data

Because we control every step of the generation process, the mapping from table columns back to ontological entities is always available as ground truth. The diversity of the input text drives the diversity of the synthetic data; the formal ontological backbone guarantees structural correctness.

## Pipeline Overview

```d2
direction: down

stage1: Stage 1: Curated Text Corpus {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  input: FineWeb PDFs Edu\n1.3T tokens, curated educational text
  llm: LLM Extraction\nGLM-4.7 / GLM-5 with structured prompting

  input -> llm: passages
}

stage2: Stage 2: Ontological Structure {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  ontology: BFO-Grounded Ontology\nClasses, relations, axioms, BFO alignment
  sysml: SysMLv2 MBSE Model\nBlocks, ports, constraints, state machines

  ontology -> sysml: formal projection
}

stage3: Stage 3: Programmatic Representation {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  objects: Python Data Objects\nDataclasses with typed fields
  schema: Relational Schema\nTables, columns, foreign keys, constraints

  objects -> schema: ORM mapping
}

stage4: Stage 4: Synthetic Data {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  generators: Value Generators\nDomain-specific, constraint-aware
  tables: Populated Tables\nRealistic rows with referential integrity

  generators -> tables: procedural generation
}

stage5: Stage 5: Training {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  serialize: Byte Serialization\nTable → byte sequence for Aegir
  train: Entity Recovery\nPredict data elements from table data
  validate: Validation\nCompare predictions against source ontology

  serialize -> train: Aegir input
  train -> validate: predicted entities
}

stage1 -> stage2: ontology extraction
stage2 -> stage3: schema projection
stage3 -> stage4: population
stage4 -> stage5: serialization + training
stage5 -> stage2: error signal refines extraction {
  style.stroke-dash: 5
  style.stroke: "#9e9e9e"
}
```

## What Is Novel

No prior work combines all five stages into a single pipeline. Each stage has precedent; the composition does not.

| Stage | Prior Art | What Exists | What Is New |
|-------|-----------|-------------|-------------|
| Text → Ontology | OntoGPT, REBEL, DeepOnto | LLM-based ontology extraction from text | Using curated educational text as seed for **training data generation** |
| BFO Grounding | Common Core Ontologies, OBO Foundry | BFO as upper ontology for domain modeling | BFO as the **generative backbone** for synthetic ML training data |
| SysMLv2 Intermediate | openCAESAR, Cameo | SysMLv2 for systems engineering | SysMLv2 MBSE as **intermediate representation** in an ML data pipeline |
| Synthetic Tables | MOSTLY.ai, SDV, NeurIPS 2024 TRL | Synthetic table generation for augmentation | Tables generated from **ontological structure** with known entity provenance |
| Entity Recovery | DODUO, TURL (masked column) | Masked language model pretraining on tables | **Ontological entity recovery** as the training objective, not next-token prediction |

The closest related work is "Enhancing Table Representations with LLM-powered Synthetic Data Generation" (NeurIPS 2024 TRL Workshop), which generates synthetic tables to improve column embedding similarity. That work generates tables for **representation learning**; Aegir generates tables for **ontological entity recovery** -- a fundamentally different objective that produces richer training signal because the ground truth includes hierarchical entity structure, cross-table relationships, and BFO-grounded type constraints.

## Why This Scales

The bottleneck in conventional table annotation is **human labeling**. The bottleneck here is **LLM inference for ontology extraction** -- which is embarrassingly parallel and decreasing in cost.

The multiplicative structure of the pipeline ensures near-unlimited training data:

| Stage | Multiplier | Source |
|-------|-----------|--------|
| Curated text | ~500M passages | FineWeb-Edu (1.3T tokens) |
| Ontology fragments | 1--5 per passage | Domain-dependent entity density |
| Database schemas | 1--10 per fragment | Varying normalization strategies |
| Table instances | 100--10,000 rows | Procedural generation with distribution control |
| **Total training examples** | **effectively unbounded** | Combinatorial product of all stages |

A single educational passage about hospital billing can produce ontology fragments for patient demographics, encounter management, diagnosis coding, insurance claims, and provider credentialing -- each of which generates distinct database schemas, each populated with different synthetic data distributions. The diversity of the training data is bounded only by the diversity of human knowledge captured in the source text.

## How This Connects to Aegir

The pretraining objective maps directly to Aegir's three target tasks:

- **Column Type Annotation (CTA)**: The per-column entity type predictions from pretraining transfer directly to CTA on SOTAB, GitTables, and WikiTables benchmarks.
- **Column Property Annotation (CPA)**: The cross-column relationship predictions learned during pretraining capture the same inter-column semantics needed for CPA.
- **Data Element Discovery**: The core pretraining objective -- grouping related columns into ontological entities across tables -- **is** data element discovery. The model learns this from synthetic data where the answer is known, then applies it to real enterprise data warehouses.

Furthermore, Aegir's agent swarm architecture enables cross-table reasoning during both pretraining and inference. Each agent processes a table, and the fused recurrent states capture inter-table relationships that no single-table model can learn.

The following sections detail each stage of the pipeline.
