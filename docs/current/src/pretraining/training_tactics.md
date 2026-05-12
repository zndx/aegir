# Training Tactics

> **Deferred framing.** This page describes self-supervised tactics
> for the entity-recovery objective in an earlier exploratory
> SysMLv2 / ORM pipeline; the operational v2 mixed-corpus pretrain
> uses next-byte prediction. Hygiene parameters (LR schedule, warmup,
> gradient clipping) for the operational run are documented in
> [Training Regime](../training_regime.md). The tactical content
> below may be revisited if any of it transfers cleanly to the
> next-byte regime; for now it is preserved in the repository for
> archival continuity.

The [training objective](./training_objective.md) defines *what* Aegir learns -- ontological entity recovery from serialized relational tables. This page defines *how*: the specific self-supervised tasks, corruption strategies, and curriculum design that compose the pretraining regiment. Each tactic is adapted from a proven LLM pretraining method but re-targeted at the structural properties of relational data with known ontological provenance.

## Tactic Overview

```d2
direction: down

core: Core Objectives {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  opm: Object Property\nMasking
  rcd: Replaced Column\nDetection
  rm: Relation\nMasking
  sc: Span Corruption\n(Entity-Level)
}

augmentation: Augmentation Strategies {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  sd: Schema\nDenoising
  csc: Cross-Schema\nContrastive
}

domain: Domain-Specific Objectives {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  ar: Axiom\nRecovery
  np: Normalization\nPrediction
  ce: Cardinality\nEstimation
}

curriculum: Difficulty Curriculum (UL2-style) {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  easy: "Easy (R)\nClear names, obvious patterns"
  medium: "Medium (S)\nAbbreviated names, multi-table"
  hard: "Hard (X)\nOpaque names, confusable types"
  expert: "Expert (Z)\nFull reconstruction, obfuscated"

  easy -> medium -> hard -> expert
}

core -> curriculum: "mixed per batch"
augmentation -> curriculum: "applied to all levels"
domain -> curriculum: "scheduled introduction"
```

Each tactic is described below with its LLM analogue, formal task specification, and the downstream capability it trains.

---

## Core Objectives

### Object Property Masking

**LLM analogue**: Masked Language Modeling (BERT)

Mask one or more properties from an ontological entity definition. The model receives the serialized tables (which still contain the data for the masked properties) and must predict what properties the source entity had.

```d2
direction: right

entity: Source Entity {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  full: "Patient\n─────────\npatient_id: str\ndate_of_birth: date\ngender: str\naddress: str"
}

masked: Masked Entity {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  partial: "Patient\n─────────\npatient_id: str\n[MASK]\ngender: str\naddress: str"
}

tables: Serialized Tables {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  data: "patient_id,date_of_birth,gender,address\na3f8c1d0,1987-03-15,Female,2847 Oak...\nb7e2a4f1,1952-11-28,Male,156 Pine..."
}

model_pred: Model Prediction {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  pred: "Masked property:\n  name: date_of_birth\n  type: BFO:Quality\n  sql_type: DATE"
}

entity -> masked: "mask property"
tables -> model_pred: "Aegir"
masked -> model_pred: "predict [MASK]" {style.stroke-dash: 5}
```

**Difficulty gradation**:

| Level | What's Masked | Challenge |
|-------|--------------|-----------|
| Easy | A column with structurally distinctive values (dates, emails) | Pattern recognition |
| Medium | A column whose type depends on co-occurring columns | Cross-column reasoning |
| Hard | A column with confusable values (UUID vs advertising ID) | Contextual disambiguation |
| Expert | Multiple properties from the same entity simultaneously | Entity structure reconstruction |

**Loss**: Cross-entropy over the property type vocabulary, plus a regression loss for predicting the property name embedding.

\\[
\mathcal{L}_{\text{OPM}} = -\frac{1}{|M|} \sum_{p \in M} \left[ \log P(y_p \mid \mathbf{h}_p) + \alpha \| \hat{\mathbf{e}}_p - \mathbf{e}_p \|^2 \right]
\\]

where \\(M\\) is the set of masked properties, \\(y_p\\) is the property's BFO type, \\(\mathbf{e}_p\\) is the property name embedding, and \\(\alpha\\) weights the name regression term.

**Trains**: Column type annotation (CTA). The model learns to identify what semantic role a column plays from its value distribution and surrounding context.

---

### Replaced Column Detection

**LLM analogue**: Replaced Token Detection (ELECTRA)

Swap columns between tables that originated from different ontological entities. A discriminator must identify which columns are imposters — present in a table they don't ontologically belong to.

```d2
direction: right

original: Original Schema {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  patient: "patient table\n─────────\npatient_id\ndate_of_birth\ngender"
  provider: "provider table\n─────────\nprovider_id\nspecialty\nlicense_number"
}

corrupted: Corrupted Schema {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  patient_c: "patient table\n─────────\npatient_id\ndate_of_birth\nlicense_number ← swapped"
  provider_c: "provider table\n─────────\nprovider_id\nspecialty\ngender ← swapped"
}

discriminator: Discriminator {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  pred: "patient.license_number: IMPOSTER\nprovider.gender: IMPOSTER\nall others: ORIGINAL"
}

original -> corrupted: "swap columns\n(generator selects\nplausible swaps)"
corrupted -> discriminator: "detect imposters"
```

The generator learns to make *plausible* swaps — columns with similar value distributions but different semantic types. This is precisely the confusable-pair problem. A naive generator might swap `patient_id` (UUID) with `encounter_date` (timestamp) — trivially detectable. A trained generator learns to swap `patient_id` with `provider_id` (both UUIDs, both foreign-keyed) — a much harder discrimination task.

**Two-phase training**:

1. **Generator**: A small model that scores candidate column swaps by value-distribution similarity and selects high-similarity pairs
2. **Discriminator**: Aegir itself, trained to detect which columns don't belong

The ELECTRA insight applies directly: the discriminator receives a training signal on *every* column (original or replaced), not just the masked positions. This is far more sample-efficient than masking-based objectives.

**Loss**: Binary cross-entropy per column.

\\[
\mathcal{L}_{\text{RCD}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log D(\mathbf{h}_i) + (1 - y_i) \log(1 - D(\mathbf{h}_i)) \right]
\\]

where \\(y_i = 1\\) if column \\(i\\) was replaced and \\(D\\) is the discriminator head.

**Trains**: Confusable type resolution. Directly addresses the hardest failure mode in production column annotation — columns with identical value patterns but different semantic roles.

---

### Relation Masking

**LLM analogue**: Next Sentence Prediction (BERT) / Sentence Order Prediction (ALBERT), extended to structural relationships

Drop a foreign key column from the serialized data and ask the model to predict that a relationship between two tables exists, which tables it connects, and what column would mediate it.

**Task variants**:

| Variant | Input | Target |
|---------|-------|--------|
| **Existence** | Two tables, FK column removed | Binary: are these tables related? |
| **Direction** | Two related tables, FK removed | Which table is parent, which is child? |
| **Column** | Tables with FK removed | Which column in the child table held the FK? |
| **Full recovery** | Multi-table schema, one FK removed | Predict source table, target table, and mediating column |

**Difficulty**: Existence is easy (value overlap between tables is a strong signal). Direction requires understanding cardinality from data distributions. Full recovery in a 10-table schema with multiple possible FK targets is genuinely hard.

**Loss**: Cross-entropy over table pairs for existence/direction, cross-entropy over columns for the FK column prediction.

\\[
\mathcal{L}_{\text{RM}} = \mathcal{L}_{\text{exist}} + \beta_1 \mathcal{L}_{\text{direction}} + \beta_2 \mathcal{L}_{\text{column}}
\\]

**Trains**: Cross-table data element discovery. The model learns to identify structural relationships between tables from data patterns alone — exactly what's needed when foreign key metadata is missing or unreliable in enterprise warehouses.

---

### Span Corruption (Entity-Level)

**LLM analogue**: Span Corruption (T5)

Mask all columns belonging to one data element across all tables and replace them with a sentinel. The model must predict what *kind* of entity is absent based on the remaining schema structure.

```d2
direction: right

full_schema: Complete Schema {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  patient: "patient (4 cols)"
  encounter: "encounter (12 cols)"
  diagnosis: "diagnosis (5 cols)"
  medication: "medication (6 cols)"
  provider: "provider (4 cols)"
}

corrupted_schema: Corrupted Schema {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  patient_c: "patient (4 cols)"
  encounter_c: "encounter (7 cols)\n← 5 cols removed"
  diagnosis_c: "[SENTINEL]"
  medication_c: "medication (6 cols)"
  provider_c: "provider (4 cols)"
}

prediction: Model Prediction {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  pred: "Missing entity:\n  type: Diagnosis ⊑ BFO:GDC\n  properties: diagnosis_id,\n    icd10_code, description,\n    is_primary\n  relations: Encounter → Diagnosis"
}

full_schema -> corrupted_schema: "remove all\nDiagnosis columns"
corrupted_schema -> prediction: "predict missing\nentity structure"
```

This is harder than single-property masking because the model must reason about the *structural hole* in the schema. A schema with patients, encounters, medications, and providers but no diagnostic information has a recognizable gap — clinical workflows always involve diagnosis. The model learns domain-level structural expectations.

**Masking strategies**:
- **Single entity**: Remove all columns from one BFO class (as above)
- **Related pair**: Remove two related entities (e.g., Diagnosis and its FK in Encounter)
- **Subtree**: Remove an entity and all its dependents in the ontological hierarchy

**Loss**: Sequence-to-sequence generation of the masked entity structure, or classification over a vocabulary of entity type templates.

**Trains**: Entity boundary detection and structural reasoning. When the model encounters a real database missing expected entities, it can predict what *should* exist — critical for data governance gap analysis.

---

## Augmentation Strategies

### Schema Denoising

**LLM analogue**: Denoising Autoencoder (BART)

Apply multiple corruptions to the serialized schema simultaneously. The model must recover the clean ontological structure from the noisy input.

**Corruption menu** (applied stochastically per training example):

| Corruption | What Changes | Real-World Analogue |
|-----------|-------------|---------------------|
| **Column renaming** | `date_of_birth` → `col_3` | Generic column names in enterprise DW |
| **Column shuffling** | Randomize column order within tables | Arbitrary column ordering conventions |
| **Table merging** | Join two tables into one wide table | Denormalization for query performance |
| **Table splitting** | Split one table into arbitrary fragments | Vertical partitioning |
| **Type coercion** | Store dates as strings, integers as floats | Legacy system type mismatches |
| **Delimiter variation** | CSV → TSV → pipe-delimited → fixed-width | Different export formats |
| **Header removal** | Drop column headers entirely | Headerless data exports |
| **Row sampling** | Keep only a random subset of rows | Partial data access |

Multiple corruptions can stack: rename columns *and* merge tables *and* switch delimiters. The model trained on this distribution becomes robust to the full range of real-world schema messiness.

**Loss**: Reconstruction loss on the original ontological labels applied to the column embeddings from the corrupted input. The corruptions change what the model *sees*; the targets remain the clean ontological structure.

**Trains**: Robustness to real-world data formats. Enterprise databases exhibit every one of these corruptions and often several simultaneously.

---

### Cross-Schema Contrastive Learning

**LLM analogue**: Contrastive Learning (SimCLR, CLIP)

Generate two different schemas from the same ontology fragment — one normalized with clear names, one denormalized with obfuscated names — and train the model to produce similar representations for both. Schemas from *different* ontology fragments should produce dissimilar representations.

```d2
direction: down

ontology: Source Ontology\nPatient + Encounter + Diagnosis {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"
}

view_a: Schema Variant A {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  desc: "3NF, descriptive names\npatient.date_of_birth\nencounter.presenting_complaint\ndiagnosis.icd10_code"
}

view_b: Schema Variant B {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  desc: "Denormalized, opaque names\npatient_encounters.col_2\npatient_encounters.col_7\npatient_encounters.col_12"
}

embed_a: Embedding A {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
}

embed_b: Embedding B {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
}

ontology -> view_a: "project +\nnormalize"
ontology -> view_b: "project +\ndenormalize +\nobfuscate"
view_a -> embed_a: "Aegir"
view_b -> embed_b: "Aegir"
embed_a -> embed_b: "should be\nsimilar" {
  style.stroke: "#2e7d32"
  style.stroke-width: 3
}
```

**Positive pairs**: Two schema variants from the same ontology fragment.
**Negative pairs**: Schemas from different ontology fragments (even within the same domain — two different healthcare schemas should still be distinguishable).

**Loss**: InfoNCE contrastive loss over schema-level representations.

\\[
\mathcal{L}_{\text{CSC}} = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \log \frac{\exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_i^b) / \tau)}{\sum_{j \in \mathcal{B}} \exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_j^b) / \tau)}
\\]

where \\(\mathbf{z}_i^a\\) and \\(\mathbf{z}_i^b\\) are schema-level embeddings (pooled from column embeddings) for the two variants of ontology fragment \\(i\\), and \\(\mathcal{B}\\) is the batch.

**Trains**: Schema-invariant representations. The model learns that the *same information* can appear in radically different structural formats — the core challenge in enterprise data integration.

---

## Domain-Specific Objectives

### Axiom Recovery

**LLM analogue**: No direct analogue — novel to this setting

Given only the populated tables (no schema metadata), predict the constraints from the source ontology.

**Target axioms**:

| Axiom Type | Example | Evidence in Data |
|-----------|---------|-----------------|
| Enum constraint | `disposition ∈ {admission, discharge, transfer, observation}` | Closed set of distinct values |
| Uniqueness | `license_number` is unique per provider | No duplicates in column |
| Cardinality | Exactly one `is_primary=true` per encounter | Group-by count pattern |
| Range | `esi_level ∈ [1, 5]` | Min/max of integer column |
| Referential | Every `encounter.patient_id` appears in `patient.patient_id` | Value subset relationship |
| Functional dependency | `zip_code → state` | Deterministic mapping in data |

**Loss**: Multi-label classification over axiom templates, parameterized by column references and value sets.

**Trains**: Constraint discovery. In production, many database constraints are implicit (enforced by application logic, not declared in the schema). A model that can infer constraints from data patterns provides direct value for data quality assessment and governance.

---

### Normalization Prediction

**LLM analogue**: No direct analogue — novel to this setting

Given a denormalized table, predict the normalized ontological entities — which groups of columns *should* be separate entities.

In the hospital example, a fully denormalized `patient_encounters` table contains patient demographics, encounter details, vital signs, diagnoses, and medications all in one wide table. The model must predict that this represents 5+ distinct ontological entities that have been collapsed.

The inverse task is also valuable: given a normalized schema, predict which tables could be meaningfully denormalized (i.e., which tables represent qualities or sub-parts of a parent entity).

**Loss**: Clustering loss over column embeddings within a single table — columns that should be factored into the same normalized entity should cluster together.

\\[
\mathcal{L}_{\text{norm}} = -\frac{1}{|\mathcal{P}_{\text{intra}}|} \sum_{(i,j) \in \mathcal{P}_{\text{intra}}} \log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}{\sum_{k \in \text{cols}(t)} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_k) / \tau)}
\\]

where \\(\mathcal{P}_{\text{intra}}\\) is the set of column pairs within a single table that originate from the same ontological entity.

**Trains**: Entity boundary detection within denormalized tables. Real enterprise data warehouses are heavily denormalized for query performance. Recovering the underlying entity structure from a 200-column fact table is a high-value governance task.

---

### Cardinality Estimation

**LLM analogue**: No direct analogue — extends relational reasoning

Given populated tables, predict the cardinality constraints from the source ontology: one-to-one, one-to-many, or many-to-many.

The model must infer cardinality from value distributions:
- **1:1**: Every FK value appears exactly once in both tables
- **1:N**: FK values in the child table repeat; each parent PK appears once
- **M:N**: Both sides have repeating values (mediated by a junction table)

**Loss**: Cross-entropy over cardinality categories per table pair.

**Trains**: Relationship characterization. Understanding cardinality is foundational for schema understanding and directly supports both CPA and data element discovery — a 1:1 relationship suggests entity decomposition, while M:N suggests an independent association.

---

## Difficulty Curriculum

Following UL2's insight that mixing objectives with explicit difficulty signals outperforms any single objective, training uses a difficulty-tagged curriculum.

```d2
direction: right

easy: R — Retrieval {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  desc: "Descriptive column names\nStructurally distinctive values\nSingle-table tasks\nClear FK declarations"
}

medium: S — Structural {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  desc: "Abbreviated column names\nMulti-table reasoning\n1-2 confusable pairs\nSome denormalization"
}

hard: X — Extreme {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  desc: "Opaque names (col_0, field_a)\nMany confusable pairs\nHeavy denormalization\nMissing headers"
}

expert: Z — Zero-Signal {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  desc: "Fully obfuscated schema\nAll columns renamed\nTables merged arbitrarily\nFull ontology reconstruction"
}

schedule: Curriculum Schedule {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  desc: "Phase 1: 70R/20S/10X/0Z\nPhase 2: 30R/40S/20X/10Z\nPhase 3: 10R/30S/30X/30Z\nPhase 4: 10R/20S/30X/40Z"
}

easy -> schedule
medium -> schedule
hard -> schedule
expert -> schedule
```

Each training example carries a difficulty tag prepended to the input. The model learns to allocate capacity differently depending on the expected difficulty — using fast pattern matching for R-level tasks and deeper structural reasoning for X/Z-level tasks.

### Curriculum Schedule

Training proceeds in four phases, progressively increasing difficulty:

| Phase | Epochs | Mix (R/S/X/Z) | Objectives Introduced |
|-------|--------|----------------|----------------------|
| 1 | 0--10 | 70/20/10/0 | OPM, RCD (easy variants) |
| 2 | 10--30 | 30/40/20/10 | + Relation Masking, Schema Denoising |
| 3 | 30--60 | 10/30/30/30 | + Span Corruption, Cross-Schema Contrastive |
| 4 | 60+ | 10/20/30/40 | + Axiom Recovery, Normalization, Cardinality |

Domain-specific objectives (axiom recovery, normalization prediction, cardinality estimation) are introduced late because they require the model to already have basic column understanding and cross-table reasoning capabilities.

## Objective Priority

The objectives are not equally important. Based on downstream task alignment:

| Objective | Priority | Downstream Impact |
|-----------|----------|-------------------|
| Object Property Masking | **Core** | Directly trains CTA |
| Replaced Column Detection | **Core** | Resolves confusable pairs — the hardest CTA failures |
| Relation Masking | **Core** | Directly trains cross-table data element discovery |
| Span Corruption | **Core** | Trains entity boundary detection |
| Schema Denoising | **High** | Robustness to real-world data — improves all tasks |
| Cross-Schema Contrastive | **High** | Schema-invariant representations — critical for transfer |
| Axiom Recovery | **Medium** | Valuable for governance but not core to CTA/DE |
| Normalization Prediction | **Medium** | Important for denormalized warehouses |
| Cardinality Estimation | **Medium** | Supports relationship characterization |

The four core objectives should compose the majority of training compute. Augmentation strategies (denoising, contrastive) are applied as data transformations rather than separate losses. Domain-specific objectives are scheduled in later phases as refinement tasks.
