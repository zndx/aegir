# Stage 4: Training Objective

The training objective is the key departure from standard pretraining: Aegir does not learn to predict the next token. It learns to recover the ontological entities -- **data elements** -- that were used to generate the relational data it observes. This is possible because the generation pipeline (Stages 1--3) preserves a complete mapping from every column back to its source ontological entity.

## Task Formulation

```d2
direction: right

input: Serialized Tables {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  bytes: "Byte-serialized\ntable data"
  note: "Model sees ONLY\nraw bytes — no\nschema metadata"
}

model: Aegir {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  encode: "Hierarchical\nencoding"
  chunk: "Dynamic\nchunking"
  heads: "Prediction\nheads"

  encode -> chunk -> heads
}

predictions: Predictions {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  cta: "Per-column types\n(CTA head)"
  de: "Data element\nclusters (DE head)"
  hier: "BFO hierarchy\n(Hierarchy head)"
}

ground_truth: Ground Truth {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
  tooltip: "Known by construction from the generation pipeline"

  ontology: "Source ontology\nfragment"
  mapping: "Column → entity\nprovenance map"
}

loss: Loss {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  compute: "Multi-task loss\nCTA + DE + hierarchy"
}

input -> model: raw bytes
model -> predictions
predictions -> loss
ground_truth -> loss: compare
loss -> model: gradient update {
  style.stroke-dash: 5
  style.stroke: "#9e9e9e"
}
```

### What the Model Sees

The model receives **byte-serialized relational tables** -- one or more tables from the same generated schema, serialized as a byte stream. The serialization format mirrors how real data would be encountered:

- CSV-style serialization with delimiters, quoting, and escape characters
- Column headers may be descriptive (`patient_id`), abbreviated (`pat_id`), or opaque (`col_0`)
- Multiple tables are concatenated with table-boundary markers
- No schema metadata (no types, no foreign key declarations, no table names beyond what appears in headers)

The model must infer semantic structure purely from the byte patterns it observes.

### What the Model Predicts

Three prediction heads operate on the column-level embeddings produced by Aegir's hierarchical encoder:

1. **Column Type Annotation (CTA)**: For each column, predict its BFO-grounded semantic type from a taxonomy. This maps directly to the CTA task on benchmarks like SOTAB and GitTables.

2. **Data Element Discovery (DE)**: Predict which columns -- potentially across different tables -- belong to the same ontological entity. This is formulated as a clustering task: columns originating from the same BFO class should receive similar embeddings.

3. **Hierarchical Consistency**: Predict the BFO hierarchy level for each column. If a column is classified as `Diagnosis` (a subclass of `GDC`), it should also be recognized as a `GenericallyDependentContinuant`. This head enforces ontological coherence.

### What We Compare Against

The ground truth comes directly from the generation pipeline:

- **CTA labels**: The `Column → BFO property` mapping from Stage 2 gives the exact semantic type of every column
- **DE labels**: The `Column → BFO class` mapping identifies which columns originated from the same ontological entity
- **Hierarchy labels**: The BFO subsumption hierarchy defines the expected parent types for every leaf prediction

## Loss Function

The total loss is a weighted combination of three terms:

\\[
\mathcal{L} = \mathcal{L}_{\text{CTA}} + \lambda_1 \mathcal{L}_{\text{DE}} + \lambda_2 \mathcal{L}_{\text{hier}}
\\]

### Column Type Annotation Loss

Standard cross-entropy over the column type taxonomy:

\\[
\mathcal{L}_{\text{CTA}} = -\frac{1}{N} \sum_{i=1}^{N} \log p(y_i \mid \mathbf{h}_i)
\\]

where \\(\mathbf{h}_i\\) is the column embedding for column \\(i\\), \\(y_i\\) is the ground truth BFO-grounded type, and \\(N\\) is the total number of columns across all tables in the batch.

### Data Element Discovery Loss

A contrastive loss that pulls together columns from the same ontological entity and pushes apart columns from different entities:

\\[
\mathcal{L}_{\text{DE}} = -\frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_k) / \tau)}
\\]

where \\(\mathcal{P}\\) is the set of positive pairs (columns from the same BFO class), \\(\text{sim}\\) is cosine similarity, and \\(\tau\\) is a temperature parameter.

This loss is what teaches the model to discover data elements: columns that the model embeds close together are predicted to belong to the same real-world entity, regardless of which table they appear in.

### Hierarchical Consistency Loss

A penalty for predictions that violate the BFO subsumption hierarchy:

\\[
\mathcal{L}_{\text{hier}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c \in \text{ancestors}(y_i)} \max(0, \delta - p(c \mid \mathbf{h}_i))
\\]

where \\(\text{ancestors}(y_i)\\) returns all BFO ancestors of the predicted type, and \\(\delta\\) is a margin. If a column is predicted as `Diagnosis`, the model should assign high probability to all ancestor types: `GDC`, `Continuant`, `Entity`.

## Training Loop

```d2
direction: down

batch: Batch Construction {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  sample: "Sample schema from pool"
  serialize: "Serialize tables to bytes"
  attach: "Attach ontological labels"

  sample -> serialize -> attach
}

forward: Forward Pass {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  encode: "Byte encoding → chunks"
  process: "Hierarchical RWKV processing"
  pool: "Column-level pooling"
  predict: "CTA + DE + hierarchy heads"

  encode -> process -> pool -> predict
}

loss: Loss Computation {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"

  cta_loss: "CTA: cross-entropy"
  de_loss: "DE: contrastive"
  hier_loss: "Hierarchy: margin"
  total: "Weighted sum"

  cta_loss -> total
  de_loss -> total
  hier_loss -> total
}

update: Gradient Update {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  backward: "Backpropagate"
  step: "Optimizer step"

  backward -> step
}

batch -> forward: "serialized bytes\n+ labels"
forward -> loss: "predictions"
loss -> update: "total loss"
update -> batch: "next batch" {
  style.stroke-dash: 5
}
```

### Batch Construction

Each training batch contains serialized tables from multiple generated schemas:

1. **Sample** a schema from the pool (with curriculum: simpler schemas early, complex multi-table schemas later)
2. **Serialize** one or more tables from the schema to bytes, using randomized serialization parameters (delimiter choice, quoting style, header format)
3. **Attach** the ontological provenance labels as training targets

### Multi-Table Batches

For cross-table data element discovery, batches include multiple tables from the same schema. The agent swarm architecture processes each table with a separate agent, and the fused recurrent states are used for the DE prediction head. This directly trains the model's cross-table reasoning capability.

## Connection to Downstream Tasks

The pretraining objective maps precisely to the three real-world tasks described in the [Introduction](../introduction.md):

| Pretraining Task | Downstream Task | Transfer Mechanism |
|-----------------|-----------------|-------------------|
| Column type prediction | CTA on SOTAB/GitTables/WikiTables | Fine-tune CTA head on benchmark taxonomy |
| Cross-column clustering | CPA on benchmark datasets | Column pair relationship classification |
| Cross-table data element prediction | Enterprise data element discovery | Direct application -- same task, real data |

The key advantage: by pretraining on synthetic data with known ground truth at massive scale, the model enters fine-tuning with strong representations for column semantics. The confusable types, cross-table relationships, and ontological hierarchies it has learned from synthetic data transfer directly to the noisy, inconsistently-named, under-documented columns in real enterprise data warehouses.

### Integration with Evidence Pipelines

In production, Aegir's predictions feed into Dempster-Shafer theory (DST) evidence fusion pipelines as a learned evidence source. The model produces:

- **Column type predictions with calibrated confidence** -- these become mass functions in the DST framework
- **Column embedding similarities** -- these provide evidence for same-entity relationships
- **Hierarchical type predictions** -- these constrain the feasible type space for conjunctive combination

The calibration quality of Aegir's confidence scores matters as much as the accuracy of its top-1 predictions. Training on diverse synthetic data with controlled difficulty (including deliberately confusable types) produces well-calibrated uncertainty estimates, because the model learns from data where the boundary between types is precisely controlled.
