# Stage 3: Synthetic Data Generation

> **Deferred framing.** This page describes Stage 3 of an earlier
> exploratory SysMLv2 / ORM pipeline for ontology-grounded
> synthetic data that preceded the project's convergence on the
> content-first derived catalog
> (`src/aegir/ontology/catalog/catalog.json`). The active
> synthetic-data pipeline is `just metaflow`
> (`src/aegir/flows/sdg_corpora_flow.py`), documented in the
> [semantic-engine authoritative
> reference](../ontology/production_state.md); the catalog is
> documented in the [Ontology chapter](../ontology.md).
> This page is preserved in the repository for archival continuity.

Given a relational schema with known ontological provenance, the third stage populates tables with realistic synthetic data. The goal is not just to fill rows -- it is to produce data distributions that exercise the same patterns and confusable types the model will encounter in real enterprise databases.

## Population Pipeline

```d2
direction: down

schema: Relational Schema {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  tables: "Tables + columns\nForeign keys\nCHECK constraints"
  provenance: "Ontological provenance\nColumn → BFO property mapping"
}

config: Generation Config {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  distributions: Distribution Control\nCardinality, null ratio, value entropy
  generators: Value Generators\nPer-type: names, dates, codes, IDs
  domain: Domain Patterns\nICD-10 codes, IBAN formats, ZIP codes
}

populate: Population Engine {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  topological: "1. Topological sort tables by FK deps"
  parent: "2. Populate parent tables first"
  child: "3. Populate children with valid FK refs"
  validate: "4. Verify all constraints satisfied"

  topological -> parent -> child -> validate
}

output: Populated Database {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  tables: "Realistic rows\nReferential integrity\nDomain-appropriate values"
  serialize: "Byte-serialized tables\nfor Aegir input"

  tables -> serialize
}

schema -> populate: "schema +\nontological types"
config -> populate: "generation\nparameters"
populate -> output: "populated\ntables"
```

## Value Generation

Each column type maps to a specialized generator that produces realistic values. The generator selection is driven by the ontological provenance of the column -- a column traced to `BFO:Quality` with domain `healthcare` produces different values than one traced to `BFO:Quality` with domain `finance`.

### Generator Categories

| Column Semantics | Generator | Example Values |
|-----------------|-----------|----------------|
| Person name | Faker (locale-aware) | "Maria Santos", "James O'Brien" |
| Date/timestamp | Range-bounded random | 2019-03-15, 2024-11-02T14:30:00 |
| Identifier (UUID) | UUIDv4 | `f47ac10b-58cc-4372-a567-0e02b2c3d479` |
| Identifier (sequential) | Auto-increment with prefix | `PAT-00001`, `ENC-2024-0042` |
| Medical code (ICD-10) | Sampled from code registry | `J18.9`, `I25.10`, `E11.65` |
| Financial code (IBAN) | Country-specific format | `DE89370400440532013000` |
| Categorical | Weighted sampling from enum | `active`, `closed`, `pending` |
| Free text | Template + Faker | "Patient presents with acute chest pain" |
| Numeric measure | Distribution-sampled | 98.6, 120/80, 72 |
| Boolean flag | Bernoulli(p) | `true`, `false` |
| Address | Locale-aware composite | "123 Main St, Springfield, IL 62704" |
| Email | Pattern-based | `maria.santos@hospital.org` |
| Phone | Country-format | `+1-555-0123` |

### Referential Integrity

Tables are populated in topological order (parents before children) to guarantee that every foreign key value references an existing parent row. The population engine:

1. Sorts tables by foreign key dependencies (detecting and breaking cycles if needed)
2. Populates root tables (no FK dependencies) first
3. For each child table, samples FK values from the parent table's primary key column
4. Respects cardinality constraints: a `NOT NULL` FK always gets a valid reference; an optional FK gets `NULL` with configurable probability

### Distribution Control

Real databases are not uniformly distributed. The generation config controls:

- **Cardinality**: How many child rows per parent (e.g., 1--30 encounters per patient, following a power-law distribution)
- **Null ratio**: What fraction of nullable columns contain NULL (typically 5--30% in real data)
- **Value entropy**: How many distinct values appear in categorical columns (a `status` column might have 3 values; a `diagnosis_code` column might have 500)
- **Skew**: Zipfian distributions for columns where a few values dominate (e.g., 80% of encounters are `status='closed'`)
- **Temporal patterns**: Dates that follow realistic patterns (weekday-heavy, seasonal, monotonically increasing)

## Diversity from Source Text

The curated text input drives diversity along two independent axes:

### Domain Diversity

Different passages produce different ontological domains, which produce structurally distinct databases:

| Source Domain | Example Tables | Distinctive Patterns |
|--------------|----------------|---------------------|
| Healthcare | patient, encounter, diagnosis, medication | ICD-10 codes, temporal encounter sequences |
| Finance | account, transaction, instrument, counterparty | IBAN/SWIFT codes, decimal precision, audit trails |
| Supply Chain | shipment, warehouse, item, carrier | GPS coordinates, weight/volume, tracking IDs |
| Education | student, course, enrollment, grade | GPA calculations, semester cycles |
| HR/Payroll | employee, department, payroll, benefit | SSN patterns, salary ranges, org hierarchies |

### Structural Diversity

Even within a single domain, different passages emphasize different relationships, producing varied schema structures:

- A passage about **emergency triage** produces schemas with acuity levels, wait times, and disposition tracking
- A passage about **chronic disease management** produces schemas with longitudinal encounters, medication histories, and care plans
- A passage about **hospital billing** produces schemas with insurance claims, procedure codes, and payment reconciliation

All three are "healthcare databases" but have substantially different table structures, column types, and relationship patterns. This structural diversity is what trains the model to generalize beyond surface patterns.

## Confusable Type Injection

A key training challenge is **confusable pairs** -- columns with nearly identical value distributions but different semantic types. The generation pipeline deliberately injects these:

| Confusable Pair | Value Pattern | Distinguishing Context |
|----------------|---------------|----------------------|
| Advertising ID vs GUID | Both UUIDv4 format | Table context (ad_events vs generic) |
| Bank account vs payment card | Both numeric strings | Length, check digit algorithm |
| Phone number vs fax number | Both `+1-XXX-XXX-XXXX` | Column name, co-occurring columns |
| ZIP code vs department code | Both 5-digit numbers | Geographic context vs org context |
| Patient ID vs provider ID | Both `XXX-NNNNN` format | Foreign key relationships |

By generating schemas where these confusable types coexist -- often in the same database -- the model learns to resolve ambiguity using cross-column and cross-table context rather than single-column pattern matching.

## Scale Arithmetic

Working through concrete numbers:

| Stage | Count | Basis |
|-------|-------|-------|
| FineWeb-Edu passages | ~500M | 1.3T tokens / ~2,600 tokens per passage |
| Ontology fragments | ~1--5 per passage | Domain-dependent entity density |
| Schemas per fragment | ~1--10 | Normalization and naming variation |
| Tables per schema | ~5--50 | Domain complexity |
| Rows per table | ~100--10,000 | Configurable per generation |
| **Total table instances** | **>10 billion** | Conservative lower bound |

The bottleneck is LLM inference for ontology extraction (Stage 1), not data generation. Once an ontology fragment exists, schema projection and data population are purely procedural and can run on commodity hardware at millions of tables per hour.
