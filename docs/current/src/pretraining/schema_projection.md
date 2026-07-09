# Stage 2: Schema Projection

> **Deferred framing.** This page describes Stage 2 of an earlier
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

Schema projection transforms BFO-grounded ontology fragments into relational database schemas through a two-step process: first into SysMLv2 systems engineering models, then into programmatic data objects and SQL schemas. The intermediate SysMLv2 representation captures structural constraints, lifecycle semantics, and system-level relationships that flat entity-relationship modeling would lose.

## Why SysMLv2 as Intermediate Representation

Using SysMLv2 (OMG, approved July 2025) as an intermediate representation between ontology and database schema is unconventional -- and deliberate. SysMLv2 provides formal constructs that bridge the gap between abstract ontological entities and concrete data structures:

| SysMLv2 Construct | Ontological Concept | Database Primitive |
|-------------------|--------------------|--------------------|
| Block Definition | Entity type | Table |
| Part Property | Composition | One-to-many FK |
| Reference Property | Association | Many-to-many junction table |
| Port | Interface/boundary | Shared column (FK target) |
| Attribute | Data property | Column |
| Constraint | Axiom | CHECK constraint |
| State Machine | Lifecycle | Status enum + temporal columns |
| Requirement | Validation rule | Application-level validation |

The [openCAESAR](https://www.opencaesar.io/) project provides an OWL2-DL ontology for SysMLv2, making the ontology-to-SysMLv2 projection formally well-defined. This means we're not hand-waving the transformation -- there's a rigorous mapping from BFO-grounded classes and relations to SysMLv2 blocks and connections.

The critical advantage: SysMLv2 models encode **systems** with internal structure, constraints, and lifecycle semantics. The generated databases aren't just flat tables with columns -- they're projections of coherent systems where referential integrity, state transitions, and constraint propagation all have formal justification in the source model.

## Projection Pipeline

```d2
direction: right

ontology: BFO Ontology Fragment {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  class_list: "Classes\n─────────\nPatient ⊑ Object\nEncounter ⊑ Process\nDiagnosis ⊑ GDC\nProvider ⊑ Role"

  relation_list: "Relations\n─────────\nhasEncounter: Patient → Encounter\nhasDiagnosis: Encounter → Diagnosis\nassignedTo: Encounter → Provider"

  axioms: "Axioms\n─────────\nEncounter.status ∈ {active, closed}\nDiagnosis.code matches ICD-10"
}

sysml: SysMLv2 Model {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  blocks: "Block Definitions\n─────────\npart def Patient { ... }\npart def Encounter { ... }\npart def Diagnosis { ... }"

  ports: "Ports & Flows\n─────────\nport patientPort : ~PatientIF\nport encounterPort : ~EncounterIF"

  states: "State Machines\n─────────\nstate def EncounterLifecycle {\n  entry; active; closed\n}"
}

code: Programmatic Objects {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  dataclasses: "Python Dataclasses\n─────────\n@dataclass\nclass Patient:\n  patient_id: str\n  date_of_birth: date\n  ..."

  schema: "SQLAlchemy Models\n─────────\nclass PatientTable(Base):\n  __tablename__ = 'patient'\n  patient_id = Column(...)\n  ..."
}

ontology -> sysml: "openCAESAR\nOWL2-DL mapping" {
  style.stroke: "#6a1b9a"
}
sysml -> code: "code generation\n(Jinja templates)" {
  style.stroke: "#1565c0"
}
```

### Step 1: Ontology → SysMLv2

Each BFO class maps to a SysMLv2 construct based on its upper-level category:

- **BFO:Object** → `part def` (a concrete block with owned parts)
- **BFO:Process** → `action def` with a state machine (lifecycle semantics)
- **BFO:Role** → `port def` (an interface that objects can fulfill)
- **BFO:Quality** → `attribute def` (a typed value property)
- **BFO:GDC (Generically Dependent Continuant)** → `part def` with `subsets informationEntity` (a record or document)

Relations map to SysMLv2 connections:
- Composition (whole-part) → `part` usage within a block
- Association → `ref` usage with multiplicity
- Participation (Object in Process) → `perform` action usage

Axioms map to `constraint def` blocks with OCL-like expressions.

### Step 2: SysMLv2 → Programmatic Objects

The SysMLv2 model is projected into Python dataclasses via template-based code generation:

```python
@dataclass
class Patient:
    patient_id: str          # from attribute def
    date_of_birth: date      # from attribute def
    gender: str              # from attribute def
    encounters: list         # from part usage (1..*)

@dataclass
class Encounter:
    encounter_id: str        # generated primary key
    patient_id: str          # from owning block (FK)
    encounter_date: datetime # from attribute def
    status: str              # from state machine states
    provider_id: str         # from ref usage (FK)
    diagnoses: list          # from part usage (1..*)

@dataclass
class Diagnosis:
    diagnosis_id: str        # generated primary key
    encounter_id: str        # from owning block (FK)
    code: str                # from attribute def
    description: str         # from attribute def
    coded_by: str            # from ref usage (FK)
```

### Step 3: Data Objects → Relational Schema

The dataclasses are mapped to SQLAlchemy models and CREATE TABLE statements:

```sql
CREATE TABLE patient (
    patient_id    VARCHAR(36) PRIMARY KEY,
    date_of_birth DATE NOT NULL,
    gender        VARCHAR(10) NOT NULL
);

CREATE TABLE encounter (
    encounter_id   VARCHAR(36) PRIMARY KEY,
    patient_id     VARCHAR(36) NOT NULL REFERENCES patient(patient_id),
    encounter_date TIMESTAMP NOT NULL,
    status         VARCHAR(20) NOT NULL CHECK (status IN ('active', 'closed')),
    provider_id    VARCHAR(36) NOT NULL REFERENCES provider(provider_id)
);

CREATE TABLE diagnosis (
    diagnosis_id VARCHAR(36) PRIMARY KEY,
    encounter_id VARCHAR(36) NOT NULL REFERENCES encounter(encounter_id),
    code         VARCHAR(10) NOT NULL,
    description  TEXT,
    coded_by     VARCHAR(36) REFERENCES provider(provider_id)
);
```

## Ontological Mapping Rules

The projection preserves ontological structure through systematic rules:

| Ontological Structure | Relational Mapping | Provenance Preserved |
|----------------------|-------------------|---------------------|
| Entity type | Table | Table name ↔ BFO class |
| Data property | Column | Column name ↔ property IRI |
| Object property (1:N) | Foreign key | FK ↔ relation IRI |
| Object property (M:N) | Junction table | Junction ↔ relation IRI |
| Subsumption hierarchy | Table-per-type inheritance | Parent FK ↔ `rdfs:subClassOf` |
| Disjointness axiom | CHECK constraint | Constraint ↔ axiom |
| Cardinality constraint | NOT NULL / UNIQUE | Column constraint ↔ cardinality |

The critical property is that **every schema element traces back to a specific ontological element**. This traceability is what makes the training objective possible: when the model predicts that two columns belong to the same data element, we can verify that prediction against the source ontology.

## Schema Variation

A single ontology fragment can produce multiple valid database schemas through controlled variation:

- **Normalization level**: 1NF, 2NF, 3NF, or fully denormalized
- **Inheritance strategy**: Table-per-type, table-per-hierarchy, or single-table with discriminator
- **Naming conventions**: `snake_case`, `camelCase`, abbreviated, or obfuscated (`col_1`, `field_a`)
- **Type mappings**: `DATE` vs `VARCHAR` for dates, `INTEGER` vs `VARCHAR` for codes

This variation is essential for training robustness. Real-world databases use all of these conventions, often mixed within a single schema. By generating diverse schemas from the same ontological source, the model learns to recognize semantic equivalence across surface-level variation.

## Formal Mapping

The schema projection is a function:

\\[
\pi: O \to \mathcal{S}
\\]

where \\(O = (C, R, A, \iota)\\) is an ontology fragment and \\(\mathcal{S} = \\{S_1, \ldots, S_k\\}\\) is a set of valid relational schemas. Each schema \\(S_i = (T, K, F, \Gamma)\\) consists of:

- \\(T = \\{t_1, \ldots, t_n\\}\\) -- tables, each with columns \\(\text{cols}(t_j)\\)
- \\(K\\) -- primary key constraints
- \\(F\\) -- foreign key constraints
- \\(\Gamma\\) -- CHECK constraints

The projection must satisfy:

\\[
\forall\, t \in T,\ \exists\, c \in C : \text{name}(t) \xleftarrow{\pi} c
\\]

\\[
\forall\, f \in F,\ \exists\, r \in R : f \xleftarrow{\pi} r
\\]

That is, every table traces to a class and every foreign key traces to a relation. This bidirectional traceability is the formal guarantee that makes ontological entity recovery a well-defined training objective.
