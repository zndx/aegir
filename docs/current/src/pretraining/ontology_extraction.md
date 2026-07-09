# Stage 1: Ontology Extraction

> **Deferred framing.** This page describes Stage 1 of an earlier
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

The first stage transforms curated educational text into formal ontological structure. A large language model reads natural language passages and produces BFO-grounded ontology fragments -- typed entity hierarchies with properties, relationships, and axioms that can be mechanically projected into database schemas.

## Input: FineWeb PDFs Edu

The source corpus is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a curated subset of Common Crawl filtered for educational content using LLaMA-3-70B-Instruct quality scoring. Key properties:

- **1.3 trillion tokens** of curated, high-quality educational text
- Spans every domain: medicine, law, finance, engineering, social sciences, natural sciences
- Already deduplicated and quality-filtered -- no need for additional curation
- PDF-extracted passages preserve document structure (headings, tables, lists)

Each passage is a self-contained description of some real-world domain -- exactly the kind of text that contains implicit ontological structure waiting to be made explicit.

## Extraction Process

```d2
direction: right

passage: Text Passage {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"
  tooltip: "A passage from FineWeb-Edu describing a real-world domain"

  text: "Hospital emergency departments\ntriage patients based on acuity\nlevels. Each encounter records\nthe presenting complaint, vital\nsigns, assigned provider, and\ndisposition decision..."
}

llm: LLM Extraction {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  prompt: Structured Prompt\nDomain identification\nEntity extraction\nBFO alignment
  model: GLM-4.7 / GLM-5\nStructured output mode
  prompt -> model
}

output: Ontology Fragment {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  class_list: "Classes:\n  Patient ⊑ BFO:Object\n  Encounter ⊑ BFO:Process\n  Provider ⊑ BFO:Role\n  AcuityLevel ⊑ BFO:Quality"
  relation_list: "Relations:\n  hasEncounter(Patient, Encounter)\n  assignedTo(Encounter, Provider)\n  hasAcuity(Encounter, AcuityLevel)"
}

validate: Validation {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  parse: OWL Parse Check
  bfo: BFO Alignment Check
  coherence: Logical Coherence

  parse -> bfo -> coherence
}

passage -> llm: raw text
llm -> output: structured extraction
output -> validate: candidate ontology
validate -> output: accepted {
  style.stroke: "#2e7d32"
}
```

The extraction uses structured prompting with a three-phase approach:

1. **Domain identification**: Classify the passage into one or more information domains (healthcare, finance, logistics, etc.) to select domain-appropriate extraction templates.

2. **Entity extraction**: Identify entity types, their properties, and inter-entity relationships. The prompt constrains outputs to BFO-compatible categories.

3. **BFO alignment**: Map each extracted entity to the appropriate BFO upper-level category, ensuring the fragment inherits BFO's formal axioms.

### Validation Gate

Not every LLM output is usable. A validation gate checks three properties:

- **Syntactic**: Does the output parse as valid OWL/RDF?
- **BFO alignment**: Is every class properly subsumed by a BFO category?
- **Coherence**: Are there contradictory axioms or dangling references?

Fragments that fail validation are discarded or re-prompted. In practice, structured output modes (JSON schema enforcement) in GLM-4.7/GLM-5 achieve >90% first-pass validation rates.

## Why BFO

[Basic Formal Ontology](https://basic-formal-ontology.org/) (ISO/IEC 21838-2:2021) is the most widely adopted upper ontology in applied information science:

- **700+ ontology projects** built on BFO across government, defense, healthcare, and industry
- **Common Core Ontologies (CCO)** used by the U.S. Department of Defense and Intelligence Community
- **OBO Foundry** biomedical ontologies (Gene Ontology, ChEBI, etc.) all align to BFO
- Formal first-order logic axiomatization ensures machine-verifiable consistency

BFO provides the upper-level categories that give our extracted ontologies a shared formal backbone. Without this grounding, extracted ontologies would be ad-hoc entity lists with no guaranteed interoperability or logical structure.

### BFO Categories for Information Systems

The BFO categories most relevant to relational data modeling:

| BFO Category | IRI | Maps To | Example |
|-------------|-----|---------|---------|
| **Generically Dependent Continuant** | `BFO:0000031` | InformationEntity | A patient record, a diagnosis code |
| **Object** | `BFO:0000030` | Concrete entity | A patient, a medical device |
| **Quality** | `BFO:0000019` | Data attribute | Acuity level, sensitivity classification |
| **Role** | `BFO:0000023` | Functional role | Data subject, provider, auditor |
| **Process** | `BFO:0000015` | Temporal event | An encounter, a transaction, a review |
| **Specifically Dependent Continuant** | `BFO:0000020` | Inherent property | A patient's blood type, a device's serial number |

These categories constrain what kinds of entities can participate in what kinds of relationships -- a Patient (Object) can *bear* a DataSubjectRole (Role), an Encounter (Process) *has participant* a Patient (Object), and so on. These constraints propagate through the pipeline: they determine which foreign key relationships are valid in the generated schemas.

## Formal Definition

An ontology fragment is a tuple:

\\[
O = (C, R, A, \iota)
\\]

where:

- \\(C = \\{c_1, \ldots, c_n\\}\\) is a set of **classes** (entity types), each with a set of **properties** \\(P(c_i) = \\{p_1, \ldots, p_k\\}\\)
- \\(R = \\{r_1, \ldots, r_m\\}\\) is a set of **relations** between classes, each \\(r_j: c_a \to c_b\\) with cardinality constraints
- \\(A\\) is a set of **axioms** -- subsumption (\\(c_i \sqsubseteq c_j\\)), disjointness (\\(c_i \sqcap c_j = \bot\\)), and property constraints (domain, range, cardinality)
- \\(\iota: C \to \text{BFO}\\) is the **BFO alignment mapping** that assigns each class to a BFO upper-level category

The alignment mapping \\(\iota\\) must satisfy BFO's axioms: if \\(\iota(c_i) = \text{BFO:Process}\\), then \\(c_i\\) inherits Process axioms (has temporal extent, can have participants, etc.). This is not merely a label -- it constrains the valid relationships and properties that \\(c_i\\) can participate in.

## Output

Each successfully validated ontology fragment becomes input to [Stage 2: Schema Projection](./schema_projection.md). A single text passage typically yields 1--5 fragments, depending on the complexity and domain diversity of the passage content.

The ontology fragments are serialized as OWL/RDF for archival and as structured JSON for downstream processing. Both representations preserve the full BFO alignment mapping, enabling validation at every subsequent stage.
