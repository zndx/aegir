# Aegir's semantic engine: an authoritative reference

*Last updated 2026-07-09. This document describes the current
operational state of Aegir's semantic engine for external/advisory
audiences. The canonical, code-exact reference for every metric, band,
gate, and membrane is the [Authors Guide](./authors_guide.md); this
document summarizes the system and its empirical footing and does not
duplicate the guide's formulas.*

> **Revision note.** Earlier revisions of this page (through
> 2026-05-12) described a four-component verifier *R(O, I)* over
> (ontology composition, corpus) pairs and a GRPO/RLVR policy trained
> against it, with a 540-template catalog as the deliverable. That
> framing — the *Concept brief / RLVR* design (see
> [Concept brief](./concept_brief.md), [RLVR](./rlvr.md)) — is the
> **long-horizon Signals M4 apparatus**: its verifier *R(O, I)* is now
> **realized as the deterministic membrane stack** (HermiT/CCO,
> OntoClean, OQuaRE) and the **content-first derivation + rigor
> program** documented here and in the Authors Guide is the
> agent-mediated propose/dispose loop building and proving that reward
> today, in direct service of M4. The RLVR pages carry the M4 research
> design. *Structural note (flagged, not performed):* this page now
> overlaps substantially with the Authors Guide and the
> [Charter](./charter.md); a future editorial pass may want to merge or
> re-scope it.

## Abstract

Aegir's semantic engine is a **BFO 2020 / CCO-grounded** domain
ontology — the **Signals Data Governance (SDG)** ontology — that is
**content-derived from FinePDFs** and **realized to a HermiT-validated
OWL artifact**, together with the closed-loop pipeline that turns it
into an ontology-grounded synthetic pretraining corpus and a
relational DDL spine. The ontology's classes are **intermediate-depth
subsumers** that serve as the annotation vocabulary for Column Type /
Column Property Annotation (CTA/CPA) over wide relational tables.

Rigor is *enforced, not asserted*, through an **agent-mediated
propose / dispose feedback loop**: an engine proposes axioms; a stack
of deterministic membranes — a Manchester **parse** membrane, a
**HermiT** reasoning-authority membrane (with CCO imported so grounding
is checked against CCO's disjointness axioms), and an **OntoClean**
meta-property membrane — disposes and returns the reason, and the agent
responds. The two strongest membranes (HermiT and OntoClean) are
*un-fakeable*: an extension cannot talk its way past a contradiction or
an anti-rigidity violation.

The realized artifact lives at `corpora/ontology/sdg-ontology.{omn,owl}`
with a consistency certificate (`HERMIT_CERTIFICATE.md`). At the time
of writing it has **828 named classes (661 `sdg:` domain classes), 0
unsatisfiable classes, and 10,570 membrane-admitted individuals**
(instance-level consistency certified by decomposition), and clears
the pre-registered rigor objectives — definitional completeness
**0.575**, BFO-grounding **0.861**, realizable-machinery **35** — with
an OQuaRE aggregate of **4.19 (GREEN)**.

## 1. Background

The substantive engineering claim is that a bespoke ontology grounded
in upper-level formal foundations (BFO 2020 + CCO) produces verifiable
*cross-context cousining* — a relation between concepts from disparate
operational domains that share an upper-class ancestry — and that this
ontology can serve as a rigorous, content-grounded CTA/CPA annotation
vocabulary. The methodological claim is that **rigor can be enforced by
deterministic disposal membranes** rather than asserted: a generic LLM
recovers taxonomy and existential restrictions (structure) but not
definitional sufficiency, full grounding, or BFO role discipline
(rigor), and the membranes are precisely what hold the line on the
latter.

The distinctive position is *generative + content-grounded +
IOF-rigorous*: the ontology is derived from text (not expert-authored
like the IOF), yet the IOF-discipline layer is applied and measured on
top of it.

## 2. Related work

The combination — a content-derived OWL ontology disposed by an
un-fakeable reasoner/OntoClean membrane, used as a CTA/CPA annotation
vocabulary and to seed a verbalization-grounded pretraining corpus —
sits at the intersection of several lines of prior work. Each
component has prior art on adjacent artifacts; the combination is the
gap.

**KELM / TEKGEN** [Agarwal et al. 2021]. The closest
verbalization-corpus recipe: verbalize a structured knowledge source
into natural-language sentences, integrate as a model training corpus,
measure downstream effect. KELM verbalizes ABox triples from Wikidata;
the present system verbalizes TBox axioms from a bespoke OWL ontology
into a byte-level pretraining slice. The structural content differs
(taxonomic and logical class expressions vs. instance-level facts) and
the integration is a pretrain mix rather than a retrieval corpus.

**OLLM** [Lo et al. 2024]. The closest end-to-end LLM ontology
generation approach: fine-tunes an LLM with a custom regulariser to
produce taxonomic backbones from scratch. The present system derives
classes from FinePDFs content and **disposes** them with a reasoner +
OntoClean membrane, emitting full OWL with restrictions and
`equivalentClass` genus-differentia definitions rather than taxonomic
backbones only.

**AutoGraph-R1** [Tsang et al. 2026] and **K2V** [Yuan et al. 2026].
The closest RL-trained graph-emitting and RLVR-with-KG-derived-reward
methods. Both establish that verifiable structured-output shaping
works; the present system's disposal authority is an *intrinsic,
semantically grounded* DL reasoner (HermiT over the realized ontology
with CCO imported) plus reasoner-invisible OntoClean checks, rather
than an extrinsic LLM-judge or QA-accuracy signal. OWL's class-axiom
expressivity is what makes the DL-reasoner membrane possible.

**OntoTune** [Liu et al. 2025] and **Zaitoun, Sagi, Peleg** [AAAI
Symposium 2024] and **OnT** [Yang et al. 2024]. Adjacent LLM
ontology / OWL-verbalization work — iterative ontology-grounded SFT,
LLM-assisted axiom verbalization for SFT pairs, and TBox-axiom-aware
embedding training respectively. The present system treats
verbalizations as a flat self-supervised byte-level corpus mixed with
general pretraining text, and treats the ontology itself as a disposed,
publishable artifact.

Secondary methodological precedents — symbolic RL on EL++ concepts,
JSON-Schema / execution-validator RL, retrieval-augmented graph
reasoning — are listed in the project's lit-review document and not
elaborated here.

## 3. The Signals Data Governance ontology

SDG is a bespoke OWL ontology designed to underwrite metadata tagging
across operational contexts that enterprise data-governance teams
ordinarily treat as separate disciplines: LIMS sample tagging,
MBSE/SysMLv2 system design, database metadata governance, kernel-trace
observability, and PROV-O / OpenLineage data lineage. It is grounded
in BFO 2020 and CCO and organized into five top-level branches plus a
belief branch:

- **Artifact** (CCO) — material things, datasets, programs.
- **Designative ICE** (CCO) — names, identifiers, designators.
- **Descriptive ICE** (CCO) — measurements, claims, lineage records;
  hosts the `sdg:BeliefStructure` (DST) branch.
- **Prescriptive ICE** (CCO; alias of `cco:ont00000965` "Prescriptive
  ICE") — requirements, controls, policies, constraints.
- **Process** (BFO 2020 `bfo:0000015`) — observation, derivation,
  governance activity.

The branches are *cross-cousined*: every domain context contributes
classes to multiple branches, anchored at shared upper-level parents.
This is the load-bearing architectural invariant — the ontology is
forced to express cross-context concepts as shared subclasses of common
BFO/CCO ancestors rather than as discipline-specific aliases for the
same real-world entity. (The full committed branch structure and
external-standard anchors are in the [Charter](./charter.md).) The
branch commitment dates to the v0.1 seed era; the fully-derived
catalog now populates the same upper anchors — at the time of writing
106 classes anchor at `bfo:0000015` (process), 68 at `bfo:0000023`
(role), and the ICE branches carry the record/measurement/directive
classes.

### 3.1 Classes as the annotation vocabulary

The purpose reframes what most of the classes *are*: not leaf terms but
**intermediate-depth subsumers** — the property-bearing classes a
heterogeneous-but-coherent column belongs to. A
`driver_stops_schedule.stops_addresses` column holds a mix (origin +
destination, residential + business shipping addresses, each bearing an
avg-time-on-site); no leaf type fits — the right annotation is the
*least common subsumer that is still property-bearing*, e.g.
`Address ⊓ ∃has-shipping-role ⊓ ∃avg-time-on-site`. **Defining these
intermediate classes well *is* building the annotation vocabulary**,
and the rigor gates exist to keep every term a coherent, grounded
annotation target.

### 3.2 Content-first derivation

The ontology is **fully derived from FinePDFs**, concept-filtered by a
ColBERT/Qdrant MaxSim aperture over a SKOS index
(`scripts/derive_ontology.py::_apply_domain_filter`,
`src/aegir/ontology/domain_index.py`). The live catalog is
`src/aegir/ontology/catalog/catalog.json` — 433 templates at the time
of writing, every one accreted by the derive→promote loop (the
hand-authored seed families are retired). The pipeline is: text → the
engine derives axiom-pattern-bound primitives
(`src/aegir/ontology/patterns.py`, tiers `fhir_minimum` / `owl2_core`
/ `odp` / `sysmlv2`), staged in `catalog.candidate.json` →
grounding-anchor retrieval supplies a real genus → the disposal
membranes admit or reject → `scripts/promote_candidates.py` threads
the derivation provenance (`pattern` / `tier` / `grounds_ddl` /
`domain` / `source_span`) into the catalog — not a fixed template
count. Classes are authored as Manchester-syntax catalog templates
with a typed slot DSL that the realizer renders, grounds, and
validates into the OWL artifact
(`scripts/build_realized_ontology.py`).

### 3.3 Cross-context cousining — concrete instances

Cousining is verifiable directly from the live catalog — the derived
templates carry a `domain` provenance tag, and templates from
different SKOS domains share upper anchors. Representative instances
(template ids from `catalog.json`):

- `bfo:0000015` (BFO 2020 `bfo:0000015`) is shared by
  `clinical_documentation_activity` (clinical),
  `vector_control_activity` (public health), `lecture_session`
  (education), `pdf_export_process` (document systems), and
  `specification_testing_event` (engineering) — 106 realized classes
  anchor here.
- `bfo:0000023` (`bfo:0000023`) is shared by `data_controller_role`
  (data governance), `police_officer_role` (public administration),
  `investor_relations_director_role` (finance), and
  `library_student_assistant_role` (education) — 68 classes; the
  role/kind discipline is what OntoClean enforces.
- `cco:ont00000958` and `cco:ont00000853` are shared
  by `personal_health_record` (clinical),
  `donation_transaction_record` (nonprofit finance),
  `sequencing_coverage_parameter` (genomics), and
  `small_group_surcharge_rule` (insurance) — records, measurements,
  and rules from disparate domains land at the same ICE ancestors
  rather than as discipline-specific aliases.

### 3.4 Atelier ↔ Aegir state-fusion via DST

The `sdg:BeliefStructure` primitives — `sdg:MassFunction`,
`sdg:BeliefInterval`, `sdg:Evidence`, `sdg:Claim` — were the seed-era
commitment (Charter Q1) to a shared structural vocabulary for
Dempster-Shafer evidence-fusion pipelines: the Aegir agent-swarm
state-fusion layer consuming belief structures emitted by **Atelier**
(a sibling project providing DST-based evidence fusion for enterprise
data-classification) using these names directly and without
translation. The belief branch was retired with the hand-authored
seed families and is **not in the current realized artifact**; its
reinstatement is a standing publish rider (the DST belief module,
#136, named in `src/aegir/flows/sdg_corpora_flow.py`). The design
intent stands: the cousining is at the explicit-uncertainty layer,
one `Evidence → Claim → BeliefInterval` triplet covering quality-tier
evidence, audit findings, lineage-edge plausibility, and column-tag
claims at calibrated confidence levels.

## 4. Rigor — the metric suite and the publish gate

Rigor is measured by `scripts/ontology_metrology.py::compute()` (pure
rdflib, JVM-free, with CCO's subClassOf backbone merged so `cco:`-chains
resolve to BFO) and gated by `scripts/ontology_oquare.py`. The exact
formulas, bands, and characteristic mapping are in
[Authors Guide §§ 3–4](./authors_guide.md#3-the-quantitative-metrics);
this section gives the operational summary.

### 4.1 The metric families

- **IOF-derived rigor dimensions** (the discriminators a shallow author
  misses): `definitional_completeness` (fraction of classes defined
  with `EquivalentTo` genus+differentia), `bfo_grounded` (fraction
  whose subClassOf/≡-genus chain reaches a BFO IRI),
  `realizable_machinery` (count of BFO role/disposition/function
  restrictions), `def_annotation_coverage` (fraction carrying an
  `iao:0000115` / `rdfs:comment` / `skos:definition`).
- **Field-standard structural metrics** (OntoQA / OQuaRE): `rr`, `ir`,
  `ar`, `aronto`, `dit`, and `tm` (tangledness, inverted).
- **OntoClean taxonomic-correctness proxies** (reasoner-invisible yet
  checkable, hence un-gameable): `subsumption_cycles`,
  `ontoclean_violations`, `sibling_disjointness`, `orphan_rate`,
  `taxonomic_cleanliness`.

### 4.2 The OQuaRE publish gate

OQuaRE (Duque-Ramos et al. 2011) adapts ISO/IEC 25000 (SQuaRE) to
ontologies: each metric is normalized to **[1,5]** against fixed,
IOF-anchored bands, aggregated into six characteristics (Structural,
FunctionalAdequacy, Reliability, Operability, Maintainability,
Transferability) and one holistic score. The **gate is GREEN only when
all three hold**: `oquare_aggregate ≥ 3.5`, `functional_adequacy ≥ 3.0`,
and `hermit_consistent == true`. The `FunctionalAdequacy ≥ 3.0` floor
is deliberate — it forces definitional rigor and BFO discipline, not
structural/grounding gains alone. The gate is wired **HARD** into
`aegir.lineup.sync._gate()`: a `sync --push` of the ontology Data
Product is refused below GREEN, so a regression cannot publish. The AIM
is **3.9**, the published OQuaRE class of Brick (3.93) /
RealEstateCore (3.91).

### 4.3 Current state

Verified against the realized artifact
(`corpora/ontology/sdg-ontology.owl` + `HERMIT_CERTIFICATE.md`,
metrology re-run 2026-07-09):

| metric | value | target |
|---|---|---|
| `definitional_completeness` | **0.575** | IOF ≈ 0.55 |
| `bfo_grounded` | **0.861** | 1.0 |
| `realizable_machinery` | **35** | IOF ≥ 14 |
| `def_annotation_coverage` | **0.877** | 1.0 |
| unsatisfiable classes | **0** | 0 (hard) |
| `subsumption_cycles` / `ontoclean_violations` | **0 / 0** | 0 (hard) |
| OQuaRE FunctionalAdequacy | **4.63** | ≥ 3.0 (floor) |
| OQuaRE aggregate | **4.19 (GREEN)** | ≥ 3.5 (floor), AIM 3.9 |

**OQ-Rigor** (`definitional_completeness ≥ 0.45` ∧
`realizable_machinery > 0`) is met outright. **OQ-Structure**'s
grounding and annotation floors (`bfo_grounded ≥ 0.95`,
`def_annotation_coverage ≥ 0.90`) sit slightly below after the
catalog's growth to 661 domain classes — the active levers, while
`ar > 0` and `oquare_aggregate ≥ 3.5` hold and the publish gate stays
GREEN. See `EVIDENCE.md` (repo root) for the full
ledger history.

### 4.4 The topic layer — measurement on the retrieval face

Since 2026-07-07 (phase-gated PASS,
[record](../roadmap/phase_gate_inverted_topic_layer.md)) the same
catalog terms are the corpus's **topic registry**: topics ≡
ontology-grounded concept anchors in qdrant (`sdg_topics` — the 29
SKOS domains plus all 433 live terms;
`src/aegir/ontology/topic_layer.py`). FinePDFs items
(anchor-proportional passage windows) associate with **one topic or
none** by hierarchical-margin ColBERT MaxSim, and every association is
content-addressed to the collection state that adjudicated it. The
instruments sit alongside OQuaRE/IOF as the retrieval-face gates:

- **τ\* = 0.1065** — the assignment gate on `rel_margin_h`,
  pre-registered as the (1−α) quantile under the shuffled-window null
  (α = 0.05, report-not-tune; lens-identity-scoped).
- **Alignment / margins** — `rel_margin_h` is the margin over the
  nearest *non-ancestor* competitor (a parent/child near-miss is not
  ambiguity); the unassigned mass is the error signal that drives
  definitional-rigor iteration on the **lexicon** side, never the
  input.
- **M1–M6, M8** (`src/aegir/ontology/annotation_membrane.py`) — the
  membranes over authored anchor surfaces (`alt_labels`,
  `scope_note`, definitions); **M8** is the durable rule: anchor
  surfaces are *positive-voice only* — definition-by-negation is an
  embedding anti-pattern.
- **M7 basin gate** (`topic_layer.basin_calibration`) — registry
  changes gated on the input side: a topic flags when its assigned
  mass explodes between pinned runs, with doc-concentration evidence
  attached; dispositions are manual and provenance-recorded.

Full-store state at the gate: 15,976 items over 453 docs, 1,629
aligned (10.2%), 175/462 topics hit, M7 PASS with zero flags. The
BERTopic-era instruments (`topic_alignment.py`, `T_I.pkl`,
`build_topic_model.py`) are deprecated — v0.3 reproducibility only;
the live successors are the topic layer and congruence (§6).

## 5. The disposal membranes

Proposed axioms pass through three membranes in order; each returns a
*reason*, so a failure is a repair instruction, not a dead end (this is
the agent-mediated feedback loop — a human author reads the same
reasons). Full detail in
[Authors Guide § 5](./authors_guide.md#5-the-disposal-membranes-what-rejects-your-extension-and-why).

1. **Parse membrane** (`evolve_rigor.validate_detailed`) — renders the
   axiom standalone and parses it under OWLAPI. Rejects malformed
   Manchester (uppercase prefixes, bare properties, undeclared
   entities, `#` comments).
2. **Reasoning-authority membrane**
   (`build_realized_ontology.consistency_check`) — imports CCO and runs
   HermiT, so grounding is validated against **CCO's disjointness
   axioms**. A class grounded to a CCO-disjoint or BFO-incompatible
   genus is unsatisfiable and rejected. *Un-fakeable.*
3. **OntoClean meta-property membrane**
   (`src/aegir/ontology/ontoclean.py`) — assigns Rigidity / Identity /
   Unity / Dependence and enforces that an **anti-rigid (role) property
   cannot subsume a rigid (kind) one**. Surfaces as
   `ontoclean_violations`. *Also un-fakeable* — reasoner-invisible yet
   checkable.

Grounding-anchor retrieval (`scripts/grounding_anchors.py`) lets the
agent ground proposals to real genera: the index spans CCO (1431
BFO-aligned classes), FHIR R5 (210 record types bridged to
`cco:ont00000958`), and **our own grounded classes** (it
accretes — each class grounded becomes a reusable anchor).

## 6. The closed-loop synthetic-data pipeline

The realized ontology drives a closed loop that converts organic input
corpora into a measured synthetic training corpus and a relational DDL
spine. **`just metaflow` is the entire pipeline**, idempotent per
input window (`src/aegir/flows/sdg_corpora_flow.py`): FinePDFs
passages are **harvested** cursor-windowed and content-hashed
(`scripts/harvest_domain_docs.py`), classified through the qdrant
ColBERT **aperture** (`src/aegir/ontology/domain_index.py`); the
engine **derives** pattern-bound primitives that the membranes dispose
and `promote_candidates` admits into the catalog; the catalog
**realizes** to the HermiT-certified OWL; the deterministic **DDL
spine** (`src/aegir/ontology/ddl.py`, `scripts/build_ddl_spine.py`)
realizes profiles from `grounds_ddl` (junction / star / normalized /
eav) into RI-true, polyglot-validated (Trino ∩ Spark) tables and
views; **dual-register prose** (natural ∥ semantic, thinking traces
retained) embeds the per-chapter constructs verbatim; and
**verify/measure** runs **congruence**
(`src/aegir/ontology/congruence.py` — input-window concepts ↔ chapter
concepts over the *same* ColBERT substrate, the BERTopic-era R_D
reborn), the sensitive-noun scan, naturalness norms, and shape EMD vs
SchemaPile. Each completed run emits one immutable run-zettel
(`src/aegir/lineup/zettel.py`) and the lineup re-projects
(`just kb-build`). The family simplicial complex is retired —
cross-entity FKs are the deriver's to *earn* through provenance,
never name-match-wired.

```d2
direction: down

input: FinePDFs stream\nharvest (content-hashed docs)\n+ qdrant ColBERT aperture {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"
}

ontology: SDG ontology\ncatalog.json (derive → promote)\nrealized + HermiT-certified OWL {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"
}

membranes: Disposal membranes\nparse → HermiT (CCO authority) → OntoClean\n+ OQuaRE publish gate {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"
}

constructs: DDL spine + verbalization\ngrounds_ddl → RI-true tables/views\n(Trino ∩ Spark) · slot-faithful frames {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"
}

output: Output corpus\ndual-register chapters (traces retained)\n+ released DDL spine {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
}

input -> ontology: content-first derivation
ontology -> membranes: propose / dispose
membranes -> ontology: reason (refine) {
  style.stroke-dash: 3
}
ontology -> constructs: TBox + provenance
constructs -> output: engine prose + materialized rows
output -> input: congruence + topic layer\n(same ColBERT substrate) {
  style.stroke-dash: 3
  style.stroke: "#6a1b9a"
}
output -> input: byte-level pretraining slice\n(H-Net + RWKV-7) {
  style.stroke-dash: 5
  style.stroke: "#9e9e9e"
}
```

**Figure 1 — The Aegir closed-loop pipeline.** FinePDFs content
derives the catalog; the disposal membranes (and the OQuaRE publish
gate) admit only rigorous additions and return their reasons; the
catalog's provenance drives RI-true constructs and slot-faithful
verbalization frames that seed dual-register chapters; congruence and
the topic layer measure the output against the input window over the
same ColBERT substrate. The dashed gray arrow indicates the downstream
pretraining application (continued-pretraining augmentation on RWKV
World v3 — *Path A*).

## 7. Repository state and reproducibility

The realized artifact and its certificate are versioned in the
`corpora` submodule (zndx/sdg-corpora); the metrics are reproducible
from the committed `.owl` with the JVM-free metrology:

```
uv run --no-sync python scripts/ontology_metrology.py corpora/ontology/sdg-ontology.owl --json
uv run --no-sync python scripts/ontology_oquare.py corpora/ontology/sdg-ontology.owl \
    --certificate corpora/ontology/HERMIT_CERTIFICATE.md --json
```

Re-deriving / re-realizing (the JVM membranes) needs the
`LD_LIBRARY_PATH` bootstrap (DeepOnto/HermiT, see the project
`CLAUDE.md` ontology notes):

```
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py --strict-grounding
just check-ontology-schema      # TTL parses, labels/definitions present, BFO ancestry, SPARQL totality
```

Consistency is independently re-verifiable: load `sdg-ontology.omn` (or
`.owl`) in any OWL reasoner (Protégé/HermiT, ROBOT, owlready2) and check
consistency against the certificate (`isConsistent: true`, 0
unsatisfiable). The Aegir repository also contains the metrology and
OQuaRE gate, the OntoClean classifier, the grounding-anchor retriever,
the content-first derivation pipeline, the corpus generation +
verification + DDL-spine tooling, and the lineup KB that surfaces the
rigor metrics. External datasets (SchemaPile, FinePDFs, SOTAB v2,
GitTables, FineWeb-Edu) are obtained via documented download scripts
with stable public distributions.

## 8. Limitations and threats to validity

- **Grounding is strong but not complete.** `bfo_grounded` is 0.861
  and `def_annotation_coverage` 0.877 — both dipped below their
  OQ-Structure floors (0.95 / 0.90) as the catalog grew to 661 domain
  classes; `orphan_rate` is 0.139 and `sibling_disjointness` near 0.
  A residual fraction of classes still ground shallowly (a bare BFO
  category where a real CCO genus would be better). These are the
  active levers, not closed problems (`realizable_machinery`, at 35,
  now clears the IOF ≥ 14 aim).
- **Definitional rigor is at the IOF band, not far beyond it.**
  `definitional_completeness` (0.575) sits just past the IOF ≈ 0.55
  frontier; the AIM is 3.9-class and the IOF discipline beyond it.
  Raising it means defining more of the *referenced* intermediate
  classes, not just the heads.
- **The corpus's relational claim is not yet demonstrated at scale.**
  The ontology is *load-bearing for slot-type prediction (CPA)* but
  trades raw FinePDFs distribution alignment; whether the
  ontology-grounded mix lifts relational + Data-Element-elucidation
  skill over a no-ontology ablation at RWKV-7-matched scale is the
  pre-registered M2 / M3 gate (`EVIDENCE.md`), **UNTESTED**. Do not
  assume a downstream benchmark target without checking the current
  plan.
- **Content origin vs. expert authorship.** SDG is derived from
  FinePDFs, not expert-authored like the IOF. The distinctive position
  is *generative + content-grounded + IOF-rigorous*; the rigor is the
  IOF-discipline layer measured on a content-grounded ontology, and the
  membranes (not assertions) are what make that measurable claim hold.

## References

**Ontology engineering & quality.**

- Duque-Ramos, A., et al. (2011). *OQuaRE: A SQuaRE-based approach for
  evaluating the quality of ontologies.*
- Smith, B., et al. (2019). *Industrial Ontologies Foundry (IOF) / BFO
  signature.*
- Guarino, N., Welty, C. *An overview of OntoClean.*
- Arp, R., Smith, B., Spear, A. (2015). *Building Ontologies with Basic
  Formal Ontology.* MIT Press.
- Common Core Ontologies (CCO). github.com/CommonCoreOntology/CommonCoreOntologies. (CC0)
- HL7 FHIR R5.
- He, Y., Chen, J., Antonyrajah, D., Horrocks, I. (2023). *DeepOnto: A
  Python package for ontology engineering with deep learning.*

**Adjacent LLM ontology / verbalization / structured-RL work.**

- Lo, A., Jiang, A. Q., Li, W., Jamnik, M. (2024). *OLLM: Generating
  ontologies from texts.* NeurIPS 2024.
- Liu, et al. (2025). *OntoTune.* WWW 2025.
- Zaitoun, A., Sagi, T., Peleg, M. (2024). LLM-assisted verbalization of
  OWL axioms. AAAI Symposium Series 2024.
- Yang, Z., Chen, J., He, Y., Gao, F., Horrocks, I. (2024). *OnT —
  Language Models as Ontology Encoders.* arXiv:2507.14334.
- Agarwal, O., Ge, H., Shakeri, S., Aharoni, R. (2021). *Knowledge
  graph based synthetic corpus generation (KELM / TEKGEN).* NAACL 2021.
- Tsang, et al. (2026). *AutoGraph-R1.* arXiv:2510.15339.
- Yuan, et al. (2026). *K2V — Knowledge-to-Verification.*

**Internal references.**

- Aegir [Authors Guide](./authors_guide.md) — the canonical,
  code-exact reference for every metric, gate, and membrane.
- Aegir [Charter](./charter.md) — outward contract, provenance
  discipline, committed branch structure and external anchors.
- Aegir [Migration](./migration.md) — vocabulary authorship history.
- Aegir [Concept brief](./concept_brief.md) / [RLVR](./rlvr.md) — the
  research design for the long-horizon Signals M4 apparatus (the RLVR
  generator whose reward is now realized as the membrane stack).
