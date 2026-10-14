# Augmented sdg-corpora for thin SKOS terms

2026-10-14. Atelier withdrew the stub relational generators in
`src/atelier/classify/insitu_extensions.py` (523 `lambda rng` tables,
commit `2ad6b46` on Atelier, removed in the withdrawal that accompanies
this memo). Each lambda emitted one table of 2–5 columns (mean 2.59):
exactly one column carried a SKOS IRI, and the other 833 columns were
siblings. Cell values came from `rng.choice` and `rng.randint`. Those
tables do not follow a published collection's structure or its value
distribution, and a model trained on them attaches that noise to the tag.

Augmented tables are new sdg-corpora collections from the ontology
projection already in this repo. Atelier will retarget its
`external/sdg-corpora` gitlink to the commit that publishes them.
Atelier will not synthesize tables, and it does not import the `aegir`
package.

Measured on the published pin `510db0b89e41b2cb564797ed27d9c1b25afbc969`
(`origin/trunk` of sdg-corpora on 2026-10-14):

| | |
|--|--|
| Published columns | 26,446, all with a `reference_code` |
| Collections | 453 (363 train / 45 test / 45 validate under Atelier's genus split) |
| Collection reference codes | 139 |
| Lambdas whose code is a collection reference code | 74, covering 785 columns |
| Lambdas for a code that is not a collection reference code | 449 |
| Order-1 neighborhood labels | 114, 0 unlabeled columns |
| Genus labels (`genus_term`) | 60, none of them `SDG.GENERIC` |
| Columns on the 26 genus labels | 14,202 |
| Columns on the four parent roots `PROCESS`, `INDEPENDENT_CONTINUANT`, `ROLE`, `GDC` | 7,142 + 2,136 + 1,465 + 62 = 10,805 |

No lambda existed for any of those 26 genus labels or for those four
roots. The mass of the corpus was uncovered, and the lambdas that did
exist were the wrong shape.

## Pipeline

Use `SdgCorporaFlow` (`src/aegir/flows/sdg_corpora_flow.py`). The step
order is fixed:

`start` → `harvest` → `derive` → `refine_escalations` → `realize` →
`ddl_stage` → `build_constructs` → `prose_natural` ∥ `prose_semantic` →
`join_verify` → `congruence` → `assemble` → `project` → `end`.

What each stage already does, and what this augmentation relies on:

- `realize` runs `scripts/realize_sdg.py`. HermiT certifies the merge.
  Kvasir emits DDL and SHACL. `realize_repairs` stays off. An
  unsatisfiable theory stops the flow. A budget trip surfaces
  recommendations and stops. It does not rewrite the ontology in place.
- `build_constructs` calls `aegir.ontology.rows.require_gittables`.
  Missing or degenerate GitTables profiles halt. Placeholder cells are
  not a fallback. SQL types in the release are inferred from the real
  cells by `scripts/project_corpus.py` (`_infer_sql_type`).
- `project` runs `python -m aegir.lineup build` and
  `aegir.governance.provenance.emit_corpus_lineage`. That step writes
  the lineup KB and the lineage record. It does not write
  `corpus/collections/`.
- Collection files are written afterwards by
  `scripts/project_corpus.py`. One construct is one collection:

```
corpus/collections/<slug>/
  README.md
  chapters/natural.md
  chapters/semantic.md
  tables/<name>.sql
  tables/<name>.csv
  terms.md
  manifest.json
```

  The slug is `{kebab(entity name)}-{construct_id[:8]}`
  (`_slug_for`). Values are GitTables particulars. The passage is
  stored as a content-hash basename.
- The answer key is `scripts/emit_column_answer_key.py`. It writes
  `corpora/corpus/tagging/answer_key.csv`, `answer_key.parquet`, and
  `summary.json`. `build_rows` walks every collection directory and
  requires `constructs/<construct_id>.json` under the run dir. The
  default run snapshot is
  `/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06`.
- Publish stays behind `aegir.lineup sync` (OQuaRE) and the P5 riders.
  The flow does not push. The augmented corpus is a new sdg-corpora
  commit.

`scripts/build_collections.py` and `scripts/relational_collections.py`
are the vestigial topic-era writers. `project_corpus.py` replaces
them. This augmentation does not extend those scripts.

## Add collections; keep the published 453

`project_corpus.py` currently `shutil.rmtree`s `--out` and rewrites it
from one run (`main`, the block marked as a wholesale replace).
Pointing that at `corpora/corpus/collections` deletes the published
tree unless `constructs/` still contains every published construct.

The run directory used for the new key must contain the corpus-v06
construct and entity JSON for every published `construct_id`, plus the
new constructs. `build_rows` exits when a published collection's
construct file is absent.

Change `project_corpus.py` so an augment write does not delete
published directories. Seed `_slug_for`'s `taken` set with the slugs
already on disk. A new construct whose `{kebab}-{cid[:8]}` collides
takes the existing `-2` suffix. Copy only the new slugs into
`corpora/corpus/collections/`. Then run `emit_column_answer_key.py`
over that union.

`corpus_mode` and `--output-dir` already keep existing chapters when
the flow accretes. `entities-from` skips `derive` and copies a prior
`entities/` directory. Either path is valid when the constructs that
come out of it still go through `realize` (repairs off), `ddl_stage`,
`build_constructs` (`require_gittables`), and the two prose registers.
A supplemental entity set that skips `require_gittables` is not a
release.

## Which terms need a second collection

Atelier's split keeps a whole collection on one side. A label that
exists in one collection has no unseen collection. A second collection
for that label is the augmentation. One new construct may realize
several of these terms. A term that still has no construct still has
no column. Coined codes are not filled in.

Labels below are order-1 `neighborhood_term` of `reference_code` on
pin `510db0b8`. "Side" is where Atelier's current genus split placed
the only collection. Column counts are published columns carrying that
label.

### 59 neighborhood labels with one collection (285 columns)

| Label | Columns | Side | Kind |
|--|--:|--|--|
| `SDG.ARTIFACT.INSCRIBED_RECORD` | 10 | train | genus |
| `SDG.ARTIFACT.MATERIAL_EQUIPMENT` | 7 | train | genus |
| `SDG.ARTIFACT.PURPOSE_FACILITY` | 11 | train | genus |
| `SDG.ARTIFACT.ROUTE_STOP` | 6 | train | genus |
| `SDG.DOM.ANNOTATION` | 1 | train | child |
| `SDG.DOM.CAMPUS_EVENT` | 4 | train | child |
| `SDG.DOM.COMPLIANCE_FRAMEWORK` | 1 | train | genus |
| `SDG.DOM.CONFIDENCE` | 1 | train | child |
| `SDG.DOM.DATASET` | 6 | train | child |
| `SDG.DOM.FINDINGS` | 1 | train | child |
| `SDG.DOM.ISSUED_DATE` | 1 | train | child |
| `SDG.DOM.LINEAGE` | 1 | train | child |
| `SDG.DOM.MANDATORY` | 1 | train | child |
| `SDG.DOM.MENTAL_HEALTH_EDUCATION_PROGRAM` | 7 | validate | child |
| `SDG.DOM.ORGANIZER` | 1 | test | child |
| `SDG.DOM.PRIORITY` | 1 | train | child |
| `SDG.DOM.SIZE_BYTES` | 1 | validate | child |
| `SDG.GENERIC.RESEARCH_ACTIVITY` | 4 | train | other |
| `SDG.ICE.DESCRIPTIVE.ACTIVITY_RECORD` | 13 | train | genus |
| `SDG.ICE.DESCRIPTIVE.ASSESSMENT_STANDARD` | 3 | test | child |
| `SDG.ICE.DESCRIPTIVE.CULTURE_DIMENSION` | 8 | train | genus |
| `SDG.ICE.DESCRIPTIVE.DECISION_SUBJECT` | 9 | train | genus |
| `SDG.ICE.DESCRIPTIVE.PRICE_SCHEDULE` | 8 | train | genus |
| `SDG.ICE.DESCRIPTIVE.SIGNIFICANCE_LEVEL` | 1 | train | child |
| `SDG.ICE.DESCRIPTIVE.TEST_STATISTIC` | 4 | train | child |
| `SDG.ICE.DESIGNATIVE.CLIENT_PROFILE` | 3 | train | child |
| `SDG.ICE.DESIGNATIVE.CONFERRED_RECOGNITION` | 11 | train | genus |
| `SDG.ICE.DESIGNATIVE.SUBJECT_CONCEPT` | 1 | train | child |
| `SDG.ICE.DIRECTIVE.CONTRACTUAL_AGREEMENT` | 11 | train | genus |
| `SDG.ICE.DIRECTIVE.HEALTH_STANDARD_COMPLIANCE` | 1 | validate | child |
| `SDG.ICE.DIRECTIVE.REGULATORY_INSTRUMENT` | 6 | train | genus |
| `SDG.ICE.HEALTH_CARE_SUMMARY` | 4 | train | child |
| `SDG.ICE.MEMBERSHIP_TRANSFER_REQUEST` | 6 | train | child |
| `SDG.ICE.PERSONAL_DATA_RECORD` | 6 | train | child |
| `SDG.ICE.PREDICATE_CONCEPT` | 1 | train | child |
| `SDG.ICE.ROLE_PERCEPTION` | 3 | validate | child |
| `SDG.ICE.SOCIAL_TAG` | 4 | validate | child |
| `SDG.ICE.STAKEHOLDER_NEED` | 4 | train | child |
| `SDG.MATERIAL_ENTITY.BOARD_OF_EQUALIZATION` | 2 | train | child |
| `SDG.MATERIAL_ENTITY.COVERED_PERSON` | 14 | train | genus |
| `SDG.MATERIAL_ENTITY.EARTH_OBSERVATION_DATA` | 5 | validate | child |
| `SDG.MATERIAL_ENTITY.ENROLLED_PARTICIPANT` | 10 | train | genus |
| `SDG.MATERIAL_ENTITY.FUNCTIONAL_SPACE` | 9 | train | genus |
| `SDG.MATERIAL_ENTITY.HOUSING_CLIENT` | 2 | train | child |
| `SDG.MATERIAL_ENTITY.INSTITUTIONAL_INVESTOR` | 6 | train | child |
| `SDG.MATERIAL_ENTITY.INTERNAL_AUDITING_DIVISION` | 4 | train | child |
| `SDG.MATERIAL_ENTITY.LATERAL_RESTRAINT_STRAP` | 4 | train | child |
| `SDG.MATERIAL_ENTITY.LEGAL_TECH_SOLUTION_LAB` | 5 | train | child |
| `SDG.MATERIAL_ENTITY.NETWORK_INFRASTRUCTURE` | 6 | validate | child |
| `SDG.MATERIAL_ENTITY.NETWORK_RESOURCE` | 2 | train | child |
| `SDG.MATERIAL_ENTITY.NONPROFIT_ORGANIZATION` | 4 | train | child |
| `SDG.MATERIAL_ENTITY.PRIM_CHEMICAL` | 1 | train | child |
| `SDG.MATERIAL_ENTITY.PUBLIC_INFORMATION_OFFICER` | 2 | test | child |
| `SDG.MATERIAL_ENTITY.QUANTITATIVE_ANALYTIC_TOOL` | 4 | test | child |
| `SDG.MATERIAL_ENTITY.SERVICE_PROVIDER` | 5 | train | child |
| `SDG.MATERIAL_ENTITY.SERVICE_RECIPIENT` | 9 | train | genus |
| `SDG.PROCESS.SCHEDULED_ACT` | 7 | train | genus |
| `SDG.PROCESS.USE_SESSION` | 5 | train | genus |
| `SDG.ROLE.INSTRUCTIONAL_ASSISTANT` | 7 | train | genus |

Eleven of those children have no train column today because their only
collection sits on test or validate: `MENTAL_HEALTH_EDUCATION_PROGRAM`,
`ORGANIZER`, `SIZE_BYTES`, `ASSESSMENT_STANDARD`,
`HEALTH_STANDARD_COMPLIANCE`, `ROLE_PERCEPTION`, `SOCIAL_TAG`,
`EARTH_OBSERVATION_DATA`, `NETWORK_INFRASTRUCTURE`,
`PUBLIC_INFORMATION_OFFICER`, `QUANTITATIVE_ANALYTIC_TOOL` (37
columns). A second collection is what lets Atelier put one bundle on
train.

### Six genus labels with one collection

These are `genus_term` labels. Their columns already lift onto a
neighborhood label that appears in other collections, so they are not
in the table above. The genus split still has no second collection:

| Genus | Columns | Side |
|--|--:|--|
| `SDG.DOM.DATASET_ENTITY` | 6 | train |
| `SDG.DOM.EMPIRICAL_RECORD` | 1 | train |
| `SDG.DOM.EPISTEMIC_UNCERTAINTY_METRIC` | 1 | train |
| `SDG.DOM.METADATA_DESCRIPTOR` | 1 | train |
| `SDG.DOM.PROVENANCE_RELATION` | 1 | train |
| `SDG.DOM.SCHEDULED_ACTIVITY` | 4 | train |

A new column whose `genus_term` is that code, in a new collection,
covers the genus. The emitter decides the code from the OWL record.

### Already spanning more than one collection

These twelve children have two or more collections and no test
collection under the current Atelier split. They are an Atelier
placement concern. They are not emit targets:

`SDG.DOM.AGENT`, `SDG.DOM.CREATED_DATE`, `SDG.DOM.OWNER`,
`SDG.ICE.INSPECTION_REPORT`, `SDG.ICE.PRODUCT_DISCLOSURE_STATEMENT`,
`SDG.ICE.QUALIFICATION_LEVEL`, `SDG.MATERIAL_ENTITY.CHILD`,
`SDG.MATERIAL_ENTITY.CLIENT`, `SDG.MATERIAL_ENTITY.FINANCIAL_PRODUCT`,
`SDG.MATERIAL_ENTITY.HEALTHCARE_ORGANIZATION`,
`SDG.MATERIAL_ENTITY.HIGHER_EDUCATION_INSTITUTION`,
`SDG.MATERIAL_ENTITY.LAW_ENFORCEMENT_AGENCY`.

Fat genera do not need more columns for count. `ARTIFACT`, `PROCESS`,
`MATERIAL_ENTITY`, `INDEPENDENT_CONTINUANT`, `ROLE`, and `ICE` already
have hundreds to thousands of train columns. Atelier's
`distribution_deficit` (about 389 train columns on the genus split,
mostly `PROCESS`) is a histogram residual. It is not the emit quota.

## Answer-key rules, unchanged

`scripts/emit_column_answer_key.py` already states them:

- `reference_code` is a code in `vocabulary/annotations.csv`.
- A coined `sdg:` genus walks `SubClassOf` to a BFO/CCO anchor
  (`genus-walk`, `domain-walk`).
- `SDG.GENERIC` is never written as a stand-in. A `SDG.GENERIC.*` code
  is written only when the recorded name is that concept and the genus
  entails nothing richer. Children of that bucket stay themselves.
- `kind-authored` applies only when the recorded genus has no anchor
  and `corpora/ontology/column_kinds.omn` covers the class. A class
  with neither an anchor nor an authored kind is an error.
- A leaf concept stays a leaf when the emitter already keeps it.
- A blind classifier must not read `corpus/tagging/`. The key is the
  scoring file. Training features are column name, SQL type, cell
  values, and sibling columns.

## Acceptance

On the new sdg-corpora commit:

1. Every slug in the 453 published collections is still present.
   For each of the 26,446 published rows at `510db0b8`,
   `(collection, table, column)` still maps to the same
   `reference_code`.
2. New slugs are disjoint from those 453. Each new slug is one
   construct: README, both chapters, `tables/*.sql`, `tables/*.csv`,
   `terms.md`, `manifest.json`. SQL types come from the cells. Cells
   are GitTables particulars with provenance.
3. Each of the 59 neighborhood labels above occurs in at least two
   collections. Each of the six genus labels above occurs as
   `genus_term(reference_code)` in at least two collections.
4. `uv run python scripts/emit_column_answer_key.py --check` passes.
   With `--run`, the run directory contains a construct file for every
   published collection, including the new ones.
5. No new row uses `SDG.GENERIC` as a stand-in. Every new
   `reference_code` is in `vocabulary/annotations.csv`.

Atelier will move the gitlink after that commit. Atelier's
`align_genus_split` assigns sides. The answer key has no split column.

## Out of scope

- Porting Atelier's `insitu_extensions.py` lambdas, the Qwen generator
  harness, or the 523 IRI procedures into Aegir.
- A second generator language, Title Words placeholders
  (`Title Words 01`), or `rng.choice` token columns.
- Restratifying Atelier's train/test/validate assignment.
- Fitting or promoting an NHSVM head. That stays in Atelier, after the
  pin moves.
- Importing Aegir from Atelier, or reading `corpus/tagging/` from a
  blind training pass.
