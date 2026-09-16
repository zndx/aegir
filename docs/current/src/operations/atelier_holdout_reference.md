# Ægir brief: holdout-target / train-reference sample pair

Atelier classifies a **holdout target** sample. NHSVM (and MaxSim
enrichment) must train on a **disjoint reference** whose BFO/CCO genus
IRIs cover that target. No shared collection slugs. When the reference
SKOS is a superset of the target’s SKOS (`skos_missing` empty),
`perfect_possible` is true and a DST run on the target can converge.

Accept gate (do not invent a second checker): Atelier
`atelier.sdg.pair.assert_pair(target_id, reference_id)` in
`/home/rch/local/src/zndx/atelier`.

## Identifiers

| | Value |
|--|--|
| Corpus pin | `b24ef9f606605dd7bb8877bddd2d9672c78282ad` (`external/sdg-corpora` in Atelier; same pin Ægir must emit against) |
| **Target** (holdout) | `sdg-corpora/b24ef9f60660_macbook` |
| Target dir | Atelier `build/sdg_sample/b24ef9f60660_macbook` |
| Target collections (**forbidden on the reference**) | `computational-material-record-044e3d9a`, `position-ac7bde24` |
| **Reference** (train) | `sdg-corpora/b24ef9f60660_reference` (emit this id) |
| Shape | both ids `sdg-corpora/<pin12>_<profile>` (`#AT.00000023.PAIRSHAPE`) |

Atelier can already *select* a genus-covering disjoint slice from the
same pin (`just sdg-sample-reference <target-id>`). That is **not**
enough for perfect NHSVM. Ægir must emit collections from **finepdfs +
SchemaPile** that realize the emit list below without using the two
target slugs.

## Required genus IRIs (`#AT.00000021.PAIRIRI`)

Reference `terms[].genus` union must cover:

```
bfo:0000002
bfo:0000003
bfo:0000004
bfo:0000015
bfo:0000016
bfo:0000023
bfo:0000040
cco:ont00000958
cco:ont00000995
```

(`bfo:0000003` is unmapped to a SKOS upper anchor on the position
collection; still required on the reference.)

Same-pin disjoint collections that already cover these IRIs (do not
reuse as the Ægir product; they are the genus-layer fallback only):
`drug-application-0614fd04`, `webinar-session-927f0888`,
`faculty-researcher-ad473424`, `inspection-event-9fdc6d58`.

## Emit list (`skos_missing` / vocabulary gaps)

Until a reference sample exists, treat the **target vocabulary_gaps**
as the emit list. After `assert_pair`, replace this list with live
`PairReport.skos_missing` (target SKOS codes absent from reference
`annotations.csv`).

**Entity gaps** (need SKOS terms + disjoint realizing collections):

- `computational-material-record-044e3d9a`: ComputationalMaterialRecord, ComputationalCode, MaterialSystem, MaterialProperty, ParsingPipeline, HighPerformanceComputerCenter, BigDataAnalyticsTool, ParsedDataStore, VisualizationService
- `position-ac7bde24`: Position, Employee, Department, CollectiveBargainingAgreement, PerformanceReview

**Column gaps** (need SKOS `SDG.DOM.*` or entity properties, not new
target tables):

`active_status`, `algorithm_type`, `approval_date`, `center_name`,
`checksum`, `chemical_formula`, `classification`, `code`, `code_name`,
`comments`, `confidence_interval`, `created_at`, `creation_date`,
`crystal_system`, `data_integrity_status`, `developer_group`,
`employment_status`, `end_date`, `end_date_time`, `endpoint_url`,
`established_date`, `expiration_date`, `file_format`, `file_size_bytes`,
`first_name`, `formation_energy`, `hire_date`, `last_name`,
`last_updated`, `lattice_parameter_a`, `lattice_parameter_b`,
`lattice_parameter_c`, `license_type`, `max_concurrent_users`,
`measurement_method`, `name`, `novel_materials_discovered`,
`numeric_value`, `operational_status`, `peak_performance_flops`,
`pipeline_name`, `property_name`, `rating`, `records_processed`,
`review_date`, `salary_range`, `service_name`, `space_group_number`,
`start_date`, `start_date_time`, `status`, `storage_size_g_b`,
`store_name`, `termination_date`, `title`, `tool_name`,
`total_compute_nodes`, `total_records`, `union_name`, `unit_of_measure`,
`updated_at`, `uptime_percentage`, `version`

Target SKOS that a reference **must include** (superset; 24 codes):

`SDG.ARTIFACT`, `SDG.DISPOSITION`, `SDG.DOM.BIRTHPLACE`,
`SDG.DOM.COUNTRY`, `SDG.DOM.CREATED_DATE`, `SDG.DOM.DATE_PUBLISHED`,
`SDG.DOM.EFFECTIVE_DATE`, `SDG.DOM.GEOGRAPHIC_PROVENANCE`,
`SDG.DOM.HOME`, `SDG.DOM.ISSUED_DATE`, `SDG.DOM.LOCATION`,
`SDG.DOM.PUBLISHED_ON_DATE`, `SDG.DOM.REGIONAL`, `SDG.DOM.SINCE`,
`SDG.DOM.STATE`, `SDG.DOM.TEMPORAL_DESCRIPTOR`, `SDG.GDC`, `SDG.GENERIC`,
`SDG.ICE`, `SDG.ICE.DESCRIPTIVE`, `SDG.INDEPENDENT_CONTINUANT`,
`SDG.MATERIAL_ENTITY`, `SDG.PROCESS`, `SDG.ROLE`

## Disjointness (`#AT.00000020.PAIROVERLAP`)

Reference collections **must not** include
`computational-material-record-044e3d9a` or `position-ac7bde24`.
No shared `manifest.collections[].slug`. Do not copy target CSVs into
the reference.

## How Atelier will `assert_pair`

```python
from atelier.sdg.pair import assert_pair
report = assert_pair(
    "sdg-corpora/b24ef9f60660_macbook",
    "sdg-corpora/b24ef9f60660_reference",
)
# report.perfect_possible  → DST on the target is allowed
# report.skos_missing      → emit list (must be empty for perfect NHSVM)
```

| Code | Fail when |
|------|-----------|
| `#AT.00000020.PAIROVERLAP` | same sample id, or shared collection slugs |
| `#AT.00000021.PAIRIRI` | target genus IRIs not ⊆ reference genera |
| `#AT.00000022.PAIRSKOS` | noted on the report when target SKOS ⊈ reference SKOS (not always raised; `perfect_possible` is false) |
| `#AT.00000023.PAIRSHAPE` | id is not `sdg-corpora/<pin>_<profile>` or dir missing |

Classification invoke after accept:

```
just classify-flow --host \
  --source-id=sdg-corpora/b24ef9f60660_macbook \
  --reference-id=sdg-corpora/b24ef9f60660_reference \
  --target=phase:nhsvm
```

NHSVM trains on the reference; the target is the holdout. Do not train
on the target. Pin must stay `b24ef9f6…`.

## Deliverable shape

A sample directory Atelier can load as
`sdg-corpora/b24ef9f60660_reference`: `manifest.json` (collections with
`slug` + `genera`), `annotations.csv`, `tables/*.csv`, RI-closed
bundles. Land under Atelier `build/sdg_sample/b24ef9f60660_reference`
or emit in sdg-corpora on this pin and let
`just sdg-sample-reference` pack it.

Sources: **finepdfs** (`scripts/emit_finepdfs_postings.py`) and
**SchemaPile** (`scripts/emit_schemapile_postings.py` /
`build_collections.py`). Do not use GitTables fixture lanes.

## Status (2026-09-16)

Reference sample is on disk. Accept gate:

```
perfect_possible True
skos_missing ()
```

| | |
|--|--|
| Sample dir | Atelier `build/sdg_sample/b24ef9f60660_reference` |
| Slugs (disjoint) | `staff-assignment-590faea4`, `crystal-assay-4fec7485` |
| SchemaPile | `sp:450157` (HR), `sp:033482` (recruiting), `sp:555914` (mzTab chemistry) |
| FinePDFs lineage | `fp:104813:51b4c0cd`, `fp:107423:8d197e5c`, `fp:81923:c7603e10` |
| Regenerate | `just emit-holdout-reference` (`scripts/emit_holdout_reference.py`) |

Live `PairReport.skos_missing` is empty. The emit list above is the
vocabulary that was driven (entity + column gaps authored as `SDG.DOM.*`
and realized on the two collections). Do not train on the target.
