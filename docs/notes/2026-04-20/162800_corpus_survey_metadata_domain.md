# Corpus survey: toward a canonical research-grade dataset for Aegir's metadata domain

Survey of public corpora suitable for pretraining a byte-level LM
specialised to relational metadata: schema/DDL, SQL variants, query
engines (Spark/Flink/Trino/Presto), storage formats (Parquet/Avro/ORC),
data-governance ontologies (Schema.org, DBpedia, BFO/CCO), and the
surrounding educational and prose context that grounds them.

Source of this survey: HF API walk-through, availability and structure
check done 2026-04-20.

## Target recipe (one-line summary)

**Compose a ~20-50 GB byte-level corpus from ~6 named sources spanning
general high-quality prose (FinePDFs-Edu), code-centric data-systems
material (Stack v2 relevant-language slices + GitHub issues), SQL task
pairs (Spider, sql-create-context), ontology/vocabulary raw text
(Schema.org OWL, DBpedia dumps, BFO/CCO OWL), and synthetic augmentation
from our ontology-grounded pipeline (docs/src/pretraining.md).**

Rationale: breadth (prose) for general language fluency, depth (code +
schemas) for syntactic competence in our domain, annotation (SQL pairs
+ ontology) for grounding, synthetic for distribution control over
rare/long-tail data-governance patterns.

## What exists publicly (verified 2026-04-20)

### General prose baseline

| Source | HF repo | Size | Role |
|---|---|---|---|
| **FinePDFs** (full) | `HuggingFaceFW/finepdfs` | ~3575 shards, several TB raw | High-quality PDF-extracted prose, multilingual. Too large for direct use; sample needed. |
| **FineWeb-Edu** (sample-10BT) | `HuggingFaceFW/fineweb-edu` | ~30 GB text (we have 8 GB already) | Curated English educational text. Already downloaded for repro experiment. |
| **FineWeb-2** (multilingual) | `HuggingFaceFW/fineweb-2` | TB-scale | Multilingual successor; useful if we want non-English data element names. |
| **peS2o** (allenai) | `allenai/peS2o` | ~40 GB | Academic papers, good for schema/stat/governance concepts. |
| **Wikipedia** (current) | `wikimedia/wikipedia` | ~20 GB (en) | Grounded definitions of entity types, useful complement to DBpedia. |
| **Wiki-40B** | `wiki40b` | 40 languages | Sentence-level cleaned wikipedia. |

Primary pick: **FinePDFs-Edu sample subset + FineWeb-Edu** as the prose
backbone. Both are in the FineWeb family with known quality filters.
The EDU variant specifically biases toward educational content which
over-represents the scientific/technical register our domain lives in.

### Code corpora — data-systems language slices

**The Stack v2 dedup** (`bigcode/the-stack-v2-dedup`) is the cleanest
available code corpus. 759 total language-shard files; the slices
directly relevant to our domain:

| Language | Shards | Priority |
|---|---|---|
| **SQL** | 1 | essential |
| **PLpgSQL** (Postgres) | 1 | essential |
| **TSQL** (SQL Server) | 1 | essential |
| **PLSQL** (Oracle) | 1 | essential |
| **SQLPL** | 1 | essential |
| **HiveQL** | 1 | essential |
| **Python** | 6 | SQLAlchemy, pandas, PySpark are Python-first |
| **Scala** | 1 | Spark-native language |
| **Java** | 15 | Flink/Kafka/Hadoop lineage |
| **YAML** | 2 | dbt/Airflow DAG + schema files |
| **JSON** | 12 | Schema.org JSON-LD, Avro schemas, Kafka Connect configs |
| **XML** | 6 | Hadoop/Hive configs, DDL exports |
| **JSONLD** | 1 | schema.org structured data |
| **TOML** | 1 | Iceberg table metadata |
| **Go** | 1 | vitess, cockroach, various DB tools |
| **Dockerfile** | — | not verified; useful if present |

Most slices are one shard each (roughly 1-10 GB per shard). The large
slices (Java, JSON, Python) are multi-shard at 10-50+ GB each. A
domain-relevant slice totals ~50-100 GB uncompressed.

The dedup version is preferred over the full Stack v2 — removes
near-duplicate code that would give the model misleading pattern
frequency.

### Schema / SQL annotation pairs (grounding data)

| Source | Size | Role |
|---|---|---|
| **Spider** (`xlangai/spider`) | ~10 k examples | Complex NL→SQL with multi-table joins, 200 DBs. Gold standard for CTA-adjacent evaluation. |
| **sql-create-context** (`b-mc2/sql-create-context`) | 78 k examples | CREATE TABLE + question + SQL triples. Directly teaches schema-to-query binding. |
| **NSText2SQL** (`NumbersStation/NSText2SQL`) | 290 k examples | Text-to-SQL at scale, lower-quality than Spider but much broader. |
| **SchemaPile / WebTables** | not found on HF but academic datasets exist | Structured schema metadata from WDC. Requires academic fetch. |

These are SMALL relative to the prose/code backbone but highly
concentrated in our domain. Include all of them. Total size < 1 GB.

### Ontology + vocabulary raw sources

Not on HuggingFace, but trivially downloadable as OWL/RDF text from
ontology hosts. These give us the GROUND TRUTH vocabulary our model
should align to:

- **Schema.org** — `https://schema.org/docs/developers.html` publishes
  all-in-one JSON-LD, Turtle, RDF/XML. ~800 types + ~1400 properties.
  Trivial: ~3 MB raw.
- **DBpedia ontology** — `https://www.dbpedia.org/resources/ontology/`
  OWL file. ~800 classes. ~2 MB.
- **BFO 2.0 / 2020** — `https://github.com/BFO-ontology/BFO-2020`. OWL
  file. ~40 classes.
- **Common Core Ontologies (CCO)** — BFO's industrial descendant,
  GitHub. Mid-level ~200 classes.
- **OBO Foundry** ontologies — biomedical, ~200+ ontologies. Specific
  picks: MONDO (disease), CHEBI (chemical), SO (sequence ontology), GO
  (gene ontology). Each is 10-100 MB RDF.
- **LOV (Linked Open Vocabularies)** — `https://lov.linkeddata.es/`.
  Meta-catalog of 700+ RDF vocabularies. Useful for "long tail"
  vocabulary exposure.

Total size: ~200 MB-1 GB if we pull the full "upper ontology + picks"
set. Worth including verbatim AND pre-processed as
"class-def ↔ human-description" pairs for grounded supervision.

### GitHub issues / docs / prose-about-code

| Source | Repo | Role |
|---|---|---|
| **GitHub issues** | `bigcode/the-stack-github-issues` | Q&A about real schema/query/data-engineering problems. High signal, messy. |
| **StackExchange dumps** | `https://archive.org/details/stackexchange` | DBA.SE, StackOverflow SQL-tagged, DataScience.SE. Not on HF but public. |

These complement The Stack (code only) with the surrounding
human-language explanation of why code is structured a certain way —
which is what metadata semantics often is.

### Existing tabular / metadata corpora we already have

| Source | On-disk | Role |
|---|---|---|
| **GitTables 1M** | `/raid/datasets/gittables/` — 24 M annotated columns | Ground-truth tables for validation, NOT for pretraining directly. |
| **SOTAB v2** | `/raid/datasets/sotab/` | Schema.org CTA benchmark. Validation. |
| **FineWeb-Edu** | `/raid/datasets/fineweb-edu/` | General prose baseline (8 shards, 26 GB). |

Don't over-train on these — they're our evaluation targets. Holding
them out of the pretraining corpus preserves the validity of the
downstream benchmarks.

### Notable things that are NOT publicly available

- PostgreSQL documentation as a clean corpus (would need to scrape
  from `https://www.postgresql.org/docs/`). Non-trivial, HTML noisy.
- Apache Spark / Flink documentation (same — ReadTheDocs-style scrapes).
- Informatica / Collibra / Alation data-governance materials —
  largely proprietary.
- Apache Atlas: source code is Apache-licensed and in The Stack (Java),
  but documentation is also not pre-packaged as a corpus.
- Hive / Impala / Trino / Presto docs — same.

**Recommendation**: a small scraper + cleaner for the big Apache
projects is ~2-3 days of work. Total corpus addition would be ~50-200
MB of high-signal technical writing. Worth doing for a research-grade
corpus. Can be deferred.

## Composition proposal

Target total byte budget: **~40 GB raw text** (before deduplication).

| Slice | Source | Byte budget | Why this proportion |
|---|---|---|---|
| **Prose — educational** | FinePDFs-Edu subset + FineWeb-Edu (existing) | 15 GB | Anchors general fluency; highest-quality reading material |
| **Prose — academic** | peS2o (sample) | 3 GB | Domain grounding for statistics, data-governance, knowledge-graph concepts |
| **SQL + DDL** | Stack v2 dedup: SQL + PLpgSQL + TSQL + PLSQL + SQLPL + HiveQL | 6 GB | Core — directly matches downstream |
| **Python / Scala / Java (data-systems subset)** | Stack v2: curated by keyword (e.g. lines mentioning Spark, Flink, Pandas, SQLAlchemy, dbt, Iceberg) | 5 GB | Teaches schema manipulation in code |
| **YAML / JSON / XML / JSONLD** | Stack v2: filter for JSON-LD, Avro schemas, Kubernetes/Airflow/dbt configs | 3 GB | Format diversity: schema-as-data |
| **Annotation pairs** | Spider + sql-create-context + NSText2SQL | 2 GB | Grounds text→schema alignment |
| **Ontology raw** | Schema.org + DBpedia + BFO/CCO + picks from OBO/LOV | 1 GB | Vocabulary + hierarchy anchor |
| **Ontology as text** | Auto-generated "class definitions as prose" from the above, one per line | 1 GB | Makes the ontology content usable by a byte LM |
| **GitHub issues (data-systems)** | the-stack-github-issues filtered by SQL/Spark/Flink/dbt keywords | 2 GB | Real-world problem formulation |
| **Synthetic tables + schema** | Our pretraining pipeline (docs/src/pretraining.md) with byte-serialized output | 2 GB | Distribution control for rare entity types |

Deduplication + shuffling post-composition. End-state: ~30 GB after
exact-dupe dedup.

Compared to FineWeb-Edu alone (current pretraining), this corpus has
roughly:
- 5% vs 0% SQL
- 10% vs 0% schema-relevant code
- 2% vs 0% ontology text
- 20% vs 100% general prose

...so ~40% of the new corpus is domain-specific, vs our current 0%.

## Synthetic augmentation — what it adds that public data doesn't

Public data has coverage gaps our downstream task will feel:

1. **Rare entity types.** SOTAB's 91 Schema.org classes include many
   tail types (Museum, Aquarium, Archive) with <100 examples in
   GitTables. Synthetic data can sample uniformly across the vocabulary.
2. **Cross-table joins with known ground truth.** Public SQL corpora
   contain queries but the SEMANTIC relationships between columns
   across tables are implicit. Our pipeline explicitly generates these.
3. **Ontology-grounded column content.** A synthetic `email` column
   IS synthesised by an ontology-aware generator that knows emails
   match `schema:email`, so the column-to-type mapping is labelled
   by construction.
4. **Schema evolution** — version-over-version schema changes with
   known mapping (column renames, type promotions). Extremely rare in
   public data; easy to synthesise.

The pipeline in `docs/src/pretraining.md` already specifies Stages 1-5
for this. Stage 4 output (populated tables + schema) feeds directly
into our byte-level pretraining stream.

## Open questions / decisions before composition

- **License posture.** Stack v2 dedup has per-language licence mix.
  For research use all is fine; for deployment we'd need to filter.
  Flag but don't block.
- **Deduplication strategy.** Within each source vs across sources?
  Across sources risks overweighting FinePDFs if it overlaps heavily
  with peS2o. MinHash LSH dedup at document level is standard.
- **Byte vs BPE tokenisation for corpus prep.** We stay byte-level
  consistent with Aegir's design. The corpus is raw text; byte-level
  exposure happens at training time.
- **Filtering: keep or drop non-English content?** Keep if we want
  Aegir to handle non-English column names (common in European
  enterprise data). Costs ~10-15% capacity. Recommend: keep.
- **Dynamic mixing at training time vs fixed mix at corpus time?**
  Fixed mix is simpler; dynamic (via streaming samplers with
  per-source weights) lets us tune proportions post hoc. Recommend:
  build as fixed-mix shards initially, add a streaming sampler in
  the training loop for Phase 2.

## Concrete next steps

1. **Cheap + informative first** — pull the small, high-signal pieces
   (Spider + sql-create-context + ontology raw files + BFO/CCO OWL) as
   a first corpus slice. Maybe 3-4 GB total. Train a probe on this
   mixed with FineWeb-Edu and measure whether domain-relevant
   perplexity improves on a held-out set of Schema.org annotations.
2. **Stack v2 SQL slice** — `bigcode/the-stack-v2-dedup` → SQL +
   PLpgSQL + TSQL + PLSQL + SQLPL + HiveQL. ~1-3 GB. Direct,
   measurable signal on query syntax.
3. **Stack v2 schema-aware code subset** — filter by keyword across
   Python/Scala/Java shards for references to `CREATE TABLE`,
   `SparkSession`, `FlinkJobConfig`, etc. ~3-5 GB. This is more work
   (we define the filter) but pays off in domain specialisation.
4. **Ontology-as-prose generator** — small script that takes the OWL
   files and emits "class `schema:Hotel` is a subtype of
   `schema:LodgingBusiness` which is a subtype of `schema:LocalBusiness`.
   Properties include..." one definition per line. ~1 GB of
   structured text from ~3 MB of OWL.
5. **Synthetic augmentation** — stand up the first stage of the
   pipeline in docs/src/pretraining.md. Start with Schema.org because
   it's the smallest and we have the benchmarks.

Each item 1-5 is a day of work and produces a measurable corpus
artefact. Items 1-2 are prerequisite; 3-5 can run in parallel.

## Format + layout on disk

Following the convention we established for FineWeb:

```
/raid/datasets/aegir-corpus-v1/
  ├── prose/
  │   ├── finepdfs-edu/    # shard_NNNN.txt files (UTF-8, \x03 docsep)
  │   └── fineweb-edu/     # (symlink to existing /raid/datasets/fineweb-edu/)
  ├── code/
  │   ├── sql/             # from Stack v2 dedup SQL + dialect slices
  │   └── data-systems/    # filtered Python/Scala/Java slices
  ├── annotations/
  │   ├── spider.jsonl
  │   └── sql-create-context.jsonl
  ├── ontology/
  │   ├── schema_org.json   # original
  │   ├── dbpedia.owl       # original
  │   ├── bfo.owl           # original
  │   └── ontology-prose.txt  # generated
  ├── synthetic/
  │   └── (generated by our pipeline)
  └── MIXTURE.json          # weights per source
```

Mixture.json gets loaded by the training pipeline's streaming sampler.

## TL;DR

The public corpus space for metadata / data-systems is healthier than
expected. The Stack v2 dedup gives us SQL dialects + supporting-language
code at clean quality. FinePDFs + FineWeb-Edu give us prose. Spider +
sql-create-context give us annotation pairs. Ontology OWL files give
us vocabulary ground truth. Total accessible public budget ~30-50 GB
before synthetic augmentation. Five concrete days of work get us from
"FineWeb-Edu only" to a research-grade domain-specialised corpus.
