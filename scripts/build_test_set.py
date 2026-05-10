"""build_test_set — generate the C1 verifier-validation test set.

Per P2 / claim C1 of the v0.5 concept brief: produce ≥ 15
known-good and ≥ 15 known-bad OWL ontologies for AUC-based
verifier validation. Outputs TTL files in
``tests/ontology_test_set/`` plus a ``labels.json`` mapping
filename → ``{"label": "good" | "bad", "kind": "...", "remark": "..."}``.

Quality profiles emitted:

**Known-good (15)**
- ``good_lab_measurement.ttl`` — lab measurement ontology with
  Sample / Instrument / Measurement / Unit; ≥ 12 classes; ≥ 5
  complex axioms.
- ``good_database_metadata.ttl`` — Dataset / Table / Column /
  Constraint with PK / FK / type-restriction patterns.
- ``good_trace_telemetry.ttl`` — Trace / Span / Service /
  AttributeKey / Resource OTel-flavored.
- ``good_ebpf_observability.ttl`` — eBPFProgram / KernelHook /
  Syscall / Event with kernel-granularity axioms.
- ``good_governance_policy.ttl`` — Policy / Control /
  Constraint / Audit governance patterns.
- ``good_sample_provenance.ttl`` — Sample lineage with PROV-O
  alignment.
- ``good_lims_clinical.ttl`` — clinical trial / patient sample /
  protocol; FinePDFs-lab corpus aligned.
- ``good_otel_semconv.ttl`` — HTTP / DB / RPC SemConv
  attribute-typed measurements.
- ``good_macrobase_outliers.ttl`` — OutlierClaim / AttributeSet /
  Lift / Aggregation Macrobase-flavored.
- ``good_sysml_blocks.ttl`` — SystemBlock / Part / State /
  Requirement SysMLv2-aligned.
- ``good_dataquality_profile.ttl`` — column profile, null rate,
  distribution, cardinality.
- ``good_iam_security.ttl`` — Identity / Role / Permission /
  Policy security model.
- ``good_workflow_orchestration.ttl`` — Job / Run / Step /
  Trigger workflow lineage.
- ``good_observation_event.ttl`` — generic ObservationProcess /
  Measurement / Result / Time pattern.
- ``good_metadata_governance.ttl`` — Tag / Classification /
  Tier / Owner data-governance pattern.

**Known-bad (15)**
- ``bad_empty.ttl`` — only Ontology declaration, no classes.
- ``bad_trivial.ttl`` — one class with one trivial axiom.
- ``bad_no_labels.ttl`` — many classes, no rdfs:label, no
  skos:definition.
- ``bad_no_complex.ttl`` — many classes but only basic
  SubClassOf, no restrictions / equivalentClass / cardinality.
- ``bad_truncated_lab.ttl`` — fragment of good_lab_measurement
  with most axioms removed.
- ``bad_random_gibberish.ttl`` — randomly-named classes with
  random subclass edges; no labels.
- ``bad_off_domain_music.ttl`` — music/art/literature ontology;
  off-domain for SchemaPile + FinePDFs-lab.
- ``bad_off_domain_cooking.ttl`` — cooking recipes / ingredients;
  off-domain.
- ``bad_circular.ttl`` — A subClassOf B subClassOf A loops.
- ``bad_only_disjoint.ttl`` — many classes but only
  DisjointClasses axioms; no positive structure.
- ``bad_single_class_chain.ttl`` — long linear subclass chain
  with no other axioms.
- ``bad_meaningless_iris.ttl`` — Foo1, Foo2, Foo3-style classes
  with no semantic content.
- ``bad_two_classes.ttl`` — only 2 classes, 1 axiom.
- ``bad_corrupted_labels.ttl`` — labels are random gibberish
  ("xqzpf", "blkrn", etc.) — verbalizes but content is
  semantically meaningless.
- ``bad_one_axiom_per_class.ttl`` — 10 classes each with single
  ``SubClassOf cco:Artifact`` axiom; no relations among them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "ontology_test_set"

# Common prefixes used across all test ontologies.
PREFIXES = """\
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix bfo:     <http://purl.obolibrary.org/obo/BFO_> .
@prefix cco:     <http://www.commoncoreontologies.org/> .
@prefix iao:     <http://purl.obolibrary.org/obo/IAO_> .
@prefix ex:      <http://example.org/test#> .
"""


# ---- known-good test ontologies ----


def good_lab_measurement() -> str:
    return PREFIXES + """
<http://example.org/test/lab-measurement> a owl:Ontology .

ex:Sample a owl:Class ;
    rdfs:label "specimen sample" ;
    skos:definition "A specimen taken from a subject for analysis." ;
    rdfs:subClassOf cco:Artifact .

ex:BloodSample a owl:Class ;
    rdfs:label "blood sample" ;
    skos:definition "A blood specimen taken from a patient for laboratory analysis." ;
    rdfs:subClassOf ex:Sample .

ex:Instrument a owl:Class ;
    rdfs:label "laboratory instrument" ;
    skos:definition "A device used for laboratory measurements." ;
    rdfs:subClassOf cco:Artifact .

ex:Spectrometer a owl:Class ;
    rdfs:label "spectrometer" ;
    skos:definition "An optical instrument that measures intensity at different wavelengths." ;
    rdfs:subClassOf ex:Instrument ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:producesMeasurement ;
                      owl:someValuesFrom ex:SpectralReading ] .

ex:Measurement a owl:Class ;
    rdfs:label "measurement" ;
    skos:definition "A quantitative observation produced by a laboratory instrument." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:SpectralReading a owl:Class ;
    rdfs:label "spectral reading" ;
    skos:definition "A measurement of light intensity at a specific wavelength." ;
    rdfs:subClassOf ex:Measurement ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasUnit ;
                      owl:someValuesFrom ex:Nanometer ] .

ex:Unit a owl:Class ;
    rdfs:label "unit of measurement" ;
    skos:definition "A standard quantity used to express measurements." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:Nanometer a owl:Class ;
    rdfs:label "nanometer unit" ;
    rdfs:subClassOf ex:Unit .

ex:Celsius a owl:Class ;
    rdfs:label "celsius unit" ;
    rdfs:subClassOf ex:Unit .

ex:LabRun a owl:Class ;
    rdfs:label "laboratory run" ;
    skos:definition "A laboratory analysis process that produces measurements from samples." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasInputSample ;
                      owl:someValuesFrom ex:Sample ] ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:produces ;
                      owl:someValuesFrom ex:Measurement ] .

ex:CalibrationProtocol a owl:Class ;
    rdfs:label "calibration protocol" ;
    skos:definition "A directive specifying instrument calibration steps." ;
    rdfs:subClassOf cco:DirectiveICE .

ex:QualityControl a owl:Class ;
    rdfs:label "quality control check" ;
    skos:definition "A process verifying that measurements meet quality criteria." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliesTo ;
                      owl:someValuesFrom ex:Measurement ] .

ex:producesMeasurement a owl:ObjectProperty ;
    rdfs:label "produces measurement" .

ex:hasUnit a owl:ObjectProperty ;
    rdfs:label "has unit" .

ex:hasInputSample a owl:ObjectProperty ;
    rdfs:label "has input sample" .

ex:produces a owl:ObjectProperty ;
    rdfs:label "produces" .

ex:appliesTo a owl:ObjectProperty ;
    rdfs:label "applies to" .
"""


def good_database_metadata() -> str:
    return PREFIXES + """
<http://example.org/test/database-metadata> a owl:Ontology .

ex:Dataset a owl:Class ;
    rdfs:label "dataset" ;
    skos:definition "A collection of structured data records organized as a table or file." ;
    rdfs:subClassOf cco:Artifact .

ex:Table a owl:Class ;
    rdfs:label "database table" ;
    skos:definition "A relational database table containing rows of typed records." ;
    rdfs:subClassOf ex:Dataset ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasColumn ;
                      owl:minCardinality 1 ] .

ex:Column a owl:Class ;
    rdfs:label "database column" ;
    skos:definition "A typed column within a relational table." ;
    rdfs:subClassOf cco:DesignativeICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasDataType ;
                      owl:someValuesFrom ex:DataType ] .

ex:DataType a owl:Class ;
    rdfs:label "column data type" ;
    skos:definition "A type specification for column values (varchar, integer, decimal, timestamp)." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:VarcharType a owl:Class ;
    rdfs:label "varchar data type" ;
    rdfs:subClassOf ex:DataType .

ex:IntegerType a owl:Class ;
    rdfs:label "integer data type" ;
    rdfs:subClassOf ex:DataType .

ex:DecimalType a owl:Class ;
    rdfs:label "decimal data type" ;
    rdfs:subClassOf ex:DataType .

ex:TimestampType a owl:Class ;
    rdfs:label "timestamp data type" ;
    rdfs:subClassOf ex:DataType .

ex:Constraint a owl:Class ;
    rdfs:label "table constraint" ;
    skos:definition "A directive that constrains valid values within a column or relation between tables." ;
    rdfs:subClassOf cco:DirectiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliesTo ;
                      owl:someValuesFrom ex:Column ] .

ex:PrimaryKeyConstraint a owl:Class ;
    rdfs:label "primary key constraint" ;
    skos:definition "A constraint requiring that values in a column uniquely identify each row." ;
    rdfs:subClassOf ex:Constraint .

ex:ForeignKeyConstraint a owl:Class ;
    rdfs:label "foreign key constraint" ;
    skos:definition "A constraint requiring that column values reference rows in another table." ;
    rdfs:subClassOf ex:Constraint ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:references ;
                      owl:someValuesFrom ex:Table ] .

ex:NotNullConstraint a owl:Class ;
    rdfs:label "not null constraint" ;
    rdfs:subClassOf ex:Constraint .

ex:UniqueConstraint a owl:Class ;
    rdfs:label "unique constraint" ;
    rdfs:subClassOf ex:Constraint .

ex:hasColumn a owl:ObjectProperty ;
    rdfs:label "has column" .
ex:hasDataType a owl:ObjectProperty ;
    rdfs:label "has data type" .
ex:appliesTo a owl:ObjectProperty ;
    rdfs:label "applies to" .
ex:references a owl:ObjectProperty ;
    rdfs:label "references" .
"""


def good_trace_telemetry() -> str:
    return PREFIXES + """
<http://example.org/test/trace-telemetry> a owl:Ontology .

ex:Trace a owl:Class ;
    rdfs:label "telemetry trace" ;
    skos:definition "A request-scoped sequence of operations recorded across services." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasSpans ;
                      owl:minCardinality 1 ] .

ex:Span a owl:Class ;
    rdfs:label "trace span" ;
    skos:definition "A single timed operation within a distributed trace." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:partOf ;
                      owl:someValuesFrom ex:Trace ] .

ex:HttpSpan a owl:Class ;
    rdfs:label "http request span" ;
    skos:definition "A span representing an HTTP request operation." ;
    rdfs:subClassOf ex:Span ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttribute ;
                      owl:someValuesFrom ex:HttpMethod ] .

ex:DatabaseSpan a owl:Class ;
    rdfs:label "database query span" ;
    rdfs:subClassOf ex:Span .

ex:Service a owl:Class ;
    rdfs:label "service" ;
    skos:definition "A deployable unit of software that emits telemetry." ;
    rdfs:subClassOf cco:Artifact .

ex:CheckoutService a owl:Class ;
    rdfs:label "checkout service" ;
    rdfs:subClassOf ex:Service .

ex:AttributeKey a owl:Class ;
    rdfs:label "telemetry attribute key" ;
    skos:definition "A typed key naming a telemetry attribute (http.method, db.system, etc.)." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:HttpMethod a owl:Class ;
    rdfs:label "http method attribute" ;
    rdfs:subClassOf ex:AttributeKey .

ex:DbSystem a owl:Class ;
    rdfs:label "database system attribute" ;
    rdfs:subClassOf ex:AttributeKey .

ex:Resource a owl:Class ;
    rdfs:label "telemetry resource" ;
    skos:definition "An entity describing the deployment unit emitting telemetry." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:KubernetesPod a owl:Class ;
    rdfs:label "kubernetes pod resource" ;
    rdfs:subClassOf ex:Resource .

ex:hasSpans a owl:ObjectProperty ;
    rdfs:label "has spans" .
ex:partOf a owl:ObjectProperty ;
    rdfs:label "part of" .
ex:hasAttribute a owl:ObjectProperty ;
    rdfs:label "has attribute" .
"""


def good_governance_policy() -> str:
    return PREFIXES + """
<http://example.org/test/governance-policy> a owl:Ontology .

ex:Policy a owl:Class ;
    rdfs:label "governance policy" ;
    skos:definition "A directive that governs handling of resources within an organization." ;
    rdfs:subClassOf cco:DirectiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliesTo ;
                      owl:minCardinality 1 ] .

ex:DataRetentionPolicy a owl:Class ;
    rdfs:label "data retention policy" ;
    skos:definition "A policy specifying how long data may be retained." ;
    rdfs:subClassOf ex:Policy ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliesTo ;
                      owl:someValuesFrom ex:Dataset ] .

ex:AccessControlPolicy a owl:Class ;
    rdfs:label "access control policy" ;
    skos:definition "A policy specifying who may read or write resources." ;
    rdfs:subClassOf ex:Policy .

ex:Control a owl:Class ;
    rdfs:label "compliance control" ;
    skos:definition "A directive that constrains operations to ensure compliance." ;
    rdfs:subClassOf cco:DirectiveICE .

ex:EncryptionControl a owl:Class ;
    rdfs:label "encryption control" ;
    rdfs:subClassOf ex:Control .

ex:AuditLoggingControl a owl:Class ;
    rdfs:label "audit logging control" ;
    rdfs:subClassOf ex:Control .

ex:Audit a owl:Class ;
    rdfs:label "compliance audit" ;
    skos:definition "A process verifying that controls are effective." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliesTo ;
                      owl:someValuesFrom ex:Control ] .

ex:Attestation a owl:Class ;
    rdfs:label "compliance attestation" ;
    skos:definition "A formal declaration that controls have been implemented." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:supports ;
                      owl:someValuesFrom ex:Control ] .

ex:Dataset a owl:Class ;
    rdfs:label "governed dataset" ;
    rdfs:subClassOf cco:Artifact .

ex:appliesTo a owl:ObjectProperty ;
    rdfs:label "applies to" .
ex:supports a owl:ObjectProperty ;
    rdfs:label "supports" .
"""


def good_lims_clinical() -> str:
    return PREFIXES + """
<http://example.org/test/lims-clinical> a owl:Ontology .

ex:ClinicalTrial a owl:Class ;
    rdfs:label "clinical trial" ;
    skos:definition "A research study evaluating a medical intervention in human participants." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasParticipant ;
                      owl:minCardinality 1 ] .

ex:Patient a owl:Class ;
    rdfs:label "patient" ;
    skos:definition "A human research participant from whom samples are collected." ;
    rdfs:subClassOf cco:Person .

ex:Visit a owl:Class ;
    rdfs:label "patient visit" ;
    skos:definition "A clinical encounter at which observations and samples are recorded." ;
    rdfs:subClassOf bfo:0000015 .

ex:BiologicalSample a owl:Class ;
    rdfs:label "biological sample" ;
    skos:definition "A specimen of biological tissue, fluid, or cell material taken for analysis." ;
    rdfs:subClassOf cco:Artifact ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:collectedFrom ;
                      owl:someValuesFrom ex:Patient ] .

ex:BloodDraw a owl:Class ;
    rdfs:label "blood draw" ;
    rdfs:subClassOf ex:BiologicalSample .

ex:UrineSample a owl:Class ;
    rdfs:label "urine sample" ;
    rdfs:subClassOf ex:BiologicalSample .

ex:Assay a owl:Class ;
    rdfs:label "laboratory assay" ;
    skos:definition "A laboratory analysis procedure run on a biological sample." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:appliedTo ;
                      owl:someValuesFrom ex:BiologicalSample ] .

ex:HematologyAssay a owl:Class ;
    rdfs:label "hematology assay" ;
    rdfs:subClassOf ex:Assay .

ex:GenotypingAssay a owl:Class ;
    rdfs:label "genotyping assay" ;
    rdfs:subClassOf ex:Assay .

ex:LabResult a owl:Class ;
    rdfs:label "laboratory result" ;
    skos:definition "A measurement value produced by a laboratory assay." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:producedBy ;
                      owl:someValuesFrom ex:Assay ] .

ex:Diagnosis a owl:Class ;
    rdfs:label "clinical diagnosis" ;
    skos:definition "A medical interpretation derived from clinical findings." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:hasParticipant a owl:ObjectProperty ; rdfs:label "has participant" .
ex:collectedFrom a owl:ObjectProperty ; rdfs:label "collected from" .
ex:appliedTo a owl:ObjectProperty ; rdfs:label "applied to" .
ex:producedBy a owl:ObjectProperty ; rdfs:label "produced by" .
"""


def good_macrobase_outliers() -> str:
    return PREFIXES + """
<http://example.org/test/macrobase-outliers> a owl:Ontology .

ex:OutlierClaim a owl:Class ;
    rdfs:label "outlier explanation claim" ;
    skos:definition "A statistical explanation pointing to attribute combinations that distinguish anomalies." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttributeSet ;
                      owl:minCardinality 1 ] .

ex:AttributeSet a owl:Class ;
    rdfs:label "explanation attribute set" ;
    skos:definition "A combination of attribute values that jointly characterize an outlier population." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttribute ;
                      owl:minCardinality 1 ] .

ex:Lift a owl:Class ;
    rdfs:label "statistical lift score" ;
    skos:definition "A ratio of foreground to background frequencies for an attribute set." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:ofAttributeSet ;
                      owl:someValuesFrom ex:AttributeSet ] .

ex:Aggregation a owl:Class ;
    rdfs:label "aggregation" ;
    skos:definition "A summary statistic over a group of underlying measurements." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:aggregates ;
                      owl:someValuesFrom ex:Measurement ] .

ex:Measurement a owl:Class ;
    rdfs:label "measurement" ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:LatencyMeasurement a owl:Class ;
    rdfs:label "latency measurement" ;
    rdfs:subClassOf ex:Measurement .

ex:CountMeasurement a owl:Class ;
    rdfs:label "count measurement" ;
    rdfs:subClassOf ex:Measurement .

ex:OutlierDetection a owl:Class ;
    rdfs:label "outlier detection process" ;
    skos:definition "A statistical analysis process that produces outlier explanation claims." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:produces ;
                      owl:someValuesFrom ex:OutlierClaim ] .

ex:hasAttributeSet a owl:ObjectProperty ; rdfs:label "has attribute set" .
ex:hasAttribute a owl:ObjectProperty ; rdfs:label "has attribute" .
ex:ofAttributeSet a owl:ObjectProperty ; rdfs:label "of attribute set" .
ex:aggregates a owl:ObjectProperty ; rdfs:label "aggregates" .
ex:produces a owl:ObjectProperty ; rdfs:label "produces" .
"""


def good_otel_semconv() -> str:
    return PREFIXES + """
<http://example.org/test/otel-semconv> a owl:Ontology .

ex:Span a owl:Class ;
    rdfs:label "telemetry span" ;
    skos:definition "A single timed operation in a distributed trace." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttribute ;
                      owl:minCardinality 1 ] .

ex:HttpServerSpan a owl:Class ;
    rdfs:label "http server span" ;
    skos:definition "A span representing handling of an inbound HTTP request." ;
    rdfs:subClassOf ex:Span ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttribute ;
                      owl:someValuesFrom ex:HttpMethodAttribute ] .

ex:DatabaseClientSpan a owl:Class ;
    rdfs:label "database client span" ;
    rdfs:subClassOf ex:Span ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasAttribute ;
                      owl:someValuesFrom ex:DbSystemAttribute ] .

ex:HttpMethodAttribute a owl:Class ;
    rdfs:label "http.method attribute" ;
    skos:definition "An attribute key naming the HTTP method (GET, POST, PUT, DELETE)." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:DbSystemAttribute a owl:Class ;
    rdfs:label "db.system attribute" ;
    skos:definition "An attribute key naming the database system (postgres, mysql, mongodb)." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:LatencyMetric a owl:Class ;
    rdfs:label "latency metric" ;
    skos:definition "A measurement of operation duration in milliseconds." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasUnit ;
                      owl:someValuesFrom ex:Millisecond ] .

ex:RequestCountMetric a owl:Class ;
    rdfs:label "request count metric" ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:Millisecond a owl:Class ;
    rdfs:label "millisecond unit" ;
    rdfs:subClassOf cco:DesignativeICE .

ex:hasAttribute a owl:ObjectProperty ; rdfs:label "has attribute" .
ex:hasUnit a owl:ObjectProperty ; rdfs:label "has unit" .
"""


def good_observation_event() -> str:
    return PREFIXES + """
<http://example.org/test/observation-event> a owl:Ontology .

ex:ObservationEvent a owl:Class ;
    rdfs:label "observation event" ;
    skos:definition "An event in which a measurement is recorded for a target entity at a specified time." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:produces ;
                      owl:someValuesFrom ex:Measurement ] .

ex:Measurement a owl:Class ;
    rdfs:label "measurement value" ;
    skos:definition "A quantitative observation of a property of an entity at a moment in time." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasValue ;
                      owl:cardinality 1 ] ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasUnit ;
                      owl:someValuesFrom ex:MeasurementUnit ] .

ex:MeasurementUnit a owl:Class ;
    rdfs:label "measurement unit" ;
    rdfs:subClassOf cco:DesignativeICE .

ex:Timestamp a owl:Class ;
    rdfs:label "observation timestamp" ;
    skos:definition "The temporal coordinate at which an observation was recorded." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:Sensor a owl:Class ;
    rdfs:label "measurement sensor" ;
    skos:definition "A device that produces measurements for a designated property." ;
    rdfs:subClassOf cco:Artifact .

ex:TemperatureSensor a owl:Class ;
    rdfs:label "temperature sensor" ;
    rdfs:subClassOf ex:Sensor .

ex:PressureSensor a owl:Class ;
    rdfs:label "pressure sensor" ;
    rdfs:subClassOf ex:Sensor .

ex:produces a owl:ObjectProperty ; rdfs:label "produces" .
ex:hasValue a owl:ObjectProperty ; rdfs:label "has value" .
ex:hasUnit a owl:ObjectProperty ; rdfs:label "has unit" .
"""


def good_dataquality_profile() -> str:
    return PREFIXES + """
<http://example.org/test/dataquality-profile> a owl:Ontology .

ex:ColumnProfile a owl:Class ;
    rdfs:label "column profile" ;
    skos:definition "A statistical summary of values in a database column." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasNullRate ;
                      owl:cardinality 1 ] ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasCardinality ;
                      owl:cardinality 1 ] .

ex:NullRate a owl:Class ;
    rdfs:label "null rate statistic" ;
    skos:definition "The fraction of values in a column that are null or missing." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:Cardinality a owl:Class ;
    rdfs:label "cardinality statistic" ;
    skos:definition "The number of distinct values in a column." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:DistributionSummary a owl:Class ;
    rdfs:label "value distribution summary" ;
    skos:definition "A histogram or quantile sketch describing how values are distributed." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:NumericProfile a owl:Class ;
    rdfs:label "numeric column profile" ;
    rdfs:subClassOf ex:ColumnProfile ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasDistribution ;
                      owl:someValuesFrom ex:DistributionSummary ] .

ex:CategoricalProfile a owl:Class ;
    rdfs:label "categorical column profile" ;
    rdfs:subClassOf ex:ColumnProfile .

ex:hasNullRate a owl:ObjectProperty ; rdfs:label "has null rate" .
ex:hasCardinality a owl:ObjectProperty ; rdfs:label "has cardinality" .
ex:hasDistribution a owl:ObjectProperty ; rdfs:label "has distribution" .
"""


def good_iam_security() -> str:
    return PREFIXES + """
<http://example.org/test/iam-security> a owl:Ontology .

ex:Identity a owl:Class ;
    rdfs:label "identity" ;
    skos:definition "An identifier representing a human or service principal in an access control system." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:UserIdentity a owl:Class ;
    rdfs:label "user identity" ;
    rdfs:subClassOf ex:Identity .

ex:ServiceIdentity a owl:Class ;
    rdfs:label "service identity" ;
    rdfs:subClassOf ex:Identity .

ex:Role a owl:Class ;
    rdfs:label "access role" ;
    skos:definition "A named collection of permissions granted to identities." ;
    rdfs:subClassOf cco:DesignativeICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasPermission ;
                      owl:minCardinality 1 ] .

ex:Permission a owl:Class ;
    rdfs:label "permission" ;
    skos:definition "An authorized operation on a target resource." ;
    rdfs:subClassOf cco:DirectiveICE .

ex:ReadPermission a owl:Class ;
    rdfs:label "read permission" ;
    rdfs:subClassOf ex:Permission .

ex:WritePermission a owl:Class ;
    rdfs:label "write permission" ;
    rdfs:subClassOf ex:Permission .

ex:DeletePermission a owl:Class ;
    rdfs:label "delete permission" ;
    rdfs:subClassOf ex:Permission .

ex:AccessPolicy a owl:Class ;
    rdfs:label "access policy" ;
    skos:definition "A directive specifying which identities may perform which operations on which resources." ;
    rdfs:subClassOf cco:DirectiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:grantsRole ;
                      owl:someValuesFrom ex:Role ] .

ex:hasPermission a owl:ObjectProperty ; rdfs:label "has permission" .
ex:grantsRole a owl:ObjectProperty ; rdfs:label "grants role" .
"""


def good_workflow_orchestration() -> str:
    return PREFIXES + """
<http://example.org/test/workflow-orchestration> a owl:Ontology .

ex:Job a owl:Class ;
    rdfs:label "data pipeline job" ;
    skos:definition "A configured data processing workflow definition." ;
    rdfs:subClassOf cco:Artifact .

ex:Run a owl:Class ;
    rdfs:label "job run" ;
    skos:definition "A single execution instance of a configured job." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:executes ;
                      owl:someValuesFrom ex:Job ] .

ex:Step a owl:Class ;
    rdfs:label "pipeline step" ;
    skos:definition "A single unit of work within a job execution." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:partOf ;
                      owl:someValuesFrom ex:Run ] .

ex:ExtractStep a owl:Class ;
    rdfs:label "data extract step" ;
    rdfs:subClassOf ex:Step .

ex:TransformStep a owl:Class ;
    rdfs:label "data transform step" ;
    rdfs:subClassOf ex:Step .

ex:LoadStep a owl:Class ;
    rdfs:label "data load step" ;
    rdfs:subClassOf ex:Step .

ex:Trigger a owl:Class ;
    rdfs:label "job trigger" ;
    skos:definition "An event or schedule that causes a job to run." ;
    rdfs:subClassOf cco:DirectiveICE .

ex:ScheduleTrigger a owl:Class ;
    rdfs:label "schedule trigger" ;
    rdfs:subClassOf ex:Trigger .

ex:DataAvailabilityTrigger a owl:Class ;
    rdfs:label "data availability trigger" ;
    rdfs:subClassOf ex:Trigger .

ex:executes a owl:ObjectProperty ; rdfs:label "executes" .
ex:partOf a owl:ObjectProperty ; rdfs:label "part of" .
"""


def good_metadata_governance() -> str:
    return PREFIXES + """
<http://example.org/test/metadata-governance> a owl:Ontology .

ex:Tag a owl:Class ;
    rdfs:label "metadata tag" ;
    skos:definition "A classification label attached to a data resource." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:Classification a owl:Class ;
    rdfs:label "data classification" ;
    skos:definition "A categorization of data sensitivity, ownership, or domain." ;
    rdfs:subClassOf cco:DescriptiveICE ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:atTier ;
                      owl:someValuesFrom ex:SensitivityTier ] .

ex:SensitivityTier a owl:Class ;
    rdfs:label "sensitivity tier" ;
    skos:definition "A graded level of data sensitivity." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:PublicTier a owl:Class ;
    rdfs:label "public tier" ;
    rdfs:subClassOf ex:SensitivityTier .

ex:InternalTier a owl:Class ;
    rdfs:label "internal tier" ;
    rdfs:subClassOf ex:SensitivityTier .

ex:RestrictedTier a owl:Class ;
    rdfs:label "restricted tier" ;
    rdfs:subClassOf ex:SensitivityTier .

ex:DataOwner a owl:Class ;
    rdfs:label "data owner" ;
    skos:definition "An identity accountable for governance of a data resource." ;
    rdfs:subClassOf cco:Person .

ex:Steward a owl:Class ;
    rdfs:label "data steward" ;
    skos:definition "An identity responsible for day-to-day operational governance of data." ;
    rdfs:subClassOf cco:Person .

ex:Glossary a owl:Class ;
    rdfs:label "business glossary entry" ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:atTier a owl:ObjectProperty ; rdfs:label "at tier" .
"""


def good_sample_provenance() -> str:
    return PREFIXES + """
<http://example.org/test/sample-provenance> a owl:Ontology .

ex:Specimen a owl:Class ;
    rdfs:label "biological specimen" ;
    skos:definition "A physical sample of biological material under tracked custody." ;
    rdfs:subClassOf cco:Artifact ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:wasDerivedFrom ;
                      owl:maxCardinality 1 ] .

ex:CollectionEvent a owl:Class ;
    rdfs:label "specimen collection event" ;
    skos:definition "A clinical event in which a biological specimen is collected from a subject." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:produced ;
                      owl:someValuesFrom ex:Specimen ] .

ex:ProcessingStep a owl:Class ;
    rdfs:label "specimen processing step" ;
    skos:definition "An action that transforms a specimen for downstream analysis." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:transforms ;
                      owl:someValuesFrom ex:Specimen ] .

ex:Aliquot a owl:Class ;
    rdfs:label "specimen aliquot" ;
    skos:definition "A sub-portion of a parent specimen prepared for a specific analysis." ;
    rdfs:subClassOf ex:Specimen .

ex:CustodyTransfer a owl:Class ;
    rdfs:label "chain of custody transfer" ;
    skos:definition "A documented transfer of specimen custody between handlers." ;
    rdfs:subClassOf bfo:0000015 .

ex:Operator a owl:Class ;
    rdfs:label "laboratory operator" ;
    skos:definition "A person who performs sample processing or analysis." ;
    rdfs:subClassOf cco:Person .

ex:wasDerivedFrom a owl:ObjectProperty ; rdfs:label "was derived from" .
ex:produced a owl:ObjectProperty ; rdfs:label "produced" .
ex:transforms a owl:ObjectProperty ; rdfs:label "transforms" .
"""


def good_ebpf_observability() -> str:
    return PREFIXES + """
<http://example.org/test/ebpf-observability> a owl:Ontology .

ex:EbpfProgram a owl:Class ;
    rdfs:label "ebpf program" ;
    skos:definition "A small kernel-attachable program that captures observability events." ;
    rdfs:subClassOf cco:Artifact .

ex:KprobeProgram a owl:Class ;
    rdfs:label "kprobe ebpf program" ;
    rdfs:subClassOf ex:EbpfProgram .

ex:XdpProgram a owl:Class ;
    rdfs:label "xdp ebpf program" ;
    rdfs:subClassOf ex:EbpfProgram .

ex:KernelHook a owl:Class ;
    rdfs:label "kernel hook attachment point" ;
    skos:definition "A site in the kernel where an ebpf program may be attached." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:Syscall a owl:Class ;
    rdfs:label "system call" ;
    skos:definition "An entry point through which a process invokes kernel functionality." ;
    rdfs:subClassOf cco:DesignativeICE .

ex:OpenSyscall a owl:Class ;
    rdfs:label "open system call" ;
    rdfs:subClassOf ex:Syscall .

ex:KernelEvent a owl:Class ;
    rdfs:label "kernel event" ;
    skos:definition "A kernel-level occurrence captured by an attached ebpf program." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:capturedBy ;
                      owl:someValuesFrom ex:EbpfProgram ] ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:atHook ;
                      owl:someValuesFrom ex:KernelHook ] .

ex:Map a owl:Class ;
    rdfs:label "ebpf map" ;
    skos:definition "An in-kernel data structure shared between an ebpf program and userspace." ;
    rdfs:subClassOf cco:Artifact .

ex:capturedBy a owl:ObjectProperty ; rdfs:label "captured by" .
ex:atHook a owl:ObjectProperty ; rdfs:label "at hook" .
"""


def good_sysml_blocks() -> str:
    return PREFIXES + """
<http://example.org/test/sysml-blocks> a owl:Ontology .

ex:SystemBlock a owl:Class ;
    rdfs:label "system block" ;
    skos:definition "A modular system element that can be parameterized and connected to other blocks." ;
    rdfs:subClassOf cco:Artifact ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty bfo:0000051 ;
                      owl:minCardinality 0 ] .

ex:Part a owl:Class ;
    rdfs:label "system part" ;
    skos:definition "A nested constituent of a containing system block." ;
    rdfs:subClassOf cco:Artifact ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty bfo:0000050 ;
                      owl:someValuesFrom ex:SystemBlock ] .

ex:Port a owl:Class ;
    rdfs:label "block port" ;
    skos:definition "An interface point on a system block enabling connection to other blocks." ;
    rdfs:subClassOf cco:Artifact .

ex:Connection a owl:Class ;
    rdfs:label "block connection" ;
    skos:definition "A relationship linking ports of different system blocks." ;
    rdfs:subClassOf cco:Artifact .

ex:Action a owl:Class ;
    rdfs:label "system action" ;
    skos:definition "A behavioral occurrence performed by a system block." ;
    rdfs:subClassOf bfo:0000015 .

ex:State a owl:Class ;
    rdfs:label "system state" ;
    skos:definition "A condition of the system at a given point in time." ;
    rdfs:subClassOf cco:DescriptiveICE .

ex:Requirement a owl:Class ;
    rdfs:label "system requirement" ;
    skos:definition "A directive specifying a property the system must satisfy." ;
    rdfs:subClassOf cco:DirectiveICE .

ex:VerificationCase a owl:Class ;
    rdfs:label "verification case" ;
    skos:definition "A test that demonstrates a requirement is satisfied." ;
    rdfs:subClassOf bfo:0000015 ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:verifies ;
                      owl:someValuesFrom ex:Requirement ] .

ex:verifies a owl:ObjectProperty ; rdfs:label "verifies" .
"""


GOOD_GENERATORS = {
    "good_lab_measurement": good_lab_measurement,
    "good_database_metadata": good_database_metadata,
    "good_trace_telemetry": good_trace_telemetry,
    "good_governance_policy": good_governance_policy,
    "good_lims_clinical": good_lims_clinical,
    "good_macrobase_outliers": good_macrobase_outliers,
    "good_otel_semconv": good_otel_semconv,
    "good_observation_event": good_observation_event,
    "good_dataquality_profile": good_dataquality_profile,
    "good_iam_security": good_iam_security,
    "good_workflow_orchestration": good_workflow_orchestration,
    "good_metadata_governance": good_metadata_governance,
    "good_sample_provenance": good_sample_provenance,
    "good_ebpf_observability": good_ebpf_observability,
    "good_sysml_blocks": good_sysml_blocks,
}


# ---- known-bad test ontologies ----


def bad_empty() -> str:
    return PREFIXES + """
<http://example.org/test/bad-empty> a owl:Ontology .
"""


def bad_trivial() -> str:
    return PREFIXES + """
<http://example.org/test/bad-trivial> a owl:Ontology .

ex:OnlyClass a owl:Class .
"""


def bad_no_labels() -> str:
    return PREFIXES + """
<http://example.org/test/bad-no-labels> a owl:Ontology .

ex:A a owl:Class .
ex:B a owl:Class ;
    rdfs:subClassOf ex:A .
ex:C a owl:Class ;
    rdfs:subClassOf ex:B .
ex:D a owl:Class ;
    rdfs:subClassOf ex:A .
ex:E a owl:Class ;
    rdfs:subClassOf ex:D .
ex:F a owl:Class ;
    rdfs:subClassOf ex:A ,
                    [ a owl:Restriction ;
                      owl:onProperty ex:p ;
                      owl:someValuesFrom ex:E ] .
ex:p a owl:ObjectProperty .
"""


def bad_no_complex() -> str:
    return PREFIXES + """
<http://example.org/test/bad-no-complex> a owl:Ontology .

ex:Top a owl:Class ; rdfs:label "top" .
ex:A a owl:Class ; rdfs:label "a" ; rdfs:subClassOf ex:Top .
ex:B a owl:Class ; rdfs:label "b" ; rdfs:subClassOf ex:Top .
ex:C a owl:Class ; rdfs:label "c" ; rdfs:subClassOf ex:A .
ex:D a owl:Class ; rdfs:label "d" ; rdfs:subClassOf ex:A .
ex:E a owl:Class ; rdfs:label "e" ; rdfs:subClassOf ex:B .
ex:F a owl:Class ; rdfs:label "f" ; rdfs:subClassOf ex:B .
ex:G a owl:Class ; rdfs:label "g" ; rdfs:subClassOf ex:C .
ex:H a owl:Class ; rdfs:label "h" ; rdfs:subClassOf ex:D .
"""


def bad_truncated_lab() -> str:
    return PREFIXES + """
<http://example.org/test/bad-truncated-lab> a owl:Ontology .

ex:Sample a owl:Class .
ex:Instrument a owl:Class .
ex:Measurement a owl:Class .
"""


def bad_random_gibberish() -> str:
    return PREFIXES + """
<http://example.org/test/bad-random> a owl:Ontology .

ex:zqxlf a owl:Class .
ex:wbpkr a owl:Class ; rdfs:subClassOf ex:zqxlf .
ex:nthgv a owl:Class ; rdfs:subClassOf ex:zqxlf .
ex:dcjmu a owl:Class ; rdfs:subClassOf ex:wbpkr .
ex:lkfpe a owl:Class ; rdfs:subClassOf ex:nthgv .
ex:vqsbo a owl:Class ; rdfs:subClassOf ex:dcjmu .
ex:rmnyt a owl:Class ; rdfs:subClassOf ex:lkfpe .
ex:hgwbz a owl:Class ; rdfs:subClassOf ex:vqsbo .
"""


def bad_off_domain_music() -> str:
    return PREFIXES + """
<http://example.org/test/bad-off-domain-music> a owl:Ontology .

ex:MusicalComposition a owl:Class ;
    rdfs:label "musical composition" ;
    skos:definition "An original work of music written for performance." .

ex:Symphony a owl:Class ;
    rdfs:label "symphony" ;
    skos:definition "An orchestral composition typically in four movements." ;
    rdfs:subClassOf ex:MusicalComposition .

ex:Concerto a owl:Class ;
    rdfs:label "concerto" ;
    skos:definition "A composition for a soloist accompanied by an orchestra." ;
    rdfs:subClassOf ex:MusicalComposition .

ex:Sonata a owl:Class ;
    rdfs:label "sonata" ;
    skos:definition "An instrumental composition for one or two performers." ;
    rdfs:subClassOf ex:MusicalComposition .

ex:Composer a owl:Class ;
    rdfs:label "composer" ;
    skos:definition "A musician who creates original musical compositions." .

ex:Performance a owl:Class ;
    rdfs:label "musical performance" ;
    skos:definition "An event in which musicians render a composition for an audience." ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:performs ;
                      owl:someValuesFrom ex:MusicalComposition ] .

ex:Conductor a owl:Class ;
    rdfs:label "orchestra conductor" ;
    skos:definition "A musician who directs an orchestral performance." .

ex:Orchestra a owl:Class ;
    rdfs:label "orchestra" ;
    skos:definition "A large ensemble of musicians performing classical music." .

ex:performs a owl:ObjectProperty ; rdfs:label "performs" .
"""


def bad_off_domain_cooking() -> str:
    return PREFIXES + """
<http://example.org/test/bad-off-domain-cooking> a owl:Ontology .

ex:Recipe a owl:Class ;
    rdfs:label "cooking recipe" ;
    skos:definition "A set of instructions for preparing a dish." ;
    rdfs:subClassOf [ a owl:Restriction ;
                      owl:onProperty ex:hasIngredient ;
                      owl:minCardinality 1 ] .

ex:Ingredient a owl:Class ;
    rdfs:label "culinary ingredient" ;
    skos:definition "A food substance used in cooking." .

ex:Vegetable a owl:Class ; rdfs:label "vegetable" ; rdfs:subClassOf ex:Ingredient .
ex:Spice a owl:Class ; rdfs:label "spice" ; rdfs:subClassOf ex:Ingredient .
ex:Tomato a owl:Class ; rdfs:label "tomato" ; rdfs:subClassOf ex:Vegetable .
ex:Garlic a owl:Class ; rdfs:label "garlic" ; rdfs:subClassOf ex:Vegetable .
ex:Basil a owl:Class ; rdfs:label "basil" ; rdfs:subClassOf ex:Spice .
ex:CookingTechnique a owl:Class ; rdfs:label "cooking technique" .
ex:Sauteing a owl:Class ; rdfs:label "sauteing" ; rdfs:subClassOf ex:CookingTechnique .

ex:hasIngredient a owl:ObjectProperty ; rdfs:label "has ingredient" .
"""


def bad_circular() -> str:
    return PREFIXES + """
<http://example.org/test/bad-circular> a owl:Ontology .

ex:A a owl:Class ; rdfs:subClassOf ex:B .
ex:B a owl:Class ; rdfs:subClassOf ex:C .
ex:C a owl:Class ; rdfs:subClassOf ex:A .
ex:D a owl:Class ; rdfs:subClassOf ex:E .
ex:E a owl:Class ; rdfs:subClassOf ex:D .
"""


def bad_only_disjoint() -> str:
    return PREFIXES + """
<http://example.org/test/bad-only-disjoint> a owl:Ontology .

ex:A a owl:Class ; rdfs:label "A" .
ex:B a owl:Class ; rdfs:label "B" .
ex:C a owl:Class ; rdfs:label "C" .
ex:D a owl:Class ; rdfs:label "D" .
ex:E a owl:Class ; rdfs:label "E" .

[] a owl:AllDisjointClasses ;
   owl:members ( ex:A ex:B ex:C ex:D ex:E ) .
"""


def bad_single_class_chain() -> str:
    return PREFIXES + """
<http://example.org/test/bad-chain> a owl:Ontology .

ex:L1 a owl:Class .
ex:L2 a owl:Class ; rdfs:subClassOf ex:L1 .
ex:L3 a owl:Class ; rdfs:subClassOf ex:L2 .
ex:L4 a owl:Class ; rdfs:subClassOf ex:L3 .
ex:L5 a owl:Class ; rdfs:subClassOf ex:L4 .
ex:L6 a owl:Class ; rdfs:subClassOf ex:L5 .
ex:L7 a owl:Class ; rdfs:subClassOf ex:L6 .
ex:L8 a owl:Class ; rdfs:subClassOf ex:L7 .
"""


def bad_meaningless_iris() -> str:
    return PREFIXES + """
<http://example.org/test/bad-meaningless> a owl:Ontology .

ex:Foo1 a owl:Class .
ex:Foo2 a owl:Class ; rdfs:subClassOf ex:Foo1 .
ex:Foo3 a owl:Class ; rdfs:subClassOf ex:Foo1 .
ex:Foo4 a owl:Class ; rdfs:subClassOf ex:Foo2 .
ex:Foo5 a owl:Class ; rdfs:subClassOf ex:Foo3 .
"""


def bad_two_classes() -> str:
    return PREFIXES + """
<http://example.org/test/bad-tiny> a owl:Ontology .

ex:Tiny a owl:Class .
ex:OtherTiny a owl:Class ; rdfs:subClassOf ex:Tiny .
"""


def bad_corrupted_labels() -> str:
    return PREFIXES + """
<http://example.org/test/bad-corrupted-labels> a owl:Ontology .

ex:A a owl:Class ; rdfs:label "xqzpf" .
ex:B a owl:Class ; rdfs:label "blkrn" ; rdfs:subClassOf ex:A .
ex:C a owl:Class ; rdfs:label "wjvts" ; rdfs:subClassOf ex:A .
ex:D a owl:Class ; rdfs:label "mqhdn" ; rdfs:subClassOf ex:B .
ex:E a owl:Class ; rdfs:label "rkftl" ; rdfs:subClassOf ex:C .
ex:F a owl:Class ; rdfs:label "ngbpu" ; rdfs:subClassOf ex:E .
"""


def bad_one_axiom_per_class() -> str:
    return PREFIXES + """
<http://example.org/test/bad-one-axiom> a owl:Ontology .

ex:A1 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A2 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A3 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A4 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A5 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A6 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A7 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A8 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A9 a owl:Class ; rdfs:subClassOf cco:Artifact .
ex:A10 a owl:Class ; rdfs:subClassOf cco:Artifact .
"""


BAD_GENERATORS = {
    "bad_empty": bad_empty,
    "bad_trivial": bad_trivial,
    "bad_no_labels": bad_no_labels,
    "bad_no_complex": bad_no_complex,
    "bad_truncated_lab": bad_truncated_lab,
    "bad_random_gibberish": bad_random_gibberish,
    "bad_off_domain_music": bad_off_domain_music,
    "bad_off_domain_cooking": bad_off_domain_cooking,
    "bad_circular": bad_circular,
    "bad_only_disjoint": bad_only_disjoint,
    "bad_single_class_chain": bad_single_class_chain,
    "bad_meaningless_iris": bad_meaningless_iris,
    "bad_two_classes": bad_two_classes,
    "bad_corrupted_labels": bad_corrupted_labels,
    "bad_one_axiom_per_class": bad_one_axiom_per_class,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict] = {}

    for name, gen in {**GOOD_GENERATORS, **BAD_GENERATORS}.items():
        ttl = gen()
        path = out_dir / f"{name}.ttl"
        path.write_text(ttl)
        kind = "good" if name.startswith("good_") else "bad"
        labels[f"{name}.ttl"] = {
            "label": kind,
            "kind": name.removeprefix("good_").removeprefix("bad_"),
        }

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")

    n_good = sum(1 for v in labels.values() if v["label"] == "good")
    n_bad = sum(1 for v in labels.values() if v["label"] == "bad")
    print(f"wrote {len(labels)} test ontologies to {out_dir}")
    print(f"  good: {n_good}")
    print(f"  bad:  {n_bad}")
    print(f"  labels: {labels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
