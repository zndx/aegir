#!/usr/bin/env python
"""Generate Batch 7 candidate JSON — long tail / benchmark gap-filling.

This batch is deliberately mechanical. We combine a small set of axiom
shapes with curated property + class lists per gap area, emitting
~150 templates with meaningful template_ids. Each gap area exercises a
distinct slice of the SDG ontology that prior batches under-covered.

Run::

    uv run --no-sync python scripts/gen_batch7.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("src/aegir/ontology/catalog/07_long_tail.candidate.json")


def ex(prop: str, anchor: str = "cco:Artifact") -> dict:
    """F2 — existential restriction template."""
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}, sdg:{prop} some {{Y:Class}}",
        "slots": {"X": "Class", "Y": "Class"},
        "family": "F2-existential",
    }


def uni(prop: str, anchor: str = "cco:Artifact") -> dict:
    """F3 — universal."""
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}, sdg:{prop} only {{Y:Class}}",
        "slots": {"X": "Class", "Y": "Class"},
        "family": "F3-universal",
    }


def card_min(prop: str, anchor: str = "cco:Artifact", n: int = 1) -> dict:
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}, sdg:{prop} min {n} {{Y:Class}}",
        "slots": {"X": "Class", "Y": "Class"},
        "family": "F4-cardinality",
    }


def card_max(prop: str, anchor: str = "cco:Artifact", n: int = 1) -> dict:
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}, sdg:{prop} max {n} {{Y:Class}}",
        "slots": {"X": "Class", "Y": "Class"},
        "family": "F4-cardinality",
    }


def basic(anchor: str = "cco:Artifact") -> dict:
    """F1 — bare subclass."""
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}",
        "slots": {"X": "Class"},
        "family": "F1-subclass",
    }


def union2(anchor: str = "cco:Artifact") -> dict:
    """F6 — equivalent to anchor and (Y or Z)."""
    return {
        "shape": f"Class: {{X:Class}} EquivalentTo: {anchor} and ({{Y:Class}} or {{Z:Class}})",
        "slots": {"X": "Class", "Y": "Class", "Z": "Class"},
        "family": "F6-union",
    }


def neg(anchor: str = "cco:Artifact") -> dict:
    """F7 — subclass with complement."""
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {anchor}, not {{Y:Class}}",
        "slots": {"X": "Class", "Y": "Class"},
        "family": "F7-negation",
    }


def cross_ex(prop: str, target_anchor: str, src_anchor: str = "cco:Artifact") -> dict:
    """Cross-branch — anchor on src, target restricted to anchor class only."""
    return {
        "shape": f"Class: {{X:Class}} SubClassOf: {src_anchor}, sdg:{prop} some {target_anchor}",
        "slots": {"X": "Class"},
        "family": "F2-existential",
    }


# ----- Gap areas --------------------------------------------------

# Schema.org / SOTAB-aligned: properties common in column-type
# annotation benchmarks (CTA target labels: Person, Organization,
# Place, Event, CreativeWork, Product, etc.).
SOTAB_GAP = [
    ("schemaorg_person_named",     ex("hasFullName")),
    ("schemaorg_person_email",     ex("hasEmail")),
    ("schemaorg_person_birthplace", ex("hasBirthplace")),
    ("schemaorg_person_affiliation", ex("hasAffiliation")),
    ("schemaorg_person_roleat",    ex("hasRoleAt")),
    ("schemaorg_org_country",      ex("hasCountry")),
    ("schemaorg_org_taxid",        ex("hasTaxID")),
    ("schemaorg_org_legalname",    ex("hasLegalName")),
    ("schemaorg_place_address",    ex("hasPostalAddress")),
    ("schemaorg_place_geocoord",   ex("hasGeoCoordinates")),
    ("schemaorg_place_latitude",   ex("hasLatitude")),
    ("schemaorg_place_longitude",  ex("hasLongitude")),
    ("schemaorg_event_starttime",  ex("hasEventStartTime")),
    ("schemaorg_event_endtime",    ex("hasEventEndTime")),
    ("schemaorg_event_location",   ex("hasEventLocation")),
    ("schemaorg_event_organizer",  ex("hasOrganizer")),
    ("schemaorg_creativework_author", ex("hasAuthor")),
    ("schemaorg_creativework_date", ex("hasDatePublished")),
    ("schemaorg_creativework_publisher", ex("hasPublisher")),
    ("schemaorg_product_brand",    ex("hasBrand")),
    ("schemaorg_product_sku",      ex("hasSKU")),
    ("schemaorg_product_price",    ex("hasPrice")),
    ("schemaorg_review_rating",    ex("hasReviewRating")),
    ("schemaorg_review_author",    ex("hasReviewAuthor")),
    ("schemaorg_review_target",    ex("reviewsItem")),
]

# CTA/CPA-aligned annotation patterns — column-tag content.
CTA_CPA_GAP = [
    ("column_tag_subclass",          ex("annotatesColumn", "cco:DescriptiveICE")),
    ("column_tag_at_confidence",     ex("hasTagConfidence", "cco:DescriptiveICE")),
    ("column_tag_from_annotator",    ex("byAnnotator", "cco:DescriptiveICE")),
    ("column_property_subclass",     ex("annotatesPropertyOf", "cco:DescriptiveICE")),
    ("column_pair_subject",          ex("hasSubjectColumn", "cco:DescriptiveICE")),
    ("column_pair_object",           ex("hasObjectColumn", "cco:DescriptiveICE")),
    ("table_topic_tag",              ex("hasTableTopic", "cco:DescriptiveICE")),
    ("entity_match_link",            ex("matchesEntity", "cco:DescriptiveICE")),
    ("foreign_key_relation",         ex("foreignKeyTo", "cco:DescriptiveICE")),
    ("primary_key_designation",      ex("isPrimaryKeyOf", "cco:DescriptiveICE")),
    ("functional_dependency_lhs",    ex("hasFDLeftHandSide", "cco:DescriptiveICE")),
    ("functional_dependency_rhs",    ex("hasFDRightHandSide", "cco:DescriptiveICE")),
    ("column_distinctness",          ex("hasDistinctnessRatio", "cco:DescriptiveICE")),
    ("column_nullability_state",     ex("hasNullabilityState", "cco:DescriptiveICE")),
    ("column_value_distribution",    ex("hasValueDistribution", "cco:DescriptiveICE")),
    ("table_provenance_source",      ex("sourcedFromTable", "cco:DescriptiveICE")),
    ("entity_type_belongs_to",       ex("entityTypeBelongsTo", "cco:DescriptiveICE")),
    ("ontology_class_match",         ex("matchesOntologyClass", "cco:DescriptiveICE")),
    ("schemaorg_label_alignment",    ex("alignedToSchemaOrgLabel", "cco:DescriptiveICE")),
    ("dbpedia_label_alignment",      ex("alignedToDBpediaLabel", "cco:DescriptiveICE")),
]

# Compliance-frame depth: NIST 800-53, ISO 27001, SOC2, GDPR.
COMPLIANCE_GAP = [
    ("nist80053_control_subclass", ex("nist80053Control", "cco:DirectiveICE")),
    ("nist80053_low_impact",       ex("atNISTImpactLevel", "cco:DirectiveICE")),
    ("nist80053_moderate_impact",  ex("atNISTImpactLevel", "cco:DirectiveICE")),
    ("nist80053_high_impact",      ex("atNISTImpactLevel", "cco:DirectiveICE")),
    ("iso27001_clause_subclass",   ex("iso27001Clause", "cco:DirectiveICE")),
    ("iso27001_annexa_subclass",   ex("iso27001AnnexA", "cco:DirectiveICE")),
    ("soc2_trust_criterion",       ex("soc2TrustCriterion", "cco:DirectiveICE")),
    ("soc2_security_subclass",     ex("forSOC2Domain", "cco:DirectiveICE")),
    ("soc2_availability_subclass", ex("forSOC2Domain", "cco:DirectiveICE")),
    ("soc2_confidentiality_subclass", ex("forSOC2Domain", "cco:DirectiveICE")),
    ("soc2_processing_integrity",  ex("forSOC2Domain", "cco:DirectiveICE")),
    ("soc2_privacy_subclass",      ex("forSOC2Domain", "cco:DirectiveICE")),
    ("gdpr_article_subclass",      ex("gdprArticle", "cco:DirectiveICE")),
    ("gdpr_lawful_basis",          ex("hasLawfulBasis", "cco:DirectiveICE")),
    ("gdpr_data_subject_right",    ex("grantsDataSubjectRight", "cco:DirectiveICE")),
    ("hipaa_safeguard_admin",      ex("hipaaSafeguard", "cco:DirectiveICE")),
    ("hipaa_safeguard_physical",   ex("hipaaSafeguard", "cco:DirectiveICE")),
    ("hipaa_safeguard_technical",  ex("hipaaSafeguard", "cco:DirectiveICE")),
    ("pci_dss_requirement",        ex("pciDssRequirement", "cco:DirectiveICE")),
    ("policy_only_for_high_impact", uni("atNISTImpactLevel", "cco:DirectiveICE")),
]

# DST combination operators.
DST_GAP = [
    ("dempster_combination_subclass", ex("combinesViaDempster", "cco:DescriptiveICE")),
    ("dempster_combinant_first",   ex("hasFirstCombinant", "cco:DescriptiveICE")),
    ("dempster_combinant_second",  ex("hasSecondCombinant", "cco:DescriptiveICE")),
    ("dempster_conflict_mass",     ex("hasConflictMass", "cco:DescriptiveICE")),
    ("conjunctive_combination",    ex("conjunctivelyCombines", "cco:DescriptiveICE")),
    ("disjunctive_combination",    ex("disjunctivelyCombines", "cco:DescriptiveICE")),
    ("yager_combination",          ex("yagerCombines", "cco:DescriptiveICE")),
    ("frame_refinement",           ex("refinesFrame", "cco:DescriptiveICE")),
    ("frame_coarsening",           ex("coarsensFrame", "cco:DescriptiveICE")),
    ("pignistic_transformation",   ex("hasPignisticTransform", "cco:DescriptiveICE")),
    ("plausibility_function_for",  ex("plausibilityForClaim", "cco:DescriptiveICE")),
    ("belief_function_for",        ex("beliefForClaim", "cco:DescriptiveICE")),
    ("evidence_independent_of",    ex("evidenceIndependentOf", "cco:DescriptiveICE")),
    ("evidence_correlated_with",   ex("evidenceCorrelatedWith", "cco:DescriptiveICE")),
    ("mass_function_normalized_form", ex("hasNormalizedForm", "cco:DescriptiveICE")),
]

# Schema evolution + dataset-version lineage.
SCHEMA_EVO_GAP = [
    ("schema_version_subclass",    ex("hasSchemaVersion")),
    ("schema_revision_of",         ex("schemaRevisionOf")),
    ("column_version_subclass",    ex("hasColumnVersion", "cco:DescriptiveICE")),
    ("column_renamed_from",        ex("renamedFrom", "cco:DescriptiveICE")),
    ("column_type_changed_from",   ex("typeChangedFrom", "cco:DescriptiveICE")),
    ("column_added_at_version",    ex("addedAtSchemaVersion", "cco:DescriptiveICE")),
    ("column_dropped_at_version",  ex("droppedAtSchemaVersion", "cco:DescriptiveICE")),
    ("dataset_snapshot_subclass",  ex("snapshotOf")),
    ("dataset_at_version",         ex("atDatasetVersion")),
    ("breaking_change_subclass",   ex("breakingChangeFor", "cco:DescriptiveICE")),
    ("nonbreaking_change_subclass", ex("nonBreakingChangeFor", "cco:DescriptiveICE")),
    ("schema_migration_subclass",  ex("migratesSchemaTo", "bfo:0000015")),
    ("rollback_relation",          ex("rollsBackTo", "cco:DescriptiveICE")),
    ("forward_compatible_with",    ex("forwardCompatibleWith", "cco:DescriptiveICE")),
    ("backward_compatible_with",   ex("backwardCompatibleWith", "cco:DescriptiveICE")),
]

# Telemetry / Span / Metric / Log specializations
TELEMETRY_GAP = [
    ("opentelemetry_span_subclass", ex("hasSpanContext", "bfo:0000015")),
    ("span_with_parent",            ex("hasParentSpan", "bfo:0000015")),
    ("span_with_trace_id",          ex("hasTraceId", "bfo:0000015")),
    ("span_with_attribute",         ex("hasSpanAttribute", "bfo:0000015")),
    ("span_with_status",            ex("hasSpanStatus", "bfo:0000015")),
    ("metric_emission_subclass",    ex("emitsMetric", "bfo:0000015")),
    ("metric_with_unit",            ex("hasMetricUnit")),
    ("metric_with_aggregation",     ex("hasAggregationTemporality")),
    ("counter_metric_subclass",     ex("hasMetricKind")),
    ("gauge_metric_subclass",       ex("hasMetricKind")),
    ("histogram_metric_subclass",   ex("hasMetricKind")),
    ("summary_metric_subclass",     ex("hasMetricKind")),
    ("log_record_subclass",         ex("hasLogSeverity")),
    ("log_with_resource_attribute", ex("hasResourceAttribute")),
    ("log_within_span_context",     ex("withinSpanContext")),
    ("trace_with_root_span",        ex("hasRootSpan")),
    ("trace_with_service_name",     ex("hasServiceName")),
    ("anomaly_in_metric",           ex("anomalyInMetric", "cco:DescriptiveICE")),
    ("baseline_for_metric",         ex("baselineFor", "cco:DescriptiveICE")),
    ("alert_triggered_by",          ex("triggeredByMetric", "bfo:0000015")),
]

# eBPF utility extensions
EBPF_GAP = [
    ("uretprobe_subclass",         ex("attachesToReturnHook")),
    ("perf_event_subclass",        ex("hasPerfEventConfig")),
    ("bpf_helper_subclass",        ex("hasBPFHelperFunction")),
    ("bpf_helper_called_by",       ex("calledByProgram")),
    ("ringbuf_map_subclass",       ex("hasRingBufferConfig")),
    ("perf_buffer_map_subclass",   ex("hasPerfBufferConfig")),
    ("lru_hash_map_subclass",      ex("hasLRUHashConfig")),
    ("xdp_action_drop",            ex("hasXDPAction")),
    ("xdp_action_pass",            ex("hasXDPAction")),
    ("xdp_action_redirect",        ex("hasXDPAction")),
    ("tc_classifier_subclass",     ex("hasTCDirection")),
    ("cgroup_skb_subclass",        ex("attachesToCgroup")),
    ("kernel_event_with_pid",      ex("withProcessId", "bfo:0000015")),
    ("kernel_event_with_tid",      ex("withThreadId", "bfo:0000015")),
    ("kernel_event_with_comm",     ex("withProcessComm", "bfo:0000015")),
]

# Cross-branch density (diagonal connections)
CROSS_GAP = [
    ("trace_supports_claim",       cross_ex("supportsClaim", "cco:DescriptiveICE", "bfo:0000015")),
    ("audit_produces_evidence",    cross_ex("producesEvidence", "cco:DescriptiveICE", "bfo:0000015")),
    ("transformation_governed_by", cross_ex("governedBy", "cco:DirectiveICE", "bfo:0000015")),
    ("ebpf_program_governed_by_directive", cross_ex("governedBy", "cco:DirectiveICE")),
    ("lineage_describes_transformation", cross_ex("describesTransformation", "bfo:0000015", "cco:DescriptiveICE")),
    ("syscall_governed_by_directive", cross_ex("governedBy", "cco:DirectiveICE", "cco:DesignativeICE")),
    ("claim_observed_by_event",    cross_ex("observedAt", "bfo:0000015", "cco:DescriptiveICE")),
    ("metric_observed_by_event",   cross_ex("observedAt", "bfo:0000015", "cco:DescriptiveICE")),
    ("policy_attests_artifact",    cross_ex("attestsToArtifact", "cco:Artifact", "cco:DirectiveICE")),
    ("evidence_for_audit",         cross_ex("evidenceForProcess", "bfo:0000015", "cco:DescriptiveICE")),
    ("provenance_agent_signs_audit", cross_ex("signsProcess", "bfo:0000015")),
    ("dataset_under_audit",        cross_ex("underAudit", "bfo:0000015")),
    ("schema_evolution_under_directive", cross_ex("governedBy", "cco:DirectiveICE", "cco:DescriptiveICE")),
    ("kernel_anomaly_supports_claim", cross_ex("supportsClaim", "cco:DescriptiveICE")),
    ("telemetry_span_observes_syscall", cross_ex("observesSyscall", "cco:DesignativeICE", "bfo:0000015")),
    ("attestation_about_compliance_claim", cross_ex("attestsToClaim", "cco:DescriptiveICE", "bfo:0000015")),
    ("dempster_combines_audit_evidences", cross_ex("combinesViaDempster", "cco:DescriptiveICE")),
    ("column_lineage_for_compliance_claim", cross_ex("supportsClaim", "cco:DescriptiveICE")),
    ("alert_governed_by_runbook",  cross_ex("governedBy", "cco:DirectiveICE", "bfo:0000015")),
    ("frozen_artifact_governed_by", cross_ex("governedBy", "cco:DirectiveICE")),
]

# Mixed-pattern boosters: union, cardinality, negation across branches.
BOOSTERS = [
    ("artifact_min_one_owner",     card_min("hasOwner", n=1)),
    ("artifact_max_one_owner",     card_max("hasOwner", n=1)),
    ("descriptive_only_about_artifact", uni("isAbout", "cco:DescriptiveICE")),
    ("directive_only_governs_artifact", uni("governs", "cco:DirectiveICE")),
    ("process_min_one_input",      card_min("hasInput", "bfo:0000015", n=1)),
    ("process_min_one_output",     card_min("hasOutput", "bfo:0000015", n=1)),
    ("process_max_one_operator",   card_max("hasOperator", "bfo:0000015", n=1)),
    ("artifact_either_active_or_archived", union2()),
    ("artifact_either_internal_or_external", union2()),
    ("descriptive_either_evidence_or_claim", union2("cco:DescriptiveICE")),
    ("artifact_not_deprecated",    neg()),
    ("artifact_not_revoked",       neg()),
    ("descriptive_not_retracted",  neg("cco:DescriptiveICE")),
    ("directive_not_superseded",   neg("cco:DirectiveICE")),
    ("process_either_started_or_completed", union2("bfo:0000015")),
]

ALL_GAPS = [
    ("Artifact.SOTAB",          SOTAB_GAP),
    ("DescriptiveICE.CTACPA",   CTA_CPA_GAP),
    ("DirectiveICE.Compliance", COMPLIANCE_GAP),
    ("DescriptiveICE.DSTOps",   DST_GAP),
    ("Artifact.SchemaEvolution", SCHEMA_EVO_GAP),
    ("Process.Telemetry",       TELEMETRY_GAP),
    ("Artifact.eBPFExtension",  EBPF_GAP),
    ("CrossBranch.density",     CROSS_GAP),
    ("AxiomFamily.boosters",    BOOSTERS),
]


def emit(branch: str, name: str, spec: dict) -> dict:
    anchor = "cco:Artifact"
    shape = spec["shape"]
    if "cco:DescriptiveICE" in shape:
        anchor = "cco:DescriptiveICE"
    elif "cco:DirectiveICE" in shape:
        anchor = "cco:DirectiveICE"
    elif "cco:DesignativeICE" in shape:
        anchor = "cco:DesignativeICE"
    elif "bfo:0000015" in shape:
        anchor = "bfo:Process"
    return {
        "template_id": name,
        "manchester_template": shape,
        "slot_types": spec["slots"],
        "is_complex": False, "verbal_template": "", "mean_verbal_length": 0.0,
        "bfo_anchor_path": [anchor],
        "provenance": {
            "author": "scaffolding",
            "date": "2026-05-10",
            "family": spec["family"],
            "branch": branch,
        },
    }


def main() -> None:
    templates = []
    seen_ids: set[str] = set()
    seen_shape_slot: set[tuple[str, frozenset]] = set()
    for branch, gap in ALL_GAPS:
        for name, spec in gap:
            base_name = name
            i = 0
            while name in seen_ids:
                i += 1
                name = f"{base_name}_v{i}"
            seen_ids.add(name)
            shape_slot_key = (spec["shape"], frozenset(spec["slots"].items()))
            if shape_slot_key in seen_shape_slot:
                shape_slot_key = (spec["shape"] + f"#{name}", frozenset(spec["slots"].items()))
            seen_shape_slot.add(shape_slot_key)
            templates.append(emit(branch, name, spec))

    out = {"version": "0.7.0-long-tail-candidate", "templates": templates}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(templates)} templates to {OUT}")
    fams: dict = {}
    brs: dict = {}
    for t in templates:
        fams[t["provenance"]["family"]] = fams.get(t["provenance"]["family"], 0) + 1
        brs[t["provenance"]["branch"]] = brs.get(t["provenance"]["branch"], 0) + 1
    print("families:", dict(sorted(fams.items())))
    print("branches:")
    for k, v in sorted(brs.items()):
        print(f"  {k:40s}{v}")


if __name__ == "__main__":
    main()
