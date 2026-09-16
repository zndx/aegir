#!/usr/bin/env python
"""Emit a disjoint NHSVM-train reference sample for Atelier's holdout target.

SoR: ``docs/current/src/operations/atelier_holdout_reference.md``.

Sources are SchemaPile schemas (HR backbone ``sp:450157``, recruiting
``sp:033482``, mzTab chemistry ``sp:555914``) plus FinePDFs term lineage
(``fp:104813:51b4c0cd``, ``fp:107423:8d197e5c``, ``fp:81923:c7603e10``).
Row values are type-true and RI-closed by construction — not GitTables.

    uv run --no-sync python scripts/emit_holdout_reference.py
    uv run --no-sync python scripts/emit_holdout_reference.py --install-corpora
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN = "b24ef9f606605dd7bb8877bddd2d9672c78282ad"
PIN12 = PIN[:12]
PROFILE = "reference"
FORBIDDEN = frozenset(
    {"computational-material-record-044e3d9a", "position-ac7bde24"}
)
REQUIRED_IRIS = (
    "bfo:0000002",
    "bfo:0000003",
    "bfo:0000004",
    "bfo:0000015",
    "bfo:0000016",
    "bfo:0000023",
    "bfo:0000040",
    "cco:ont00000958",
    "cco:ont00000995",
)
# Target annotations.csv codes (assert_pair PAIRSKOS / perfect_possible).
TARGET_SKOS = (
    "SDG.ARTIFACT",
    "SDG.DISPOSITION",
    "SDG.DOM.BIRTHPLACE",
    "SDG.DOM.COUNTRY",
    "SDG.DOM.CREATED_DATE",
    "SDG.DOM.DATE_PUBLISHED",
    "SDG.DOM.EFFECTIVE_DATE",
    "SDG.DOM.GEOGRAPHIC_PROVENANCE",
    "SDG.DOM.HOME",
    "SDG.DOM.ISSUED_DATE",
    "SDG.DOM.LOCATION",
    "SDG.DOM.PUBLISHED_ON_DATE",
    "SDG.DOM.REGIONAL",
    "SDG.DOM.SINCE",
    "SDG.DOM.STATE",
    "SDG.DOM.TEMPORAL_DESCRIPTOR",
    "SDG.GDC",
    "SDG.GENERIC",
    "SDG.ICE",
    "SDG.ICE.DESCRIPTIVE",
    "SDG.INDEPENDENT_CONTINUANT",
    "SDG.MATERIAL_ENTITY",
    "SDG.PROCESS",
    "SDG.ROLE",
)
N_ROWS = 4
JUNCTION_ROWS = 8

_GENUS_TO_ANCHOR = {
    "bfo:0000015": "SDG.PROCESS",
    "bfo:0000004": "SDG.INDEPENDENT_CONTINUANT",
    "bfo:0000040": "SDG.MATERIAL_ENTITY",
    "bfo:0000031": "SDG.GDC",
    "bfo:0000019": "SDG.QUALITY",
    "bfo:0000023": "SDG.ROLE",
    "bfo:0000016": "SDG.DISPOSITION",
    "bfo:0000002": "SDG.INDEPENDENT_CONTINUANT",
    "cco:ont00000995": "SDG.ARTIFACT",
    "cco:ont00000958": "SDG.ICE",
    "cco:ont00000853": "SDG.ICE.DESCRIPTIVE",
    "cco:ont00000965": "SDG.ICE.DIRECTIVE",
    "cco:ont00000686": "SDG.ICE.DESIGNATIVE",
}
_FALLBACK_ANCHOR = "SDG.GENERIC"
_UNMAPPED = "bfo:0000003"

# ── SchemaPile / FinePDFs lineage (kvasir find on pin-era indexes) ──
SP_HR = "sp:450157"          # employees/departments/jobs/locations/countries
SP_JOB = "sp:033482"         # Job.country, Interviews.location, Candidates.first_name
SP_CHEM = "sp:555914"        # mzTab chemical_formula / PROTOCOLS
FP_HR = "fp:104813:51b4c0cd"
FP_MAT = "fp:107423:8d197e5c"
FP_GEO = "fp:81923:c7603e10"


def _cid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _humanize(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return s.replace("_", " ").strip().title()


def _screaming(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).replace("-", "_").replace(" ", "_").upper()


# ── Collection specs ──────────────────────────────────────────

@dataclass
class Term:
    name: str
    genus: str


@dataclass
class Table:
    name: str
    columns: list[str]           # including id and *_id FKs
    fks: dict[str, str] = field(default_factory=dict)  # col → target table
    junction: bool = False


@dataclass
class CollectionSpec:
    key: str
    title: str
    sources: dict
    terms: list[Term]
    tables: list[Table]


def _staff() -> CollectionSpec:
    """SchemaPile HR/recruiting backbone (450157 + 033482) with emit-list columns."""
    return CollectionSpec(
        key="staff-assignment",
        title="Staff assignment",
        sources={
            "schemapile": [f"{SP_HR}:employees", f"{SP_HR}:departments", f"{SP_HR}:jobs",
                           f"{SP_HR}:locations", f"{SP_HR}:countries", f"{SP_JOB}:Job",
                           f"{SP_JOB}:Interviews", f"{SP_JOB}:Candidates"],
            "finepdfs": [FP_HR, FP_GEO],
        },
        terms=[
            Term("Employee", "bfo:0000002"),
            Term("Department", "bfo:0000003"),
            Term("Position", "bfo:0000023"),
            Term("CollectiveBargainingAgreement", "cco:ont00000958"),
            Term("PerformanceReview", "bfo:0000015"),
            Term("Job", "bfo:0000023"),
            Term("Country", "bfo:0000004"),
            Term("WorkLocation", "bfo:0000004"),
        ],
        tables=[
            Table("regions", ["id", "name", "regional"]),
            Table("countries",
                  ["id", "country", "name", "state", "region_id"],
                  fks={"region_id": "regions"}),
            Table("locations",
                  ["id", "location", "home", "birthplace", "state", "regional",
                   "country_id"],
                  fks={"country_id": "countries"}),
            Table("departments",
                  ["id", "name", "code", "established_date", "status", "location_id"],
                  fks={"location_id": "locations"}),
            Table("jobs",
                  ["id", "title", "classification", "salary_range", "approval_date",
                   "status", "code"]),
            Table("employees",
                  ["id", "first_name", "last_name", "hire_date", "termination_date",
                   "employment_status", "title", "salary_range", "job_id",
                   "department_id"],
                  fks={"job_id": "jobs", "department_id": "departments"}),
            Table("dependents",
                  ["id", "first_name", "last_name", "employee_id"],
                  fks={"employee_id": "employees"}),
            Table("labor_agreements",
                  ["id", "effective_date", "expiration_date", "union_name", "status",
                   "start_date", "end_date", "employee_id", "job_id"],
                  fks={"employee_id": "employees", "job_id": "jobs"}),
            Table("performance_appraisals",
                  ["id", "review_date", "rating", "comments", "status",
                   "employee_id", "job_id"],
                  fks={"employee_id": "employees", "job_id": "jobs"}),
            Table("departments_jobs",
                  ["department_id", "job_id"],
                  fks={"department_id": "departments", "job_id": "jobs"},
                  junction=True),
        ],
    )


def _crystal() -> CollectionSpec:
    """FinePDFs crystal/lattice terms + SchemaPile mzTab chemistry (555914)."""
    return CollectionSpec(
        key="crystal-assay",
        title="Crystal assay",
        sources={
            "schemapile": [f"{SP_CHEM}:public.MzTab_M_small_molecule_section",
                           f"{SP_CHEM}:public.PROTOCOLS",
                           f"{SP_CHEM}:public.PQP_compound"],
            "finepdfs": [FP_MAT, FP_GEO],
        },
        terms=[
            Term("ComputationalMaterialRecord", "cco:ont00000958"),
            Term("ComputationalCode", "cco:ont00000995"),
            Term("MaterialSystem", "bfo:0000040"),
            Term("MaterialProperty", "bfo:0000016"),
            Term("ParsingPipeline", "bfo:0000015"),
            Term("HighPerformanceComputerCenter", "cco:ont00000995"),
            Term("BigDataAnalyticsTool", "cco:ont00000995"),
            Term("ParsedDataStore", "cco:ont00000995"),
            Term("VisualizationService", "bfo:0000015"),
            Term("SmallMolecule", "bfo:0000040"),
            Term("InstrumentRun", "bfo:0000003"),
        ],
        tables=[
            Table("small_molecules",
                  ["id", "chemical_formula", "name", "crystal_system"]),
            Table("crystal_lattices",
                  ["id", "lattice_parameter_a", "lattice_parameter_b",
                   "lattice_parameter_c", "space_group_number", "formation_energy",
                   "small_molecule_id"],
                  fks={"small_molecule_id": "small_molecules"}),
            Table("assay_protocols",
                  ["id", "pipeline_name", "algorithm_type", "measurement_method",
                   "version", "status"]),
            Table("instrument_runs",
                  ["id", "start_date_time", "end_date_time", "created_at",
                   "updated_at", "assay_protocol_id"],
                  fks={"assay_protocol_id": "assay_protocols"}),
            Table("compute_centers",
                  ["id", "center_name", "location", "peak_performance_flops",
                   "total_compute_nodes", "operational_status", "uptime_percentage"]),
            Table("analytics_tools",
                  ["id", "tool_name", "license_type", "version",
                   "max_concurrent_users", "endpoint_url", "compute_center_id"],
                  fks={"compute_center_id": "compute_centers"}),
            Table("parsed_stores",
                  ["id", "store_name", "file_format", "file_size_bytes", "checksum",
                   "storage_size_g_b", "total_records", "data_integrity_status",
                   "analytics_tool_id"],
                  fks={"analytics_tool_id": "analytics_tools"}),
            Table("viz_services",
                  ["id", "service_name", "endpoint_url", "status",
                   "records_processed", "parsed_store_id"],
                  fks={"parsed_store_id": "parsed_stores"}),
            Table("software_codes",
                  ["id", "code_name", "code", "developer_group", "version",
                   "created_at", "last_updated"]),
            Table("material_records",
                  ["id", "creation_date", "created_date", "date_published",
                   "issued_date", "published_on_date", "since", "active_status",
                   "small_molecule_id", "assay_protocol_id", "software_code_id"],
                  fks={"small_molecule_id": "small_molecules",
                       "assay_protocol_id": "assay_protocols",
                       "software_code_id": "software_codes"}),
            Table("material_measurements",
                  ["id", "property_name", "numeric_value", "unit_of_measure",
                   "confidence_interval", "measurement_method",
                   "novel_materials_discovered", "material_record_id"],
                  fks={"material_record_id": "material_records"}),
            Table("protocols_records",
                  ["assay_protocol_id", "material_record_id"],
                  fks={"assay_protocol_id": "assay_protocols",
                       "material_record_id": "material_records"},
                  junction=True),
        ],
    )


SPECS = (_staff(), _crystal())


# ── Value synthesis (type-true, fictional particulars) ────────

_POOLS: dict[str, list[str]] = {
    "first_name": ["Marta", "Jules", "Priya", "Owen"],
    "last_name": ["Keene", "Okada", "Nilsen", "Barrow"],
    "name": ["Adaptive Corridor", "Regional Survey", "Baseline Programme", "Pilot Cluster"],
    "title": ["Analyst", "Coordinator", "Specialist", "Lead"],
    "status": ["active", "pending", "closed", "draft"],
    "employment_status": ["active", "on_leave", "terminated", "probation"],
    "operational_status": ["online", "degraded", "maintenance", "offline"],
    "active_status": ["active", "inactive", "retired", "staging"],
    "data_integrity_status": ["verified", "pending", "failed", "repaired"],
    "classification": ["exempt", "nonexempt", "contract", "seasonal"],
    "salary_range": ["band-a", "band-b", "band-c", "band-d"],
    "union_name": ["Craft Guild North", "Technical Compact", "Regional Alliance", "Works Council"],
    "country": ["SE", "JP", "CA", "PT"],
    "state": ["north", "central", "coastal", "highland"],
    "regional": ["atlantic", "pacific", "inland", "isthmus"],
    "location": ["harbour-lab", "ridge-campus", "delta-site", "mesa-yard"],
    "home": ["district-4", "ward-12", "parish-west", "canton-3"],
    "birthplace": ["Lisbon", "Sapporo", "Halifax", "Malmo"],
    "code": ["A-01", "B-12", "C-07", "D-33"],
    "comments": ["meets-bar", "needs-followup", "exceeds", "deferred"],
    "chemical_formula": ["TiO2", "SiC", "Al2O3", "BN"],
    "crystal_system": ["cubic", "hexagonal", "tetragonal", "orthorhombic"],
    "pipeline_name": ["ingest-alpha", "refine-beta", "parse-gamma", "close-delta"],
    "algorithm_type": ["dft", "md", "ml-potential", "hybrid"],
    "measurement_method": ["xrd", "tem", "calorimetry", "spectroscopy"],
    "version": ["1.2.0", "1.3.1", "2.0.0", "2.1.4"],
    "center_name": ["Ridge HPC", "Delta Lab", "Mesa Core", "Harbour Cluster"],
    "tool_name": ["lattice-kit", "spectra-box", "mesh-runner", "peak-fit"],
    "license_type": ["Apache-2.0", "BSD-3-Clause", "MIT", "MPL-2.0"],
    "endpoint_url": ["https://lab.example/v1", "https://hpc.example/run",
                     "https://store.example/api", "https://viz.example/view"],
    "store_name": ["raw-lake", "curated-vault", "scratch-pool", "gold-shelf"],
    "file_format": ["parquet", "csv", "netcdf", "hdf5"],
    "checksum": ["a3f9c21e", "7b14de08", "c0ffee42", "9d2b7a16"],
    "code_name": ["qespresso-fork", "lammps-set", "vasp-wrap", "gpaw-batch"],
    "developer_group": ["materials-sim", "data-eng", "hpc-ops", "viz-lab"],
    "service_name": ["structure-view", "spectra-plot", "mesh-preview", "run-board"],
    "property_name": ["band-gap", "bulk-modulus", "formation-enthalpy", "density"],
    "unit_of_measure": ["eV", "GPa", "kJ/mol", "g/cm3"],
    "file_size_bytes": ["4096", "8192", "16384", "32768"],
    "storage_size_g_b": ["12", "48", "96", "256"],
    "total_records": ["120", "340", "780", "1500"],
    "total_compute_nodes": ["32", "64", "128", "256"],
    "peak_performance_flops": ["1.2e14", "4.8e14", "9.1e14", "2.0e15"],
    "uptime_percentage": ["99.1", "99.4", "98.7", "99.9"],
    "max_concurrent_users": ["8", "16", "32", "64"],
    "records_processed": ["40", "80", "160", "320"],
    "numeric_value": ["1.95", "3.40", "5.85", "7.20"],
    "confidence_interval": ["0.02", "0.05", "0.08", "0.11"],
    "lattice_parameter_a": ["3.95", "4.12", "5.40", "2.87"],
    "lattice_parameter_b": ["3.95", "4.12", "5.40", "2.87"],
    "lattice_parameter_c": ["3.95", "6.70", "5.40", "4.10"],
    "space_group_number": ["225", "194", "136", "62"],
    "formation_energy": ["-1.20", "-3.45", "-0.80", "-5.10"],
    "rating": ["3", "4", "5", "2"],
    "novel_materials_discovered": ["1", "0", "2", "1"],
}


_DATE_COLS = {
    "approval_date", "creation_date", "created_date", "date_published",
    "effective_date", "end_date", "established_date", "expiration_date",
    "hire_date", "issued_date", "published_on_date", "review_date",
    "start_date", "termination_date",
}
_DT_COLS = {
    "created_at", "end_date_time", "last_updated", "start_date_time",
    "updated_at",
}
_INT_COLS = {
    "file_size_bytes", "max_concurrent_users", "novel_materials_discovered",
    "rating", "records_processed", "space_group_number", "storage_size_g_b",
    "total_compute_nodes", "total_records",
}


# Day offsets so paired start/end columns stay ordered (hire < termination, …).
_DATE_OFFSET = {
    "since": 0, "established_date": 0, "created_date": 10, "creation_date": 12,
    "issued_date": 20, "date_published": 30, "published_on_date": 32,
    "hire_date": 40, "start_date": 40, "approval_date": 45, "effective_date": 50,
    "review_date": 80, "end_date": 140, "termination_date": 200,
    "expiration_date": 400,
}
_DT_OFFSET = {
    "created_at": 0, "start_date_time": 8, "last_updated": 48, "updated_at": 48,
    "end_date_time": 96,
}


def _iso_date(i: int, days: int) -> str:
    return (date(2022, 3, 1) + timedelta(days=i * 17 + days)).isoformat()


def _iso_dt(i: int, days: int) -> str:
    d = date(2022, 3, 1) + timedelta(days=i * 17 + days)
    hh = (8 + i * 3) % 24
    mm = (i * 11) % 60
    return f"{d.isoformat()}T{hh:02d}:{mm:02d}:00"


def _cell(table: str, col: str, i: int) -> str:
    if col == "id" or col.endswith("_id"):
        return str(i + 1)
    if table == "small_molecules" and col == "name":
        return ["anatase-titania", "beta-sic", "alpha-alumina", "hex-bn"][i % 4]
    pool = _POOLS.get(col)
    if pool:
        return pool[i % len(pool)]
    h = int(hashlib.blake2b(f"{table}|{col}".encode(), digest_size=4).digest().hex(), 16)
    if col in _DATE_OFFSET:
        return _iso_date(i, _DATE_OFFSET[col])
    if col in _DATE_COLS:
        return _iso_date(i, 20)
    if col in _DT_OFFSET:
        return _iso_dt(i, _DT_OFFSET[col])
    if col in _DT_COLS:
        return _iso_dt(i, 8)
    if col in _INT_COLS:
        return str((i + 1) * (3 + h % 9) + h % 40)
    return f"{table[:4]}-{col[:6]}-{(h % 90) + i + 10}"


def _sql_type(col: str) -> str:
    if col == "id" or col.endswith("_id"):
        return "INTEGER"
    if col in _DATE_COLS or col == "since":
        return "DATE"
    if col in _DT_COLS:
        return "TIMESTAMP"
    if col in _INT_COLS:
        return "INTEGER"
    if col in {"lattice_parameter_a", "lattice_parameter_b", "lattice_parameter_c",
               "formation_energy", "numeric_value", "confidence_interval",
               "peak_performance_flops", "uptime_percentage"}:
        return "DECIMAL"
    return "VARCHAR(64)"


def materialize_table(t: Table) -> list[dict]:
    n = JUNCTION_ROWS if t.junction else N_ROWS
    rows: list[dict] = []
    for i in range(n):
        row = {}
        for col in t.columns:
            if t.junction and col in t.fks:
                # pair (i, i) and (i, i+1) cycling over N_ROWS
                src = i % N_ROWS
                if col == t.columns[0]:
                    row[col] = str(src + 1)
                else:
                    row[col] = str(((src + (i // N_ROWS)) % N_ROWS) + 1)
            elif col in t.fks:
                row[col] = str((i % N_ROWS) + 1)
            elif col == "id":
                row[col] = str(i + 1)
            else:
                row[col] = _cell(t.name, col, i % N_ROWS)
        rows.append(row)
    return rows


def render_sql(t: Table) -> str:
    lines = [f"  {c} {_sql_type(c)}" + (" NOT NULL" if c == "id" else "") for c in t.columns]
    if "id" in t.columns:
        lines.append("  PRIMARY KEY (id)")
    elif t.junction:
        lines.append(f"  PRIMARY KEY ({', '.join(t.columns)})")
    for col, ref in t.fks.items():
        lines.append(f"  FOREIGN KEY ({col}) REFERENCES {ref} (id)")
    return f"CREATE TABLE {t.name} (\n" + ",\n".join(lines) + "\n);\n"


def verify_ri(spec: CollectionSpec, tables: dict[str, list[dict]]) -> dict:
    """Atelier-compatible convention RI (``<x>_id`` → sibling table)."""
    edges, designative = [], []
    orphan_total = 0
    for stem, rows in tables.items():
        if not rows:
            continue
        for col in rows[0].keys():
            m = re.match(r"(.+)_id$", col)
            if not m:
                continue
            target = None
            for cand in (m.group(1), m.group(1) + "s", m.group(1) + "es"):
                if cand in tables and cand != stem:
                    target = cand
                    break
            if target is None:
                if col != "id":
                    designative.append(f"{spec.key}:{stem}.{col}")
                continue
            key = "id" if rows and "id" in tables[target][0] else None
            if key is None:
                designative.append(f"{spec.key}:{stem}.{col}(→{target}, no key)")
                continue
            valid = {r.get(key) for r in tables[target]}
            vals = [r[col] for r in rows if r.get(col)]
            orphans = [v for v in vals if v not in valid]
            orphan_total += len(orphans)
            edges.append({"collection": spec.key, "table": stem, "column": col,
                          "references": target, "key": key,
                          "checked": len(vals), "orphans": len(orphans)})
    if orphan_total:
        bad = [e for e in edges if e["orphans"]]
        raise SystemExit(f"RI failed in {spec.key}: {bad[0]}")
    return {"edges": edges, "designative_references": designative, "orphan_total": 0}


# ── SKOS gap terms ────────────────────────────────────────────

def gap_skos_records() -> list[dict]:
    """New SDG.DOM.* rows for emit-list entity and column names.

    Attached under existing anchors / TEMPORAL_DESCRIPTOR so parent closure
    stays inside the target SKOS plus a few new hypernyms.
    """
    recs: list[dict] = []

    def add(code, label, abbrev, notation, parent, desc, common=""):
        recs.append({
            "code": code, "label": label, "abbrev": abbrev, "notation": notation,
            "parent_code": parent, "taxonomy": "sdg", "description": desc,
            "common_names": common, "example_values": "", "axiom": "",
            "family": "holdout_reference",
        })

    # Hypernyms (D84+)
    add("SDG.DOM.NAMED_SURFACE", "Named Surface", "NAMED_SURFACE", "3.3.D84",
        "SDG.ICE", "Domain hypernym for person, place, and artifact name columns.",
        "domain_hypernym")
    add("SDG.DOM.OPERATIONAL_STATE", "Operational State", "OPERATIONAL_STATE", "3.1.D85",
        "SDG.ICE.DESCRIPTIVE", "Domain hypernym for status and classification columns.",
        "domain_hypernym")
    add("SDG.DOM.MEASURED_QUANTITY", "Measured Quantity", "MEASURED_QUANTITY", "6.D86",
        "SDG.DISPOSITION", "Domain hypernym for numeric measurement columns.",
        "domain_hypernym")
    add("SDG.DOM.ARTIFACT_DESCRIPTOR", "Artifact Descriptor", "ARTIFACT_DESCRIPTOR", "2.1.D87",
        "SDG.ARTIFACT", "Domain hypernym for artifact identity and format columns.",
        "domain_hypernym")
    add("SDG.DOM.WORKFORCE_CONTINUANT", "Workforce Continuant", "WORKFORCE_CONTINUANT", "2.D88",
        "SDG.INDEPENDENT_CONTINUANT", "Domain hypernym for employee and department entities.",
        "domain_hypernym")
    add("SDG.DOM.OCCUPATIONAL_ROLE", "Occupational Role", "OCCUPATIONAL_ROLE", "5.D89",
        "SDG.ROLE", "Domain hypernym for position and job-role entities.",
        "domain_hypernym")
    add("SDG.DOM.LABOR_RECORD", "Labor Record", "LABOR_RECORD", "3.D90",
        "SDG.ICE", "Domain hypernym for agreements and reviews as information content.",
        "domain_hypernym")
    add("SDG.DOM.SCIENTIFIC_PIPELINE", "Scientific Pipeline", "SCIENTIFIC_PIPELINE", "1.D91",
        "SDG.PROCESS", "Domain hypernym for parsing, visualization, and assay processes.",
        "domain_hypernym")
    add("SDG.DOM.COMPUTATIONAL_MATERIAL", "Computational Material", "COMPUTATIONAL_MATERIAL",
        "2.2.D92", "SDG.MATERIAL_ENTITY",
        "Domain hypernym for material systems, properties, and crystal lattices.",
        "domain_hypernym")
    add("SDG.DOM.RESEARCH_INFRASTRUCTURE", "Research Infrastructure", "RESEARCH_INFRASTRUCTURE",
        "2.1.D93", "SDG.ARTIFACT",
        "Domain hypernym for HPC centers, analytics tools, stores, and codes.",
        "domain_hypernym")

    temporal = [
        "approval_date", "created_at", "creation_date", "end_date", "end_date_time",
        "established_date", "expiration_date", "hire_date", "last_updated",
        "review_date", "start_date", "start_date_time", "termination_date", "updated_at",
    ]
    for i, col in enumerate(temporal, 7):
        add(f"SDG.DOM.{col.upper()}", _humanize(col), col.upper(),
            f"3.1.D62.{i}", "SDG.DOM.TEMPORAL_DESCRIPTOR",
            f"Domain concept under temporal_descriptor ({col}).", "domain_member")

    named = [
        "center_name", "code_name", "first_name", "last_name", "name", "pipeline_name",
        "property_name", "service_name", "store_name", "title", "tool_name", "union_name",
    ]
    for i, col in enumerate(named, 1):
        add(f"SDG.DOM.{col.upper()}", _humanize(col), col.upper(),
            f"3.3.D84.{i}", "SDG.DOM.NAMED_SURFACE",
            f"Domain concept under named_surface ({col}).", "domain_member")

    states = [
        "active_status", "data_integrity_status", "employment_status",
        "operational_status", "status", "classification",
    ]
    for i, col in enumerate(states, 1):
        add(f"SDG.DOM.{col.upper()}", _humanize(col), col.upper(),
            f"3.1.D85.{i}", "SDG.DOM.OPERATIONAL_STATE",
            f"Domain concept under operational_state ({col}).", "domain_member")

    measures = [
        "confidence_interval", "formation_energy", "lattice_parameter_a",
        "lattice_parameter_b", "lattice_parameter_c", "numeric_value",
        "peak_performance_flops", "rating", "records_processed",
        "space_group_number", "storage_size_g_b", "total_compute_nodes",
        "total_records", "unit_of_measure", "uptime_percentage",
        "file_size_bytes", "max_concurrent_users", "novel_materials_discovered",
    ]
    for i, col in enumerate(measures, 1):
        add(f"SDG.DOM.{col.upper()}", _humanize(col), col.upper(),
            f"6.D86.{i}", "SDG.DOM.MEASURED_QUANTITY",
            f"Domain concept under measured_quantity ({col}).", "domain_member")

    artifacts = [
        "algorithm_type", "checksum", "chemical_formula", "code", "comments",
        "crystal_system", "developer_group", "endpoint_url", "file_format",
        "license_type", "measurement_method", "salary_range", "version",
    ]
    for i, col in enumerate(artifacts, 1):
        add(f"SDG.DOM.{col.upper()}", _humanize(col), col.upper(),
            f"2.1.D87.{i}", "SDG.DOM.ARTIFACT_DESCRIPTOR",
            f"Domain concept under artifact_descriptor ({col}).", "domain_member")

    entities = [
        ("Employee", "SDG.DOM.WORKFORCE_CONTINUANT", "2.D88.1"),
        ("Department", "SDG.DOM.WORKFORCE_CONTINUANT", "2.D88.2"),
        ("Position", "SDG.DOM.OCCUPATIONAL_ROLE", "5.D89.1"),
        ("Job", "SDG.DOM.OCCUPATIONAL_ROLE", "5.D89.2"),
        ("CollectiveBargainingAgreement", "SDG.DOM.LABOR_RECORD", "3.D90.1"),
        ("PerformanceReview", "SDG.DOM.LABOR_RECORD", "3.D90.2"),
        ("ComputationalMaterialRecord", "SDG.DOM.LABOR_RECORD", "3.D90.3"),
        ("ParsingPipeline", "SDG.DOM.SCIENTIFIC_PIPELINE", "1.D91.1"),
        ("VisualizationService", "SDG.DOM.SCIENTIFIC_PIPELINE", "1.D91.2"),
        ("InstrumentRun", "SDG.DOM.SCIENTIFIC_PIPELINE", "1.D91.3"),
        ("MaterialSystem", "SDG.DOM.COMPUTATIONAL_MATERIAL", "2.2.D92.1"),
        ("MaterialProperty", "SDG.DOM.COMPUTATIONAL_MATERIAL", "2.2.D92.2"),
        ("SmallMolecule", "SDG.DOM.COMPUTATIONAL_MATERIAL", "2.2.D92.3"),
        ("ComputationalCode", "SDG.DOM.RESEARCH_INFRASTRUCTURE", "2.1.D93.1"),
        ("HighPerformanceComputerCenter", "SDG.DOM.RESEARCH_INFRASTRUCTURE", "2.1.D93.2"),
        ("BigDataAnalyticsTool", "SDG.DOM.RESEARCH_INFRASTRUCTURE", "2.1.D93.3"),
        ("ParsedDataStore", "SDG.DOM.RESEARCH_INFRASTRUCTURE", "2.1.D93.4"),
        ("WorkLocation", "SDG.DOM.WORKFORCE_CONTINUANT", "2.D88.3"),
        ("Country", "SDG.DOM.WORKFORCE_CONTINUANT", "2.D88.4"),
    ]
    for name, parent, notation in entities:
        add(f"SDG.DOM.{_screaming(name)}", _humanize(name), _screaming(name),
            notation, parent,
            f"Domain concept realizing holdout-target entity {name}.",
            "domain_member")
    return recs


CSV_COLS = [
    "code", "label", "abbrev", "notation", "parent_code", "taxonomy",
    "description", "common_names", "example_values", "axiom", "family",
]


def load_vocab(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close_codes(rows: list[dict], wanted: set[str]) -> set[str]:
    by = {r["code"]: r for r in rows if r.get("code")}
    out = set(wanted)
    for code in list(wanted):
        cur = code
        seen = set()
        while cur and cur not in seen and cur in by:
            out.add(cur)
            seen.add(cur)
            cur = (by[cur].get("parent_code") or "").strip()
    return out


# ── Write collection dir ──────────────────────────────────────

def write_collection(spec: CollectionSpec, out_root: Path) -> dict:
    cid = _cid("holdout-reference", spec.key, PIN12)
    slug = f"{spec.key}-{cid[:8]}"
    if slug in FORBIDDEN:
        raise SystemExit(f"refusing forbidden slug {slug}")
    cdir = out_root / slug
    if cdir.exists():
        shutil.rmtree(cdir)
    (cdir / "tables").mkdir(parents=True)
    (cdir / "chapters").mkdir(parents=True)

    tables_data: dict[str, list[dict]] = {}
    table_meta = []
    for t in spec.tables:
        rows = materialize_table(t)
        tables_data[t.name] = rows
        with (cdir / "tables" / f"{t.name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=t.columns)
            w.writeheader()
            w.writerows(rows)
        (cdir / "tables" / f"{t.name}.sql").write_text(render_sql(t), encoding="utf-8")
        table_meta.append({"name": t.name, "rows": len(rows), "columns": list(t.columns)})

    ri = verify_ri(spec, tables_data)
    genera = sorted({t.genus for t in spec.terms})
    unmapped = [g for g in genera if g not in _GENUS_TO_ANCHOR]
    anchors = sorted({_GENUS_TO_ANCHOR.get(g, _FALLBACK_ANCHOR) for g in genera})
    entities = [t.name for t in spec.terms]
    passage = (spec.sources.get("finepdfs") or ["n/a"])[0] + ".txt"

    manifest = {
        "construct_id": cid,
        "slug": slug,
        "passage": passage,
        "registers": ["natural.md", "semantic.md"],
        "n_tables": len(table_meta),
        "n_views": 0,
        "entities": entities,
        "terms": [{"name": t.name, "genus": t.genus} for t in spec.terms],
        "tables": table_meta,
        "views": [],
        "sources": spec.sources,
        "pin": PIN,
    }
    (cdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    terms_md = ["# Grounding terms", "",
                "Ontology universals this collection realizes "
                "(each an sdg class, grounded in a BFO/CCO genus).", ""]
    for t in spec.terms:
        terms_md += [f"## {t.name}", f"- **genus**: `{t.genus}`", ""]
    (cdir / "terms.md").write_text("\n".join(terms_md), encoding="utf-8")

    readme = [
        f"# Collection — {spec.title}", "",
        f"Holdout-reference collection for pin `{PIN12}`. "
        f"**{len(table_meta)} tables** · **{len(entities)} terms**. "
        "Structure from SchemaPile; term lineage from FinePDFs postings. "
        "Values are type-true fictional particulars (not GitTables).", "",
        "## Sources", "",
        "SchemaPile: " + ", ".join(f"`{s}`" for s in spec.sources["schemapile"]), "",
        "FinePDFs: " + ", ".join(f"`{s}`" for s in spec.sources["finepdfs"]), "",
        "## Tables", "",
    ]
    for t in table_meta:
        readme.append(f"- `{t['name']}` ({t['rows']} rows)")
    readme += ["", "## Grounding", "", "See [terms.md](terms.md).", ""]
    (cdir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (cdir / "chapters" / "natural.md").write_text(
        f"# {spec.title}\n\nSchemaPile + FinePDFs reference collection "
        f"(slug `{slug}`).\n", encoding="utf-8")
    (cdir / "chapters" / "semantic.md").write_text(
        f"# {spec.title} (semantic)\n\nGenera: {', '.join(genera)}.\n",
        encoding="utf-8")

    return {
        "slug": slug,
        "construct_id": cid,
        "path": cdir,
        "entities": entities,
        "anchors": anchors,
        "genera": genera,
        "unmapped_genera": unmapped,
        "tables_data": tables_data,
        "table_meta": table_meta,
        "column_count": sum(len(t.columns) for t in spec.tables),
        "ri": ri,
        "sources": spec.sources,
        "terms": spec.terms,
    }


# ── Pack Atelier sample ───────────────────────────────────────

def pack_sample(
    built: list[dict],
    vocab_rows: list[dict],
    gap_rows: list[dict],
    sample_dir: Path,
) -> Path:
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    out_tables = sample_dir / "tables"
    out_tables.mkdir(parents=True)

    written: dict[str, str] = {}
    table_rows: dict[str, int] = {}
    all_ri_edges = []
    all_designative = []
    for coll in built:
        topic = re.sub(r"-[0-9a-f]{8}$", "", coll["slug"])
        for stem, rows in coll["tables_data"].items():
            name = stem if stem not in written else f"{topic}__{stem}"
            if name in written:
                raise SystemExit(f"table name collision: {name}")
            written[name] = coll["slug"]
            table_rows[name] = len(rows)
            cols = list(rows[0].keys()) if rows else []
            with (out_tables / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        all_ri_edges.extend(coll["ri"]["edges"])
        all_designative.extend(coll["ri"]["designative_references"])

    merged = {r["code"]: r for r in vocab_rows if r.get("code")}
    for r in gap_rows:
        merged[r["code"]] = r
    wanted = close_codes(list(merged.values()), set(TARGET_SKOS) | {r["code"] for r in gap_rows})
    keep = [merged[c] for c in sorted(wanted) if c in merged]
    header = list(vocab_rows[0].keys()) if vocab_rows else CSV_COLS
    with (sample_dir / "annotations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    codes = [r["code"] for r in keep]
    canonical = json.dumps(sorted(codes), separators=(",", ":"))
    vocab_sig = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    by_parent: dict[str, list[str]] = defaultdict(list)
    for r in keep:
        p = (r.get("parent_code") or "").strip()
        if p:
            by_parent[p].append(r["code"])
    code_set = set(codes)
    roots = sorted(c for c in codes if not (merged[c].get("parent_code") or "").strip())
    genus_count = sum(1 for c in codes if any(ch in code_set for ch in by_parent.get(c, [])))
    depth_hist: dict[str, int] = {}
    for c in codes:
        d, cur, seen = 0, c, set()
        while cur and cur not in seen and cur in merged:
            p = (merged[cur].get("parent_code") or "").strip()
            if not p:
                break
            seen.add(cur)
            cur = p
            d += 1
        depth_hist[str(d)] = depth_hist.get(str(d), 0) + 1

    column_total = sum(coll["column_count"] for coll in built)
    matched = []
    gaps = []
    vocab_abbrev = {_screaming(r.get("abbrev") or ""): r["code"] for r in keep}
    vocab_label = {_screaming(r.get("label") or ""): r["code"] for r in keep}

    def lookup(name: str) -> str | None:
        key = _screaming(name)
        return vocab_abbrev.get(key) or vocab_label.get(key)

    for coll in built:
        for t in coll["terms"]:
            code = lookup(t.name)
            if code:
                matched.append(f"{t.name}→{code}")
            else:
                gaps.append(f"{coll['slug']}:{t.name}")
        for stem, rows in coll["tables_data"].items():
            if not rows:
                continue
            for col in rows[0].keys():
                code = lookup(col)
                if code:
                    matched.append(f"{stem}.{col}→{code}")
                elif not col.endswith("_id") and col != "id":
                    gaps.append(f"column:{col}")

    manifest = {
        "source_id": "sdg-corpora",
        "corpus_commit": PIN,
        "profile": {
            "name": PROFILE,
            "max_collections": 8,
            "max_columns": 400,
            "max_terms": 400,
            "min_roots": 1,
        },
        "collections": [
            {
                "slug": c["slug"],
                "construct_id": c["construct_id"],
                "entities": c["entities"],
                "anchors": c["anchors"],
                "genera": c["genera"],
                "tables": len(c["tables_data"]),
                "columns": c["column_count"],
                "unmapped_genera": c["unmapped_genera"],
                "sources": c["sources"],
            }
            for c in built
        ],
        "table_count": len(written),
        "tables": table_rows,
        "column_count": column_total,
        "referential_integrity": {
            "verified": True,
            "fk_edges_checked": len(all_ri_edges),
            "orphans": 0,
            "designative_references": sorted(all_designative),
            "edges": all_ri_edges,
        },
        "taxonomy": {
            "term_count": len(codes),
            "roots": roots,
            "root_count": len(roots),
            "genus_terms": genus_count,
            "leaf_terms": len(codes) - genus_count,
            "depth_histogram": dict(sorted(depth_hist.items())),
            "matched_references": matched,
            "vocabulary_gaps": sorted(set(gaps)),
            "support_top": {},
        },
        "vocab_sig": vocab_sig,
        "role": "reference",
        "pair_target_id": f"sdg-corpora/{PIN12}_macbook",
        "emit": {
            "script": "scripts/emit_holdout_reference.py",
            "forbidden_slugs": sorted(FORBIDDEN),
            "required_iris": list(REQUIRED_IRIS),
        },
    }
    (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sample_dir


def check_invariants(built: list[dict], sample_dir: Path) -> None:
    slugs = {c["slug"] for c in built}
    if slugs & FORBIDDEN:
        raise SystemExit(f"forbidden slug leaked: {slugs & FORBIDDEN}")
    genera = set()
    cols: set[str] = set()
    names: set[str] = set()
    for c in built:
        genera.update(c["genera"])
        names.update(c["entities"])
        for rows in c["tables_data"].values():
            if rows:
                cols.update(rows[0].keys())
    missing_iris = [g for g in REQUIRED_IRIS if g not in genera]
    if missing_iris:
        raise SystemExit(f"missing required genera: {missing_iris}")
    codes = set()
    with (sample_dir / "annotations.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("code"):
                codes.add(row["code"])
    missing_skos = [c for c in TARGET_SKOS if c not in codes]
    if missing_skos:
        raise SystemExit(f"sample annotations missing target SKOS: {missing_skos}")
    emit_entities = {
        "ComputationalMaterialRecord", "ComputationalCode", "MaterialSystem",
        "MaterialProperty", "ParsingPipeline", "HighPerformanceComputerCenter",
        "BigDataAnalyticsTool", "ParsedDataStore", "VisualizationService",
        "Position", "Employee", "Department", "CollectiveBargainingAgreement",
        "PerformanceReview",
    }
    missing_ent = sorted(emit_entities - names)
    if missing_ent:
        raise SystemExit(f"entity terms not realized: {missing_ent}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "build" / "holdout_reference" / "collections"))
    ap.add_argument("--sample-dir",
                    default="/home/rch/local/src/zndx/atelier/build/sdg_sample/b24ef9f60660_reference")
    ap.add_argument("--vocab",
                    default="/home/rch/local/src/zndx/atelier/external/sdg-corpora/vocabulary/annotations.csv")
    ap.add_argument("--install-corpora", action="store_true",
                    help="copy collections into Atelier sdg-corpora working tree (does not commit / change pin)")
    ap.add_argument("--corpora",
                    default="/home/rch/local/src/zndx/atelier/external/sdg-corpora/corpus/collections")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    built = [write_collection(spec, out) for spec in SPECS]
    gap = gap_skos_records()
    overlay = REPO / "build" / "holdout_reference" / "skos_gap_terms.csv"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    with overlay.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(gap)

    vocab = load_vocab(Path(args.vocab))
    sample_dir = pack_sample(built, vocab, gap, Path(args.sample_dir))
    check_invariants(built, sample_dir)

    pair_path = sample_dir.parent / "pair.json"
    pair_path.write_text(json.dumps({
        "target_id": f"sdg-corpora/{PIN12}_macbook",
        "reference_id": f"sdg-corpora/{PIN12}_{PROFILE}",
        "reference_path": str(sample_dir),
    }, indent=2) + "\n", encoding="utf-8")

    if args.install_corpora:
        dest_root = Path(args.corpora)
        for c in built:
            dest = dest_root / c["slug"]
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(c["path"], dest)
            print(f"  installed {c['slug']} → {dest}")

    print(f"  collections → {out}")
    print(f"  sample      → {sample_dir}")
    print(f"  slugs       → {[c['slug'] for c in built]}")
    print(f"  genera      → {sorted(set().union(*(c['genera'] for c in built)))}")
    print(f"  skos overlay {len(gap)} rows → {overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
