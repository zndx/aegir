#!/usr/bin/env python3
"""Corpus item 5: synthetic Schema.org tables with known-ground-truth labels.

This is the scaffolded (no-LLM-required) first pass of the ontology-grounded
synthetic-data pipeline described in docs/current/src/pretraining.md. The full
pipeline uses LLM calls (GLM-4.7 / Cerebras) to generate semantically-
rich text from BFO-aligned ontologies; here we replace the LLM with a
templated generator that still produces useful byte-level pretraining
data WITH column-to-type ground truth.

Pipeline:
  1. Parse Schema.org JSON-LD for (class → property list, property range).
  2. For each selected domain-relevant class (Person, Organization,
     LocalBusiness, Product, Hotel, Restaurant, ...), generate N
     synthetic tables, each with:
       - 5-20 columns drawn from the class's declared + inherited properties
       - 10-100 rows of realistic values keyed by property range
         (Text/Number/Date/URL/Email/...)
  3. Emit each table as (schema_decl + N rows) in byte-level text format
     with well-known delimiters.

Value generators: all stdlib, no extra dep. Realistic enough that a
byte-level LM can learn format patterns while the ground-truth
``(column_name, schema:property, schema:range_type)`` mapping gives
the downstream task clean supervision.

Output: /raid/datasets/aegir-corpus-v1/synthetic/schemaorg-tables.txt
with documents separated by \\x03, alongside a JSONL sidecar
``schemaorg-tables-labels.jsonl`` that records the ground-truth per
table (for downstream evaluation, not pretraining).

Usage:
    uv run --no-sync python scripts/gen_corpus_item5.py \
        --schemaorg /raid/datasets/aegir-corpus-v1/ontology/schemaorg.jsonld \
        --out-dir /raid/datasets/aegir-corpus-v1/synthetic \
        --tables 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import string
import time
from pathlib import Path
from typing import Iterable

log = logging.getLogger("corpus_item5")

DOC_DELIM = b"\x03"
TABLE_DELIM = "\n--TABLE--\n"
ROW_DELIM = "\n"
CELL_DELIM = "\t"


# ── Schema.org parse ────────────────────────────────────────────


def _schemaorg_classes_and_props(jsonld_path: Path) -> tuple[dict, dict]:
    """Parse Schema.org JSON-LD, return (classes, properties) dicts.

    classes[id] = {"label": ..., "comment": ..., "parents": [ids]}
    properties[id] = {"label": ..., "comment": ..., "domains": [ids],
                      "ranges": [label_names]}
    """
    data = json.loads(jsonld_path.read_text())
    graph = data.get("@graph", [])
    classes: dict[str, dict] = {}
    properties: dict[str, dict] = {}

    def _extract(field) -> list[str]:
        if field is None:
            return []
        if isinstance(field, dict):
            return [field.get("@id", "")]
        if isinstance(field, list):
            return [
                item.get("@id", "") if isinstance(item, dict) else str(item)
                for item in field
            ]
        return [str(field)]

    def _label(node: dict) -> str:
        lab = node.get("rdfs:label", "")
        if isinstance(lab, dict):
            lab = lab.get("@value", "")
        if isinstance(lab, list):
            lab = lab[0] if lab else ""
            if isinstance(lab, dict):
                lab = lab.get("@value", "")
        return str(lab)

    def _comment(node: dict) -> str:
        c = node.get("rdfs:comment", "")
        if isinstance(c, dict):
            c = c.get("@value", "")
        if isinstance(c, list):
            c = c[0] if c else ""
            if isinstance(c, dict):
                c = c.get("@value", "")
        return re.sub(r"\s+", " ", str(c)).strip()

    for node in graph:
        if not isinstance(node, dict):
            continue
        nid = node.get("@id", "")
        ntype = node.get("@type")
        if ntype == "rdfs:Class":
            classes[nid] = {
                "label": _label(node) or nid.split(":")[-1],
                "comment": _comment(node),
                "parents": _extract(node.get("rdfs:subClassOf")),
            }
        elif ntype == "rdf:Property":
            properties[nid] = {
                "label": _label(node) or nid.split(":")[-1],
                "comment": _comment(node),
                "domains": _extract(node.get("schema:domainIncludes")),
                "ranges": _extract(node.get("schema:rangeIncludes")),
            }
    return classes, properties


def _inherited_props(
    class_id: str, classes: dict, property_by_domain: dict[str, list[str]]
) -> list[str]:
    """Collect all property ids applicable to class_id (direct + ancestor)."""
    seen = set()
    result = []
    stack = [class_id]
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        for prop_id in property_by_domain.get(cid, []):
            if prop_id not in result:
                result.append(prop_id)
        cnode = classes.get(cid, {})
        for parent in cnode.get("parents", []):
            stack.append(parent)
    return result


# ── Value generators by Schema.org range ────────────────────────


FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
    "Nancy", "Daniel", "Margaret", "Matthew", "Lisa", "Anthony", "Betty",
    "Donald", "Dorothy", "Mark", "Sandra", "Paul", "Ashley", "Steven",
    "Kimberly", "Andrew", "Donna", "Kenneth", "Emily",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
]
CITIES = [
    "New York", "London", "Tokyo", "Paris", "Berlin", "San Francisco",
    "Seattle", "Boston", "Austin", "Denver", "Chicago", "Toronto",
    "Madrid", "Rome", "Sydney", "Singapore", "Mumbai", "Buenos Aires",
    "Cape Town", "Dubai",
]
COUNTRIES = [
    "USA", "UK", "Japan", "France", "Germany", "Canada", "Australia",
    "Spain", "Italy", "Singapore", "India", "Argentina", "South Africa",
    "UAE", "Brazil", "Mexico", "China", "Netherlands",
]
STREET_SUFFIXES = [
    "St", "Ave", "Rd", "Blvd", "Ln", "Dr", "Way", "Pl", "Ct", "Pkwy",
]
CORP_SUFFIXES = [
    "Inc", "Corp", "LLC", "Ltd", "GmbH", "SA", "BV", "Group", "Holdings",
    "Industries", "Systems", "Solutions",
]
ORG_STEMS = [
    "Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Wonka",
    "Cyberdyne", "Tyrell", "Massive", "Aperture", "Oceanic", "Oscorp",
    "Pied Piper", "Hooli", "Weyland", "Monarch", "Veridian", "Goliath",
    "Vandelay",
]


def _rand_first_name(r: random.Random) -> str:
    return r.choice(FIRST_NAMES)


def _rand_last_name(r: random.Random) -> str:
    return r.choice(LAST_NAMES)


def _rand_person_name(r: random.Random) -> str:
    return f"{_rand_first_name(r)} {_rand_last_name(r)}"


def _rand_org_name(r: random.Random) -> str:
    return f"{r.choice(ORG_STEMS)} {r.choice(CORP_SUFFIXES)}"


def _rand_email(r: random.Random) -> str:
    u = _rand_first_name(r).lower() + "." + _rand_last_name(r).lower()
    dom = r.choice(["example.com", "mail.com", "corp.io", "domain.org", "site.net"])
    return f"{u}@{dom}"


def _rand_phone(r: random.Random) -> str:
    return f"+{r.randint(1, 99)}-{r.randint(100, 999)}-{r.randint(100, 999)}-{r.randint(1000, 9999)}"


def _rand_url(r: random.Random, org: bool = False) -> str:
    slug = "".join(r.choices(string.ascii_lowercase, k=r.randint(4, 12)))
    tld = r.choice(["com", "org", "net", "io", "co"])
    return f"https://www.{slug}.{tld}"


def _rand_address(r: random.Random) -> str:
    return f"{r.randint(1, 9999)} {r.choice(['Main', 'Oak', 'Maple', 'Pine', 'Cedar', 'Elm'])} {r.choice(STREET_SUFFIXES)}"


def _rand_postal(r: random.Random) -> str:
    return f"{r.randint(10000, 99999)}"


def _rand_date(r: random.Random) -> str:
    y = r.randint(1950, 2025)
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _rand_datetime(r: random.Random) -> str:
    return _rand_date(r) + f"T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"


def _rand_number(r: random.Random) -> str:
    return str(r.randint(0, 10_000_000))


def _rand_float(r: random.Random) -> str:
    return f"{r.uniform(0, 10_000):.2f}"


def _rand_price(r: random.Random) -> str:
    return f"${r.uniform(0.99, 9999.99):.2f}"


def _rand_bool(r: random.Random) -> str:
    return r.choice(["true", "false"])


def _rand_text(r: random.Random, min_words: int = 3, max_words: int = 15) -> str:
    words = [
        "quick", "brown", "fox", "lazy", "dog", "data", "system", "model",
        "column", "row", "table", "value", "entry", "record", "result",
        "analysis", "report", "summary", "description", "title", "content",
    ]
    n = r.randint(min_words, max_words)
    return " ".join(r.choices(words, k=n))


# Map Schema.org range IRIs / labels → value generator
_RANGE_GENERATORS = {
    "schema:Text": _rand_text,
    "schema:URL": _rand_url,
    "schema:Integer": _rand_number,
    "schema:Number": _rand_float,
    "schema:Float": _rand_float,
    "schema:Date": _rand_date,
    "schema:DateTime": _rand_datetime,
    "schema:Time": lambda r: f"{r.randint(0,23):02d}:{r.randint(0,59):02d}:{r.randint(0,59):02d}",
    "schema:Boolean": _rand_bool,
    "schema:Person": _rand_person_name,
    "schema:Organization": _rand_org_name,
    "schema:LocalBusiness": _rand_org_name,
    "schema:Place": lambda r: r.choice(CITIES),
    "schema:City": lambda r: r.choice(CITIES),
    "schema:PostalAddress": _rand_address,
    "schema:Country": lambda r: r.choice(COUNTRIES),
    "schema:MonetaryAmount": _rand_price,
    "schema:Duration": lambda r: f"PT{r.randint(1, 24)}H",
    "schema:PropertyValue": _rand_text,
    "schema:Role": _rand_text,
    "schema:Rating": lambda r: f"{r.uniform(1, 5):.1f}",
    "schema:Language": lambda r: r.choice(["en", "fr", "de", "ja", "es", "pt"]),
    "schema:Thing": _rand_text,
}

# Property-name heuristics for ranges that aren't typed specifically enough
_PROPERTY_NAME_GENERATORS = {
    "email": _rand_email,
    "telephone": _rand_phone,
    "phone": _rand_phone,
    "url": _rand_url,
    "website": _rand_url,
    "address": _rand_address,
    "postalCode": _rand_postal,
    "zip": _rand_postal,
    "firstName": _rand_first_name,
    "givenName": _rand_first_name,
    "lastName": _rand_last_name,
    "familyName": _rand_last_name,
    "name": _rand_person_name,
    "price": _rand_price,
    "date": _rand_date,
    "datetime": _rand_datetime,
    "birthDate": _rand_date,
    "foundingDate": _rand_date,
    "postalAddress": _rand_address,
    "city": lambda r: r.choice(CITIES),
    "country": lambda r: r.choice(COUNTRIES),
}


def _pick_generator(prop_label: str, prop_ranges: list[str], r: random.Random):
    """Choose the most specific generator for a property."""
    # Name-based heuristics first — they override generic ranges
    lower = prop_label.lower()
    for key, gen in _PROPERTY_NAME_GENERATORS.items():
        if key.lower() in lower:
            return gen
    # Range-based
    for rng in prop_ranges:
        if rng in _RANGE_GENERATORS:
            return _RANGE_GENERATORS[rng]
    # Default to text
    return _rand_text


# ── Table generation ────────────────────────────────────────────


# Domain-relevant seed classes (those most likely to appear in real
# relational data). Augment further based on the Schema.org graph.
SEED_CLASSES = [
    "schema:Person", "schema:Organization", "schema:LocalBusiness",
    "schema:Product", "schema:Hotel", "schema:Restaurant", "schema:Book",
    "schema:Article", "schema:Movie", "schema:MusicAlbum",
    "schema:MusicRecording", "schema:Event", "schema:SportsEvent",
    "schema:MedicalEntity", "schema:Drug", "schema:MedicalCondition",
    "schema:Place", "schema:City", "schema:Country", "schema:Recipe",
    "schema:CreativeWork", "schema:Review", "schema:Rating",
    "schema:Offer", "schema:Vehicle", "schema:SoftwareApplication",
    "schema:WebSite", "schema:WebPage", "schema:Dataset",
    "schema:EducationalOrganization", "schema:Action",
]


def _gen_table(
    class_id: str,
    classes: dict,
    properties: dict,
    property_by_domain: dict,
    r: random.Random,
) -> tuple[str, list[dict]]:
    """Generate one synthetic table for the given class."""
    cls = classes.get(class_id, {})
    cls_label = cls.get("label", class_id.split(":")[-1])
    all_props = _inherited_props(class_id, classes, property_by_domain)
    if not all_props:
        return "", []
    # Pick 5-20 properties for the table (schema-like density)
    n_cols = min(len(all_props), r.randint(5, 20))
    picked = r.sample(all_props, n_cols)

    column_labels = []
    generators = []
    label_records = []
    for p in picked:
        pnode = properties.get(p, {})
        plabel = pnode.get("label", p.split(":")[-1])
        pranges = pnode.get("ranges", [])
        # Occasionally use a realistic renamed column (snake_case, abbrev)
        display = plabel
        if r.random() < 0.3:
            display = re.sub(r"(?<!^)([A-Z])", r"_\1", plabel).lower()
        if r.random() < 0.15 and len(plabel) > 4:
            display = plabel[:4].lower()
        column_labels.append(display)
        generators.append(_pick_generator(plabel, pranges, r))
        label_records.append({
            "column": display,
            "schema_property": p,
            "schema_property_label": plabel,
            "schema_ranges": pranges,
        })

    # Header row: CREATE TABLE + columns
    table_name = f"{cls_label.lower()}_{r.randint(1000,9999)}"
    lines = [
        f"-- Table: {table_name} (class: {class_id})",
        "CREATE TABLE " + table_name + " (",
    ]
    for i, col in enumerate(column_labels):
        comma = "," if i < len(column_labels) - 1 else ""
        lines.append(f"  {col} TEXT{comma}")
    lines.append(");")
    lines.append("")

    # Data rows
    n_rows = r.randint(10, 100)
    lines.append(CELL_DELIM.join(column_labels))  # header
    for _ in range(n_rows):
        row_vals = [gen(r) for gen in generators]
        # quick string escape for tab/newline
        row_vals = [
            str(v).replace("\t", " ").replace("\n", " ")[:120]
            for v in row_vals
        ]
        lines.append(CELL_DELIM.join(row_vals))

    doc = "\n".join(lines)
    return doc, [
        {
            "table_name": table_name,
            "class_id": class_id,
            "class_label": cls_label,
            "columns": label_records,
        }
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--schemaorg",
        type=Path,
        default=Path("/raid/datasets/aegir-corpus-v1/ontology/schemaorg.jsonld"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/datasets/aegir-corpus-v1/synthetic"),
    )
    ap.add_argument("--tables", type=int, default=5000,
                    help="How many synthetic tables to generate")
    ap.add_argument("--seed", type=int, default=4649)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Parsing %s ...", args.schemaorg)
    classes, properties = _schemaorg_classes_and_props(args.schemaorg)
    log.info("  %d classes, %d properties", len(classes), len(properties))

    # Build domain index: property → list of domain-class-ids
    property_by_domain: dict[str, list[str]] = {}
    for pid, pnode in properties.items():
        for d in pnode["domains"]:
            property_by_domain.setdefault(d, []).append(pid)

    # Expand seed classes with additional domain-relevant ones from the graph
    extra = [
        cid for cid in classes
        if any(p in cid for p in (
            "LocalBusiness", "Organization", "Product", "Store", "Restaurant",
            "Hotel", "Person", "MedicalEntity",
        ))
    ]
    seeds = list(dict.fromkeys(SEED_CLASSES + extra))
    seeds = [s for s in seeds if s in classes]
    log.info("  %d seed classes available", len(seeds))
    if not seeds:
        log.error("No seed classes found in Schema.org parse — aborting")
        return 1

    out_txt = args.out_dir / "schemaorg-tables.txt"
    out_lbl = args.out_dir / "schemaorg-tables-labels.jsonl"
    if out_txt.exists() and out_txt.stat().st_size > 0:
        log.info("SKIP (exists): %s (%.2f MB)",
                 out_txt.name, out_txt.stat().st_size / 2**20)
        return 0

    rng = random.Random(args.seed)
    t0 = time.time()
    n_written = 0
    total_bytes = 0
    with open(out_txt, "wb") as ftxt, open(out_lbl, "w") as flbl:
        for i in range(args.tables):
            class_id = rng.choice(seeds)
            doc, label_records = _gen_table(
                class_id, classes, properties, property_by_domain, rng,
            )
            if not doc:
                continue
            b = doc.encode("utf-8", errors="replace")
            ftxt.write(b)
            ftxt.write(DOC_DELIM)
            for rec in label_records:
                flbl.write(json.dumps(rec, ensure_ascii=False))
                flbl.write("\n")
            n_written += 1
            total_bytes += len(b) + 1
            if n_written % 500 == 0:
                log.info("  %d tables, %.2f MB, %.1fs",
                         n_written, total_bytes / 2**20, time.time() - t0)
    log.info("Done. %d tables, %.2f MB (+ labels sidecar %s) in %.1fs",
             n_written, total_bytes / 2**20, out_lbl.name, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
