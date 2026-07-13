"""Deterministic, **referentially-closed** row synthesis for the DDL spine.

Correct-by-construction relational data (Track A): given the lowered spine
(``list[SpineTable]`` from :func:`ddl.template_to_table`) and the sanctioned FK
edges (:func:`ddl.cross_family_fks`), materialize each table's ``rows`` so that

  * **RI = 1.0 by construction** — every FK cell is drawn from the referenced
    table's primary-key pool (no after-the-fact validation; cf. the verification
    *membrane*: generate-correct-by-construction inside the loop);
  * values are **type-true** (xsd range → typed literal) and **realistic**
    (ontology-grounded enums + curated semantic pools, not ``val_1``);
  * **NOT-NULL** / ``some`` / ``min 1`` columns are always populated.

Pure + deterministic (stdlib only — ``hashlib``/``random`` seeded per cell); no
LLM, no JVM, no GPU. We guarantee *type / range / NOT-NULL / RI*; we do **not**
attempt arbitrary OWL semantics (``only`` / disjointness / cross-row functional
constraints) — that boundary is documented and (optionally) spot-checked by HermiT
elsewhere. Operates on duck-typed ``SpineTable`` (``.table`` / ``.not_null`` /
``.template``) so there is no import cycle with :mod:`aegir.ontology.ddl`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # typing only — no runtime import (avoids ddl↔rows cycle)
    from aegir.ontology.ddl import SpineTable

MIN_ROWS, MAX_ROWS = 4, 8
_FK_TARGET_MIN_ROWS = 6                 # FK targets need enough distinct parents for organic fan-out
_JUNCTION_MIN_ROWS = 12                 # association-class (M:N) tables get denser fan-out than base tables
_GLOBAL_SEED = 0xAE61
logger = logging.getLogger(__name__)

_ENTITY = {"Class", "Individual", "NamedIndividual"}
_XSD_NONSTRING = {"xsd:dateTime", "xsd:date", "xsd:integer", "xsd:int", "xsd:long",
                  "xsd:decimal", "xsd:double", "xsd:float", "xsd:boolean"}
_GENERIC_TOK = {"class", "subclass", "basic", "template", "the", "and", "for",
                "foundation", "long", "tail", "complex", "axiom"}

# Curated, ontology-native value pools keyed by the DataProperty column name (the
# `_dataprop_col` form). Definition-derived enums (parse_enum_from_definition) take
# precedence when a skos:definition enumerates a value set; these are the fallback.
SEMANTIC_VALUE_POOLS: dict[str, list[str]] = {
    "status": ["pending", "running", "complete", "failed", "superseded"],
    "state": ["active", "inactive", "archived", "draft"],
    "phase": ["initiation", "execution", "review", "closeout"],
    "unit": ["mg/L", "nm", "ms", "deg_C", "count", "ratio", "kg", "m/s"],
    "format": ["ISO-8601", "RFC-3339", "UUID", "E.164", "JSON", "CSV"],
    "language": ["en", "de", "fr", "ja", "es"],
    "currency": ["USD", "EUR", "GBP", "JPY"],
    "country": ["US", "DE", "FR", "JP", "GB"],
    "severity": ["low", "medium", "high", "critical"],
    "category": ["primary", "secondary", "tertiary"],
    "method": ["manual", "automated", "hybrid"],
    "type": ["nominal", "ordinal", "interval", "ratio"],
    "code": ["A-01", "B-12", "C-07", "D-33", "E-21"],
    "role": ["owner", "reviewer", "contributor", "observer"],
    # value sets whose tokens carry slashes/hyphens/dots the skos:definition enum parser
    # (parse_enum_from_definition, lowercase-alnum only) cannot represent — curated here so
    # they still land as domain-meaningful cells rather than "<Concept> NN" placeholders.
    "mime_type": ["application/json", "text/csv", "application/parquet", "text/plain",
                  "application/xml", "application/avro", "application/octet-stream"],
    "license": ["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0", "MPL-2.0", "CC-BY-4.0", "proprietary"],
    "owner": ["platform-team", "data-engineering", "sre", "analytics", "ml-infra", "governance"],
    "host_name": ["node-a01", "node-b14", "worker-07", "edge-03", "gw-12", "ingest-21"],
    "uri": ["s3://lake/raw", "s3://lake/curated", "abfss://prod/silver", "gs://warehouse/gold",
            "hdfs://cluster/staging"],
    "checksum": ["a3f9c21e", "7b14de08", "c0ffee42", "9d2b7a16", "5e8f3c91", "1a4b6c2d"],
    # Generic string attributes whose values are domain-real regardless of the table's concept
    # (a location is a location, an identifier an identifier) — honest deterministic domain cells for
    # every family. The CONCEPT-specific columns (subject heads, named relations, `name`) are instead
    # seeded per-template by the local engine (scripts/seed_entity_values.py); see entity_value_pools.json.
    "identifier": ["urn:uuid:9f2a", "doi:10.1109/x", "ARN:res/41", "gid://svc/77", "oid:1.3.6.1", "ref-8842"],
    "location": ["us-east-1", "eu-west-3", "rack-7", "zone-b", "ap-south-2", "on-prem-dc1"],
    "label_text": ["nightly summary", "pre-release note", "calibration record", "audit excerpt",
                   "change rationale", "intake form"],
}


def parse_enum_from_definition(defn: str | None) -> list[str] | None:
    """Extract an enumerated value set from a skos:definition (ontology-grounded).

    Fires when the definition gives a closed list of ≥2 short value tokens — a
    parenthetical ``"(pending / running / complete / failed)"`` or an explicit
    ``"status: a, b, c"`` cue. **Illustrative** parentheticals (``"(e.g. ISO, RFC)"``)
    are *not* enums and are skipped (the caller then uses the curated pool). Returns
    ``None`` when no closed set is present, so the type generator / pool takes over.
    """
    if not defn:
        return None
    cand: str | None = None
    for grp in re.findall(r"\(([^)]*)\)", defn):                 # prefer an explicit value list
        g = grp.strip()
        if re.match(r"(?i)\s*(e\.?g\.?|i\.?e\.?|such as|including|for example)", g):
            continue                                            # examples, not a closed set
        if re.search(r"[,/|]| or ", g):
            cand = g
            break
    if cand is None:                                            # or an explicit "<cue>: a, b, c"
        m = re.search(r"(?:one of|values?|states?|status(?:es)?)\s*[:\-]\s*([A-Za-z0-9 ,/|]+)", defn, re.I)
        cand = m.group(1) if m else None
    if cand is None:
        return None
    vals: list[str] = []
    for p in re.split(r"\s*[,/|]\s*|\s+or\s+", cand):
        p = p.strip().strip(".").strip().lower()
        if (re.fullmatch(r"[a-z][a-z0-9 ]{1,20}", p) and p not in _GENERIC_TOK
                and p not in ("a", "an", "of", "etc", "such", "value", "the")):
            vals.append(p)
    return vals if len(vals) >= 2 else None


def _seed(*parts, base: int = _GLOBAL_SEED) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in (base, *parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def _pk_index(st: "SpineTable") -> int:
    for i, c in enumerate(st.table.columns):
        if c.slot_ref == "__pk__":
            return i
    return 0


def _col_index(st: "SpineTable", name: str) -> int | None:
    for i, c in enumerate(st.table.columns):
        if c.name == name:
            return i
    return None


def _meaningful_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z]+", text or "") if len(t) >= 3 and t.lower() not in _GENERIC_TOK]


def _pk_prefix(st: "SpineTable") -> str:
    tid = getattr(getattr(st, "template", None), "template_id", "") or st.table.name
    toks = _meaningful_tokens(tid)
    word = toks[-1] if toks else (re.sub(r"[^A-Za-z]", "", st.table.name) or "row")
    return word[:4].upper()


def pk_value(st: "SpineTable", i: int) -> str:
    """A readable, table-stable surrogate primary key, e.g. ``OBSE-0003``."""
    return f"{_pk_prefix(st)}-{i + 1:04d}"


def row_count_for(st: "SpineTable", seed: int = _GLOBAL_SEED) -> int:
    return MIN_ROWS + (_seed(st.table.name, "rowcount", base=seed) % (MAX_ROWS - MIN_ROWS + 1))


def _concept_noun(st: "SpineTable") -> str:
    """A short concept noun for naming generic entity instances (subject/related)."""
    tid = getattr(getattr(st, "template", None), "template_id", "") or st.table.name
    toks = _meaningful_tokens(tid)[-2:]
    if toks:
        return " ".join(toks).title()
    v = re.sub(r"\{[^}]+\}", "", getattr(getattr(st, "template", None), "verbal_template", "") or "")
    vw = _meaningful_tokens(v)[:2]
    return " ".join(vw).title() if vw else "Entity"


def _entity_value(col, st: "SpineTable", row_ix: int) -> str:
    name = col.name
    if name in ("subject", "related") or name.startswith(("subject_", "related_")):
        noun = _concept_noun(st)
    else:
        noun = name.replace("_", " ").title()
    return f"{noun} {row_ix + 1:02d}"


def _xsd_value(t: str, name: str, rng: random.Random) -> str:
    # TOKEN match, not substring — else "ratio" matches "du(ratio)n" and a duration reads as [0,1].
    toks = {w for w in re.split(r"[^a-z0-9]+", name) if w}

    def has(*ws: str) -> bool:
        return bool(toks & set(ws))

    if t == "xsd:dateTime":
        d = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 899), seconds=rng.randint(0, 86399))
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    if t == "xsd:date":
        return (date(2023, 1, 1) + timedelta(days=rng.randint(0, 899))).isoformat()
    if t in ("xsd:integer", "xsd:int"):
        if has("version"):
            return str(rng.randint(1, 12))
        if has("priority", "rank", "level", "severity"):
            return str(rng.randint(1, 5))
        if has("count", "number", "size", "quantity", "n"):
            return str(rng.randint(0, 500))
        return str(rng.randint(1, 1000))
    if t == "xsd:long":
        return str(rng.randint(10 ** 6, 10 ** 9))
    if t in ("xsd:decimal", "xsd:double", "xsd:float"):
        if has("confidence", "ratio", "probability", "score", "fraction", "rate", "weight"):
            return f"{rng.uniform(0.0, 1.0):.3f}"
        if has("duration", "seconds", "latency", "elapsed", "ms", "milliseconds"):
            return f"{rng.uniform(0.1, 7200.0):.2f}"
        return f"{rng.uniform(0.0, 1000.0):.2f}"
    if t == "xsd:boolean":
        return rng.choice(["true", "false"])
    return _entity_value_str(name, rng)


def _entity_value_str(name: str, rng: random.Random) -> str:
    return f"{name.replace('_', ' ').title()} {rng.randint(1, 99):02d}"


# ── intra-row temporal coherence (start < end, end = start + duration) ─────────
# value_for is per-cell, so a start/end pair drawn independently is start>end ~half
# the time (the rows.py bug Comp 4 fixes). This is a same-row, cross-COLUMN constraint
# (cheap to enforce) — distinct from the cross-ROW / `only` semantics we deliberately
# leave to HermiT at the membrane. We re-derive each `end_*` from its `start_*` plus a
# positive span (the row's `duration_*` column when present, so the triple is mutually
# consistent), making the temporal columns realistic instead of merely type-true.
_TEMPORAL_TYPES = {"xsd:dateTime", "xsd:date"}
_DUR_TYPES = {"xsd:decimal", "xsd:double", "xsd:float", "xsd:integer", "xsd:long"}


def _temporal_role(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("start", "begin", "created", "issued", "effective", "opened")):
        return "start"
    if any(k in n for k in ("end", "finish", "stop", "closed", "completed", "expir")):
        return "end"
    return None


def _parse_temporal(s: str) -> "tuple[datetime, bool] | tuple[None, bool]":
    """(datetime, is_datetime) from an ISO date / dateTime cell, else (None, False)."""
    try:
        if "T" in s:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S"), True
        return datetime.strptime(s, "%Y-%m-%d"), False
    except (TypeError, ValueError):
        return None, False


def _fmt_temporal(dt: datetime, xsd_type: str) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if xsd_type == "xsd:dateTime" else dt.date().isoformat()


def _enforce_temporal_coherence(st: "SpineTable", row: list[str], *, seed: int, row_ix: int) -> None:
    """Rewrite ``end_*`` cells so every start/end pair is chronologically ordered and (when a
    duration column is present) ``end == start + duration``. Mutates ``row`` in place; no-op when
    the table has no start+end temporal pair."""
    cols = st.table.columns
    starts = [(i, c) for i, c in enumerate(cols)
              if c.slot_type in _TEMPORAL_TYPES and _temporal_role(c.name) == "start"]
    ends = [(i, c) for i, c in enumerate(cols)
            if c.slot_type in _TEMPORAL_TYPES and _temporal_role(c.name) == "end"]
    if not starts or not ends:
        return
    si, _ = starts[0]
    start_dt, _ = _parse_temporal(row[si] if si < len(row) else "")
    if start_dt is None:
        return
    rng = random.Random(_seed(st.table.name, "tcoh", row_ix, base=seed))
    dur_i = next((i for i, c in enumerate(cols) if c.slot_type in _DUR_TYPES
                  and any(k in c.name.lower() for k in ("duration", "seconds", "elapsed", "latency"))), None)
    dur_s = None
    if dur_i is not None and dur_i < len(row):
        try:
            dur_s = float(row[dur_i])
        except (TypeError, ValueError):
            dur_s = None
    if dur_s is None or dur_s <= 0:
        dur_s = rng.uniform(60.0, 7200.0)
    for ei, ecol in ends:
        if ei >= len(row):
            continue
        if ecol.slot_type == "xsd:date":            # day granularity: ≥1 day after start
            end_dt = start_dt + timedelta(days=max(1, int(dur_s // 86400) + rng.randint(1, 14)))
        else:
            end_dt = start_dt + timedelta(seconds=max(1.0, dur_s))
        row[ei] = _fmt_temporal(end_dt, ecol.slot_type)


# ── GitTables empirical value pools (real values, PROVENANCE-retained; RH 2026-07-13) ──────────────────────
# Ground value generation in the empirical distribution of real GitTables columns instead of rng.uniform /
# "{Column} NN". Values are admitted with retained lineage (the sampling strategy is a discrete versioned
# sdg-strategy entry → Atlas), never sanitized. Type-checked so a real value never violates the xsd type;
# a corpus run passes profiles in (loaded from build/gittables/value_profiles.json). [[gittables_value_realism]]
_GT_PROFILES = Path(__file__).resolve().parent.parent.parent.parent / "build" / "gittables" / "value_profiles.json"
# column-name keyword → (semantic_type, xsd-compat class). ORDER MATTERS — specific compounds (company_name,
# product_name) MUST precede the generic `name` or they'd fall into person_name. Restricted to the pools that
# curate reliably clean; job_title/category/geo_place are DEFERRED (residual noise) and email/phone to the
# sensitivity membrane (high-risk PII), temporal to the mechanical date-gen (format-correct).
_GT_NAME_RULES: "list[tuple[str, re.Pattern, str]]" = [
    ("organization", re.compile(r"company|organi|org|employer|vendor|supplier|firm|agency|brand|manufacturer"), "str"),
    ("product", re.compile(r"product|item|model|goods|part"), "str"),
    ("person_name", re.compile(r"name|employee|person|contact|author|customer|client|owner|holder"), "str"),
    ("url", re.compile(r"url|link|website|homepage|uri"), "str"), ("color", re.compile(r"colou?r"), "str"),
    ("identifier", re.compile(r"\b(id|code|ref|sku|isbn|barcode|serial)\b"), "str"),
    ("monetary", re.compile(r"salary|pay|wage|price|cost|amount|revenue|budget|fee|income|balance"), "num"),
    ("measurement", re.compile(r"weight|height|length|width|depth|temperature|distance|speed|size|dimension"), "num"),
    ("quantity", re.compile(r"count|quantity|qty|total|units|stock"), "int"),
]
_GT_XSD_CLASS = {"xsd:decimal": "num", "xsd:double": "num", "xsd:float": "num",
                 "xsd:integer": "int", "xsd:int": "int", "xsd:long": "int",
                 "xsd:date": "date", "xsd:dateTime": "date"}


@functools.lru_cache(maxsize=1)
def _gt_pools() -> dict:
    """{semantic_type: [values]} loaded once from the profiles; {} if not built (falls back to mechanism)."""
    try:
        d = json.loads(_GT_PROFILES.read_text())
        return {st: [v for v, _src in p["values"]] for st, p in d.items() if p.get("values")}
    except (OSError, ValueError, KeyError):
        return {}


# matched (so they claim the column ahead of the generic `name` rule) but NOT injected — routed to the
# mechanical fallback pending stronger pool curation (product pool still admits lorem/macro noise).
_GT_DEFERRED = {"product"}
_GT_NUMERIC = {"monetary", "measurement", "quantity"}  # pools are magnitude-mixed → require the aligner's range


def require_gittables(min_types: int = 6, min_values: int = 40) -> None:
    """LOUD availability BOUNDARY (RH: no silent degradation) — call at corpus-gen entry. If the GitTables value
    profiles are missing or degenerate (transiently unavailable / un-built), HALT rather than let realistic
    columns silently revert to mechanical placeholders. Build with scripts/build_gittables_value_profiles.py."""
    healthy = [st for st, vs in _gt_pools().items() if len(vs) >= min_values]
    if len(healthy) < min_types:
        raise RuntimeError(
            f"GitTables value profiles unavailable/degenerate ({len(healthy)} healthy types < {min_types}); "
            f"value realism would silently degrade — refusing to generate. Run "
            f"`uv run --no-sync python scripts/build_gittables_value_profiles.py`. ({_GT_PROFILES})")


def _semantic_type_of(name: str, xsd_type: str) -> "str | None":
    """The GitTables semantic type an injected column claims, or None (a legitimate non-match → mechanical).
    ``name`` is taken in its ORIGINAL case: camelCase/underscored heads are split to spaced-lowercase so the
    word-boundary rules fire on ``employeeName`` / ``employee_name`` / ``employee name`` alike — the SINGLE
    normalization both value paths (rows.value_for + entities._cell) share, so realism is identical across them."""
    want = _GT_XSD_CLASS.get(xsd_type, "str")  # string/entity slots → "str"
    norm = re.sub(r"[_\s]+", " ", re.sub(r"(?<!^)(?=[A-Z])", " ", name)).lower()  # camelCase/snake → boundaries
    for st, name_re, compat in _GT_NAME_RULES:
        if name_re.search(norm) and (compat == want or (want == "str" and compat == "str")):
            return st
    return None


def _gittables_value(name: str, xsd_type: str, rng: random.Random, constraint=None) -> "str | None":
    """A real, PROVENANCE-retained GitTables value — type-checked + (optionally) plausibility-CONSTRAINED by the
    agent-mediated aligner (subtractive). None → the column claims no injected type (legitimate mechanical), or a
    deferred type (claimed but curation-pending). If a constraint empties the pool, we sample the nearest REAL
    values rather than a mechanical placeholder — the cell stays real + provenanced, never a silent fallback."""
    st = _semantic_type_of(name, xsd_type)
    if st is None or st in _GT_DEFERRED:
        return None
    # a NUMERIC pool is magnitude-mixed ($5 unit price beside $2M revenue) → only draw from it when the round-trip
    # actually supplied a RANGE. An absent OR empty (engine-off) constraint keeps the safe mechanical generator —
    # a non-None-but-rangeless constraint must NOT re-open the magnitude-mixed pool (the base_salary=2.97M trap).
    if st in _GT_NUMERIC and not (constraint is not None
                                  and (constraint.num_min is not None or constraint.num_max is not None)):
        return None
    pool = _gt_pools().get(st)
    if not pool:
        return None
    if constraint is not None:
        from aegir.ontology import value_alignment  # noqa: PLC0415
        filtered = value_alignment.apply(pool, constraint)
        if filtered:                                   # plausible real values survive → sample a plausible one
            return rng.choice(filtered)
        # constraint emptied the pool → still a REAL value from the full pool (provenanced), never mechanical
    return rng.choice(pool)


def value_for(col, st: "SpineTable", *, row_ix: int, seed: int = _GLOBAL_SEED,
              definition: str | None = None, pools: dict[str, list[str]] | None = None,
              entity_values: list[str] | None = None, gittables: bool = True, constraint=None) -> str:
    """A single deterministic, type-true, realistic cell value for ``col`` at ``row_ix``.

    FK columns are NOT produced here — :func:`materialize_rows` overwrites them from
    the referenced table's PK pool. Order of preference: authoritative xsd numeric/
    temporal/boolean type → ontology-grounded definition enum → concept-specific
    LLM-seeded ``entity_values`` pool → curated semantic pool → realistic entity/label
    instance. ``entity_values`` is the Comp 4 domain layer: concept-real values for the
    entity/string columns that would otherwise be ``"<Concept> NN"`` placeholders.
    """
    pools = SEMANTIC_VALUE_POOLS if pools is None else pools
    t = col.slot_type
    name = col.name.lower()
    rng = random.Random(_seed(st.table.name, col.name, row_ix, "v", base=seed))
    enum = parse_enum_from_definition(definition)
    if enum:                                       # ontology-grounded value set — most specific, any type
        return rng.choice(enum)
    if t in _XSD_NONSTRING:                        # numeric/temporal/boolean: the xsd type is authoritative
        if gittables:                              # real value, type-checked to this xsd type (realism > rng.uniform)
            gv = _gittables_value(col.name, t, rng, constraint)   # original case → camelCase heads detected
            if gv is not None:
                return gv
        return _xsd_value(t, name, rng)
    if entity_values:                              # concept-specific LLM-seeded domain values
        return rng.choice(entity_values)
    if name in pools:                              # curated semantic pool
        return rng.choice(pools[name])
    if gittables:                                  # real GitTables value in place of a "{Concept} NN" placeholder
        gv = _gittables_value(col.name, t, rng, constraint)       # original case → camelCase heads detected
        if gv is not None:
            return gv
    return _entity_value(col, st, row_ix)          # mechanical fallback (xsd:string / untyped / unknown)


def _col_idx(st: "SpineTable", name: str) -> int | None:
    for i, c in enumerate(st.table.columns):
        if c.name == name:
            return i
    return None


def _fill_eav_registry(st: "SpineTable", *, seed: int) -> None:
    """An EAV attribute-registry table: one row per attribute (id, attr_name, attr_type)."""
    attrs = st.realize_meta.get("attributes", [])
    pki, ni, ti = _pk_index(st), _col_idx(st, "attr_name"), _col_idx(st, "attr_type")
    rows = []
    for i, (name, xtype) in enumerate(attrs):
        row = ["" for _ in st.table.columns]
        row[pki] = pk_value(st, i)
        if ni is not None:
            row[ni] = name
        if ti is not None:
            row[ti] = xtype
        rows.append(row)
    st.table.rows = rows


def _fill_eav_values(st: "SpineTable", by_name: dict, *, seed: int) -> None:
    """An EAV value table: the entity × attribute cross-product (the row explosion that IS EAV growth),
    each cell typed by the attribute and RI-safe (entity_id from the entity PK pool, attr_id from the
    registry PK pool). Values reuse value_for so enums/curated pools still ground them."""
    from types import SimpleNamespace
    meta = st.realize_meta
    ent, reg = by_name.get(meta.get("entity_table")), by_name.get(meta.get("registry_table"))
    if not ent or not reg or not ent.table.rows or not reg.table.rows:
        st.table.rows = []
        return
    members = list(meta.get("members") or [])
    rpk, rni = _pk_index(reg), _col_idx(reg, "attr_name")
    name2pk = {r[rni]: r[rpk] for r in reg.table.rows if rni is not None and r[rni] in members}
    ent_pks = [r[_pk_index(ent)] for r in ent.table.rows]
    pki, ei, ai, vi = (_pk_index(st), _col_idx(st, "entity_id"), _col_idx(st, "attr_id"), _col_idx(st, "value"))
    vtype = st.table.columns[vi].slot_type if vi is not None else "xsd:string"
    rows, k = [], 0
    for epk in ent_pks:
        for name in sorted(name2pk):
            rng = random.Random(_seed(st.table.name, epk, name, "eav", base=seed))
            row = ["" for _ in st.table.columns]
            row[pki] = pk_value(st, k)
            if ei is not None:
                row[ei] = epk
            if ai is not None:
                row[ai] = name2pk[name]
            if vi is not None:
                col_like = SimpleNamespace(name=name, slot_type=vtype, slot_ref="data:value")
                row[vi] = value_for(col_like, ent, row_ix=k, seed=seed)
            rows.append(row)
            k += 1
    st.table.rows = rows


_EAV_KINDS = {"eav_registry", "eav_value"}


def resolve_constraints(spine: Sequence["SpineTable"], definitions: dict[str, dict[str, str]] | None = None,
                        domain: str = "", *, engine: bool = True) -> dict:
    """Agent-mediated round-trip (the SUBTRACTIVE aligner): for each NUMERIC injected column, resolve a
    plausibility Constraint ONCE (cached, deduped across the corpus) → ``{(table, col): Constraint}``. A numeric
    GitTables pool is magnitude-mixed ($5 unit-price beside $2M revenue), so this range is what makes a numeric
    column REAL rather than mechanical. Graceful per-column: an engine hiccup is LOGGED and that column stays
    mechanical (not silent, not catastrophic) — the availability guarantee is ``require_gittables``, this is the
    quality refinement on top. [[gittables_value_realism]] [[signal_boundary_machinery]]"""
    from aegir.ontology import value_alignment  # noqa: PLC0415
    definitions = definitions or {}
    out: dict = {}
    seen: dict[tuple[str, str], object] = {}
    for st in spine:
        tname = st.table.name
        col_defs = definitions.get(tname, {})
        for col in st.table.columns:
            sem = _semantic_type_of(col.name, col.slot_type)
            if sem is None or sem not in _GT_NUMERIC:
                continue
            ckey = (sem, col.name.lower())
            if ckey not in seen:
                pool = _gt_pools().get(sem) or []
                try:
                    seen[ckey] = value_alignment.constrain(
                        col.name, col_defs.get(col.name, ""), sem, pool[:12], domain, engine=engine)
                except Exception as exc:  # noqa: BLE001 — a hiccup degrades THIS column (logged), not the run
                    logger.warning(f"value-alignment skipped for '{col.name}' ({sem}): {exc}")
                    seen[ckey] = None
            if seen[ckey] is not None:
                out[(tname, col.name)] = seen[ckey]
    return out


def materialize_rows(spine: Sequence["SpineTable"], fks: Sequence, *, seed: int = _GLOBAL_SEED,
                     definitions: dict[str, dict[str, str]] | None = None,
                     entity_pools: dict[str, dict[str, list[str]]] | None = None,
                     align: bool = False, domain: str = "") -> None:
    """Populate ``st.table.rows`` for every table so that RI = 1.0 by construction.

    Three passes: (A) synthesize each table's PK pool; (B) typed/realistic non-FK
    cells; (C) FK cells drawn (seeded, non-uniform fan-out) from the referenced
    table's PK pool. ``definitions`` optionally maps ``table_name → {col_name →
    skos:definition}`` so ontology-enumerated value sets win (supplied by the
    caller / #58); absent → curated pools + type generators. ``entity_pools`` maps
    ``table_name → {col_name → [domain values]}`` — concept-specific, RI-safe (non-FK
    only) instance values (LLM-seeded + committed, see ``scripts/seed_entity_values.py``)
    that replace ``"Process 01"`` placeholders for entity/string columns. Mutates in place.
    """
    definitions = definitions or {}
    entity_pools = entity_pools or {}
    by_name = {st.table.name: st for st in spine}
    fk_targets = {e.dst_table for e in fks}
    src_fk: dict[str, dict[str, str]] = {}
    for e in fks:
        src_fk.setdefault(e.src_table, {})[e.src_col] = e.dst_table

    # counts (FK targets get enough distinct parents; junctions get denser M:N fan-out)
    counts = {st.table.name: row_count_for(st, seed) for st in spine}
    for tname in fk_targets:
        if tname in counts:
            counts[tname] = max(counts[tname], _FK_TARGET_MIN_ROWS)
    for st in spine:
        if getattr(st, "kind", "entity") == "junction":                       # dense many-to-many association rows
            counts[st.table.name] = max(counts[st.table.name], _JUNCTION_MIN_ROWS)

    # Pass A — PK pools (EAV registry/value tables are filled by their dedicated passes below)
    for st in spine:
        if getattr(st, "kind", "entity") in _EAV_KINDS:
            continue
        n, ncol, pk = counts[st.table.name], len(st.table.columns), _pk_index(st)
        st.table.rows = [["" for _ in range(ncol)] for _ in range(n)]
        for i in range(n):
            st.table.rows[i][pk] = pk_value(st, i)

    # EAV passes — registry first (defines attr PKs), then value tables (entity × attr cross-product).
    for st in spine:
        if getattr(st, "kind", "entity") == "eav_registry":
            _fill_eav_registry(st, seed=seed)
    for st in spine:
        if getattr(st, "kind", "entity") == "eav_value":
            _fill_eav_values(st, by_name, seed=seed)

    # Round-trip alignment (agent-mediated) — resolve per-numeric-column plausibility constraints once, so a
    # numeric GitTables value lands in a domain-plausible range (else it stays the safe mechanical generator).
    cmap = resolve_constraints(spine, definitions, domain) if align else {}

    # Pass B + C — non-PK cells, then FK overwrite from target PK pool (EAV tables already complete)
    for st in spine:
        if getattr(st, "kind", "entity") in _EAV_KINDS:
            continue
        tname = st.table.name
        pk, fkmap = _pk_index(st), src_fk.get(tname, {})
        col_defs = definitions.get(tname, {})
        col_pools = entity_pools.get(tname, {})
        for i, row in enumerate(st.table.rows):
            for ci, col in enumerate(st.table.columns):
                if ci == pk:
                    continue
                if col.name in fkmap:
                    dst = by_name.get(fkmap[col.name])
                    if dst and dst.table.rows:
                        dpk = _pk_index(dst)
                        pool = [r[dpk] for r in dst.table.rows]
                        row[ci] = pool[_seed(tname, col.name, i, "fk", base=seed) % len(pool)]
                        continue
                row[ci] = value_for(col, st, row_ix=i, seed=seed, definition=col_defs.get(col.name),
                                    entity_values=col_pools.get(col.name), constraint=cmap.get((tname, col.name)))
            _enforce_temporal_coherence(st, row, seed=seed, row_ix=i)


def assert_referential_integrity(spine: Sequence["SpineTable"], fks: Sequence) -> None:
    """Hard invariant: every FK cell is a PK value of the referenced table. Raises on violation."""
    by_name = {st.table.name: st for st in spine}
    for e in fks:
        src, dst = by_name.get(e.src_table), by_name.get(e.dst_table)
        if src is None or dst is None:
            continue
        si, di = _col_index(src, e.src_col), _col_index(dst, e.dst_col)
        if si is None or di is None:
            continue
        dst_pks = {r[di] for r in dst.table.rows}
        for r in src.table.rows:
            if si < len(r) and r[si] not in dst_pks:
                raise AssertionError(
                    f"RI violation: {e.src_table}.{e.src_col}={r[si]!r} ∉ {e.dst_table}.{e.dst_col}")


def table_rows_as_records(st: "SpineTable") -> list[dict[str, str]]:
    """Rows as column-name→value dicts (for the prompt preview + the verifiable footer)."""
    names = [c.name for c in st.table.columns]
    return [{names[i]: (r[i] if i < len(r) else "") for i in range(len(names))} for r in st.table.rows]


if __name__ == "__main__":  # deterministic smoke: RI=1.0, type-true values, reproducible
    from types import SimpleNamespace

    from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec

    def _fixture():
        tA = TableSpec(name="t_sample", ref="x", columns=[
            ColumnSpec("id", "Class", "__pk__"),
            ColumnSpec("collected_at", "xsd:dateTime", "data:collectedAt"),
            ColumnSpec("status", "DataProperty", "data:hasStatus"),
        ], rows=[])
        stA = SimpleNamespace(table=tA, not_null=set(),
                              template=SimpleNamespace(template_id="02_observation_sample_record",
                                                       verbal_template="a sample record"))
        tB = TableSpec(name="t_process", ref="y", columns=[
            ColumnSpec("id", "Class", "__pk__"),
            ColumnSpec("subject", "Class", "S"),
            ColumnSpec("processed_by", "Class", "P"),
            ColumnSpec("confidence", "xsd:decimal", "data:hasConfidence"),
        ], rows=[])
        stB = SimpleNamespace(table=tB, not_null={"processed_by"},
                              template=SimpleNamespace(template_id="04_kernel_process_run",
                                                       verbal_template="a process run"))
        fks = [FKEdge("t_process", "processed_by", "t_sample", "id", "processedBy")]
        return [stA, stB], fks

    spine, fks = _fixture()
    defs = {"t_sample": {"status": "operational status: pending, running, complete, failed"}}
    materialize_rows(spine, fks, seed=123, definitions=defs)        # type: ignore[arg-type]  # duck-typed stubs
    assert_referential_integrity(spine, fks)                        # type: ignore[arg-type]

    spine2, fks2 = _fixture()
    materialize_rows(spine2, fks2, seed=123, definitions=defs)      # type: ignore[arg-type]
    assert [s.table.rows for s in spine] == [s.table.rows for s in spine2], "not deterministic"

    sample = next(s for s in spine if s.table.name == "t_sample")
    proc = next(s for s in spine if s.table.name == "t_process")
    print("t_sample:", [c.name for c in sample.table.columns])
    for r in sample.table.rows[:3]:
        print("  ", r)
    print("t_process:", [c.name for c in proc.table.columns])
    for r in proc.table.rows[:3]:
        print("  ", r)
    print(f"RI=1.0 ✓  deterministic ✓  (status from ontology enum; collected_at dateTime; confidence∈[0,1])")
