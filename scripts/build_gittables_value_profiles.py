"""build_gittables_value_profiles — curate GitTables into per-semantic-type value pools, PROVENANCE-retained.

The value generator (rows.py) was pure mechanism — `rng.uniform(0,1000)` for a salary, `"{Column} 42"` for a
name — so tables read as nonsense (RH's "essence of values" concern). This grounds value generation in the
EMPIRICAL distribution of 562k real GitTables tables, WITHOUT sanitizing: real values are admitted because their
LINEAGE is retained (RH 2026-07-13 — provenance ≻ exclusion for an enterprise-lineage corpus). Each value carries
its source (table content-hash + column); the SEGMENTATION RULES + run parameters below are emitted as a discrete
lineage manifest so a published corpus can pin the exact sampling strategy (→ Apache Atlas). [[atlas_age_provenance_graph]]

    uv run --no-sync python scripts/build_gittables_value_profiles.py --sample 8000 --seed 13

Output: build/gittables/value_profiles.json (pools by semantic type, capped, with per-value provenance) +
        build/gittables/lineage.json (the segmentation rules, filter heuristics, snapshot, run params — the
        provenance record that becomes a discrete sdg-strategy entry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GITTABLES = Path("/raid/datasets/gittables")
OUT = REPO / "build" / "gittables"

# ── SEGMENTATION RULES (the regex layer — a DISCRETE lineage entry; versioned) ─────────────────────────────
# semantic_type → (column-name token pattern, optional value-shape validator). Column NAME is the primary
# signal (reliable); the value validator guards against a mis-named column polluting a pool.
# v3: PER-VALUE validator gate — a value enters a pool only if its OWN shape corroborates the type, not merely
# because its column name did (v2 admitted a name-column's stray junk rows / a single-word place in an org column).
SEG_VERSION = "seg-v3-2026-07-13"
_NAME = re.compile(r"^[A-Z][a-z]+(?:[ '\-][A-Z][a-z]+){1,2}$")
_MONEY = re.compile(r"^\$?\d{1,3}(?:,\d{3})+(?:\.\d{2})?$|^\$\d+(?:\.\d{2})?$|^\d+\.\d{2}$")  # needs $, comma-thousands, or cents
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
_URL = re.compile(r"^https?://", re.I)
_DATEISH = re.compile(r"^\D*\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_TITLECASE = re.compile(r"^[A-Z][A-Za-z]+(?: [A-Z][A-Za-z&/-]+){0,3}$")        # Title-Case phrase (each word capitalized)
_ORG = re.compile(r"\b(Inc|LLC|Ltd|Corp|Company|Bank|University|GmbH|PLC|Foundation|Institute|Games|Systems|"
                  r"Technologies|Solutions|Holdings|Industries|Enterprises|Associates|Partners)\b|"
                  r"^[A-Z][A-Za-z0-9&.'\-]+ [A-Z][A-Za-z0-9&.'\- ]+$")          # org suffix OR ≥2 Capitalized words
_LOREM = re.compile(r"\b(lorem|ipsum|dolor|amet|consectetur|adipiscing|sed|nullam|porttitor|potenti|"
                    r"vestibulum|euismod|tincidunt|fermentum|malesuada|pellentesque|vivamus|aliquam)\b", re.I)
# registration / scrape placeholders that pass the shape validators but read as nonsense in a real column
# (WHOIS redaction, proxy privacy, "not disclosed", test/dummy stand-ins) — reject broadly.
_PLACEHOLDER = re.compile(r"\b(redacted|privacy|whois|undisclosed|not\s+disclosed|not\s+applicable|"
                          r"data\s+protected|domains?\s+by\s+proxy|placeholder|dummy|test\s*data)\b", re.I)
_IDENT = re.compile(r"^(?=.*[A-Za-z0-9])(?!\d+\.\d+$)[A-Za-z0-9][A-Za-z0-9\-_/]{2,}$")  # alnum code, not a float
_QUANT = re.compile(r"^\d+$")                                                  # a plain count
_DECIMAL = re.compile(r"^-?\d+\.\d+$")
RULES: "dict[str, tuple[re.Pattern, re.Pattern | None]]" = {
    "person_name":   (re.compile(r"\b(name|employee|person|contact|author|customer|client|owner|manager|holder)\b", re.I), _NAME),
    "organization":  (re.compile(r"\b(company|organi[sz]ation|org|employer|vendor|supplier|institution|firm|agency|manufacturer|brand)\b", re.I), _ORG),
    "monetary":      (re.compile(r"\b(salary|pay|wage|price|cost|amount|revenue|budget|fee|income|balance|usd|dollar)\b", re.I), _MONEY),
    "temporal":      (re.compile(r"\b(date|time|year|created|updated|start|end|dob|birth|timestamp|period)\b", re.I), _DATEISH),
    "email":         (re.compile(r"\b(e?mail)\b", re.I), _EMAIL),
    "url":           (re.compile(r"\b(url|link|website|homepage|uri)\b", re.I), _URL),
    "geo_place":     (re.compile(r"\b(city|state|country|region|province|county|location|address|street|place|nation)\b", re.I), _TITLECASE),
    "job_title":     (re.compile(r"\b(title|position|role|job|occupation|designation)\b", re.I), _TITLECASE),
    "category":      (re.compile(r"\b(status|state|type|category|kind|class|level|grade|tier|group)\b", re.I), _TITLECASE),
    "product":       (re.compile(r"\b(product|item|model|goods|service|part)\b", re.I), None),
    "identifier":    (re.compile(r"\b(id|code|ref|sku|isbn|barcode|serial)\b", re.I), _IDENT),
    "phone":         (re.compile(r"\b(phone|tel|mobile|fax|cell)\b", re.I), re.compile(r"\d{3}[-.\s]\d{3,4}[-.\s]\d{4}")),
    "measurement":   (re.compile(r"\b(weight|height|length|width|depth|temperature|distance|speed|size|dimension)\b", re.I), _DECIMAL),
    "quantity":      (re.compile(r"\b(count|quantity|qty|total|units|stock)\b", re.I), _QUANT),
    "color":         (re.compile(r"\b(colou?r)\b", re.I), _TITLECASE),
}
# semantic types that carry genuine PII — admitted (provenance-retained) but FLAGGED for the governance
# membrane; a corpus run can route these through the sensitivity net (RH: catch the dangerous few).
PII_TYPES = {"person_name", "email", "phone", "geo_place"}
# FILTER heuristics (the curation layer — also a lineage entry): reject scraped/stats junk.
FILTER = {"max_value_len": 60, "min_distinct": 4, "min_rows": 5, "max_null_frac": 0.5,
          "junk_chars": r"[\\{}$^~]|:\s*,\s*:", "cap_per_type": 4000, "cap_per_col": 30}
_JUNK = re.compile(FILTER["junk_chars"])


def _clean(v: str) -> "str | None":
    v = re.sub(r"\s+", " ", str(v)).strip().strip("'\"").strip()  # collapse newlines/ws + strip stray quotes
    if not v or v.lower() in ("nan", "none", "null", "na", "n/a", "-", "--") or len(v) > FILTER["max_value_len"]:
        return None
    if _JUNK.search(v) or _LOREM.search(v) or _PLACEHOLDER.search(v):  # formatting junk / lorem / redaction noise
        return None
    return v


def _semantic_type(col: str, vals: "list[str]") -> "str | None":
    for st, (name_re, val_re) in RULES.items():
        if name_re.search(col):
            if val_re is None:
                return st
            hits = sum(1 for v in vals[:20] if val_re.match(v))
            if hits >= max(2, len(vals[:20]) // 3):  # value shape must corroborate the name
                return st
    return None


def _table_hash(f: Path) -> str:
    return hashlib.sha1(f.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=8000, help="number of GitTables tables to profile")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    files = sorted(p for p in GITTABLES.iterdir() if p.suffix == ".parquet")
    import random
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.sample, len(files)))

    pools: "dict[str, list]" = defaultdict(list)          # semantic_type → [(value, source)]
    prov: "dict[str, dict]" = defaultdict(lambda: {"tables": set(), "cols": 0})
    scanned = kept_cols = 0
    for f in sample:
        try:
            df = pd.read_parquet(f, columns=None)
        except Exception:  # noqa: BLE001 — GitTables has malformed parquet; skip
            continue
        scanned += 1
        if df.shape[0] < FILTER["min_rows"]:
            continue
        th = None
        for col in df.columns:
            cname = str(col)
            if not cname or cname.lower().startswith(("unnamed", "na", "col")):
                continue
            s = df[col]
            if s.isna().mean() > FILTER["max_null_frac"]:
                continue
            vals = [c for c in (_clean(v) for v in s.dropna().astype(str)) if c]
            if len(set(vals)) < FILTER["min_distinct"]:
                continue
            st = _semantic_type(cname, vals)
            if not st or len(pools[st]) >= FILTER["cap_per_type"]:
                continue
            # PER-VALUE gate: the column NAME classified the type, but only values whose SHAPE also corroborates
            # the type enter the pool — else a mostly-name column's stray junk row ('Incr Critical%') or a
            # single-word place in an org column ('Guéckédou') pollutes it. Column-level match ≠ value admission.
            val_re = RULES[st][1]
            clean = [v for v in vals if val_re is None or val_re.match(v)]
            if not clean:
                continue
            th = th or _table_hash(f)
            for v in list(dict.fromkeys(clean))[: FILTER["cap_per_col"]]:  # dedup within col
                pools[st].append([v, f"{th}:{cname}"])
            prov[st]["tables"].add(th)
            prov[st]["cols"] += 1
            kept_cols += 1

    OUT.mkdir(parents=True, exist_ok=True)
    profiles = {st: {"values": vs, "n": len(vs), "pii": st in PII_TYPES} for st, vs in pools.items()}
    (OUT / "value_profiles.json").write_text(json.dumps(profiles, indent=1))
    lineage = {
        "artifact": "gittables_value_profiles", "created": "2026-07-13", "segmentation_version": SEG_VERSION,
        "gittables_root": str(GITTABLES), "gittables_total_tables": len(files),
        "run": {"sample": args.sample, "seed": args.seed, "tables_scanned": scanned, "columns_kept": kept_cols},
        "filter_heuristics": FILTER,
        "segmentation_rules": {st: {"name_pattern": r.pattern, "value_validated": v is not None}
                               for st, (r, v) in RULES.items()},
        "per_type": {st: {"n_values": len(pools[st]), "n_source_tables": len(prov[st]["tables"]),
                          "n_source_columns": prov[st]["cols"]} for st in sorted(pools)},
        "snapshot_sha1": hashlib.sha1(
            "".join(sorted(p.name for p in sample)).encode()).hexdigest()[:16],
    }
    (OUT / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"scanned {scanned} tables · kept {kept_cols} columns · {len(pools)} semantic types")
    for st in sorted(pools, key=lambda s: -len(pools[s])):
        ex = [v for v, _ in pools[st][:3]]
        pii = " [PII]" if st in PII_TYPES else ""
        print(f"  {st:14s} {len(pools[st]):5d} values from {len(prov[st]['tables'])} tables{pii}  e.g. {ex}")
    print(f"→ {OUT}/value_profiles.json + lineage.json (snapshot {lineage['snapshot_sha1']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
