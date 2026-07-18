"""differentia_harvest — the GitTables value-harvest pre-pass over the DIFFERENTIA-TYPE SPECS.

The contract is the sdg-strategy component ``targets/differentia_value_specs`` (353 specs, one per
distinct differentia property of the authored taxonomy) — consumed BY STRATEGY REF via
``strategy.manifest.read_component``, exactly as the component's harvest_contract states: the strategy
tree carries the what-to-harvest; the wire (or this local baseline) carries only results.

Two harvesters behind one interface (Increment staging per the contract):
- ``local`` (Increment 1, this module): an honest name+seed-corroboration matcher over a seeded
  GitTables sample. REAL values with REAL per-value lineage only — it is a baseline the Atelier
  ensemble strictly improves on, never a fabricating mock. Reuses the profiles builder's cleaning
  (importlib, not replicated) so pool hygiene is identical to the gtvs pools.
- ``federated`` (Increment 2): Atelier's maxsim/NHSVM/CatBoost ensemble behind the zndx.engine.v1
  federation face (``AEGIR_HARVEST_TARGET``, e.g. atelier :50251) once the Harvest RPC lands in
  signals-protocol (additive-only). Until then this raises loudly rather than degrade silently.

Output: build/gittables/differentia_profiles.json — pools keyed by the SPEC's projected snake column
name, each value ``[value, "<table_sha1>:<column>"]`` (the gtvs lineage shape), consumed at generation
time by the SAME unified core (rows._gittables_value) ahead of the name-rule semantic types. The harvest
is EMITTED to aegir_hx as an OpenLineage ``differentia_harvest`` run with token-grain value nodes
(governance.provenance) — the graph answers noun-admission; the artifact stays the rebuildable truth. Floors:
min_distinct >= 4 per pool; below-floor specs stay FLAGGED, never fabricated (the same below-frontier
discipline as the residual differentiae). PII-shaped spec columns are DEFERRED to the sensitivity
membrane per the gtvs policy. [[gittables_value_realism]] [[differentia_sufficiency_taxonomy]]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
GITTABLES = Path("/raid/datasets/gittables")
OUT = REPO / "build" / "gittables" / "differentia_profiles.json"
SPEC_COMPONENT = "targets/differentia_value_specs.json"
HARVESTER_VERSION = "local-baseline-v1-2026-07-17"

MIN_DISTINCT = 4          # gtvs filter_heuristics pool floor — below it the spec stays flagged
CAP_PER_COL = 30
CAP_PER_SPEC = 2000
# PII-shaped differentia columns → deferred to the sensitivity membrane (gtvs pii_governance)
_PII_COL = re.compile(r"\b(name|email|phone|address|birth|dob|ssn)\b")
# Baseline-harvest value rejections (the 8000-scan lessons; the increment-2 ensemble subsumes all three):
# lorem TAIL vocabulary the gtvs list misses ('quam','justo','sapien' polluted item_category). Applied per
# value AND as a COLUMN gate (>=2 lorem-hit values → the whole column is a mockaroo dump, drop it — a real
# domain column never has two latin-gibberish values, so the gate is robust even if the vocab is partial).
_LOREM_EXT = re.compile(r"\b(quam|justo|nunc|nisl|duis|condimentum|curabitur|sem|praesent|risus|vitae|eget|"
                        r"ligula|sagittis|augue|turpis|magna|ornare|integer|donec|etiam|morbi|proin|quisque|"
                        r"mauris|fusce|cras|velit|tortor|lectus|rhoncus|blandit|ultrices|molestie|ante|vel|"
                        r"sapien|libero|leo|odio|posuere|cubilia|curae|semper|luctus|dapibus|habitasse|platea|"
                        r"dictumst|iaculis|gravida|suscipit|sodales|ullamcorper|viverra|tristique|convallis|"
                        r"laoreet|faucibus|pharetra|hendrerit|nibh|arcu|orci|purus|felis|urna|erat|massa|"
                        r"tellus|metus|mattis|cursus|egestas|feugiat|pretium|volutpat|dignissim|imperdiet|"
                        r"venenatis|vulputate|scelerisque|penatibus|natoque|ridiculus|nascetur)\b", re.I)
# SQL/schema type names echoed as values ('VarChar' in country_code);
_SQL_TYPE = re.compile(r"^(var)?char\d*$|^(n?varchar|int(eger)?|bigint|smallint|tinyint|text|datetime2?|"
                       r"timestamp|decimal|numeric|float|double|boolean|bool|blob|clob)$", re.I)
# identifier-shaped values (3+ snake tokens, no digits: 'response_button_text' in zone_type) — read as
# framework identifiers, never as domain values; legit short codes (LEA, DE, H.E.A.T.) don't match.
_IDENTIFIER_SHAPE = re.compile(r"^[a-z]+(_[a-z]+){2,}$")


def _reject_value(v: str, col_tokens: "set[str]") -> bool:
    """True → drop: lorem tail, SQL type echo, identifier-shape, or a HEADER ECHO of the column itself
    ('code of the country' / 'Tab.Const.Country' as a country_code value)."""
    if _LOREM_EXT.search(v) or _SQL_TYPE.match(v) or _IDENTIFIER_SHAPE.match(v):
        return True
    vtoks = set(re.split(r"[^a-z0-9]+", v.lower())) - {""}
    return bool(col_tokens and col_tokens <= vtoks)      # every column token present in the value → echo


def _builder():
    """The gtvs profiles builder's cleaning layer — imported, not replicated (identical pool hygiene)."""
    spec = importlib.util.spec_from_file_location(
        "build_gittables_value_profiles", REPO / "scripts" / "build_gittables_value_profiles.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_component(ref: "str | None" = None) -> dict:
    """The spec component BY STRATEGY REF (working tree for main, `git show` for shadows)."""
    from aegir.strategy.manifest import read_component
    return json.loads(read_component(SPEC_COMPONENT, ref))


def harvestable(component: dict) -> "dict[str, dict]":
    """The specs the value harvest applies to: data-kind, string-class xsd (numeric stays behind the
    aligner gate), non-PII-shaped. Object-kind specs project as FK columns — no value pool."""
    out, deferred = {}, []
    for prop, s in component["specs"].items():
        if s["kind"] != "data" or s["xsd"] not in ("string", "anyURI", "token"):
            continue
        if _PII_COL.search(s["column"]["snake"].replace("_", " ")):
            deferred.append(prop)
            continue
        out[prop] = s
    if deferred:
        print(f"  deferred to sensitivity membrane (PII-shaped): {len(deferred)} specs")
    return out


def _norm_col(name: str) -> str:
    """camelCase / snake / spaced / plural-tolerant → canonical snake (the spec's column key form)."""
    n = re.sub(r"[_\s]+", " ", re.sub(r"(?<!^)(?=[A-Z])", " ", str(name))).lower().strip()
    n = re.sub(r"s\b", "", n)                       # plural-strip per token (funding sources ≡ funding source)
    return n.replace(" ", "_")


def harvest_local(specs: "dict[str, dict]", sample: int = 8000, seed: int = 13) -> "tuple[dict, dict]":
    """The Increment-1 baseline: scan a seeded GitTables sample; a candidate column matches a spec by
    NAME (canonical-snake equality, or spec-tokens ⊆ column-tokens WITH seed-exemplar corroboration);
    values are admitted through the gtvs cleaner with per-value lineage. Returns (pools, run_meta)."""
    b = _builder()
    import random
    files = sorted(p for p in GITTABLES.iterdir() if p.suffix == ".parquet")
    rng = random.Random(seed)
    picked = rng.sample(files, min(sample, len(files)))

    by_col: "dict[str, list[tuple[str, dict]]]" = defaultdict(list)      # canonical col key → [(prop, spec)]
    tokens: "dict[str, set]" = {}
    seeds_cf: "dict[str, set]" = {}
    for prop, s in specs.items():
        key = _norm_col(s["column"]["snake"])
        by_col[key].append((prop, s))
        tokens[prop] = set(key.split("_"))
        seeds_cf[prop] = {v.casefold() for v in s["seed_exemplars"]}

    import pandas as pd
    pools: "dict[str, dict]" = {}
    stats = {"tables_scanned": 0, "columns_matched": 0, "name_exact": 0, "token_subset": 0}
    for f in picked:
        try:
            df = pd.read_parquet(f)
        except Exception:  # noqa: BLE001 — GitTables has malformed parquet; skip
            continue
        stats["tables_scanned"] += 1
        if df.shape[0] < b.FILTER["min_rows"]:
            continue
        th = None
        for col in df.columns:
            ckey = _norm_col(col)
            ctoks = set(ckey.split("_"))
            hits = list(by_col.get(ckey, []))
            exact = bool(hits)
            if not hits:                                # token-subset match needs seed corroboration below
                hits = [(p, s) for p, toks in tokens.items() if len(toks) > 1 and toks <= ctoks
                        for s in (specs[p],)]
            if not hits:
                continue
            s_ = df[col]
            if s_.isna().mean() > b.FILTER["max_null_frac"]:
                continue
            raw_vals = [c for c in (b._clean(v) for v in s_.dropna().astype(str)) if c]
            if sum(1 for v in raw_vals[:40] if _LOREM_EXT.search(v)) >= 2:
                continue                                # column-level lorem gate: a mockaroo dump, drop whole col
            vals = list(dict.fromkeys(c for c in raw_vals if not _reject_value(c, ctoks)))
            if len(vals) < MIN_DISTINCT:
                continue
            for prop, spec in hits:
                overlap = sum(1 for v in vals if v.casefold() in seeds_cf[prop])
                # Admission: a MULTI-token exact name (funding_source, consent_status) is specific enough
                # to stand alone; a SINGLE-token generic (type, category, status) or a token-subset match
                # must be seed-corroborated — else any domain's 'type' column pollutes the pool (the same
                # rationale gtvs uses to DEFER category/status; the RaygunBreadcrumb lesson).
                if not overlap and not (exact and "_" in _norm_col(spec["column"]["snake"])):
                    continue
                pool = pools.setdefault(spec["column"]["snake"], {
                    "property": prop, "xsd": spec["xsd"], "values": [], "n_source_columns": 0,
                    "seed_overlap_columns": 0, "method": "name+seed-overlap"})
                if len(pool["values"]) >= CAP_PER_SPEC:
                    continue
                th = th or b._table_hash(f)
                for v in vals[:CAP_PER_COL]:
                    pool["values"].append([v, f"{th}:{col}"])
                pool["n_source_columns"] += 1
                pool["seed_overlap_columns"] += 1 if overlap else 0
                stats["columns_matched"] += 1
                stats["name_exact" if exact else "token_subset"] += 1

    # floors + baseline confidence (documented crude — the ensemble replaces it)
    for c in list(pools):
        p = pools[c]
        p["values"] = list({v: src for v, src in p["values"]}.items())        # dedup across columns
        if len(p["values"]) < MIN_DISTINCT:
            del pools[c]                                                       # below floor → stays flagged
            continue
        p["n"] = len(p["values"])
        p["confidence"] = round(min(1.0, 0.45 + 0.15 * min(p["n_source_columns"], 3)
                                    + (0.25 if p["seed_overlap_columns"] else 0.0)), 2)
    meta = {"harvester": HARVESTER_VERSION, "sample": sample, "seed": seed, **stats,
            "snapshot_sha1": hashlib.sha1("".join(sorted(p.name for p in picked)).encode()).hexdigest()[:16]}
    return pools, meta


def harvest_federated(specs: "dict[str, dict]") -> "tuple[dict, dict]":
    """Increment 2: Atelier's ensemble behind the zndx.engine.v1 federation face. Loud until it lands —
    a silent local fallback here would misreport the harvester in lineage."""
    target = os.environ.get("AEGIR_HARVEST_TARGET", "")
    raise NotImplementedError(
        "federated harvest needs the Harvest RPC in signals-protocol (additive) + the Atelier servicer"
        + (f" at {target}" if target else " (set AEGIR_HARVEST_TARGET, e.g. atelier :50251)")
        + " — Increment 2 of targets/differentia_value_specs.harvest_contract.staging")


def write_profiles(pools: dict, run_meta: dict, component: dict) -> Path:
    """Content-addressed pools + the reproducibility triple the contract pins: (spec component version+
    hash, GitTables snapshot, harvester version)."""
    from aegir.strategy.manifest import declared
    man = declared() or {}
    doc = {"_meta": {"artifact": "differentia_value_profiles",
                     "spec_component": {"version": component["version"],
                                        "sha": (man.get("components") or {}).get(SPEC_COMPONENT),
                                        "strategy_id": man.get("strategy_id")},
                     **run_meta},
           "pools": {c: pools[c] for c in sorted(pools)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    # the artifact is the rebuildable TRUTH; the aegir_hx graph is the queryable Atlas+OL projection —
    # emit best-effort (a down graph never breaks a harvest; provenance.backfill() re-projects later)
    try:
        from aegir.governance.provenance import emit_pool_lineage
        r = emit_pool_lineage("differentia_harvest", f"{run_meta['snapshot_sha1']}:{run_meta['harvester']}",
                              run_meta["snapshot_sha1"],
                              {f"differentia/{c}": pools[c]["values"] for c in pools},
                              {"harvester": run_meta["harvester"]})
        print(f"lineage → aegir_hx: OL run {r['run_id'][:8]}… · {r['outputs']} pool datasets · {r['tokens']} tokens")
    except Exception as e:  # noqa: BLE001
        print(f"lineage emission deferred (graph unavailable: {e}) — run governance.provenance.backfill()")
    return OUT
