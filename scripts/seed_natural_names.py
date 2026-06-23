#!/usr/bin/env python
"""L2 — natural physical naming for the canonical-deliverable corpus (Path A C2.5).

The ontology-derived column/table names are semantic (`attestation_process`, `specimen`) — great for
Atlas/lineage *reference*, but (a) they echo the concept, so a model training on headered tables can learn
concept-from-header (a shortcut that doesn't transfer), and (b) downstream/Atelier consumers want realistic
DB schemas, not ontology jargon. This script asks the local engine (Qwen3.6) for **natural physical names** —
the way a working DBA would name the table + its columns — *decoupled from* the ontology label but plausible
for the column's meaning. Output is a committed map ``natural_names.json`` (template_id → {table, cols}) that
``build_ddl_spine`` reads to materialize the **natural** (canonical, trained) variant alongside the
**semantic** (reference) one; the natural↔canonical(Data Element) map is the lineage edge (Atlas `naming`).

Sentinel-rigorous + deterministic (cached) — same contract as seed_entity_values / derive_domain_taxonomy
(Qwen3.6 leaks reasoning; only tagged lines are read). Names are valid snake_case SQL identifiers, distinct
within a table.

    just engine-serve
    uv run --no-sync python scripts/seed_natural_names.py --limit 8        # smoke (dry)
    uv run --no-sync python scripts/seed_natural_names.py --write-resource  # full (cached, resumable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import ddl as D  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

_CACHE = REPO / "build" / "natural_name_cache.json"
_TRACES = REPO / "build" / "natural_name_traces.jsonl"
_RESOURCE = REPO / "src" / "aegir" / "ontology" / "natural_names.json"

_IDENT = re.compile(r"^[a-z][a-z0-9_]{1,38}$")
_TABLE = re.compile(r"^\s*NAT_TABLE\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*$")
_COL = re.compile(r"^\s*NAT_COL\s+([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z][A-Za-z0-9_]*)\s*$")

_SYSTEM = (
    "You are a senior data engineer naming a relational table the way it would appear in a real production "
    "database. Given the table's concept and its current (ontology-derived) column names, produce NATURAL "
    "physical names: a table name and one name per column. Hard rules: (1) snake_case, valid SQL identifiers, "
    "≤ ~32 chars; (2) realistic DB style a DBA would actually type (e.g. sample_id, material_type, "
    "collected_at, status_code, owner_ref) — NOT ontology jargon; (3) the natural name must be PLAUSIBLE for "
    "the column's meaning but NOT a verbatim copy of the ontology concept — decouple the surface from the "
    "label (e.g. concept 'specimen' → 'sample_id' or 'material_ref', not 'specimen'); (4) names distinct "
    "within the table; surrogate keys → 'id'; (5) OUTPUT CONTRACT: first a line 'NAT_TABLE: <name>', then one "
    "line per column 'NAT_COL <current_column> -> <natural_name>'. Put ALL reasoning on OTHER lines; a line "
    "without a NAT_ tag is ignored."
)


def _concept(template) -> str:
    toks = [t for t in re.split(r"[^A-Za-z]+", template.template_id) if len(t) >= 3]
    return " ".join(toks[:4]) if toks else (template.template_id or "entity")


def name_targets(template, family: str) -> "tuple[str, list[str]]":
    """(semantic table name, [semantic column names]) — the names L2 will give natural aliases."""
    st = D.template_to_table(template, family)
    return st.table.name, [c.name for c in st.table.columns]


def _prompt(concept: str, verbalization: str, table: str, cols: "list[str]") -> str:
    vb = f'\nThe table records: "{verbalization}".' if verbalization else ""
    return (f"Table concept: {concept}.{vb}\nCurrent table name: {table}\n"
            f"Current columns:\n" + "\n".join(f"  - {c}" for c in cols) +
            "\nProduce NAT_TABLE + one NAT_COL line per column.")


def parse_response(raw: str, want_cols: "set[str]") -> "dict | None":
    table = None
    cols: dict[str, str] = {}
    used: set[str] = set()
    for line in raw.splitlines():
        mt = _TABLE.match(line)
        if mt and table is None and _IDENT.match(mt.group(1).lower()):
            table = mt.group(1).lower()
            continue
        mc = _COL.match(line)
        if mc and mc.group(1) in want_cols:
            nat = mc.group(2).lower()
            if _IDENT.match(nat) and nat not in used:
                cols[mc.group(1)] = nat
                used.add(nat)
    if not table or len(cols) < max(1, len(want_cols) // 2):
        return None
    return {"table": table, "cols": cols}


def _key(tid: str, table: str, cols: "list[str]") -> str:
    return f"{tid}::{hashlib.md5((table + '|' + ','.join(sorted(cols))).encode()).hexdigest()[:8]}"


def seed_one(template, family: str, *, capability: str, temperature: float):
    from aegir.engine.client import complete_detailed
    table, cols = name_targets(template, family)
    cols = [c for c in cols if c != "id"]  # surrogate key is always 'id'
    if not cols:
        return None, table, cols, "", ""
    verbalization = (template.frames()[0] if template.frames() else template.verbal_template) or ""
    out = complete_detailed(_prompt(_concept(template), verbalization, table, cols), capability=capability,
                            system_prompt=_SYSTEM, max_tokens=4096, temperature=temperature)
    parsed = parse_response(out["text"], set(cols))
    return parsed, table, cols, out["text"], out.get("reasoning_content", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-resource", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    cat_dir = REPO / a.catalog_dir if not Path(a.catalog_dir).is_absolute() else Path(a.catalog_dir)
    files = sorted(p for p in cat_dir.glob("0*.json") if ".candidate" not in p.name and "combined" not in p.name)
    templates: list[tuple] = []
    for path in files:
        cat = load_catalog(path)
        ts = cat.templates[:a.per_family] if a.per_family else cat.templates
        templates += [(t, path.stem) for t in ts]
    if a.limit:
        templates = templates[:a.limit]

    cache = json.loads(_CACHE.read_text()) if (_CACHE.exists() and not a.no_cache) else {}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    tfh = _TRACES.open("a")
    n_called = n_cached = n_cols = 0
    for i, (t, fam) in enumerate(templates):
        table, cols = name_targets(t, fam)
        cols_nopk = [c for c in cols if c != "id"]
        if not cols_nopk:
            continue
        k = _key(t.template_id, table, cols_nopk)
        if k in cache and not a.no_cache:
            rec = cache[k]
            n_cached += 1
        else:
            try:
                rec, table, _cols, raw, reasoning = seed_one(t, fam, capability=a.capability, temperature=a.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"  [{t.template_id}] engine error: {e}", file=sys.stderr)
                continue
            n_called += 1
            cache[k] = rec
            tfh.write(json.dumps({"template_id": t.template_id, "rec": rec, "raw": raw,
                                  "reasoning_content": reasoning}) + "\n")
            if not a.no_cache:
                _CACHE.write_text(json.dumps(cache, indent=1))
        if rec:
            n_cols += len(rec["cols"])
            if i < 6 or a.limit:
                print(f"[{t.template_id}] {table} → {rec['table']}")
                for s, nat in list(rec["cols"].items())[:6]:
                    print(f"    {s} → {nat}")
    tfh.close()
    print(f"\nnatural-named: called={n_called} cached={n_cached} | columns renamed={n_cols}")

    if a.write_resource:
        resource = json.loads(_RESOURCE.read_text()) if _RESOURCE.exists() else {}
        for (t, fam) in templates:
            table, cols = name_targets(t, fam)
            cols_nopk = [c for c in cols if c != "id"]
            if not cols_nopk:
                continue
            rec = cache.get(_key(t.template_id, table, cols_nopk))
            if rec:
                resource[t.template_id] = rec
        _RESOURCE.write_text(json.dumps(resource, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(resource)} templates' natural names → {_RESOURCE.relative_to(REPO)}")
    else:
        print("(dry run — pass --write-resource to merge into natural_names.json)")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
