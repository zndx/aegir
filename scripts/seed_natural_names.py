#!/usr/bin/env python
"""Convert 1c — natural physical naming IN-LOOP (membrane-gated, provenance-stamped).

The ontology-derived column/table names are semantic (`attestation_process`, `specimen`) — great for
Atlas/lineage *reference*, but (a) they echo the concept, so a model training on headered tables can learn
concept-from-header (a shortcut that doesn't transfer), and (b) downstream/Atelier consumers want realistic
DB schemas, not ontology jargon; Atelier's ``name_match`` evidence channel matches column names against the
very vocabulary we emit, so a semantic name IS the answer key — the natural register is what makes their
benchmark measure comprehension. This script asks the local engine (Qwen3.8-27B) for **natural physical names** —
the way a working DBA would name the table + its columns — decoupled from the ontology label but plausible
for the column's meaning, and DISPOSES each proposal through ``natural_naming.check_names`` (identifier
validity, FULL coverage, distinctness, de-echo, aggregate echo rate), re-prompting with the returned reason
up to ``--rounds`` times ([[agent_mediated_feedback_loop]]). Admissions merge into ``natural_names.json``
with per-template provenance {run, model, membrane, rounds, source} — the successor of the pre-membrane
static map, exactly as the individual registry succeeded the frozen value pools (Convert 1b).

``build_ddl_spine --naming natural`` (default) threads the map into schema realization so the whole
subgraph — base tables, junction/EAV/star sub-tables, FK columns, views, SQL — is constructed in the
natural register from birth; the natural↔semantic map is the lineage edge (Atlas ``naming``).

Sentinel-rigorous (Qwen3.8-27B leaks reasoning; only NAT_-tagged lines are read); cache is versioned so
pre-membrane (static-era) entries never satisfy a membrane run.

    just engine-serve
    uv run --no-sync python scripts/seed_natural_names.py --limit 8        # smoke (dry)
    uv run --no-sync python scripts/seed_natural_names.py --write-resource  # full (cached, resumable)
"""
from __future__ import annotations

import argparse
import datetime as _dt
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

_TABLE = re.compile(r"^\s*NAT_TABLE\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*$")
_COL = re.compile(r"^\s*NAT_COL\s+([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z][A-Za-z0-9_]*)\s*$")

_SYSTEM = (
    "You are a senior data engineer naming a relational table the way it would appear in a real production "
    "database. Given the table's concept and its current (ontology-derived) column names, produce NATURAL "
    "physical names: a table name and one name per column. Hard rules: (1) snake_case, valid SQL identifiers, "
    "≤ ~32 chars; (2) realistic DB style a DBA would actually type (e.g. sample_id, material_type, "
    "collected_at, status_code, owner_ref) — NOT ontology jargon; (3) the natural name must be PLAUSIBLE for "
    "the column's meaning but NOT a copy of the ontology concept — decouple the surface from the label "
    "(e.g. concept 'specimen' → 'sample_id' or 'material_ref', not 'specimen'); dropping a 't_' prefix, "
    "pluralizing, or re-inflecting the concept does NOT count as decoupling — abbreviate or rephrase; "
    "(4) names distinct within the table; NEVER propose bare 'id' (it is the reserved surrogate key) — a "
    "subject/identity column becomes a '<stem>_id' business key; (5) OUTPUT CONTRACT: first a line "
    "'NAT_TABLE: <name>', then one line per column 'NAT_COL <current_column> -> <natural_name>'. Put ALL "
    "reasoning on OTHER lines; a line without a NAT_ tag is ignored."
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
    """Permissive sentinel parse — the MEMBRANE (natural_naming.check_names) decides admissibility and
    returns the precise reason (missing cols, echo, dupes …); parsing only extracts what was proposed."""
    table = None
    cols: dict[str, str] = {}
    for line in raw.splitlines():
        mt = _TABLE.match(line)
        if mt and table is None:
            table = mt.group(1).lower()
            continue
        mc = _COL.match(line)
        if mc and mc.group(1) in want_cols:
            cols[mc.group(1)] = mc.group(2).lower()
    if not table and not cols:
        return None
    return {"table": table or "", "cols": cols}


def _key(tid: str, table: str, cols: "list[str]") -> str:
    # v2: membrane-era cache namespace — pre-membrane (static-era) entries must never satisfy this run
    return f"v2:{tid}::{hashlib.md5((table + '|' + ','.join(sorted(cols))).encode()).hexdigest()[:8]}"


def seed_one(template, family: str, *, capability: str, temperature: float, rounds: int = 3):
    """Propose → membrane → re-prompt-with-reason, up to ``rounds`` ([[agent_mediated_feedback_loop]]).
    Returns (admitted_rec | None, semantic_table, semantic_cols, trace_rows)."""
    from aegir.engine.client import complete_detailed
    from aegir.ontology.natural_naming import check_names
    table, cols = name_targets(template, family)
    cols = [c for c in cols if c != "id"]  # surrogate key is always 'id'
    if not cols:
        return None, table, cols, []
    verbalization = (template.frames()[0] if template.frames() else template.verbal_template) or ""
    prompt = _prompt(_concept(template), verbalization, table, cols)
    traces: list[dict] = []
    for rnd in range(1, rounds + 1):
        out = complete_detailed(prompt, capability=capability, system_prompt=_SYSTEM,
                                max_tokens=4096, temperature=temperature)
        parsed = parse_response(out["text"], set(cols))
        ok, reason, cleaned = check_names(table, cols, parsed)
        traces.append({"round": rnd, "ok": ok, "reason": reason, "rec": cleaned or parsed,
                       "raw": out["text"], "reasoning_content": out.get("reasoning_content", "")})
        if ok and cleaned:
            cleaned["provenance"] = {"run": _dt.date.today().isoformat(), "model": capability,
                                     "membrane": "natural_naming.check_names", "rounds": rnd,
                                     "source": "engine-derived"}
            return cleaned, table, cols, traces
        prompt = (_prompt(_concept(template), verbalization, table, cols) +
                  f"\n\n[PRIOR ATTEMPT REJECTED — {reason} — fix and re-emit ALL NAT_ lines]")
    return None, table, cols, traces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=3, help="membrane re-prompt rounds per template")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-resource", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    cat_dir = REPO / a.catalog_dir if not Path(a.catalog_dir).is_absolute() else Path(a.catalog_dir)
    from aegir.ontology.schema import catalog_files
    files = catalog_files(cat_dir)
    templates: list[tuple] = []
    for path in files:
        cat = load_catalog(path)
        ts = cat.templates[:a.per_family] if a.per_family else cat.templates
        templates += [(t, path.stem) for t in ts]
    if a.limit:
        templates = templates[:a.limit]

    cache = json.loads(_CACHE.read_text()) if (_CACHE.exists() and not a.no_cache) else {}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    resource = json.loads(_RESOURCE.read_text()) if (_RESOURCE.exists() and a.write_resource) else {}
    tfh = _TRACES.open("a")
    n_called = n_cached = n_cols = 0
    unresolved: list[str] = []
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
                rec, table, _cols, traces = seed_one(t, fam, capability=a.capability,
                                                     temperature=a.temperature, rounds=a.rounds)
            except Exception as e:  # noqa: BLE001
                print(f"  [{t.template_id}] engine error: {e}", file=sys.stderr, flush=True)
                continue
            n_called += 1
            cache[k] = rec
            for tr in traces:
                tfh.write(json.dumps({"template_id": t.template_id, **tr}) + "\n")
            tfh.flush()
            if not a.no_cache:
                _CACHE.write_text(json.dumps(cache, indent=1))
        if rec:
            n_cols += len(rec["cols"])
            if a.write_resource:  # incremental accretion — a killed run keeps its admissions
                resource[t.template_id] = rec
                _RESOURCE.write_text(json.dumps(resource, indent=1, sort_keys=True) + "\n")
            if i < 6 or a.limit:
                print(f"[{t.template_id}] {table} → {rec['table']}", flush=True)
                for s, nat in list(rec["cols"].items())[:6]:
                    print(f"    {s} → {nat}")
        else:
            unresolved.append(t.template_id)
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{len(templates)} (called={n_called} cached={n_cached} "
                  f"unresolved={len(unresolved)})", flush=True)
    tfh.close()
    print(f"\nnatural-named: called={n_called} cached={n_cached} | columns renamed={n_cols} | "
          f"unresolved after {a.rounds} rounds: {len(unresolved)}")
    if unresolved:
        print("  unresolved: " + ", ".join(unresolved[:12]) + (" …" if len(unresolved) > 12 else ""))

    if a.write_resource:
        print(f"wrote {len(resource)} templates' natural names → {_RESOURCE.relative_to(REPO)}")
    else:
        print("(dry run — pass --write-resource to merge into natural_names.json)")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
