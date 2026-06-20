#!/usr/bin/env python
"""LLM-seeded, RI-safe domain values for entity/name columns (Semantic-Layer-Upkeep Comp 4).

The deterministic row generator (:mod:`aegir.ontology.rows`) grounds *typed* columns (xsd ranges),
*enumerated* columns (skos:definition value sets) and *generic-string* columns (curated pools). What
it cannot ground are the **concept-specific entity columns** — a table's subject head (``labrun``,
``ebpfprogram``, ``mass``), its named relation targets, and its ``name`` attribute — which fall back to
``"<Concept> NN"`` placeholders. This script asks the **local capability engine** (Qwen3.6 via the gRPC
client — strict layering, never vLLM directly) for a pool of realistic *domain* instance values per such
column, seeded by the template's concept + verbalization, and caches them to a committed resource the
generator reads.

RI-safety (load-bearing): the seeded pools feed **non-FK** columns only. FK cells are always overwritten
from the referenced table's PK pool in :func:`rows.materialize_rows` (RI = 1.0 by construction), so a
seeded value can never break referential integrity even if a relation column later carries an FK.

Invariants (mirror scripts/elaborate_verbalizations.py):
  * **Sentinel contract** — the model emits ``VALUES <col>: a | b | c`` lines; untagged lines (reasoning
    leakage) are ignored. Values must be non-empty, distinct, not themselves ``"<Word> NN"`` placeholders.
  * **Cached + deterministic** — results cache to ``build/entity_value_seed_cache.json`` keyed by
    (template_id, col, concept-hash); ``--write-resource`` merges the cache into the committed
    ``src/aegir/ontology/entity_value_pools.json`` (template_id → {col → [values]}). Re-runs skip cached.
  * **Thinking-trace retained** — reasoning_content → ``build/entity_value_seed_traces.jsonl`` (corpus
    value-add, cf. Cerebras GLM), not discarded. Resource note: the engine is LOCAL (not gated).

    just engine-serve   # (in another shell — the engine must be up)
    uv run --no-sync python scripts/seed_entity_values.py --per-family 8 --write-resource   # the gate cohort
    uv run --no-sync python scripts/seed_entity_values.py --write-resource                  # full 540 (multi-hour; resumable)
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
from aegir.ontology.rows import SEMANTIC_VALUE_POOLS, parse_enum_from_definition  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

_CACHE = REPO / "build" / "entity_value_seed_cache.json"
_TRACES = REPO / "build" / "entity_value_seed_traces.jsonl"
_RESOURCE = REPO / "src" / "aegir" / "ontology" / "entity_value_pools.json"

_ENTITY_OWL = {"Class", "Individual", "NamedIndividual"}
_STRING_OWL = {"xsd:string", "DataProperty", "Literal"}
_GENERIC = {"class", "subclass", "basic", "template", "generic", "foundation", "long", "tail",
            "complex", "axiom", "the", "and", "for", "with", "via", "to", "of", "an", "from"}
# "<TitleCase words> NN" — the very placeholder shape we are replacing; reject if the model echoes it.
_PLACEHOLDER = re.compile(r"^[A-Za-z][A-Za-z]*( [A-Za-z]+)* +\d{2,}$")
_VALUES = re.compile(r"^\s*VALUES\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")

_SYSTEM = (
    "You invent realistic example DATA VALUES for columns of a relational table in a technical domain "
    "(data engineering, telemetry, lab/observation, governance, provenance). For each requested column "
    "you produce a pipe-separated list of distinct, concrete, domain-plausible cell values — the kind a "
    "real database would hold, NOT placeholders like 'Item 01'. Hard rules: (1) values are short (1-4 "
    "words / a realistic identifier), concrete, and specific to the column's meaning and the table's "
    "concept; (2) no numbering like 'X 01', no 'example', no 'value', no ontology jargon; (3) OUTPUT "
    "CONTRACT: emit exactly one line per requested column, beginning with the literal tag 'VALUES "
    "<column>: ' followed by 8-12 values separated by ' | '. Put any thinking on OTHER lines; a line "
    "without the tag is ignored."
)


def _concept(template) -> str:
    toks = [t for t in re.split(r"[^A-Za-z]+", template.template_id)
            if len(t) >= 3 and t.lower() not in _GENERIC]
    return " ".join(toks[:3]) if toks else (template.template_id or "entity")


def _has_enum(col, dp_meta: dict) -> bool:
    iri = col.slot_ref[len("data:"):] if col.slot_ref.startswith("data:") else None
    defn = dp_meta.get(iri, ("", "", ""))[2] if iri else None
    return bool(parse_enum_from_definition(defn))


def seed_columns(template, family: str, dp_meta: dict) -> "list[tuple[str, str]]":
    """The (column, semantic-hint) pairs worth LLM-seeding: concept-specific entity heads, named
    relation targets, and un-pooled string attributes (whatever would otherwise be a placeholder)."""
    table = D.template_to_table(template, family)
    concept = _concept(template)
    head = concept.split()[0] if concept else concept          # the subject entity noun (e.g. "labrun")
    restr_targets = {r.target_slot for r in D.parse_restrictions(template)}
    by_slot_name = {}  # slot → col for restriction-target detection
    for c in table.table.columns:
        by_slot_name[c.slot_ref] = c
    out: list[tuple[str, str]] = []
    seen_subject = False
    for c in table.table.columns:
        if c.slot_ref == "__pk__":
            continue
        nm = c.name
        if c.slot_type in _ENTITY_OWL:
            if c.slot_ref in restr_targets:
                hint = f"entities that serve as the '{nm.replace('_', ' ')}' of a {concept}"
            elif not seen_subject:
                seen_subject = True
                hint = f"distinct real-world instances of a {head} (this column is the table's subject)"
            else:
                hint = f"real-world '{nm.replace('_', ' ')}' entities for a {concept}"
            out.append((nm, hint))
        elif c.slot_type in _STRING_OWL and nm not in SEMANTIC_VALUE_POOLS and not _has_enum(c, dp_meta):
            out.append((nm, f"realistic '{nm.replace('_', ' ')}' values for a {concept}"))
    return out


def _prompt(concept: str, verbalization: str, cols: "list[tuple[str, str]]") -> str:
    lines = "\n".join(f"  - {nm}: {hint}" for nm, hint in cols)
    vb = f'\nThe table records: "{verbalization}".' if verbalization else ""
    return (f"Table concept: {concept}.{vb}\n"
            f"Produce a 'VALUES <column>: ...' line for EACH of these columns:\n{lines}")


def parse_response(raw: str, want: "set[str]") -> "dict[str, list[str]]":
    out: dict[str, list[str]] = {}
    for line in raw.splitlines():
        m = _VALUES.match(line)
        if not m or m.group(1) not in want:
            continue
        col = m.group(1)
        vals: list[str] = []
        for v in m.group(2).split("|"):
            v = v.strip().strip('"').strip("'").strip()
            if v and not _PLACEHOLDER.match(v) and 1 <= len(v) <= 48 and v.lower() != col.lower():
                vals.append(v)
        vals = list(dict.fromkeys(vals))            # dedupe, keep order
        if len(vals) >= 3:
            out[col] = vals
    return out


def _key(tid: str, concept: str, cols: "list[tuple[str, str]]") -> str:
    sig = concept + "|" + ",".join(sorted(nm for nm, _ in cols))
    return f"{tid}::{hashlib.md5(sig.encode()).hexdigest()[:8]}"


def seed_one(template, family: str, dp_meta: dict, *, capability: str, temperature: float):
    """Returns (col→values dict, columns, raw_text, reasoning). Engine call; validates the response."""
    from aegir.engine.client import complete_detailed
    cols = seed_columns(template, family, dp_meta)
    if not cols:
        return {}, cols, "", ""
    verbalization = (template.frames()[0] if template.frames() else template.verbal_template) or ""
    # Generous budget: Qwen3.6 thinking is RETAINED (engine policy) and leaks into content for this
    # prompt shape, so the 'VALUES …' lines land AFTER a long reasoning preamble — the sentinel parser
    # skips the preamble but the budget must be large enough to reach the tagged lines (we wait for it).
    out = complete_detailed(_prompt(_concept(template), verbalization, cols), capability=capability,
                            system_prompt=_SYSTEM, max_tokens=8192, temperature=temperature)
    pools = parse_response(out["text"], {nm for nm, _ in cols})
    return pools, cols, out["text"], out.get("reasoning_content", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    ap.add_argument("--per-family", type=int, default=None, help="cap templates per family (gate cohort = 8)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N templates (smoke)")
    ap.add_argument("--write-resource", action="store_true",
                    help="merge the cache into src/aegir/ontology/entity_value_pools.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cat_dir = REPO / args.catalog_dir if not Path(args.catalog_dir).is_absolute() else Path(args.catalog_dir)
    files = sorted(p for p in cat_dir.glob("0*.json") if ".candidate" not in p.name and "combined" not in p.name)
    dp_meta = D.dataprop_meta()

    templates: list[tuple] = []
    for path in files:
        cat = load_catalog(path)
        ts = cat.templates[:args.per_family] if args.per_family else cat.templates
        for t in ts:
            templates.append((t, path.stem))
    if args.limit:
        templates = templates[:args.limit]

    cache = {}
    if _CACHE.exists() and not args.no_cache:
        cache = json.loads(_CACHE.read_text())
    _CACHE.parent.mkdir(parents=True, exist_ok=True)

    n_called = n_cached = n_cols = n_vals = 0
    trace_fh = _TRACES.open("a")
    for i, (t, fam) in enumerate(templates):
        cols = seed_columns(t, fam, dp_meta)
        if not cols:
            continue
        k = _key(t.template_id, _concept(t), cols)
        if k in cache and not args.no_cache:
            pools = cache[k]["pools"]
            n_cached += 1
        else:
            try:
                pools, cols, raw, reasoning = seed_one(t, fam, dp_meta,
                                                       capability=args.capability, temperature=args.temperature)
            except Exception as e:  # noqa: BLE001 — engine down / RPC error: report, keep going
                print(f"  [{t.template_id}] engine error: {e}", file=sys.stderr)
                continue
            n_called += 1
            cache[k] = {"template_id": t.template_id, "pools": pools}
            trace_fh.write(json.dumps({"template_id": t.template_id, "concept": _concept(t),
                                       "pools": pools, "raw_output": raw,
                                       "reasoning_content": reasoning}) + "\n")
            if not args.no_cache:
                _CACHE.write_text(json.dumps(cache, indent=1))
        n_cols += len(pools)
        n_vals += sum(len(v) for v in pools.values())
        if i < 6 or args.limit:
            print(f"[{t.template_id}]")
            for col, vals in pools.items():
                print(f"    {col}: {' | '.join(vals[:6])}{' …' if len(vals) > 6 else ''}")

    trace_fh.close()
    print(f"\nseeded: called={n_called} cached={n_cached} | columns={n_cols} values={n_vals} "
          f"(avg {n_vals/max(1,n_cols):.1f} values/column)")

    if args.write_resource:
        resource = {}
        if _RESOURCE.exists():
            resource = json.loads(_RESOURCE.read_text())
        for rec in cache.values():
            if rec["pools"]:
                resource.setdefault(rec["template_id"], {}).update(rec["pools"])
        _RESOURCE.write_text(json.dumps(resource, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(resource)} templates' pools → {_RESOURCE.relative_to(REPO)}")
    else:
        print("(dry run — pass --write-resource to merge into entity_value_pools.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
