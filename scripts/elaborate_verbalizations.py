#!/usr/bin/env python
"""LLM elaboration of template verbalizations (Semantic-Layer-Upkeep Comp 3d).

Takes each template's base verbalization frame(s) and asks the **local capability engine** (Qwen3.8-27B via
the gRPC client — strict layering, never vLLM directly) for additional *procedural*, semantically-faithful
rephrasings in distinct syntactic styles. This is the natural-language diversity layer on top of the
deterministic DeepOnto parse-tree frames (build_verbalization_frames.py): the deterministic frames raise
the per-template count off the floor; the LLM flattens the global skeleton distribution with genuinely
varied prose.

Invariants:
  * **Slot-faithful** — every elaboration must carry EXACTLY the base's ``{slot}`` set (validated; non-
    conforming lines dropped). The slots are what the generator substitutes; dropping/renaming one breaks
    the pipeline.
  * **Cached + deterministic** — results cache to ``build/verbalization_elaborations.json`` keyed by
    (template_id, base-hash); re-runs skip cached templates. The catalog write is reproducible.
  * **Thinking-trace retained** — the model's reasoning_content is captured to
    ``build/verbalization_elaboration_traces.jsonl`` (a corpus value-add, cf. Cerebras GLM), not discarded.

Writes the merged set back into the catalog's ``verbal_templates`` (dedup of base frames + accepted
elaborations). Resource note: the engine is LOCAL (not gated); paid APIs are not used here.

    just engine-serve   # (in another shell — the engine must be up)
    uv run --no-sync python scripts/elaborate_verbalizations.py --limit 5      # smoke
    uv run --no-sync python scripts/elaborate_verbalizations.py --n 2          # full catalog, 2 elabs each
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import load_catalog, save_catalog  # noqa: E402

_SLOT_RE = re.compile(r"\{[^}]*\}")
_CACHE = REPO / "build" / "verbalization_elaborations.json"
_TRACES = REPO / "build" / "verbalization_elaboration_traces.jsonl"

_SYSTEM = (
    "You rephrase formal ontology axioms as clear sentences for a technical textbook on relational data "
    "modeling and domain ontologies. Hard rules: (1) Preserve EVERY placeholder token in curly braces "
    "exactly as written (e.g. {X}, {p}, {Y}) — identical spelling, identical braces, each appearing once; "
    "never add or drop a placeholder. (2) Stay strictly faithful to the axiom's meaning — do not weaken "
    "'only' to 'some', invent relations, or change quantifiers. (3) Make each rephrasing a DISTINCT "
    "syntactic style (definitional, procedural/operational, normative 'must', relational-fronted). "
    "(4) OUTPUT CONTRACT: emit each final rephrasing on its own line beginning with the exact tag "
    "'REPHRASING: ' and nothing else on that line. Put any thinking OUTSIDE these lines; a line without "
    "the tag is ignored. Emit no other tagged lines."
)

_TAG = re.compile(r"^\s*REPHRASING:\s*(.+?)\s*$")
# meta-fragments that must never be accepted even if tagged (defensive: reasoning that mimics the tag)
_META = re.compile(r"\b(draft|let'?s|wait|check|i'?ll|i will|note:|axiom:|placeholder|constraint \d|"
                   r"verify|step \d|first,|okay|here'?s)\b", re.I)


def slots_of(text: str) -> set[str]:
    return set(_SLOT_RE.findall(text))


def _prompt(base: str, want: set[str], n: int) -> str:
    toks = ", ".join(sorted(want))
    return (f'Axiom (placeholders in braces): "{base}"\n'
            f"Placeholders that MUST each appear exactly once: {toks}\n"
            f"Write {n} distinct, procedurally-phrased rephrasings, each on its own "
            f"'REPHRASING: ' line.")


def _key(tid: str, base: str) -> str:
    import hashlib
    return f"{tid}::{hashlib.md5(base.encode()).hexdigest()[:8]}"


def elaborate_one(base: str, n: int, *, capability: str, temperature: float):
    """Returns (accepted_lines, raw_text, reasoning_content). Validates slot-faithfulness."""
    from aegir.engine.client import complete_detailed
    want = slots_of(base)
    out = complete_detailed(_prompt(base, want, n), capability=capability,
                            system_prompt=_SYSTEM, max_tokens=3072, temperature=temperature)
    raw = out["text"]
    accepted = []
    for line in raw.splitlines():
        m = _TAG.match(line)
        if not m:
            continue  # only sentinel-tagged lines are candidates — reasoning leakage can't pass
        cand = m.group(1).strip().strip('"').strip()
        if (slots_of(cand) == want and cand.lower() != base.lower()
                and 10 <= len(cand) <= 400 and not _META.search(cand)):
            accepted.append(cand)
    # dedupe, preserve order
    seen: set[str] = set()
    accepted = [a for a in accepted if not (a in seen or seen.add(a))]
    return accepted, raw, out.get("reasoning_content", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--n", type=int, default=2, help="elaborations to request per template")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N templates (0 = all)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--write-catalog", action="store_true",
                    help="merge accepted elaborations into the catalog's verbal_templates and save")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cat = load_catalog(Path(args.catalog))
    templates = [t for t in cat.templates if t.frames()]
    if args.limit:
        templates = templates[:args.limit]

    cache = {}
    if _CACHE.exists() and not args.no_cache:
        cache = json.loads(_CACHE.read_text())
    _CACHE.parent.mkdir(parents=True, exist_ok=True)

    n_called = n_accepted = n_cached = 0
    trace_fh = _TRACES.open("a")
    for i, t in enumerate(templates):
        base = t.frames()[0]
        k = _key(t.template_id, base)
        if k in cache and not args.no_cache:
            elabs = cache[k]["elaborations"]
            n_cached += 1
        else:
            try:
                elabs, raw, reasoning = elaborate_one(base, args.n, capability=args.capability,
                                                      temperature=args.temperature)
            except Exception as e:  # noqa: BLE001 — engine down / RPC error: report, keep going
                print(f"  [{t.template_id}] engine error: {e}", file=sys.stderr)
                continue
            n_called += 1
            cache[k] = {"template_id": t.template_id, "base": base, "elaborations": elabs}
            trace_fh.write(json.dumps({"template_id": t.template_id, "base": base,
                                       "elaborations": elabs, "raw_output": raw,
                                       "reasoning_content": reasoning}) + "\n")
            if not args.no_cache:
                _CACHE.write_text(json.dumps(cache, indent=1))
        n_accepted += len(elabs)
        if i < 8 or args.limit:
            print(f"[{t.template_id}] base={base!r}")
            for e in elabs:
                print(f"    + {e}")
    trace_fh.close()

    print(f"\nelaborated: called={n_called} cached={n_cached} | accepted elaborations={n_accepted} "
          f"(avg {n_accepted/max(1,len(templates)):.1f}/template)")

    if args.write_catalog:
        by_key = {_key(t.template_id, t.frames()[0]): t for t in templates}
        for k, rec in cache.items():
            t = by_key.get(k)
            if not t:
                continue
            merged = list(dict.fromkeys(t.frames() + rec["elaborations"]))
            t.verbal_templates = merged
        save_catalog(cat, Path(args.catalog))
        print(f"wrote merged verbal_templates → {args.catalog}")
    else:
        print("(dry run — pass --write-catalog to merge into the catalog)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
