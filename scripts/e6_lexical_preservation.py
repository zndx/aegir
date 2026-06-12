#!/usr/bin/env python
"""E6 Channel B (EVIDENCE.md): lexical preservation of DeepOnto verbalizations in entity NAMES.

For each template: V = content terms of its (placeholder-stripped) verbal_template; N = tokens of
its table name + column names (snake-split; surrogate id excluded; anchor-attribute columns scored
against their DataProperty labels separately — they are vocab-derived, not template-derived).

  recall    = |N ∩ V| / |V|   (expressiveness: verbalization terms reflected in names)
  precision = |N ∩ V| / |N|   (integrity: name tokens grounded in the verbalization)

Cross-pairing null: names scored against a rotated template's V. Table-columns and VIEW-columns
reported separately — views are currently ABSENT; their 0 is the baseline that drives Phase-2.
Deterministic; no GPU/LLM.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
from aegir.ontology.ddl import template_to_table, _data_properties  # noqa: E402

STOP = set("a an the of in on to for with by is are was be been has have had that this it its "
           "some only min max exactly and or not at as from which when where each every".split())
SLOT_RE = re.compile(r"\{[^}]+\}")
WORD_RE = re.compile(r"[a-z]{2,}")


def terms(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(SLOT_RE.sub(" ", text or "").lower()) if w not in STOP}


def name_tokens(name: str) -> set[str]:
    return {w for w in re.split(r"[_\W]+", name.lower()) if len(w) >= 2 and w not in STOP}


def score(names: set[str], verbal: set[str]) -> tuple[float, float]:
    if not verbal or not names:
        return 0.0, 0.0
    hit = names & verbal
    return len(hit) / len(verbal), len(hit) / len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", default=None, help="optional generated .candidate.json to score")
    args = ap.parse_args()

    # DataProperty labels (for anchor-attribute columns: vocab-derived, scored apart)
    dp_label_tokens: dict[str, set[str]] = {}
    for dom, props in _data_properties().items():
        for col, _rng, iri in props:
            dp_label_tokens[col] = name_tokens(iri.split(":")[-1])

    def catalog_rows(tmpls: list[CatalogTemplate], fam: str):
        rows = []
        for t in tmpls:
            v = terms(t.verbal_template)
            stab = template_to_table(t, fam)
            tname = name_tokens(stab.table.name.removeprefix("t_"))
            slot_cols, attr_cols = set(), set()
            for c in stab.table.columns:
                if c.slot_ref == "__pk__":
                    continue
                (attr_cols if c.slot_ref.startswith("data:") else slot_cols).update(name_tokens(c.name))
            rows.append({"v": v, "tname": tname, "slot": slot_cols, "attr": attr_cols})
        return rows

    seed_rows, gen_rows = [], []
    for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))):
        if "candidate" in f or "combined" in f:
            continue
        seed_rows.extend(catalog_rows(load_catalog(f).templates, Path(f).stem))
    if args.generated and Path(args.generated).exists():
        gen = json.load(open(args.generated))
        gen_rows = catalog_rows([CatalogTemplate(**d) for d in gen["templates"]], "generated")

    def report(label: str, rows: list[dict]):
        if not rows:
            print(f"{label}: (none)")
            return
        for kind, key in [("table name", "tname"), ("slot/table-columns", "slot")]:
            rs = [score(r[key], r["v"]) for r in rows]
            nulls = [score(rows[i][key], rows[(i + 7) % len(rows)]["v"]) for i in range(len(rows))]
            print(f"  {label:>9} {kind:<19} recall {st.mean(x[0] for x in rs):.3f} "
                  f"prec {st.mean(x[1] for x in rs):.3f}   (null recall {st.mean(x[0] for x in nulls):.3f})")
        # anchor-attribute columns vs their own DataProperty labels (integrity by construction)
        rs = [score(r["attr"], set().union(*(dp_label_tokens.get(w, {w}) for w in r["attr"])) or set())
              for r in rows if r["attr"]]
        if rs:
            print(f"  {label:>9} {'attr-cols vs dp-labels':<19} recall {st.mean(x[0] for x in rs):.3f}")

    print("E6-B lexical preservation (verbalization terms → entity names)\n")
    report("SEED", seed_rows)
    report("GENERATED", gen_rows)
    print(f"\n  VIEW-columns: ABSENT (0 view artifacts exist) — recall/precision = 0.000 by absence.")
    print("  This zero is the pre-registered baseline; Phase-2 views must move it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
