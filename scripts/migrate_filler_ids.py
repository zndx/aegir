#!/usr/bin/env python
"""migrate_filler_ids — namespace egalitarianism, structurally (RH 2026-07-21).

The filler_* template ids are implementation history wearing identity: they seed the SKOS
IRIs, the term-panel ids, and the DDL spine's table names (t_<template_id> in the semantic
register; engine-derived natural names echo the prefix they were prompted with) — so a
PUBLISHABLE corpus was shipping caste-marked schema. The 251 filler_* templates are logically
first-class (251/251 EquivalentTo + verbalized); their ids now say so:

    filler_<blob>  →  snake(defined head class)      e.g. filler_datacollectionprocess
                                                        →  data_collection_process
    collision (the class already has a non-filler template) → <snake>_definition

Provenance records renamed_from + the migration stamp; catalog-internal references
(parent_template) rewrite. Downstream identity (SKOS codes/IRIs, spine names) heals on
regeneration; build_skos_vocab emits DEPRECATED bridge concepts for the old IRIs
(dct:isReplacedBy → new) so previously-minted references keep resolving — references
never break, the archive/deprecation carries history.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAMP = "egalitarian-ids-2026-07-21"


def snake(x: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()


def main() -> int:
    cat_p = REPO / "src/aegir/ontology/catalog/catalog.json"
    cat = json.loads(cat_p.read_text())
    tpls = cat["templates"]
    ids = {t["template_id"] for t in tpls}
    rename: "dict[str, str]" = {}
    for t in tpls:
        tid = t["template_id"]
        if not tid.startswith("filler_"):
            continue
        m = (re.search(r"Class:\s*\{(\w+):Class\}", t.get("manchester_template") or "")
             or re.search(r"Class:\s*sdg:(\w+)", t.get("manchester_template") or ""))
        if not m:
            print(f"  ! {tid}: no head — left as-is")
            continue
        new = snake(m.group(1))
        if new in ids or new in rename.values():
            new = f"{new}_definition"
        rename[tid] = new
    for t in tpls:
        tid = t["template_id"]
        if tid in rename:
            t["template_id"] = rename[tid]
            prov = t.setdefault("provenance", {})
            prov["renamed_from"] = tid
            prov["renamed"] = STAMP
        pt = t.get("provenance", {}).get("parent_template")
        if pt in rename:
            t["provenance"]["parent_template"] = rename[pt]
    cat_p.write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    n_def = sum(1 for v in rename.values() if v.endswith("_definition"))
    print(f"renamed {len(rename)} templates ({n_def} collision-suffixed _definition) — {STAMP}")
    (REPO / "build/filler_id_migration.json").write_text(json.dumps(
        {"stamp": STAMP, "renames": rename}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
