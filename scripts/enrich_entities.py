#!/usr/bin/env python
"""enrich_entities — merge the authored genus+differentia taxonomy INTO the run entities, so the flow
RE-REALIZES the taxonomy-enriched ontology and RE-PROJECTS the corpus with the differentia columns.

For each species that got an accepted differentia: set its genus to the sdg: mid-tier genus and add the
differentia as a real entity feature — a value/typed ATTRIBUTE (→ a discriminating column) or an object
RELATION (→ a discriminating FK join, the multi-table Data Elements). The 32 genera and any object-filler
classes that aren't already entities are added as minimal entities (so the FKs resolve). Writes an enriched
entities/ tree the SdgCorporaFlow consumes via --entities-from.

    uv run --no-sync python scripts/enrich_entities.py --taxonomy build/taxonomy/v05 --entities <run>/entities \
        --out /raid/.../entities-taxonomy
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def _diff_feature(prop: str, kind: str, restriction: str):
    """→ ('attr', {name,xsd,enum}) | ('rel', {prop,target,card}) — the differentia as a real entity feature."""
    vals = re.findall(r'value\s+"([^"]+)"', restriction)
    if kind == "data":
        if vals:
            return "attr", {"name": prop, "xsd": "string", "enum": sorted(set(vals))}
        xm = re.search(r"xsd:(\w+)", restriction)
        return "attr", {"name": prop, "xsd": xm.group(1) if xm else "string", "enum": []}
    fm = re.search(r"(?:some|only|min \d+|max \d+|exactly \d+)\s+(?:sdg:)?([A-Z][A-Za-z0-9]*)", restriction)
    return "rel", {"prop": prop, "target": fm.group(1) if fm else "Thing", "card": "some"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default="build/taxonomy/v05")
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # authored taxonomy → per-species (genus, differentia); per-genus mode BFO/CCO anchor from its members
    sp_genus_anchor: "dict[str, str]" = {}
    for f in glob.glob(str(Path(a.entities) / "*.json")):
        for e in json.loads(Path(f).read_text()).get("entities", []):
            sp_genus_anchor.setdefault(e["name"], e.get("genus", "cco:ont00000995"))
    sp_diff: "dict[str, tuple[str, dict]]" = {}
    genus_members: "dict[str, list[str]]" = {}
    for gf in sorted(glob.glob(str(Path(a.taxonomy) / "genera" / "*.json"))):
        r = json.loads(Path(gf).read_text())
        g = r["genus"]
        for s, d in r["differentiae"].items():
            if d["accepted"] and s != g:
                sp_diff[s] = (g, d)
                genus_members.setdefault(g, []).append(s)
    genus_anchor = {g: Counter(sp_genus_anchor.get(s, "cco:ont00000995") for s in mem).most_common(1)[0][0]
                    for g, mem in genus_members.items()}

    existing = set(sp_genus_anchor)
    fillers_needed: "set[str]" = set()
    n_attr = n_rel = n_species = 0
    for f in sorted(glob.glob(str(Path(a.entities) / "*.json"))):
        doc = json.loads(Path(f).read_text())
        for e in doc.get("entities", []):
            if e["name"] not in sp_diff:
                continue
            g, d = sp_diff[e["name"]]
            e["genus"] = f"sdg:{g}"                                   # set the mid-tier genus
            kind, feat = _diff_feature(d["property"], d["kind"], d["restriction"])
            if kind == "attr":
                if not any(x["name"] == feat["name"] for x in e.setdefault("attributes", [])):
                    e["attributes"].append(feat); n_attr += 1
            else:
                if not any(x["prop"] == feat["prop"] for x in e.setdefault("relations", [])):
                    e["relations"].append(feat); n_rel += 1
                    if feat["target"] not in existing:
                        fillers_needed.add(feat["target"])
            n_species += 1
        (out / Path(f).name).write_text(json.dumps(doc, indent=1))

    # the 32 genus classes + any object-filler classes not already entities → a taxonomy entities file
    tax_entities = []
    for g, anc in genus_anchor.items():
        tax_entities.append({"name": g, "label": g, "genus": anc,
                             "definition": f"The genus of {g}-kind Data Elements.", "attributes": [], "relations": []})
    for fc in sorted(fillers_needed):
        tax_entities.append({"name": fc, "label": fc, "genus": "bfo:0000001",
                             "definition": f"A {fc} referenced as a differentiating relation target.",
                             "attributes": [], "relations": []})
    (out / "taxonomy_overlay.json").write_text(json.dumps({"entities": tax_entities}, indent=1))

    print(f"enriched {n_species} species ({n_attr} value/typed attrs + {n_rel} object relations) · "
          f"+{len(genus_anchor)} genus classes + {len(fillers_needed)} filler classes → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
