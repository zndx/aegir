#!/usr/bin/env python
"""Derive the SKOS column-type vocabulary from the ontology (the shared type-system key).

One controlled vocabulary, four consumers: the ontology slot-types, the sdg-corpora column
labels, Atelier's classification category_set (ReferenceCategory), the Aegir ontology-CPA /
edge-probe label space, and the Atlas governance taxonomy. It is *grounded* — the codes are
the ontology, not arbitrary: the 7 BFO/CCO ``bfo_anchor`` classes are the upper concepts and
each of the 540 templates is a leaf concept under its anchor.

Emits (default → ``corpora/vocabulary/``):
  - annotations.csv / annotations.parquet  — Atelier "universal" ReferenceCategory records
    (code, label, abbrev, notation, parent_code, taxonomy, description, common_names,
    example_values); parquet is the release-asset form Atelier consumes.
  - vocabulary.ttl                         — SKOS ConceptScheme (skos:Concept / prefLabel /
    notation / definition / broader / inScheme) for interop + the Atlas glossary projection.

Deterministic: no LLM, no GPU, no network.
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402

# The 7 ``bfo_anchor`` classes as upper SKOS concepts + their CCO/BFO subsumption.
# (code, notation, prefLabel, abbrev, parent_code, definition)
ANCHORS: dict[str, tuple[str, str, str, str, str | None, str]] = {
    "bfo:Process": ("SDG.PROCESS", "1", "Process", "PROC", None,
                    "A BFO process (occurrent): an entity that unfolds in time, e.g. an "
                    "observation, a program execution, a governance activity."),
    "bfo:IndependentContinuant": ("SDG.INDEPENDENT_CONTINUANT", "2", "Independent Continuant",
                    "INDCONT", None,
                    "A BFO independent continuant: a bearer of qualities that persists through time."),
    "cco:Artifact": ("SDG.ARTIFACT", "2.1", "Artifact", "ARTIFACT", "SDG.INDEPENDENT_CONTINUANT",
                    "A CCO artifact: an object intentionally made to realize a function "
                    "(instruments, systems, datasets-as-objects)."),
    "cco:InformationContentEntity": ("SDG.ICE", "3", "Information Content Entity", "ICE", None,
                    "A CCO information content entity: a generically dependent continuant that "
                    "is about, or carries information concerning, some entity."),
    "cco:DescriptiveICE": ("SDG.ICE.DESCRIPTIVE", "3.1", "Descriptive Information Content Entity",
                    "DESCICE", "SDG.ICE",
                    "An ICE that describes an entity or state of affairs (measurements, records, "
                    "observations)."),
    "cco:DirectiveICE": ("SDG.ICE.DIRECTIVE", "3.2", "Directive Information Content Entity",
                    "DIRICE", "SDG.ICE",
                    "An ICE that prescribes or directs (plans, rules, requirements, policies)."),
    "cco:DesignativeICE": ("SDG.ICE.DESIGNATIVE", "3.3", "Designative Information Content Entity",
                    "DESIGICE", "SDG.ICE",
                    "An ICE that designates or names an entity (identifiers, codes, keys)."),
}
# Anchorless templates (empty bfo_anchor_path) attach here.
GENERIC = ("SDG.GENERIC", "0", "Generic Entity", "GENERIC", None,
           "Foundational / domain-agnostic axiom patterns without a specific BFO/CCO anchor.")
SCHEME_IRI = "https://signals360.example.org/sdg/scheme"


def _humanize(tid: str) -> str:
    return re.sub(r"[_\W]+", " ", tid).strip().title()


def _iri(code: str) -> str:
    return f"https://signals360.example.org/sdg#{code.replace('.', '_')}"


def build_records() -> list[dict]:
    files = [f for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0*.json")))
             if "candidate" not in f and "combined" not in f]
    records: list[dict] = []

    # 1) upper concepts (anchors + generic)
    for code, notation, label, abbrev, parent, definition in [*ANCHORS.values(), GENERIC]:
        records.append({"code": code, "label": label, "abbrev": abbrev, "notation": notation,
                        "parent_code": parent or "", "taxonomy": "sdg", "description": definition,
                        "common_names": "", "example_values": ""})

    # 2) one leaf concept per template, under its anchor
    counters: dict[str, int] = {}
    for path in files:
        family = Path(path).stem
        for t in load_catalog(path).templates:
            anchor = t.bfo_anchor_path[-1] if t.bfo_anchor_path else None
            a = ANCHORS.get(anchor, GENERIC) if anchor else GENERIC
            parent_code, parent_notation = a[0], a[1]
            counters[parent_code] = counters.get(parent_code, 0) + 1
            class_slots = [s for s, ty in t.slot_types.items() if ty != "ObjectProperty"]
            desc = (t.verbal_template or "").strip()
            desc = (desc + " — " if desc else "") + t.manchester_template.strip()
            records.append({
                "code": f"{parent_code}.{t.template_id.upper()}",
                "label": _humanize(t.template_id),
                "abbrev": t.template_id.upper(),
                "notation": f"{parent_notation}.{counters[parent_code]}",
                "parent_code": parent_code,
                "taxonomy": "sdg",
                "description": desc[:1000],
                "common_names": f"family={family}; slots={','.join(class_slots)}",
                "example_values": "",  # harvested from the corpus later
            })
    return records


def write_csv(records: list[dict], path: Path) -> None:
    cols = ["code", "label", "abbrev", "notation", "parent_code", "taxonomy",
            "description", "common_names", "example_values"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)


def write_parquet(records: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["code", "label", "abbrev", "notation", "parent_code", "taxonomy",
            "description", "common_names", "example_values"]
    pq.write_table(pa.table({c: [r[c] for r in records] for c in cols}), path)


def write_ttl(records: list[dict], path: Path) -> None:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix sdg:  <https://signals360.example.org/sdg#> .", "",
        f"<{SCHEME_IRI}> a skos:ConceptScheme ;",
        '    skos:prefLabel "SDG column-type vocabulary (ontology-derived)" .', "",
    ]
    for r in records:
        lines.append(f"<{_iri(r['code'])}> a skos:Concept ;")
        lines.append(f'    skos:prefLabel "{esc(r["label"])}" ;')
        lines.append(f'    skos:notation "{r["notation"]}" ;')
        lines.append(f'    skos:altLabel "{esc(r["abbrev"])}" ;')
        if r["description"]:
            lines.append(f'    skos:definition "{esc(r["description"])}" ;')
        if r["parent_code"]:
            lines.append(f"    skos:broader <{_iri(r['parent_code'])}> ;")
        lines.append(f"    skos:inScheme <{SCHEME_IRI}> .")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "corpora" / "vocabulary"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = build_records()
    write_csv(records, out / "annotations.csv")
    write_parquet(records, out / "annotations.parquet")
    write_ttl(records, out / "vocabulary.ttl")

    n_upper = len(ANCHORS) + 1
    print(f"SKOS vocabulary: {len(records)} concepts ({n_upper} upper + {len(records) - n_upper} "
          f"template leaves) → {out}")
    print("  annotations.csv / annotations.parquet (Atelier ReferenceCategory) + vocabulary.ttl (SKOS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
