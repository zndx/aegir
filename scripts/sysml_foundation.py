"""sysml_foundation.py — extend the BFO/CCO/FHIR foundation with SysML v2 domain vocabulary.

The SysML v2 standard library (OMG / Systems-Modeling, **EPL-2.0**) is a systems-engineering
MODELING LANGUAGE, not an ontology. We **draw** its conceptual vocabulary (Parts, Items, Ports,
Connections, Actions, Requirements, Constraints; the Geometry/CSG library; the ISQ/SI quantity
library) and **cite** it as provenance — we do NOT import or redistribute the EPL files into the
clean (CC0-track) artifact.

For each construct we ontologize its **ontological referent**, not the metamodel element: a
`part def` denotes a physical part → cco:ont00000995 (not "the SysML PartDefinition model element").
SysML supplies the *domain vocabulary*; BFO/CCO supplies the *being*; the propose/dispose membrane
(Stage 2) grounds one in the other and HermiT validates it against CCO's disjointness axioms.

EPL-cleanliness (enforced here): only concept **names** + **taxonomy** (the `:>` supertype graph,
uncopyrightable facts) are read from the library and the doc comments are used for *understanding
only* — they are NEVER persisted. Every gloss emitted to `sysml_foundation.json` is authored by us.
The raw `.sysml` files are cached under gitignored `build/sysml/` (a local build cache, not shipped).

Stage 1 (this module): fetch → parse → assign BFO/CCO genus (curated top-level bridge + per-file
inheritance) → emit `build/sysml/sysml_foundation.json`. Stage 2 (follow-up): feed these into the
grounding-anchor index + the qdrant SKOS conceptual-filter index, and route the long tail through
the HermiT/CCO membrane for our own ≡ definitions.

Usage:  uv run --no-sync python scripts/sysml_foundation.py
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/Systems-Modeling/SysML-v2-Release/master/sysml.library/"
CACHE = Path("build/sysml")

# The concept-bearing library files we draw from (Systems + Geometry + the ISQ/SI quantity backbone).
FILES = [
    "Systems Library/Items.sysml",
    "Systems Library/Parts.sysml",
    "Systems Library/Ports.sysml",
    "Systems Library/Connections.sysml",
    "Systems Library/Interfaces.sysml",
    "Systems Library/Actions.sysml",
    "Systems Library/States.sysml",
    "Systems Library/Requirements.sysml",
    "Systems Library/Constraints.sysml",
    "Systems Library/AnalysisCases.sysml",
    "Systems Library/Allocations.sysml",
    "Domain Libraries/Geometry/ShapeItems.sysml",
    "Domain Libraries/Geometry/SpatialItems.sysml",
    "Domain Libraries/Quantities and Units/Quantities.sysml",
    "Domain Libraries/Quantities and Units/ISQ.sysml",
    "Domain Libraries/Quantities and Units/SI.sysml",
]

# Curated top-level bridge: SysML construct NAME -> (BFO/CCO genus short-form, OUR gloss, domain).
# Short-forms resolve to IRIs in Stage 2 (cco: -> CCO https IRIs, bfo: -> purl BFO_). Glosses are ours.
BRIDGE: dict[str, tuple[str, str, str]] = {
    "Item": ("bfo:0000002", "a thing (continuant) that can flow through, occupy, or compose a system", "core"),
    "Part": ("cco:ont00000995", "a designed physical part or component of a system — a material artifact", "manufacturing"),
    "Port": ("bfo:0000016", "a connection point of a part — a disposition to interface with another part", "manufacturing"),
    "Interface": ("bfo:0000016", "a defined interaction capability realized between connected parts", "manufacturing"),
    "Connection": ("bfo:0000145", "a structural connection holding between parts of a system", "manufacturing"),
    "Action": ("bfo:0000015", "an activity or behavior a system performs — a process", "core"),
    "StateAction": ("bfo:0000015", "a stative condition a system maintains over an interval", "core"),
    "RequirementCheck": ("cco:ont00000965", "a required condition or capability a system shall satisfy", "engineering"),
    "ConstraintCheck": ("cco:ont00000958", "a constraint expression asserted over system properties", "engineering"),
    "AnalysisCase": ("cco:ont00000958", "an analysis specification evaluated against a system model", "engineering"),
    "ShapeItem": ("bfo:0000006", "a geometric shape occupying a spatial region (incl. CSG solids)", "geometry"),
    "SpatialItem": ("bfo:0000006", "a spatially-extended item with position and orientation", "geometry"),
    "ScalarQuantityValue": ("cco:ont00000853", "a measured scalar quantity (ISQ) borne by an entity", "measurement"),
    "QuantityValue": ("cco:ont00000853", "a measurable quantity (ISQ) borne by an entity", "measurement"),
    "MeasurementUnit": ("cco:ont00000120", "a unit of measure (SI/ISQ) for a quantity", "measurement"),
    "UnitOfMeasure": ("cco:ont00000120", "a unit of measure (SI/ISQ) for a quantity", "measurement"),
}

# Per-file default genus (inheritance fallback for concepts not named in BRIDGE).
FILE_GENUS: dict[str, tuple[str, str]] = {
    "Items": ("bfo:0000002", "core"),
    "Parts": ("cco:ont00000995", "manufacturing"),
    "Ports": ("bfo:0000016", "manufacturing"),
    "Connections": ("bfo:0000145", "manufacturing"),
    "Interfaces": ("bfo:0000016", "manufacturing"),
    "Actions": ("bfo:0000015", "core"),
    "States": ("bfo:0000015", "core"),
    "Requirements": ("cco:ont00000965", "engineering"),
    "Constraints": ("cco:ont00000958", "engineering"),
    "AnalysisCases": ("cco:ont00000958", "engineering"),
    "Allocations": ("cco:ont00000958", "engineering"),
    "ShapeItems": ("bfo:0000006", "geometry"),
    "SpatialItems": ("bfo:0000006", "geometry"),
    "Quantities": ("cco:ont00000853", "measurement"),
    "ISQ": ("cco:ont00000853", "measurement"),
    "SI": ("cco:ont00000120", "measurement"),
}

# `[abstract] <kind> def <Name>[ :> <Super1>, <Super2>]` — names + taxonomy only (EPL-clean facts).
DEF_RE = re.compile(
    r"^\s*(?:abstract\s+)?(part|item|attribute|port|interface|connection|action|state|"
    r"occurrence|calc|constraint|requirement|analysis|verification|view|viewpoint|"
    r"rendering|metadata|enum|allocation)\s+def\s+([A-Za-z_]\w*)\s*(?::>+\s*([^{;]+))?",
    re.MULTILINE,
)

PROVENANCE = "SysML v2 standard library (EPL-2.0); concept name + taxonomy drawn, definition authored by Aegir"


def fetch(rel: str) -> str:
    """Fetch a .sysml file, caching under gitignored build/sysml/ (doc comments used for understanding only)."""
    dest = CACHE / rel
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    url = BASE + urllib.parse.quote(rel)
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (trusted host)
        text = r.read().decode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text


def parse_concepts(text: str) -> list[dict]:
    """Extract (name, kind, supertypes) — uncopyrightable structural facts; doc comments NOT retained."""
    out = []
    for m in DEF_RE.finditer(text):
        kind, name, supers = m.group(1), m.group(2), (m.group(3) or "")
        sup = [s.strip() for s in supers.split(",") if s.strip()]
        out.append({"name": name, "kind": kind, "supertypes": sup})
    return out


def ground(concept: dict, file_stem: str) -> dict:
    """Assign a BFO/CCO genus: curated bridge by name, else per-file inheritance default."""
    name = concept["name"]
    if name in BRIDGE:
        genus, gloss, domain = BRIDGE[name]
    else:
        genus, domain = FILE_GENUS.get(file_stem, ("bfo:0000002", "core"))
        gloss = ""  # authored downstream (Stage 2 membrane / engine); never copied from EPL docs
    return {
        "name": name,
        "kind": concept["kind"],
        "supertypes": concept["supertypes"],
        "genus": genus,
        "gloss": gloss,
        "domain": domain,
        "sysml_file": file_stem,
        "provenance": PROVENANCE,
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    terms: list[dict] = []
    seen: set[str] = set()
    for rel in FILES:
        stem = Path(rel).stem
        try:
            text = fetch(rel)
        except Exception as exc:  # noqa: BLE001 — report + continue (a missing file shouldn't abort)
            print(f"  ! skip {rel}: {exc}")
            continue
        concepts = parse_concepts(text)
        for c in concepts:
            if c["name"] in seen:
                continue
            seen.add(c["name"])
            terms.append(ground(c, stem))
        print(f"  {stem:16s} {len(concepts):3d} concepts")

    out = CACHE / "sysml_foundation.json"
    out.write_text(json.dumps(terms, indent=2), encoding="utf-8")

    by_domain: dict[str, int] = {}
    bridged = 0
    for t in terms:
        by_domain[t["domain"]] = by_domain.get(t["domain"], 0) + 1
        if t["gloss"]:
            bridged += 1
    print(f"\n{len(terms)} grounded terms ({bridged} curated-gloss, {len(terms) - bridged} inherited-genus)")
    print("by domain: " + ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items())))
    print(f"-> {out}  (EPL-clean: names+taxonomy+our-genera only; no SysML doc text persisted)")


if __name__ == "__main__":
    main()
