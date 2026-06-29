"""domain_foundations.py — curated WITSML (energy/drilling) + BRL-CAD (CSG-design) foundation vocabularies.

Companion to sysml_foundation.py. Both EXTEND the BFO/CCO/FHIR foundation with domain-standard vocabulary so
the qdrant late-interaction classifier can DISCRIMINATE inbound FinePDFs docs among the targeted domains
(FHIR/clinical, SysMLv2/manufacturing, WITSML/energy, CSG-design, LIMS) BEFORE the content-first derive.

Breadth-not-depth: ground each standard's top-level constructs to a BFO/CCO genus (the bridge) + author OUR
OWN glosses; the names + taxonomy are uncopyrightable facts drawn from the standards and CITED. Genera here
are provisional discrimination hints — the HermiT/CCO membrane validates them when these become realized
classes (the SysML-foundation reasoner pass generalizes to these).

  WITSML  — Energistics well-data standard; permissive **Apache** PDS distribution (github.com/pds-technology).
  BRL-CAD — US Army Research Laboratory CSG solid modeler (US-gov / LGPL / BSD); 41 primitives + the CSG tree.

Emits build/foundation/{witsml,brlcad}_foundation.json (gitignored); grounding_anchors loads them.

  uv run --no-sync python scripts/domain_foundations.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("build/foundation")

# WITSML 2.0 data objects → BFO/CCO genus + our gloss (energy/drilling). Apache (PDS); names drawn + cited.
WITSML = {
    "Well": ("bfo:site", "a well — a constructed feature for accessing subsurface hydrocarbons"),
    "Wellbore": ("bfo:site", "a wellbore — the drilled hole within a well"),
    "WellboreGeometry": ("bfo:spatial_region", "the as-built geometry of a wellbore"),
    "Target": ("bfo:site", "a subsurface drilling target"),
    "Trajectory": ("bfo:spatial_region", "the path of a wellbore through space"),
    "TrajectoryStation": ("bfo:spatial_region", "a surveyed point on a wellbore trajectory"),
    "Log": ("cco:ICE", "a well log — a record of measurements along a wellbore"),
    "WellLog": ("cco:ICE", "a well-log information object"),
    "MudLog": ("cco:ICE", "a record of formation/cuttings observations made while drilling"),
    "ChannelSet": ("cco:ICE", "a set of indexed measurement channels"),
    "Channel": ("cco:Measurement", "a single indexed measurement series along a wellbore"),
    "FormationMarker": ("cco:ICE", "a geological formation pick/marker"),
    "OpsReport": ("cco:ICE", "a drilling operations report"),
    "FluidsReport": ("cco:ICE", "a drilling-fluids report"),
    "BhaRun": ("cco:ICE", "a record of a bottom-hole-assembly run"),
    "Risk": ("cco:ICE", "a recorded operational risk"),
    "Message": ("cco:ICE", "an operational message/log entry"),
    "DrillReport": ("cco:DirectiveICE", "a drilling report against a program"),
    "SurveyProgram": ("cco:DirectiveICE", "a planned wellbore survey program"),
    "Rig": ("cco:Artifact", "a drilling rig — the equipment assembly that drills a well"),
    "Tubular": ("cco:Artifact", "a tubular component (drill pipe, casing) in the wellbore"),
    "Bit": ("cco:Artifact", "a drill bit"),
    "CementJob": ("bfo:process", "a cementing operation in the wellbore"),
    "StimJob": ("bfo:process", "a well-stimulation operation"),
}

# BRL-CAD CSG solid primitives → bfo:spatial_region (shapes). US-gov/LGPL/BSD; names drawn + cited.
BRLCAD_PRIMS = {
    "arb8": "an arbitrary convex polyhedron of up to 8 vertices",
    "arbn": "an arbitrary convex polyhedron bounded by N planes",
    "sph": "a sphere", "ell": "an ellipsoid", "tor": "a torus", "superell": "a superellipsoid",
    "tgc": "a truncated general cone", "rec": "a right elliptical cylinder",
    "rhc": "a right hyperbolic cylinder", "rpc": "a right parabolic cylinder",
    "ehy": "an elliptical hyperboloid", "epa": "an elliptical paraboloid", "eto": "an elliptical torus",
    "hyp": "a hyperboloid of one sheet", "half": "a half-space bounded by a plane",
    "part": "a particle (rounded capsule) solid", "pipe": "a piecewise pipe/tube solid",
    "bot": "a bag-of-triangles polygonal mesh", "nmg": "an n-manifold boundary-representation solid",
    "brep": "a NURBS boundary-representation solid", "bspline": "a B-spline/NURBS surface solid",
    "sketch": "a 2D sketch profile", "extrude": "a solid formed by extruding a sketch",
    "revolve": "a solid of revolution", "metaball": "an implicit blobby (metaball) surface",
    "ars": "a stack of arbitrary cross-sections (waterlines)", "dsp": "a displacement-mapped terrain surface",
    "ebm": "an extruded-bitmap solid", "vol": "a volumetric (voxel) solid", "hrt": "a heart-shaped solid",
}
# BRL-CAD CSG combination machinery → the boolean tree.
BRLCAD_CSG = {
    "region": ("bfo:spatial_region", "a CSG region — a combination of primitives forming a solid with material"),
    "combination": ("bfo:spatial_region", "a CSG combination — a boolean tree over primitives/sub-combinations"),
    "BooleanUnion": ("bfo:process", "the CSG boolean union of two solids"),
    "BooleanIntersection": ("bfo:process", "the CSG boolean intersection of two solids"),
    "BooleanDifference": ("bfo:process", "the CSG boolean difference (subtraction) of two solids"),
}

WITSML_PROV = ("WITSML 2.0 data model (Energistics; permissive Apache PDS distribution, github.com/pds-technology); "
               "object names drawn + cited, definitions authored by Aegir")
BRLCAD_PROV = ("BRL-CAD solid-modeling vocabulary (US Army Research Laboratory; US-gov / LGPL / BSD); "
               "primitive/CSG names drawn + cited, definitions authored by Aegir")


def emit(name: str, terms: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_foundation.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")
    by: dict[str, int] = {}
    for t in terms:
        by[t["domain"]] = by.get(t["domain"], 0) + 1
    print(f"  {name:8s} {len(terms):3d} terms  {by}  -> {OUT / (name + '_foundation.json')}")


def main() -> None:
    witsml = [{"name": n, "kind": "object", "supertypes": [], "genus": g, "gloss": d,
               "domain": "energy", "source": "witsml", "provenance": WITSML_PROV}
              for n, (g, d) in WITSML.items()]
    brlcad = [{"name": n, "kind": "primitive", "supertypes": [], "genus": "bfo:spatial_region", "gloss": d,
               "domain": "csg-design", "source": "brlcad", "provenance": BRLCAD_PROV}
              for n, d in BRLCAD_PRIMS.items()]
    brlcad += [{"name": n, "kind": "csg", "supertypes": [], "genus": g, "gloss": d,
                "domain": "csg-design", "source": "brlcad", "provenance": BRLCAD_PROV}
               for n, (g, d) in BRLCAD_CSG.items()]
    emit("witsml", witsml)
    emit("brlcad", brlcad)
    print(f"{len(witsml) + len(brlcad)} domain-foundation terms (witsml energy + brlcad csg-design); Apache/US-gov-clean")


if __name__ == "__main__":
    main()
