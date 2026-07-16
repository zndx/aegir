#!/usr/bin/env python
"""emit_taxonomy — assemble the authored genus+differentia taxonomy into an ontology OVERLAY, validate it
GLOBALLY with HermiT, and render the SHACL shapes that carry each differentia (the SHACL arc, #16).

Input: build/taxonomy/<tag>/genera/*.json (from author_taxonomy) + the run entities (for each species' BFO/CCO
genus, so the induced genus grounds to the mode anchor of its members). Outputs, under the same dir:
  overlay.omn          genera (SubClassOf mode BFO/CCO anchor) + species (SubClassOf genus + differentia) +
                       declared properties/fillers + DisjointClasses per genus  →  HermiT-checked globally.
  overlay_shapes.ttl   one sh:NodeShape per species whose sh:property carries the DIFFERENTIA (the disambiguator
                       now IN the shape) — feeds discriminability: the authored siblings become distinguishable.

The DisjointClasses are nearly free — the per-genus HermiT membrane already certified pairwise-distinctness;
this emits + GLOBALLY re-certifies them (all 32 genera at once catches any cross-genus conflict).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
             "Ontology: <https://signals.zndx.org/sdg/taxonomy-overlay>\n")
_SH = "https://signals.zndx.org/sdg#"


def _species_genus(entities_dir: Path) -> "dict[str, str]":
    out: "dict[str, str]" = {}
    for f in glob.glob(str(entities_dir / "*.json")):
        for e in json.loads(Path(f).read_text()).get("entities", []):
            n, g = e.get("name"), e.get("genus")
            if n and g and n not in out:
                out[n] = g
    return out


def _norm_restriction(restriction: str) -> "tuple[str, str, list[str]]":
    """→ (valid_manchester, inferred_kind, [filler_locals]). Kind is INFERRED from the restriction shape (robust
    to the model mislabelling it), bare boolean/numeric literals are typed, '|'/'or' alternatives collapse to
    the first clause, and object fillers get the sdg: prefix + a declaration."""
    r = re.split(r"\s*\|\s*|\s+or\s+", restriction.strip(), maxsplit=1)[0].strip()
    m = re.match(r"value\s+(.+)$", r, re.S)
    if m:
        v = m.group(1).strip()
        if v.lower() in ("true", "false"):
            return f'value "{v.lower()}"^^xsd:boolean', "data", []
        if re.fullmatch(r"-?\d+", v):
            return f'value "{v}"^^xsd:integer', "data", []
        if re.fullmatch(r"-?\d+\.\d+", v):
            return f'value "{v}"^^xsd:decimal', "data", []
        qm = re.match(r'"((?:[^"\\]|\\.)*)"', v)                 # first quoted string, else quote the bareword
        val = qm.group(1) if qm else v.strip('"')
        return f'value "{val}"', "data", []
    m = re.match(r"(some|only|min \d+|max \d+|exactly \d+)\s+(\S+)", r)
    if m:
        card, filler = m.group(1), m.group(2)
        if filler.startswith("xsd:"):
            return f"{card} {filler}", "data", []
        fl = filler.split(":")[-1]
        return f"{card} sdg:{fl}", "object", [fl]
    return f'value "{r[:40]}"', "data", []                       # fallback: a plain string value


def build_overlay(genera_dir: Path, sp_genus: "dict[str, str]") -> "tuple[str, dict]":
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(genera_dir / "*.json")))]
    props, fillers, disjoints = {}, set(), []
    anchors, species_all, frames = set(), set(), []
    for r in recs:
        g = r["genus"]
        accepted = {s: d for s, d in r["differentiae"].items() if d["accepted"]}
        anchor = Counter(sp_genus.get(s, "bfo:0000001") for s in accepted).most_common(1)
        anc = anchor[0][0] if anchor else "bfo:0000001"
        anchors.add(anc)
        frames.append(f"Class: sdg:{g}\n    SubClassOf: {anc}")
        emitted = []
        for s, d in accepted.items():
            if s == g:                                            # genus label == a member (fallback naming) →
                continue                                          # keep it as the GENUS only, not a disjoint species
            rr, kind, fs = _norm_restriction(d["restriction"])
            if props.setdefault(d["property"], kind) != kind:     # property already fixed to the other kind → skip
                continue
            frames.append(f"Class: sdg:{s}\n    SubClassOf: sdg:{g}, sdg:{d['property']} {rr}")
            fillers.update(fs)
            species_all.add(s)
            emitted.append(s)
        if len(emitted) > 1:                                       # disjoint only the species actually DECLARED
            disjoints.append(sorted(emitted))
    # DECLARATIONS FIRST (Manchester needs property kind + fillers/anchors known before the restrictions use them)
    lines = [_PREFIXES]
    for p, k in sorted(props.items()):
        lines.append(f"{'ObjectProperty' if k == 'object' else 'DataProperty'}: sdg:{p}")
    for anc in sorted(anchors):
        lines.append(f"Class: {anc}")
    for fc in sorted(fillers - species_all):
        lines.append(f"Class: sdg:{fc}\n    SubClassOf: owl:Thing")
    lines.extend(frames)
    for grp in disjoints:
        lines.append("DisjointClasses: " + ", ".join(f"sdg:{s}" for s in grp))
    stats = {"genera": len(recs), "species": len(species_all), "properties": len(props),
             "object_properties": sum(1 for k in props.values() if k == "object"),
             "disjoint_axioms": len(disjoints),
             "disjoint_pairs": sum(len(g) * (len(g) - 1) // 2 for g in disjoints)}
    return "\n".join(lines), stats


def hermit(omn_text: str, budget_s: int = 900) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(omn_text)
        path = f.name
    helper = ("import sys, json; sys.path.insert(0,'src'); "
              "from aegir.ontology.deeponto_harness import ensure_jvm; ensure_jvm(); "
              "import importlib.util as u; "
              "spec=u.spec_from_file_location('bro','scripts/build_realized_ontology.py'); "
              "m=u.module_from_spec(spec); spec.loader.exec_module(m); "
              "r=m._reason(open(sys.argv[1]).read()); "
              "print(json.dumps({'consistent': bool(r[2]), 'n_classes': r[3], 'unsat': [x.split('#')[-1] for x in r[4]]}))")
    import os
    env = {**os.environ, "LD_LIBRARY_PATH": str(REPO / "build" / "jvm-libs"), "AEGIR_REASON_BUDGET_S": str(budget_s)}
    p = subprocess.run(["uv", "run", "--no-sync", "python", "-c", helper, path],
                       cwd=str(REPO), capture_output=True, text=True, timeout=budget_s + 180, env=env)
    if p.returncode == 3:
        return {"consistent": None, "reason": "hermit budget exceeded"}
    lines = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {"consistent": None, "reason": (p.stderr or "")[-200:]}


def emit_shapes(genera_dir: Path) -> str:
    """One sh:NodeShape per differentiated species — its sh:property IS the differentia (the disambiguator now
    in the shape). This is what closes the discriminability projection-gap for the authored siblings."""
    out = ["@prefix sh: <http://www.w3.org/ns/shacl#> .", "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
           f"@prefix sdg: <{_SH}> .", ""]
    for f in sorted(glob.glob(str(genera_dir / "*.json"))):
        r = json.loads(Path(f).read_text())
        for s, d in r["differentiae"].items():
            if not d["accepted"]:
                continue
            rr, kind, fs = _norm_restriction(d["restriction"])
            out.append(f"sdg:{s}Shape a sh:NodeShape ; sh:targetClass sdg:{s} ;")
            if kind == "data":
                vm = re.search(r'value\s+"([^"]*)"', rr)
                if vm:
                    out.append(f'  sh:property [ sh:path sdg:{d["property"]} ; sh:hasValue "{vm.group(1)}" ] .')
                else:
                    out.append(f'  sh:property [ sh:path sdg:{d["property"]} ; sh:datatype xsd:string ] .')
            else:
                tgt = f"sdg:{fs[0]}" if fs else "sdg:Thing"
                out.append(f'  sh:property [ sh:path sdg:{d["property"]} ; sh:class {tgt} ; sh:minCount 1 ] .')
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default=str(REPO / "build" / "taxonomy" / "v05"))
    ap.add_argument("--entities", required=True)
    ap.add_argument("--budget", type=int, default=900)
    a = ap.parse_args()
    tdir = Path(a.taxonomy)
    sp_genus = _species_genus(Path(a.entities))
    overlay, stats = build_overlay(tdir / "genera", sp_genus)
    (tdir / "overlay.omn").write_text(overlay)
    (tdir / "overlay_shapes.ttl").write_text(emit_shapes(tdir / "genera"))
    print(f"overlay: {stats['genera']} genera · {stats['species']} species · {stats['properties']} properties "
          f"({stats['object_properties']} object) · {stats['disjoint_axioms']} DisjointClasses "
          f"({stats['disjoint_pairs']} pairs)")
    print("running GLOBAL HermiT over the whole overlay (all genera at once)…", flush=True)
    v = hermit(overlay, a.budget)
    print(f"HermiT: consistent={v.get('consistent')} · classes={v.get('n_classes')} · "
          f"unsat={len(v.get('unsat', []))} {v.get('unsat', [])[:8]}" + (f" · {v.get('reason','')}" if v.get('reason') else ""))
    (tdir / "overlay_hermit.json").write_text(json.dumps({**stats, "hermit": v}, indent=2))
    print(f"→ {tdir}/overlay.omn · overlay_shapes.ttl · overlay_hermit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
