#!/usr/bin/env python
"""verify_property_domains — prove the created domains before they bite (#27 discipline).

Isolated assembly (never production): numeric BFO backbone + armed signatures + the GENERATED
DOMAINS_OMN + the catalog templates' axioms with sdg coins grounded → HermiT under the budget
watchdog. Unsat = the violator worklist (mechanical/CAS re-authoring, never dropped); the
AEGIR_PROPERTY_DOMAINS default flips ON only at 0 unsat. Run-side properties get their
verification at the W2 unified-realization shakedown (this pass exercises the catalog-used
subset).

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run python scripts/verify_property_domains.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.property_domains import DOMAINS_OMN  # noqa: E402
from aegir.ontology.relation_signatures import SIGNATURES_OMN, grounding_frames  # noqa: E402

_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
             "Ontology: <https://signals.zndx.org/sdg/property-domain-verify>\n")

_HEAD = re.compile(r"Class:\s*\{(\w+):Class\}\s*SubClassOf:\s*([a-z]+:[\w]+)")
_REST = re.compile(r"sdg:(\w+)\s+(some|only|min \d+|max \d+|exactly \d+)\s+\{(\w+):Class\}")


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    cat = json.loads((REPO / "src/aegir/ontology/catalog/catalog.json").read_text())
    frames, props, fillers, heads = [], set(), set(), set()
    for t in cat.get("templates", []):
        mt = t.get("manchester_template") or ""
        hm = _HEAD.search(mt)
        if not hm or not hm.group(2).startswith("bfo:"):
            continue
        head, anchor = hm.group(1), hm.group(2)
        rests = _REST.findall(mt)
        if not rests:
            continue
        heads.add(head)
        parts = [anchor]
        for prop, quant, filler in rests:
            props.add(prop)
            fillers.add(filler)
            parts.append(f"sdg:{prop} {quant} sdg:{filler}")
        frames.append(f"Class: sdg:{head}\n    SubClassOf: " + ",\n        ".join(parts))
    grounded, loose = grounding_frames(props)
    # only DOMAINS_OMN frames for properties this fragment actually uses (isolated relevance)
    dom_frames = []
    for fm in DOMAINS_OMN.strip().split("\n\n") if "\n\n" in DOMAINS_OMN else \
            re.split(r"\n(?=(?:Object|Data)Property:)", DOMAINS_OMN.strip()):
        pm = re.match(r"(?:Object|Data)Property:\s*sdg:(\w+)", fm)
        if pm and pm.group(1) in props:
            dom_frames.append(fm.strip())
    bro = _mod(REPO / "scripts/build_realized_ontology.py", "bro_pd")
    et = _mod(REPO / "scripts/emit_taxonomy.py", "et_pd")
    omn = (_PREFIXES + bro.NUMERIC_BFO + "\n" + SIGNATURES_OMN + "\n"
           + "\n".join(dom_frames) + "\n"
           + "\n".join(f"Class: sdg:{f}\n    SubClassOf: owl:Thing"
                       for f in sorted(fillers - heads)) + "\n"
           + grounded + "\n"
           + "\n".join(f"ObjectProperty: sdg:{p}" for p in loose
                       if not any(f"sdg:{p}\n" in d or f"sdg:{p} " in d for d in dom_frames))
           + "\n" + "\n".join(frames))
    print(f"{len(frames)} axioms · {len(dom_frames)} created-domain frames in scope · "
          f"running HermiT…", flush=True)
    v = et.hermit(omn, budget_s=int(__import__('os').environ.get('AEGIR_REASON_BUDGET_S', 900)))
    unsat = v.get("unsat") or []
    print(f"HermiT: consistent={v.get('consistent')} · classes={v.get('n_classes')} · "
          f"UNSAT={len(unsat)}")
    if unsat:
        print("violators (the re-authoring worklist):")
        for u in unsat[:16]:
            print(f"  ✘ {u}")
    (REPO / "build/property_domain_verify.json").write_text(json.dumps(
        {"n_axioms": len(frames), "n_domain_frames": len(dom_frames),
         "consistent": v.get("consistent"), "unsat": unsat}, indent=1))
    print("→ build/property_domain_verify.json")
    return 0 if not unsat else 3


if __name__ == "__main__":
    raise SystemExit(main())
