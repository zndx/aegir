#!/usr/bin/env python
"""verify_relation_signatures — prove the arming fires: signatures + grounded coins → HermiT (#27 move 1).

Assembles an ISOLATED ontology (never production): the numeric BFO backbone + the new relation
SIGNATURES + the anchored-head lint templates' axioms with their sdg relations GROUNDED
(SubPropertyOf the nominal BFO parents). If the instrument works, HermiT refutes approximately the
lint's DOMAIN_MISCAST set (a role that sdg:realizes anything is now a role that bfo:realizes —
domain bfo:process — while being a bfo:role — disjointness backbone → UNSAT). The unsat list is the
move-2 worklist, cross-checked against build/relation_lint.json.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run python scripts/verify_relation_signatures.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.relation_signatures import SIGNATURES_OMN, grounding_frames  # noqa: E402

_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
             "Ontology: <https://signals.zndx.org/sdg/relation-signature-verify>\n")

_HEAD = re.compile(r"Class:\s*\{(\w+):Class\}\s*SubClassOf:\s*([a-z]+:[\w]+)")
_REST = re.compile(r"sdg:(\w+)\s+(some|only|min \d+|max \d+|exactly \d+)\s+\{(\w+):Class\}")


def _bro():
    spec = importlib.util.spec_from_file_location("bro", REPO / "scripts/build_realized_ontology.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    cat = json.loads((REPO / "src/aegir/ontology/catalog/catalog.json").read_text())
    tpls = cat.get("templates", cat if isinstance(cat, list) else [])
    frames, props, fillers, heads = [], set(), set(), []
    for t in tpls:
        mt = t.get("manchester_template") or ""
        hm = _HEAD.search(mt)
        if not hm:
            continue
        head, anchor = hm.group(1), hm.group(2)
        if not anchor.startswith("bfo:"):
            continue                                        # numeric backbone only for this probe
        rests = _REST.findall(mt)
        if not rests:
            continue
        heads.append(head)
        parts = [anchor]
        for prop, quant, filler in rests:
            props.add(prop)
            fillers.add(filler)
            parts.append(f"sdg:{prop} {quant} sdg:{filler}")
        frames.append(f"Class: sdg:{head}\n    SubClassOf: " + ",\n        ".join(parts))
    grounded, loose = grounding_frames(props)
    bro = _bro()
    omn = (_PREFIXES + bro.NUMERIC_BFO + "\n" + SIGNATURES_OMN + "\n"
           + "\n".join(f"Class: sdg:{f}\n    SubClassOf: owl:Thing" for f in sorted(fillers - set(heads)))
           + "\n" + grounded + "\n"
           + "\n".join(f"ObjectProperty: sdg:{p}" for p in loose) + "\n"
           + "\n".join(frames))
    print(f"{len(frames)} axioms · {len(props)} sdg relations ({len(loose)} left bare/long-tail) — "
          f"running HermiT…", flush=True)
    spec = importlib.util.spec_from_file_location("et", REPO / "scripts/emit_taxonomy.py")
    assert spec and spec.loader
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    v = et.hermit(omn, budget_s=600)
    unsat = v.get("unsat") or []
    print(f"HermiT: consistent={v.get('consistent')} · classes={v.get('n_classes')} · UNSAT={len(unsat)}")
    lint = json.loads((REPO / "build/relation_lint.json").read_text())
    lint_heads = {x["head"] for x in lint["violations"] if x["kind"] == "DOMAIN_MISCAST"}
    hit = lint_heads & set(unsat)
    print(f"cross-check vs lint DOMAIN_MISCAST heads: {len(hit)}/{len(lint_heads)} refuted by HermiT")
    extra = sorted(set(unsat) - lint_heads)[:8]
    if extra:
        print(f"unsat beyond the lint set (range-side / inherited): {extra}")
    (REPO / "build/relation_signature_verify.json").write_text(json.dumps(
        {"n_axioms": len(frames), "unsat": unsat, "lint_heads": sorted(lint_heads),
         "lint_refuted": sorted(hit), "loose_props": loose}, indent=1))
    print("→ build/relation_signature_verify.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
