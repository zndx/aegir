"""Define the intermediate domain classes — the property-bearing subsumers that real columns annotate to.

These are NOT "fillers" (a slot-mechanics artifact of them filling {Slot:Class} differentia positions): they
are intermediate-depth hierarchical classes — the CTA/CPA annotation TARGETS. A column such as
driver_stops_schedule.stops_addresses holds a heterogeneous-but-coherent mix (origin + destination,
residential + business shipping addresses, each bearing an avg-time-on-site); no leaf type fits — the column
belongs to the least common SUBSUMER that is still property-bearing. Defining these classes well IS building
the annotation vocabulary. ~131 of ~210 classes are intermediate classes Phase A grounds but leaves PRIMITIVE;
the lever is to DEFINE them (def_completeness = ≡-defined / ALL classes).

The agent-mediated propose/dispose loop, CLOSED over BOTH membranes (RH 2026-06-29):
  • PARSE membrane (`evolve_rigor.validate_detailed`) — admits well-formed Manchester, returns the reason.
  • REASONING-AUTHORITY membrane (`build_realized_ontology.consistency_check`) — imports CCO and runs HermiT,
    so a class grounded to a CCO-disjoint / BFO-incompatible genus is REJECTED and fed back to the agent.
Each intermediate class carries its top-k retrieved grounding anchors (CCO/FHIR/our own); rejects are
re-prompted WITH the specific failure, up to --rounds. The agent RESPONDS to parser AND reasoner.

    just engine-serve
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/define_intermediate_classes.py --rounds 4 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology.schema import CatalogTemplate, load_catalog, save_catalog  # noqa: E402
from evolve_rigor import parse_slots, validate_detailed  # noqa: E402
from grounding_anchors import Retriever  # noqa: E402
from build_realized_ontology import consistency_check  # noqa: E402

_SLOT = re.compile(r"\{(\w+):(\w+)\}")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _decamel(s: str) -> str:
    """camelCase filler name → retrieval query: 'ParasiticPlant' -> 'parasitic plant'."""
    return _CAMEL.sub(" ", s).replace("_", " ").lower()
# the syntax rules the membrane enforces (diagnosed 2026-06-29 — the engine's first pass violated all three)
_SYNTAX = (
    "SYNTAX (axioms that break these FAIL to parse — the membrane will reject them):\n"
    "1. prefixes are LOWERCASE only: cco: bfo: fhir: sdg: — NEVER CCO: or BFO:.\n"
    "2. EVERY property is prefixed: write `sdg:derivesNutrientsFrom some X`, never bare `participates_in`.\n"
    "3. Use the EXACT cco:/fhir: IRIs from the provided anchors (e.g. cco:ont00000871) — do NOT INVENT cco:/bfo: "
    "names. Coin sdg: (camelCase) only for genuinely new classes/properties. The genus must be a provided ANCHOR "
    "(cco:/fhir:/sdg:) OR a BFO category (bfo:0000040 material entity / bfo:0000015 process / bfo:0000031 "
    "generically dependent continuant / bfo:0000019 quality / bfo:0000023 role) OR a {Slot:Class}.\n"
)
_SYS = (
    "You are defining REFERENCED domain classes (fillers) of a BFO 2020 / CCO ontology. Each is a primitive "
    "placeholder. For EACH, author a genus-differentia definition as ONE Manchester axiom: `Class: "
    "{Name:Class} EquivalentTo: <genus> and <≥1 differentiating restriction>` — necessary AND sufficient, from "
    "domain world-knowledge + the given contexts. The head slot MUST be the filler's EXACT given name (for "
    "`ParasiticPlant` emit `Class: {ParasiticPlant:Class} EquivalentTo: …`). Define ONLY a filler you can "
    "genuinely characterize by a sufficient condition; if it is a bare natural kind you cannot define, set "
    "keep=true and skip it.\n\n" + _SYNTAX +
    "\nGROUNDING — each filler is given CANDIDATE GROUNDED ANCHORS (real classes: cco: real-world genera, "
    "fhir: clinical/record types, sdg: our own). For the GENUS and EACH differentia filler, PREFER the "
    "best-fitting anchor over a generic bfo: category or a coined sdg: term — the anchors are already "
    "BFO-grounded, so reusing them grounds your definition meaningfully. The GENUS must be a class BROADER "
    "than the filler (a parent kind) — NEVER the filler itself or a near-synonym; for an information/record "
    "concept ground to cco:ont00000958 (Information Content Entity) or a fhir: type. Fall back to a bare bfo: "
    "category only when NO anchor fits.\n"
    "\nOUTPUT exactly one ```json block {\"definitions\":[{\"manchester_template\":\"…\",\"keep\":false,"
    "\"verbal\":\"one-sentence gloss\"}]}. Reasoning OUTSIDE the json."
)


def _parse_defs(text: str) -> list:
    m = re.search(r"```json\s*(.+?)```", text, re.S) or re.search(r"```\s*(\{.+?\})\s*```", text, re.S)
    if m:
        blob = m.group(1)
    else:
        m2 = re.search(r"\{[\s\S]*\"definitions\"[\s\S]*\}", text)
        blob = m2.group(0) if m2 else None
    if not blob:
        return []
    try:
        return json.loads(blob).get("definitions", [])
    except ValueError:
        return []


def _propose(chunk: list, contexts: dict, feedback: dict, retriever: "Retriever", cap: str, temp: float) -> dict:
    """One proposal pass over ``chunk`` → {filler: manchester}, binding on the head slot. Each filler carries
    its top-k retrieved grounding anchors (the boundary's domain-vocabulary signal); re-tries carry the gate's
    prior verdict so the agent can RESPOND to it."""
    lines = []
    for f in chunk:
        ctx = ", ".join(sorted(set(contexts[f]))[:4])
        ex = {_decamel(f), f.lower(), f"sdg:{f}".lower()}  # the genus must be BROADER — never the filler itself
        alist = " · ".join(f"{a['curie']} ({a['label']})" for a in retriever.retrieve(_decamel(f), k=7, exclude=ex))
        block = f"- {f}  (appears in: {ctx})\n  anchors (prefer for genus + differentia fillers): {alist}"
        if f in feedback:
            block += f"\n  PRIOR ATTEMPT REJECTED — {feedback[f]}  → fix and re-emit."
        lines.append(block)
    out = complete_detailed("Define these referenced fillers:\n\n" + "\n".join(lines),
                            capability=cap, system_prompt=_SYS, max_tokens=16000, temperature=temp)
    proposed = {}
    for d in _parse_defs(out["text"]):
        man = d.get("manchester_template") or ""
        if d.get("keep") or "EquivalentTo" not in man:
            continue
        hm = re.search(r"Class:\s*\{(\w+)", man)
        match = next((f for f in chunk if hm and f.lower() == hm.group(1).lower()), None)
        if match:
            proposed[match] = (man, d.get("verbal") or f"A {match}.")
    return proposed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--out", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--batch", type=int, default=14)
    ap.add_argument("--rounds", type=int, default=3, help="max feedback rounds per filler")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    cat = load_catalog(args.catalog)
    heads, contexts = set(), {}
    for t in cat.templates:
        man = t.manchester_template or ""
        slots = _SLOT.findall(man)
        if slots:
            heads.add(slots[0][0])
        hm = re.search(r"Class:\s*\{(\w+)", man)
        head = hm.group(1) if hm else "?"
        for name, _typ in slots[1:]:
            contexts.setdefault(name, []).append(head)
    defined_heads = {hm.group(1) for t in cat.templates
                     if (hm := re.search(r"Class:\s*\{(\w+)", t.manchester_template or ""))
                     and "EquivalentTo" in (t.manchester_template or "")}
    fillers = sorted(set(contexts) - heads - defined_heads)
    if args.limit:
        fillers = fillers[:args.limit]
    print(f"define-intermediate-classes: {len(fillers)} intermediate classes · two-membrane loop ≤{args.rounds} rounds (heads {len(heads)})")
    retriever = Retriever()  # the grounding-anchor signal source (CCO + FHIR + our accreting classes)

    accepted: dict[str, CatalogTemplate] = {}
    feedback: dict[str, str] = {}
    for rnd in range(args.rounds):
        pending = [f for f in fillers if f not in accepted]
        if not pending:
            break
        proposed: dict[str, tuple] = {}
        for bi in range(0, len(pending), args.batch):
            chunk = pending[bi:bi + args.batch]
            try:
                proposed.update(_propose(chunk, contexts, feedback, retriever, args.capability, args.temperature))
            except Exception as e:  # noqa: BLE001
                print(f"  round {rnd + 1} batch @{bi}: engine error {type(e).__name__}")
        # membrane — admit + capture the REASON for the rejects (the feedback channel)
        prov = {f"filler_{f.lower()}": CatalogTemplate(
            template_id=f"filler_{f.lower()}", manchester_template=man, slot_types=parse_slots(man),
            is_complex=True, verbal_template=verbal, verbal_templates=[], mean_verbal_length=0.0,
            bfo_anchor_path=[], broader=[], provenance={"source": "define_fillers"})
            for f, (man, verbal) in proposed.items()}
        detailed = validate_detailed([(tid, t.manchester_template) for tid, t in prov.items()], prov) if prov else {}
        n_new = 0
        for f, (man, _v) in proposed.items():
            tid = f"filler_{f.lower()}"
            ok, reason = detailed.get(tid, (False, "not validated"))
            if ok and f not in accepted:
                accepted[f] = prov[tid]
                feedback.pop(f, None)
                n_new += 1
            elif not ok:
                feedback[f] = f"`{man}` → {reason}"
        # REASONING-AUTHORITY membrane — HermiT validates the round's grounding against CCO's disjointness;
        # an intermediate class grounded to a disjoint/incompatible genus is DEMOTED back to the agent.
        n_demoted = 0
        if accepted:
            _consistent, unsat = consistency_check(list(cat.templates) + list(accepted.values()))
            unsat_local = {u.rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower() for u in unsat}
            for f in list(accepted):
                if f.lower() in unsat_local:
                    gm = re.search(r"EquivalentTo:\s*(\S+)", accepted[f].manchester_template or "")
                    feedback[f] = (f"HermiT: '{f}' is UNSATISFIABLE under CCO disjointness — the genus "
                                   f"{gm.group(1) if gm else '?'} is incompatible with how it is used (disjoint "
                                   "CCO parents, or a continuant genus where a process/ICE is required). "
                                   "Re-ground to a single COMPATIBLE parent from the anchors.")
                    del accepted[f]
                    n_demoted += 1
        print(f"  round {rnd + 1}/{args.rounds}: proposed {len(proposed)} · +{n_new} admitted · "
              f"{n_demoted} HermiT-demoted · {len(accepted)}/{len(fillers)} total · {len(feedback)} pending")

    cat.templates.extend(accepted.values())
    save_catalog(cat, args.out)
    print(f"\nAPPLIED: {len(accepted)} intermediate-class ≡-definitions (parse + HermiT membranes) → {Path(args.out).name}")
    print("DISPOSE next: build_realized_ontology.py --strict-grounding (HermiT) → ontology_metrology / ontology_oquare")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
