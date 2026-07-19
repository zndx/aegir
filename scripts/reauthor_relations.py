#!/usr/bin/env python
"""reauthor_relations — #27 move 2: fix the 39 reasoner-certified relation miscasts.

Two tiers, per the RH-ratified plan:

  2a MECHANICAL — per restriction, the correct grounded relation is chosen from
     (subject-category × filler-category), regardless of the coined stem's claim (the
     RegisteredManagerRole finding: stems and fillers can be CROSS-SWAPPED):
        role-subject + occurrent-cue filler   → sdg:realizedIn   (BFO_0000054)
        role-subject + agent/IC-cue filler    → sdg:inheresIn    (BFO_0000197)
     Confident rewrites apply directly; anything ambiguous falls to 2b.

  2b CAS — engine re-authoring with the NAMED-INTERMEDIATE pattern (never anonymous nesting:
     named classes project to DDL, nested expressions project to nothing): a continuant target
     moves behind a new named process class (`Role realizedIn some XProcess` +
     `XProcess hasParticipant some Target`). Membranes: parse/namespace (sdg: only) → the ARMED
     HermiT fragment (backbone + signatures + the candidate) — accepted only when the head
     VANISHES from unsat. Reject ⇒ re-prompt with the reason; exhaustion ⇒ escalate (never drop).

Both tiers: catalog updated in place with provenance ({"reauthored": …, "previous_axiom": …});
STALE verbalizations scrubbed for changed templates and regenerated from the pure recovery path
(merge_manchester_restrictions + compose_frames) — the v2 elaboration cache invalidates by
base-hash, so a follow-up v2 run re-elaborates exactly the changed set.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run python scripts/reauthor_relations.py
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
from aegir.ontology.verbalization import (VerbalizationParts, compose_frames,  # noqa: E402
                                          merge_manchester_restrictions)

PROMPT_VERSION = "reauthor-relations-2026-07-19"
_REST = re.compile(r"(sdg:\w+|bfo:\d{7})\s+(some|only|min \d+|max \d+|exactly \d+)\s+\{(\w+):Class\}")
_HEAD = re.compile(r"Class:\s*\{(\w+):Class\}\s*SubClassOf:\s*(bfo:\d{7})")

_OCC = re.compile(r"(Process|Activity|Provision(?!er)|Operation|Session|Event|Procedure|Investigation|"
                  r"Monitoring|Assessment|Training|Delivery|Execution|Review|Sequence|Ceremony|"
                  r"Intervention|Programme|Program\b|Workflow|Task)\b")
_AGENT = re.compile(r"(Person|Adult|Child|Individual|Provider|Organi[sz]ation|Agency|Officer|Manager|"
                    r"Employee|Nurse|Practitioner|Committee|Board|Team|Institution|Company|Agent|"
                    r"Professional|Specialist|Worker|Staff|User|Member|Holder)\b")
_REALIZ = re.compile(r"^(sdg:(role[_]?)?realiz\w*|bfo:0000055|bfo:0000054)$")
_INHER = re.compile(r"^(sdg:(inheres\w*|borne[_]?by|bears\w*|has[_]?bearer\w*|has[_]?role\w*|"
                    r"has[_]?borne\w*)|bfo:000019[67])$")

_JUDGE_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "axioms": {"type": "array", "items": {"type": "object", "properties": {
            "head": {"type": "string"}, "subclass_of": {"type": "array", "items": {"type": "string"}}},
            "required": ["head", "subclass_of"]}},
        "new_classes": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "anchor": {"type": "string"}, "definition": {"type": "string"}},
            "required": ["name", "anchor", "definition"]}},
        "why": {"type": "string"},
    }, "required": ["axioms", "new_classes", "why"]})

_SYSTEM_2B = """You repair BFO-grounded OWL axioms whose relations are category-miscast. Rules:
- NEVER mint bfo:/cco: terms; new classes are sdg: CamelCase with real definitions.
- Relations available (with signatures): sdg:realizedIn (realizable→process), sdg:inheresIn
  (role/quality→independent continuant), sdg:participatesIn (continuant→process),
  sdg:hasParticipant (process→continuant), sdg:outputOf (continuant→process),
  sdg:concretizes (independent continuant→generically dependent continuant).
- THE NAMED-INTERMEDIATE PATTERN (preferred for continuant targets of realization): a role is
  realizedIn a named PROCESS class; the process hasParticipant the target organization/person.
  Example: {LegalTechServiceProvision} realizes {LegalTechSolutionLab}  ⇒
    axiom 1: head LegalTechServiceProvision, subclass_of ["bfo:0000023",
             "sdg:borneBy exactly 1 {ServiceProvider:Class}",
             "sdg:realizedIn some {LegalTechServiceDeliveryProcess:Class}"]
    new class: LegalTechServiceDeliveryProcess anchor bfo:0000015, plus
    axiom 2: head LegalTechServiceDeliveryProcess, subclass_of ["bfo:0000015",
             "sdg:hasParticipant some {LegalTechSolutionLab:Class}"]
- Preserve every {Slot:Class} placeholder form; keep correct existing restrictions unchanged.
- Anchors: roles bfo:0000023, processes bfo:0000015, independent continuants bfo:0000004."""


def _et():
    spec = importlib.util.spec_from_file_location("et", REPO / "scripts/emit_taxonomy.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bro_backbone() -> str:
    spec = importlib.util.spec_from_file_location("bro", REPO / "scripts/build_realized_ontology.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.NUMERIC_BFO


_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Ontology: <https://signals.zndx.org/sdg/reauthor-verify>\n")

_EXTRA_PROPS = """
ObjectProperty: sdg:outputOf
    SubPropertyOf: bfo:0000056
ObjectProperty: sdg:concretizes
"""


def fragment_check(et, backbone: str, manchester_list: "list[str]", heads: "list[str]") -> "list[str]":
    """Armed fragment → the subset of ``heads`` that are UNSAT (empty = accepted)."""
    props, fillers, declared = set(), set(), set(h for h in heads)
    frames = []
    for mt in manchester_list:
        # head + anchor found ANYWHERE (2b candidates may order parts freely)
        hh = re.search(r"Class:\s*\{(\w+):Class\}", mt)
        aa = re.search(r"\b(bfo:\d{7})\b(?!\s+(?:some|only|min|max|exactly))", mt)
        if not hh or not aa:
            return ["<unparseable head>"]
        head, anchor = hh.group(1), aa.group(1)
        declared.add(head)
        parts = [anchor]
        for prop, quant, filler in _REST.findall(mt):
            if prop.startswith("sdg:"):
                props.add(prop.split(":", 1)[1])
            fillers.add(filler)
            parts.append(f"{prop} {quant} sdg:{filler}")
        frames.append(f"Class: sdg:{head}\n    SubClassOf: " + ",\n        ".join(parts))
    grounded, loose = grounding_frames(props - {"outputOf", "concretizes"})
    omn = (_PREFIXES + backbone + SIGNATURES_OMN + _EXTRA_PROPS
           + "\n".join(f"Class: sdg:{f}\n    SubClassOf: owl:Thing" for f in sorted(fillers - declared))
           + "\n" + grounded + "\n"
           + "\n".join(f"ObjectProperty: sdg:{p}" for p in loose) + "\n"
           + "\n".join(frames))
    v = et.hermit(omn, budget_s=300)
    unsat = set(v.get("unsat") or [])
    return [h for h in heads if h in unsat]


def mechanical_rewrite(mt: str, anchor: str) -> "tuple[str, list[str]] | None":
    """2a: (subject×filler)-driven relation reassignment. → (new_manchester, notes) or None (→2b)."""
    if anchor != "bfo:0000023":                     # confident tier: role subjects only
        return None
    notes, out, ambiguous = [], mt, False
    for prop, quant, filler in _REST.findall(mt):
        occ, agent = bool(_OCC.search(filler)), bool(_AGENT.search(filler))
        want = None
        if _REALIZ.match(prop):
            if occ and not agent:
                want = "sdg:realizedIn"
            elif agent and not occ:
                want = "sdg:inheresIn"              # cross-swapped stem (RegisteredManagerRole)
            else:
                ambiguous = True
        elif _INHER.match(prop):
            if occ and not agent:
                want = "sdg:realizedIn"             # cross-swapped the other way
            elif agent and not occ:
                want = "sdg:inheresIn" if prop != "sdg:borneBy" else None   # borneBy already correct
            else:
                ambiguous = True
        if want and want != prop:
            out = out.replace(f"{prop} {quant} {{{filler}:Class}}",
                              f"{want} {quant} {{{filler}:Class}}")
            notes.append(f"{prop}→{want} ({filler})")
    if ambiguous:
        return None
    return (out, notes) if notes else None


def propose_2b(template_id: str, mt: str, reason: str, feedback: str) -> "dict | str":
    """One engine proposal. → candidate dict, or a rejection-reason string (schema/namespace)."""
    from aegir.engine.client import complete_detailed
    prompt = (f"MISCAST AXIOM (template `{template_id}`):\n{mt}\n\nWhy it is wrong: {reason}\n"
              + (f"\nYour previous attempt was REJECTED: {feedback}\nFix exactly that.\n" if feedback else "")
              + "\nRepair it (named-intermediate pattern where a realization targets a continuant). "
              "Every axiom head and every filler MUST use the {Name:Class} placeholder form.")
    try:
        r = complete_detailed(prompt, capability="instruct", system_prompt=_SYSTEM_2B,
                              max_tokens=1400, temperature=0.4, json_schema=_JUDGE_SCHEMA)
        v = json.loads(r.get("text") or "")
    except Exception as e:  # noqa: BLE001
        return f"engine/parse error: {str(e)[:80]}"
    # names may arrive in slot form ({X:Class}) — the prompt itself demands that form; normalize first
    bad = [c["name"] for c in v.get("new_classes", [])
           if not re.fullmatch(r"[A-Z][A-Za-z0-9]+", c.get("name", "").strip("{}").split(":")[0])
           or c.get("anchor", "").split(":")[0] not in ("bfo",)]
    if bad:
        return f"new classes must be sdg CamelCase anchored to bfo numerals: {bad}"
    return v


def main() -> int:
    cat_path = REPO / "src/aegir/ontology/catalog/catalog.json"
    cat = json.loads(cat_path.read_text())
    tpls = {t["template_id"]: t for t in cat["templates"]}
    ver = json.loads((REPO / "build/relation_signature_verify.json").read_text())
    lintv = {x["head"]: x for x in json.loads((REPO / "build/relation_lint.json").read_text())["violations"]}
    def snake(s): return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    # head → template by the MANCHESTER HEAD SLOT (template_id ≠ snake(head) for some — AcademicRole
    # lives in faculty_member_role; the naive lookup silently escalated those)
    by_head = {}
    for _t in cat["templates"]:
        _hm = re.search(r"Class:\s*\{(\w+):Class\}", _t.get("manchester_template") or "")
        if _hm:
            by_head.setdefault(_hm.group(1), _t)
    et = _et()
    backbone = _bro_backbone()

    only = None
    if "--only-escalated" in sys.argv:
        prev = json.loads((REPO / "build/reauthor_relations.json").read_text())
        only = {e["head"] for e in prev.get("escalated", [])}
        print(f"re-running ONLY the {len(only)} escalated heads", flush=True)

    swapped, authored, escalated, new_templates = [], [], [], []
    for head in sorted(ver["unsat"]):
        if only is not None and head not in only:
            continue
        t = by_head.get(head) or tpls.get(snake(head))
        if not t:
            escalated.append({"head": head, "reason": "template not found by head slot"})
            print(f"  ⇢ {head}: template NOT FOUND", flush=True)
            continue
        tid = t["template_id"]
        mt = t["manchester_template"]
        hm = _HEAD.search(mt)
        anchor = hm.group(2) if hm else "?"
        # 2a
        mech = mechanical_rewrite(mt, anchor)
        accepted = False
        attempts: "list[tuple[list[str], list[dict], str]]" = []
        if mech:
            attempts.append(([mech[0]], [], f"2a: {'; '.join(mech[1])}"))
        else:
            reason = lintv.get(head, {}).get("why", "domain/range miscast under BFO signatures")
            feedback = ""
            for _ in range(3):                          # inform-and-refine: EVERY membrane feeds back
                v = propose_2b(tid, mt, reason, feedback)
                if isinstance(v, str):
                    feedback = v
                    print(f"  · {head} round reject: {v[:110]}", flush=True)
                    continue
                mans = [f"Class: {{{ax['head'].strip('{}').split(':')[0]}:Class}}\n    SubClassOf: "
                        + ", ".join(ax.get("subclass_of", []))
                        for ax in v.get("axioms", []) if ax.get("subclass_of")]
                if not mans:
                    feedback = "no axioms with subclass_of parts"
                    print(f"  · {head} round reject: {feedback}", flush=True)
                    continue
                heads_chk = [re.search(r"\{(\w+):Class\}", m) for m in mans]
                if not all(heads_chk):
                    feedback = "axiom heads must use the {Name:Class} placeholder form"
                    print(f"  · {head} round reject: {feedback}", flush=True)
                    continue
                still = fragment_check(et, backbone, mans, [h.group(1) for h in heads_chk])
                if still:
                    feedback = (f"HermiT refutes {still} under the BFO signatures — the category "
                                f"assignment is still wrong; re-check domain/range of every relation")
                    print(f"  ✘ {head}: still unsat {still} — re-prompting", flush=True)
                    continue
                attempts.append((mans, v.get("new_classes", []), f"2b: {v.get('why', '')[:100]}"))
                break
        for mans, newcls, note in attempts:
            heads_in = [m.group(1) for m in (re.search(r"\{(\w+):Class\}", x) for x in mans) if m]
            still = fragment_check(et, backbone, mans, heads_in) if note.startswith("2a") else []
            if still:
                print(f"  ✘ {head}: candidate still unsat {still}", flush=True)
                continue
            # apply: first axiom replaces the template; extra axioms become new templates
            old = t["manchester_template"]
            t["manchester_template"] = mans[0]
            prov = t.setdefault("provenance", {})
            prov["reauthored"] = PROMPT_VERSION
            prov["previous_axiom"] = old
            # scrub stale verbalizations → regenerate from the pure recovery path
            parts = merge_manchester_restrictions(
                VerbalizationParts(subject="{" + heads_in[0] + "}"), mans[0])
            frames = compose_frames(parts)
            t["verbal_template"] = frames[0] if frames else ""
            t["verbal_templates"] = frames
            for extra_m, extra_head in list(zip(mans, heads_in))[1:]:
                new_templates.append({
                    "template_id": snake(extra_head), "manchester_template": extra_m,
                    "slot_types": {h: "Class" for h in re.findall(r"\{(\w+):Class\}", extra_m)},
                    "is_complex": False, "verbal_template": "", "verbal_templates": [],
                    "provenance": {"pattern": "reauthor_intermediate", "reauthored": PROMPT_VERSION,
                                   "parent_template": tid}})
            for c in newcls:
                pass                                    # definitions ride the new templates' axioms
            (swapped if note.startswith("2a") else authored).append({"head": head, "note": note})
            accepted = True
            print(f"  ✓ {head}: {note}", flush=True)
            break
        if not accepted:
            escalated.append({"head": head, "reason": "no candidate cleared the armed membrane"})
            print(f"  ⇢ {head}: ESCALATED", flush=True)

    # regenerate verbal for the new intermediate templates
    for nt in new_templates:
        parts = merge_manchester_restrictions(
            VerbalizationParts(subject="{" + re.search(r"\{(\w+):Class\}", nt["manchester_template"]).group(1) + "}"),
            nt["manchester_template"])
        fr = compose_frames(parts)
        nt["verbal_template"] = fr[0] if fr else ""
        nt["verbal_templates"] = fr
    cat["templates"].extend(new_templates)
    cat_path.write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    out = {"prompt_version": PROMPT_VERSION, "swapped": swapped, "authored": authored,
           "escalated": escalated, "new_templates": [n["template_id"] for n in new_templates]}
    (REPO / "build/reauthor_relations.json").write_text(json.dumps(out, indent=1))
    print(f"\n2a swapped {len(swapped)} · 2b authored {len(authored)} · escalated {len(escalated)} · "
          f"new intermediates {len(new_templates)} → catalog + build/reauthor_relations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
