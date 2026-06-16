"""H₁ — the first DISCOVERED harness (meta-harness inc-2c).

Proposed by the Opus coding-agent proposer (the Meta-Harness proposer is Claude
Code/Opus; Lee et al. 2026) reading candidates/000/'s harness + traces + scores on
the SEARCH set only. Diagnosis from those traces: H₀ promotes every search topic in
ONE mint (cost is already at the floor), so the only headroom is per-construct R1 =
weighted-Jaccard(construct terms, topic TF-IDF terms). H₀'s constructs carried rich
template_ids but under-mined the SLOT names (R1 0.26/0.14/0.13). The change is
concentrated in build_prompt (the mutable surface; next_objective + run + the
frozen-executor boundary are byte-identical to H₀):

  1. show the FULL topic salient-term signature (H₀ capped at 25) — more of V_t to hit;
  2. require EVERY slot (class / object-property / data-property) be named from those
     salient terms, and mint 6-8 of them (≥3 data columns) — construct_terms tokenizes
     slot names, so this directly grows |V_c ∩ V_t|;
  3. surface the topic's NEAREST existing catalog template as a structural scaffold.

This is legitimate optimization, not gaming: the conjunctive contract (DeepOnto
verbalize+complex, HermiT coherence, polyglot DDL, novelty, schema) still gates every
construct, and the held-out topics (never seen by the proposer) are the honest test.

Interface identical to H₀: run(topic, gate, complete, exemplars, max_iters=6, log=None).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import generate_ontology as go  # noqa: E402  (SLOT_DSL, ANCHORS, parse_templates — frozen infra)

_FAIL_SIG = {"has_construct": True, "deeponto_ok": False, "deeponto_complex": False,
             "consistent": False, "polyglot_ok": False, "novelty_ok": False,
             "schema_ok": False, "r1_on": 0.0, "r1_ci_low": -1.0, "n_cols": 0}

_OBJ_HINT = {
    "enrich_domain_terms": "Your terms are too generic — R1 is low. REWRITE so EVERY slot name reuses "
                           "words from the salient-terms list (compound 2 of them per name).",
    "fix_verbalizability": "DeepOnto could not verbalize it — fix the Manchester syntax/IRIs.",
    "fix_nontriviality": "It is a bare subsumption — add a restriction (e.g. `{p:ObjectProperty} some {Y:Class}`).",
    "fix_consistency": "HermiT found it INCOHERENT (a named class is unsatisfiable — usually anchored "
                       "across DISJOINT BFO categories, e.g. Process ⊓ Continuant). Re-anchor to ONE category.",
    "fix_ddl": "Its DDL does not parse — simplify to a clean typed table shape.",
    "diversify": "It duplicates a seed — make it materially different.",
    "de_can": "Too few/uniform columns — add typed DataProperty slots (values/dates/counts).",
    "refine": "Refine it to fully satisfy the contract.",
}


def build_prompt(topic: dict, objective: str, feedback: dict, prev: dict | None,
                 exemplars: list) -> str:
    """H₁'s mint-prompt strategy: maximize V_c ∩ V_t by naming every slot from the
    full salient-term signature, with a nearest-template structural scaffold."""
    import json
    src = (topic.get("topic_repr_text") or "")[:2500]
    top_terms = sorted(topic.get("_top_terms") or [])
    nearest = topic.get("top_template_id") or ""
    ex = "\n".join(
        f"- {t.template_id}: {t.manchester_template}  (anchor {t.bfo_anchor_path[-1:]})"
        for t in exemplars[:2])
    contract = (
        "CONTRACT (your construct must pass ALL — this is a conjunctive membrane):\n"
        "  • DeepOnto verbalizes it AND it asserts a COMPLEX class (a restriction/intersection — "
        "not a bare `X SubClassOf: anchor`).\n"
        "  • LOGICALLY COHERENT (HermiT, sound+complete): no class unsatisfiable — do NOT anchor across "
        "DISJOINT BFO categories (e.g. a Process is an Occurrent and CANNOT also be a Continuant/Artifact/ICE).\n"
        "  • It lowers to a valid relational table (≥3 typed columns; Trino+Spark-parseable DDL).\n"
        "  • NOVEL (not a near-duplicate of the seed catalog).\n"
        "  • R1 (the binding one): R1 = Jaccard(your minted term set, the topic's salient terms). Your "
        "minted term set = the words tokenized out of your template_id AND EVERY slot name. So EACH slot "
        "name you choose is scored. Generic names (Article, Document, Person, hasName, X, Y, p) score ~0 "
        "and FAIL. Cover as many salient terms as you can across your slot names.\n"
        f"  TOPIC SALIENT TERMS — reuse these EXACT words in your minted names (more coverage ⇒ higher R1):\n"
        f"    {', '.join(top_terms)}\n"
    )
    scaffold = (f"NEAREST EXISTING CATALOG TEMPLATE to this topic (imitate its slot STRUCTURE, then "
                f"refill every slot with the salient terms above): {nearest}\n" if nearest else "")
    if objective in ("draft_initial",) or not prev:
        task = ("Author ONE new Manchester axiom template for the SOURCE TEXT's domain, grounded in "
                "BFO/CCO. Name EVERY slot — the head Class, each ObjectProperty, and each DataProperty — "
                "with a TOPIC-SPECIFIC term built from the salient-terms list (compound 2 salient words, "
                "e.g. `MedicinalHerbalPlant`, `hasActiveCompound`, `antiInflammatoryEffect`). Mint 6-8 "
                "such slots including ≥3 DataProperty columns (values / dates / counts). Use NO generic "
                "placeholders (X/Y/p) and NO generic words — they score 0 on R1.")
    else:
        pv = json.dumps({k: v for k, v in (prev.get("template_dict") or {}).items()
                         if k in ("template_id", "manchester_template", "slot_types", "bfo_anchor_path")})
        fails = ", ".join(f"{k}={feedback.get(k)}" for k in
                          ("deeponto_ok", "deeponto_complex", "consistent", "polyglot_ok", "novelty_ok",
                           "schema_ok", "r1_on", "r1_ci_low") if k in feedback)
        hint = _OBJ_HINT.get(objective, "Refine it to satisfy the contract.")
        task = (f"Your PREVIOUS construct was:\n{pv}\nGate feedback: {fails}\n"
                f"OBJECTIVE [{objective}]: {hint}\nReturn an improved single construct.")
    return (
        "You are an ontology engineer extending a BFO 2020 / CCO grounded ontology.\n\n"
        f"{go.SLOT_DSL}\n\n{contract}\n{scaffold}"
        f"Family exemplars (imitate shape + grounding):\n{ex}\n\n"
        f"{task}\n\n"
        f'SOURCE TEXT (real document from the target topic):\n"""{src}"""\n\n'
        "Output ONLY a fenced ```json block with a JSON list containing ONE object: "
        '{"template_id": str (lowercase_snake, descriptive of the DOMAIN), "manchester_template": str, '
        '"slot_types": {slot: "Class|ObjectProperty|DataProperty|Individual"}, "bfo_anchor_path": [iri]}. '
        f"bfo_anchor_path must end at one of {sorted(go.ANCHORS)}."
    )


# ── below this line: byte-identical to H₀ (the proposer changed only build_prompt) ──
def next_objective(s: dict) -> str:
    g = lambda k, d=False: s.get(k, d)  # noqa: E731
    if (g("has_construct") and g("deeponto_ok") and g("deeponto_complex") and g("consistent")
            and g("polyglot_ok") and g("novelty_ok") and g("schema_ok") and s.get("r1_ci_low", -1) > 0):
        return "PROMOTE"
    if not g("has_construct"):
        return "draft_initial"
    if not g("deeponto_ok"):
        return "fix_verbalizability"
    if not g("consistent"):
        return "fix_consistency"
    if not g("deeponto_complex"):
        return "fix_nontriviality"
    if not g("polyglot_ok"):
        return "fix_ddl"
    if s.get("r1_ci_low", -1) <= 0:
        return "enrich_domain_terms"
    if not g("novelty_ok"):
        return "diversify"
    if not g("schema_ok"):
        return "de_can"
    return "refine"


def _parse_construct(resp: str, topic: dict, objective: str, prev: dict | None) -> dict:
    templates = go.parse_templates(resp)
    if not templates:
        return prev or {"template": None, "template_id": None, "raw_len": len(resp)}
    from dataclasses import asdict
    fam = topic.get("top_family") or "07_long_tail"
    t = templates[0]
    t.provenance = {"generated_from_topic": str(topic.get("topic_id")), "family": fam,
                    "model": "grok-acp", "objective": objective, "harness": "h1"}
    return {"template": t, "template_id": t.template_id, "template_dict": asdict(t),
            "raw_len": len(resp)}


def run(topic: dict, gate, complete, exemplars: dict, max_iters: int = 6, log=None) -> dict:
    fam = topic.get("top_family") or "07_long_tail"
    ex = exemplars.get(fam, [])
    construct, signals, prev = None, {"has_construct": False}, None
    objective = next_objective(signals)
    history, objectives = [], []
    outcome = "give_up"
    for it in range(1, max_iters + 1):
        objectives.append(objective)
        prompt = build_prompt(topic, objective, signals, prev, ex)
        resp = complete(prompt)
        construct = _parse_construct(resp, topic, objective, prev) if resp else (prev or {"template": None})
        if not construct or construct.get("template") is None:
            signals = {**_FAIL_SIG, "iterations": it}
        else:
            signals = {**gate(construct, topic), "has_construct": True, "iterations": it}
        history.append(signals.get("r1_on"))
        if log:
            log({"event": "iter", "iter": it, "objective": objective,
                 "construct_id": (construct or {}).get("template_id"),
                 "r1_on": signals.get("r1_on"), "r1_ci_low": signals.get("r1_ci_low"),
                 "deeponto_ok": signals.get("deeponto_ok"), "consistent": signals.get("consistent"),
                 "polyglot_ok": signals.get("polyglot_ok"), "novelty_ok": signals.get("novelty_ok"),
                 "schema_ok": signals.get("schema_ok")})
        objective = next_objective(signals)
        if objective == "PROMOTE":
            outcome = "promote"
            break
        prev = construct
    return {"outcome": outcome, "iters": len(history), "construct": construct,
            "signals": signals, "history": history, "objectives": objectives}
