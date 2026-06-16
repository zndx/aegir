"""H₀ — the seed harness (meta-harness inc-2b).

A SINGLE-FILE harness program in the Meta-Harness sense (Lee, Khattab, Finn 2026,
`build/resources/2603.28052.pdf`). It wraps two FROZEN executors — Grok (raw
completion over ACP) and the verifier/reasoner stack (the `gate`) — and is the ONLY
artifact the outer-loop proposer (inc-2c) rewrites. So everything the proposer may
mutate lives HERE, in one readable file, with no hidden control flow behind imports:

  • build_prompt   — the mint-prompt STRATEGY (contract membrane + topic salient
                     terms + family exemplars + gate feedback). This is the main
                     surface the paper says harnesses optimize.
  • next_objective — the OBJECTIVE-SELECTION control logic: a flat transcription of
                     the RETE/FSM seed_rules priority order in `fsm_rete.py`. That
                     spine remains the tested reference; this is its legible, mutable
                     distillation (the paper beats fixed hand-designed scaffolds, so
                     H₀ exposes the logic as plain code the proposer can rewrite).
  • run            — the iterate loop (mint → gate → reorient → repeat / promote).

The frozen executor (`complete`) and the parser (`parse_templates`) are NOT mutated
(they are the model + I/O plumbing), so they are imported / injected, not inlined.
The `gate` is the FIXED reward/membrane, passed in. This keeps H₀ self-contained for
the proposer while honoring the frozen-executor boundary.

Interface contract (a proposed harness MUST preserve this):

    run(topic, gate, complete, exemplars, max_iters=6, log=None) -> dict
      topic     : coverage-audit topic dict (topic_repr_text, _top_terms, top_family,
                  topic_id)
      gate      : (construct:dict, topic:dict) -> signal dict        [FROZEN reward]
      complete  : (prompt:str) -> str                                [FROZEN Grok]
      exemplars : {family: [CatalogTemplate]}  (fixed catalog material)
      returns   : {outcome:'promote'|'give_up', iters, construct, signals, history,
                   objectives}
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import generate_ontology as go  # noqa: E402  (SLOT_DSL, ANCHORS, parse_templates — frozen infra)

# Signal vector for a failed/empty mint — drives a bounded retry (not a fake pass).
_FAIL_SIG = {"has_construct": True, "deeponto_ok": False, "deeponto_complex": False,
             "consistent": False, "polyglot_ok": False, "novelty_ok": False,
             "schema_ok": False, "r1_on": 0.0, "r1_ci_low": -1.0, "n_cols": 0}

_OBJ_HINT = {
    "enrich_domain_terms": "Your terms are too generic — R1 is low. REWRITE with topic-specific "
                           "minted class/property names drawn from the salient terms.",
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
    """The mint-prompt strategy (mutable surface). Mirrors inc-1's
    mediate_acp.build_mint_prompt; the proposer diverges this copy."""
    import json
    src = (topic.get("topic_repr_text") or "")[:2500]
    top_terms = topic.get("_top_terms") or []
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
        "  • R1 (the binding one): its DOMAIN TERMS must match THIS topic's salient terms. "
        "R1 scores your minted class/property/slot names against the topic vocabulary below. "
        "Generic names (Article, Document, Person, hasName) score ~0 and FAIL. Mint "
        "TOPIC-SPECIFIC concepts (e.g. for herbal medicine: MedicinalPlant, hasActiveCompound, "
        "AntiInflammatoryEffect).\n"
        f"  TOPIC SALIENT TERMS (match these): {', '.join(sorted(top_terms)[:25])}\n"
    )
    if objective in ("draft_initial",) or not prev:
        task = ("Author ONE new Manchester axiom template capturing the SPECIFIC domain concepts of "
                "the SOURCE TEXT below, grounded in BFO/CCO, with topic-specific class/property names.")
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
        f"{go.SLOT_DSL}\n\n{contract}\n"
        f"Family exemplars (imitate shape + grounding):\n{ex}\n\n"
        f"{task}\n\n"
        f'SOURCE TEXT (real document from the target topic):\n"""{src}"""\n\n'
        "Output ONLY a fenced ```json block with a JSON list containing ONE object: "
        '{"template_id": str (lowercase_snake, descriptive of the DOMAIN), "manchester_template": str, '
        '"slot_types": {slot: "Class|ObjectProperty|DataProperty|Individual"}, "bfo_anchor_path": [iri]}. '
        f"bfo_anchor_path must end at one of {sorted(go.ANCHORS)}."
    )


def next_objective(s: dict) -> str:
    """Flat transcription of fsm_rete.seed_rules, in salience order. Returns the
    next objective, or 'PROMOTE' when the full conjunctive contract is satisfied.
    (Budget/give-up is handled by run's loop bound — see the FSM's budget_exhausted
    at salience 110, just under contract_satisfied at 120.)"""
    g = lambda k, d=False: s.get(k, d)  # noqa: E731
    if (g("has_construct") and g("deeponto_ok") and g("deeponto_complex") and g("consistent")
            and g("polyglot_ok") and g("novelty_ok") and g("schema_ok") and s.get("r1_ci_low", -1) > 0):
        return "PROMOTE"                                # contract_satisfied (120)
    if not g("has_construct"):
        return "draft_initial"                          # no_construct (100)
    if not g("deeponto_ok"):
        return "fix_verbalizability"                    # deeponto_fail (90)
    if not g("consistent"):
        return "fix_consistency"                        # inconsistent (89)  [HermiT]
    if not g("deeponto_complex"):
        return "fix_nontriviality"                      # not_complex (88)
    if not g("polyglot_ok"):
        return "fix_ddl"                                # polyglot_fail (80)
    if s.get("r1_ci_low", -1) <= 0:
        return "enrich_domain_terms"                    # r1_not_specific (70)
    if not g("novelty_ok"):
        return "diversify"                              # novelty_low (60)
    if not g("schema_ok"):
        return "de_can"                                 # schema_canned (50)
    return "refine"                                     # fallthrough (unreached when contract holds)


def _parse_construct(resp: str, topic: dict, objective: str, prev: dict | None) -> dict:
    """Parse Grok's fenced-JSON response into a construct dict, or fall back to prev."""
    templates = go.parse_templates(resp)
    if not templates:
        return prev or {"template": None, "template_id": None, "raw_len": len(resp)}
    from dataclasses import asdict
    fam = topic.get("top_family") or "07_long_tail"
    t = templates[0]
    t.provenance = {"generated_from_topic": str(topic.get("topic_id")), "family": fam,
                    "model": "grok-acp", "objective": objective, "harness": "h0"}
    return {"template": t, "template_id": t.template_id, "template_dict": asdict(t),
            "raw_len": len(resp)}


def run(topic: dict, gate, complete, exemplars: dict, max_iters: int = 6, log=None) -> dict:
    """The H₀ control loop. mint (build_prompt → complete → parse) → gate → reorient
    via next_objective → repeat until promote or budget. Returns the run record."""
    fam = topic.get("top_family") or "07_long_tail"
    ex = exemplars.get(fam, [])
    construct, signals, prev = None, {"has_construct": False}, None
    objective = next_objective(signals)                 # 'draft_initial'
    history, objectives = [], []
    outcome = "give_up"
    for it in range(1, max_iters + 1):
        objectives.append(objective)
        prompt = build_prompt(topic, objective, signals, prev, ex)
        resp = complete(prompt)                         # FROZEN Grok
        construct = _parse_construct(resp, topic, objective, prev) if resp else (prev or {"template": None})
        if not construct or construct.get("template") is None:
            signals = {**_FAIL_SIG, "iterations": it}
        else:
            signals = {**gate(construct, topic), "has_construct": True, "iterations": it}  # FROZEN reward
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
