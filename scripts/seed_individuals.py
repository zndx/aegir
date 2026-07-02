#!/usr/bin/env python
"""seed_individuals.py — in-loop, per-domain individual seeding (Convert 1b, [[convert_priority_cas]]).

The loop-closed successor of ``seed_entity_values.py`` (whose proven bones — column selection, VALUES
sentinel contract, parse — it reuses). What converts:

  * **static → in-loop**: values are derived per (domain, template) against the CURRENT catalog, not
    frozen into a global pool. The registry (``individual_registry.json``) accretes admitted survivors —
    accretion is inheritance; a frozen pool is pre-wiring.
  * **one-shot → membrane loop**: :func:`individuals.check_column` DISPOSES with a REASON (generic filler,
    mechanical stems, near-duplicates); rejected columns are re-prompted WITH the reason, up to --rounds —
    the agent-mediated feedback pattern ([[agent_mediated_feedback_loop]]).
  * **strings → individuals**: Class-typed columns' admitted values are recorded with the class they
    instantiate (``entity_classes``) so ``build_realized_ontology`` emits them as OWL ``Individual:``
    frames — the ontology INSTANTIATED, HermiT certifying instance-level consistency.
  * **provenance stamped** (convert contract clause c): every record carries {domain, run, model, rounds,
    membrane verdicts} — the authenticity audit travels with the artifact.

    just engine-serve   # the engine must be up (local Qwen3.6; not gated)
    uv run --no-sync python scripts/seed_individuals.py --limit 20 --rounds 3   # cohort
    uv run --no-sync python scripts/seed_individuals.py --rounds 3              # full catalog (multi-hour)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.ontology import ddl as D  # noqa: E402
from aegir.ontology import individuals as IND  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402
from seed_entity_values import _SYSTEM, _concept, _prompt, parse_response, seed_columns  # noqa: E402

_TRACES = REPO / "build" / "individual_seed_traces.jsonl"
_ENTITY_OWL = {"Class", "Individual", "NamedIndividual"}


def entity_col_classes(template, family: str) -> "dict[str, str]":
    """{column → class local-name} for the template's Class-typed columns — the slot name IS the class
    under the realizer's slot→global-IRI unification, so admitted values become its individuals."""
    table = D.template_to_table(template, family)
    out = {}
    for c in table.table.columns:
        if c.slot_type in _ENTITY_OWL and c.slot_ref not in ("__pk__",) and not c.slot_ref.startswith("data:"):
            out[c.name] = c.slot_ref
    return out


def seed_template(template, family: str, dp_meta: dict, *, rounds: int, capability: str,
                  temperature: float, trace_fh) -> "tuple[dict, dict, dict]":
    """The membrane loop for one template → (admitted {col: values}, final feedback {col: reason},
    stats). Rejected columns are re-prompted WITH the membrane's reason each round."""
    from aegir.engine.client import complete_detailed
    cols = seed_columns(template, family, dp_meta)
    if not cols:
        return {}, {}, {}
    concept = _concept(template)
    verbalization = (template.frames()[0] if template.frames() else template.verbal_template) or ""
    hints = dict(cols)
    pending = list(hints)
    admitted: dict[str, list] = {}
    feedback: dict[str, str] = {}
    model = ""
    for rnd in range(rounds):
        if not pending:
            break
        ask = [(nm, hints[nm] + (f" [PRIOR ATTEMPT REJECTED — {feedback[nm]} — fix and re-emit]"
                                 if nm in feedback else "")) for nm in pending]
        out = complete_detailed(_prompt(concept, verbalization, ask), capability=capability,
                                system_prompt=_SYSTEM, max_tokens=8192, temperature=temperature)
        model = out.get("model") or model
        pools = parse_response(out["text"], set(pending))
        trace_fh.write(json.dumps({"template_id": template.template_id, "round": rnd + 1,
                                   "asked": [nm for nm, _ in ask], "parsed": {k: len(v) for k, v in pools.items()},
                                   "reasoning_content": out.get("reasoning_content", "")[:2000]}) + "\n")
        for nm in list(pending):
            vals = pools.get(nm)
            if not vals:
                feedback[nm] = "no usable 'VALUES <col>:' line parsed — follow the output contract exactly"
                continue
            ok, reason, kept = IND.check_column(nm, vals)
            if ok:
                admitted[nm] = kept
                pending.remove(nm)
                feedback.pop(nm, None)
            else:
                feedback[nm] = reason
    return admitted, feedback, {"model": model, "n_cols": len(cols)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/08_derived.json"))
    ap.add_argument("--rounds", type=int, default=3, help="max membrane feedback rounds per template")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N uncovered templates")
    ap.add_argument("--refresh", action="store_true", help="re-seed templates already covered by the registry")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    cat = load_catalog(args.catalog)
    family = Path(args.catalog).stem
    dp_meta = D.dataprop_meta()
    reg = IND.load_registry()
    run_stamp = datetime.date.today().isoformat()

    todo = []
    for t in cat.templates:
        if not args.refresh and (reg.get("templates", {}).get(t.template_id, {}).get("columns")):
            continue
        todo.append(t)
    if args.limit:
        todo = todo[:args.limit]
    print(f"seed-individuals: {len(todo)} templates to seed (registry covers "
          f"{len(reg.get('templates', {}))}) · membrane loop ≤{args.rounds} rounds")

    _TRACES.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = _TRACES.open("a")
    n_admitted = n_rejected = 0
    reject_reasons: Counter = Counter()
    for i, t in enumerate(todo):
        dom = (t.provenance or {}).get("domain") or {}
        dom_label = dom.get("label", "") if isinstance(dom, dict) else str(dom)
        try:
            admitted, feedback, stats = seed_template(t, family, dp_meta, rounds=args.rounds,
                                                      capability=args.capability,
                                                      temperature=args.temperature, trace_fh=trace_fh)
        except Exception as e:  # noqa: BLE001 — engine error: report, keep going
            print(f"  [{t.template_id}] engine error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            continue
        if admitted:
            ent = entity_col_classes(t, family)
            IND.admit(reg, t.template_id,
                      columns=admitted,
                      entity_classes={c: ent[c] for c in admitted if c in ent},
                      domain=dom_label,
                      provenance={"run": run_stamp, "model": stats.get("model", ""),
                                  "membrane": {c: "ok" for c in admitted} | dict(feedback),
                                  "source": "seed_individuals"})
            IND.save_registry(reg)  # accrete incrementally — an interrupt loses nothing
        n_admitted += len(admitted)
        n_rejected += len(feedback)
        for r in feedback.values():
            reject_reasons[r.split(" — ")[0][:60]] += 1
        if i < 6 or args.limit:
            print(f"[{t.template_id}] ({dom_label or 'no-domain'}) +{len(admitted)} col(s)"
                  + (f" · {len(feedback)} unresolved: {list(feedback)[:3]}" if feedback else ""))

    trace_fh.close()
    print(f"\nADMITTED {n_admitted} columns · {n_rejected} unresolved after {args.rounds} rounds "
          f"· registry now covers {len(reg.get('templates', {}))} templates, "
          f"{IND.n_individuals(reg)} distinct individuals")
    if reject_reasons:
        print("membrane signals (rejection reasons):")
        for r, n in reject_reasons.most_common(6):
            print(f"   {n:>3}× {r}")
    print("DISPOSE next: build_ddl_spine (registry-fed cells) · build_realized_ontology (ABox → HermiT)")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
