"""reauthor_unsat.py — close the loop on the realize boundary's SIGNAL: re-author, don't shed.

build_realized_ontology's HermiT check (the BOUNDARY, [[signal_boundary_machinery]]) emits
``build/realize_signals.json`` — for each class it narrowed out of the domain, the minimal JUSTIFICATION
(axioms) + a legible cause (e.g. "realize a role in a Process/occurrent, not a Function/continuant").
This script is the agent-mediated RESPONSE: the engine RE-AUTHORS the offending conjunct of the source
template — guided by that signal — so the class becomes satisfiable against BFO + π(CCO), rather than being
silently dropped. Cogent signals managed with agency, not a last-resort bailout.

Two membranes DISPOSE (exactly as in define_intermediate_classes.py — the same agent-mediated pattern):
  • PARSE membrane (evolve_rigor.validate_detailed) — admits well-formed Manchester + returns the reason.
  • REASONING-AUTHORITY membrane (build_realized_ontology.consistency_check) — imports π(CCO), runs HermiT
    over the catalog with the re-authoring applied, so a rewrite that is STILL unsatisfiable (or newly
    clashes) is REJECTED. Rejects are re-prompted WITH the failure, up to --rounds. The agent RESPONDS to
    the reasoner. We narrow the domain of the offending CLASS (fix its conjunct); we never relax the theory.

    just engine-serve                       # the gRPC engine must be up (Qwen3.6)
    # first produce a signal (a realize that narrows a class writes build/realize_signals.json):
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/reauthor_unsat.py --rounds 4
    # then re-realize to confirm the loop closed (fewer/no narrowed classes):
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology.schema import load_catalog, save_catalog  # noqa: E402
from evolve_rigor import parse_slots, validate_detailed  # noqa: E402
from build_realized_ontology import SIGNALS_OUT, consistency_check  # noqa: E402

_HEAD = re.compile(r"Class:\s*\{(\w+)")

# the syntax rules the membrane enforces (mirrors define_intermediate_classes._SYNTAX — the engine's first
# pass historically violated all three; stating them up front saves a rejection round)
_SYNTAX = (
    "SYNTAX (axioms that break these FAIL to parse — the membrane rejects them):\n"
    "1. prefixes are LOWERCASE only: cco: bfo: fhir: sdg: — NEVER CCO: or BFO:.\n"
    "2. EVERY property is prefixed: write `sdg:realized_in some X`, never a bare `realized_in`.\n"
    "3. Use EXACT cco:/fhir: IRIs (never invent them); coin sdg: (camelCase/snake) only for genuinely new "
    "terms. A genus/filler must be a bfo: category, a real cco:/fhir: class, or a {Slot:Class}.\n"
)
_SYS = (
    "You are repairing a BFO 2020 / CCO ontology. Each class below is UNSATISFIABLE against the theory "
    "(BFO + a Common Core module): its axiom forces its instances to be empty. You are given the class's "
    "CURRENT axiom, the reasoner's minimal JUSTIFICATION (the exact axioms that collide), and a one-line "
    "diagnosis. RE-AUTHOR the axiom so the class is SATISFIABLE — keep the SAME head class, keep every "
    "conjunct that is fine, and fix ONLY the offending one.\n\n"
    "PRINCIPLE: narrow the domain of THIS class to what is logically coherent; do NOT weaken the theory. "
    "The usual fix is to make a filler's TYPE match the property's RANGE — e.g. if `realized_in` ranges over "
    "processes (occurrents) but the filler is a Function (a continuant), realize the role in a PROCESS filler "
    "(a bfo:0000015 subclass), or use a property whose range is a continuant (e.g. `bfo:0000178 has_continuant_part` / a "  # coined-ok: legacy/RO relation IRI — VERSION_DRIFT, resolved in-context by the sweep
    "`has_function` relation). Prefer a FAITHFUL type fix over deleting the conjunct; delete a conjunct only "
    "if no coherent repair preserves the class's meaning.\n\n" + _SYNTAX +
    "\nOUTPUT exactly one ```json block {\"repairs\":[{\"name\":\"<ClassName>\",\"manchester_template\":\"Class: "
    "{<ClassName>:<Type>} EquivalentTo: …\",\"verbal\":\"one-sentence gloss\",\"note\":\"what you changed and "
    "why\"}]}. The head slot MUST be the EXACT given class name. Reasoning OUTSIDE the json."
)


def load_signals(path: Path) -> "dict[str, dict]":
    """Read the boundary's signal record → {local_class_name: {"iri", "axioms", "why"}}, keyed by local
    name (the sdg IRI fragment) so it maps to a template head slot."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for iri, sig in raw.items():
        name = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        out[name] = {"iri": iri, "axioms": sig.get("axioms", []), "why": sig.get("why", "")}
    return out


def _head(t) -> "str | None":
    m = _HEAD.search(t.manchester_template or "")
    return m.group(1) if m else None


def map_targets(signals: "dict[str, dict]", cat) -> "tuple[dict, list[str]]":
    """Find, for each signalled class, the source template whose HEAD slot is that class (the directly
    authored axiom to repair). Returns ({name: CatalogTemplate}, [unmappable names]). An unmappable signal
    is a class with no defining template of its own (a bare filler / Phase-A grounding artifact) — a
    different lever (--strict-grounding), reported not repaired."""
    by_head = {}
    for t in cat.templates:
        h = _head(t)
        if h:
            by_head.setdefault(h, t)  # first template that heads this class
    targets, unmapped = {}, []
    for name in signals:
        if name in by_head:
            targets[name] = by_head[name]
        else:
            unmapped.append(name)
    return targets, unmapped


def _parse_repairs(text: str) -> list:
    m = re.search(r"```json\s*(.+?)```", text, re.S) or re.search(r"```\s*(\{.+?\})\s*```", text, re.S)
    blob = m.group(1) if m else None
    if not blob:
        m2 = re.search(r"\{[\s\S]*\"repairs\"[\s\S]*\}", text)
        blob = m2.group(0) if m2 else None
    if not blob:
        return []
    try:
        return json.loads(blob).get("repairs", [])
    except ValueError:
        return []


def _propose(chunk: "list[str]", targets: dict, signals: dict, feedback: dict, cap: str, temp: float,
             proposer=None) -> dict:
    """One proposal pass over ``chunk`` (class names) → {name: (manchester, verbal)}. Each item carries its
    current axiom + the boundary SIGNAL (why + justification); re-tries carry the prior verdict so the agent
    RESPONDS to it. ``proposer`` overrides the engine call (dependency injection for tests)."""
    lines = []
    for name in chunk:
        sig = signals[name]
        cur = (targets[name].manchester_template or "").strip()
        just = "\n      ".join(sig["axioms"][:8])
        block = (f"- {name}\n  current axiom: {cur}\n  diagnosis: {sig['why']}\n"
                 f"  reasoner justification (the colliding axioms):\n      {just}")
        if name in feedback:
            block += f"\n  PRIOR ATTEMPT REJECTED — {feedback[name]}  → fix and re-emit."
        lines.append(block)
    prompt = "Repair these unsatisfiable classes:\n\n" + "\n\n".join(lines)
    call = proposer or (lambda p: complete_detailed(p, capability=cap, system_prompt=_SYS,
                                                    max_tokens=12000, temperature=temp)["text"])
    text = call(prompt)
    proposed = {}
    for d in _parse_repairs(text):
        man = (d.get("manchester_template") or "").strip()
        hm = _HEAD.search(man)
        match = next((n for n in chunk if hm and n.lower() == hm.group(1).lower()), None)
        if match and "EquivalentTo" in man:
            proposed[match] = (man, d.get("verbal") or targets[match].verbal_template or f"A {match}.")
    return proposed


def _apply_and_check(targets: dict, cat, proposals: dict) -> "set[str]":
    """Apply ``proposals`` to a CANDIDATE view of the catalog, run the reasoning-authority membrane once,
    and return the set of proposed names that are now SATISFIABLE (no longer unsatisfiable vs the theory).
    Non-destructive: the real catalog is only mutated by the caller for accepted names."""
    cand = []
    for t in cat.templates:
        h = _head(t)
        if h in proposals:
            man, verbal = proposals[h]
            cand.append(dataclasses.replace(t, manchester_template=man, slot_types=parse_slots(man),
                                            verbal_template=verbal))
        else:
            cand.append(t)
    _consistent, unsat = consistency_check(cand)
    unsat_local = {u.rsplit("#", 1)[-1].rsplit("/", 1)[-1] for u in unsat}
    return {n for n in proposals if n not in unsat_local}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=str(SIGNALS_OUT), help="the boundary signal record (JSON)")
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--out", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4, help="max feedback rounds per class")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    signals = load_signals(Path(args.signals))
    if not signals:
        print(f"no boundary signals at {Path(args.signals).relative_to(REPO) if Path(args.signals).is_relative_to(REPO) else args.signals}"
              " — nothing to re-author (a clean realize emits none). Done.")
        return 0
    cat = load_catalog(args.catalog)
    targets, unmapped = map_targets(signals, cat)
    if args.limit:
        targets = dict(list(targets.items())[:args.limit])
    print(f"reauthor-unsat: {len(signals)} signalled class(es) · {len(targets)} mapped to a source template · "
          f"{len(unmapped)} unmappable (grounding artifacts — use --strict-grounding)")
    if unmapped:
        print(f"   unmapped: {unmapped[:12]}")
    if not targets:
        print("   nothing mappable to re-author. Done.")
        return 0

    accepted: set[str] = set()
    feedback: dict[str, str] = {}
    for rnd in range(args.rounds):
        pending = [n for n in targets if n not in accepted]
        if not pending:
            break
        proposed: dict[str, tuple] = {}
        for bi in range(0, len(pending), args.batch):
            chunk = pending[bi:bi + args.batch]
            try:
                proposed.update(_propose(chunk, targets, signals, feedback, args.capability, args.temperature))
            except Exception as e:  # noqa: BLE001 — an engine error must not abort the round
                print(f"  round {rnd + 1} batch @{bi}: engine error {type(e).__name__}: {str(e)[:80]}")
        # PARSE membrane — admit well-formed Manchester, capture the reason for rejects (the feedback channel)
        prov = {n: dataclasses.replace(targets[n], manchester_template=man, slot_types=parse_slots(man),
                                       verbal_template=verbal)
                for n, (man, verbal) in proposed.items()}
        detailed = validate_detailed([(n, t.manchester_template) for n, t in prov.items()], prov) if prov else {}
        parse_ok = {}
        for n, (man, verbal) in proposed.items():
            ok, reason = detailed.get(n, (False, "not validated"))
            if ok:
                parse_ok[n] = (man, verbal)
            else:
                feedback[n] = f"`{man}` → {reason}"
        # REASONING-AUTHORITY membrane — HermiT over the catalog with the repairs applied; a class still
        # unsatisfiable is re-prompted WITH its signal (the agent responds to the reasoner).
        fixed = _apply_and_check(targets, cat, parse_ok) if parse_ok else set()
        n_new = 0
        for n, (man, verbal) in parse_ok.items():
            if n in fixed:
                tgt = targets[n]  # mutate in place → updates cat.templates
                tgt.manchester_template, tgt.slot_types, tgt.verbal_template = man, parse_slots(man), verbal
                accepted.add(n)
                feedback.pop(n, None)
                n_new += 1
            else:
                feedback[n] = f"`{man}` STILL unsatisfiable vs the theory — {signals[n]['why']}"
        print(f"  round {rnd + 1}/{args.rounds}: proposed {len(proposed)} · parse-ok {len(parse_ok)} · "
              f"+{n_new} repaired · {len(accepted)}/{len(targets)} total · {len(feedback)} pending")

    save_catalog(cat, args.out)
    print(f"\nAPPLIED: {len(accepted)}/{len(targets)} classes re-authored to satisfiability (parse + HermiT "
          f"membranes) → {Path(args.out).name}")
    still = sorted(set(targets) - accepted)
    if still:
        print(f"UNREPAIRED after {args.rounds} rounds ({len(still)}): {still[:12]} — these narrow the domain "
              "on the next realize (or raise --rounds / inspect the signal).")
    print("DISPOSE next: build_realized_ontology.py → confirm fewer/no narrowed classes (loop closed)")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
