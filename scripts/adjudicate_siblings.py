"""Adjudicate sibling pairs — the closed loop for the sibling-adjudication objective (RH 2026-07-09).

For every ADJUDICABLE family (siblings sharing a SPECIFIC genus — CCO/sdg, or a BFO
realizable; see ``aegir.ontology.adjudication``), the agent nominates the DISJOINT pairs
over an explicit overlap-by-default (roles co-inhere: one bearer may hold two roles —
blanket disjointness is the anti-pattern the adjudication objective replaced). Membranes
dispose, with reasons:

  M-A shape/membership  — nominated pairs must be family-internal, well-formed, rationaled.
  M-B ABox refutation   — a nominated-disjoint pair with a CO-TYPED INDIVIDUAL in the
                          realized ABox is refuted DETERMINISTICALLY, the individual named
                          (the 10,570-individual artifact is ground truth against
                          overzealous disjointness).
  M-C HermiT (batch)    — accepted nominations are appended to the realized OMN as
                          DisjointClasses frames and the WHOLE theory re-reasoned once per
                          round; any unsat names its offending pair(s), which are rejected
                          with that evidence and re-prompted.

Accepted verdicts land in ``src/aegir/ontology/catalog/adjudications.json`` (family
records: members, default=overlap, disjoint_pairs+rationales, provenance). The realizer
emits the DisjointClasses frames from there on every future build — adjudications stay
inside the HermiT boundary forever. Kvasir reads them natively from the .omn.

  uv run --no-sync python scripts/adjudicate_siblings.py --limit-families 4 --dry-run  # smoke
  uv run --no-sync python scripts/adjudicate_siblings.py                               # full
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology import adjudication as ADJ  # noqa: E402

OWL_PATH = REPO / "corpora" / "ontology" / "sdg-ontology.owl"
OMN_PATH = REPO / "corpora" / "ontology" / "sdg-ontology.omn"

_SYS = """You adjudicate sibling relationships among ontology classes that share a genus.
The DEFAULT verdict for every pair is OVERLAP-PERMITTED (an individual may instantiate both —
roles co-inhere, records can satisfy several descriptions). Your job is to nominate ONLY the
pairs that are GENUINELY DISJOINT: no possible individual could be a member of both, as a matter
of what the classes MEAN (identity-incompatible kinds), not as a matter of rarity.

Be conservative: a wrong disjointness is a false logical claim that a reasoner will enforce
against real data forever; a missed one is merely a weaker theory. When in doubt, do not
nominate. For each nomination give a one-sentence rationale naming the incompatibility.
Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "families": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "genus": {"type": "string"},
                    "disjoint_pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                                           "rationale": {"type": "string"}},
                            "required": ["a", "b", "rationale"],
                        },
                    },
                },
                "required": ["genus", "disjoint_pairs"],
            },
        }
    },
    "required": ["families"],
}


def _abox_refuter(owl_path: Path):
    """M-B, closure-aware: returns ``refute(a, b) -> reason|None``.

    Refutes a nominated disjointness when (i) one class sits in the other's entailed
    descendant closure (siblings that SUBSUME can't be disjoint), or (ii) any individual
    is typed — directly or via the closure — in BOTH classes (an inferred co-member;
    asserting the disjointness would make the whole theory inconsistent, which HermiT
    reports globally in seconds but without pair-level names — this membrane names the
    pair AND the witness)."""
    import rdflib
    from rdflib import OWL, RDF, URIRef
    g = rdflib.Graph()
    g.parse(str(owl_path))
    desc = ADJ.descendant_closure(graph=g)
    # individual -> asserted sdg types (locals)
    typed: dict[str, set] = {}
    for i in g.subjects(RDF.type, OWL.NamedIndividual):
        ts = {str(c).split("#")[-1] for c in g.objects(i, RDF.type)
              if isinstance(c, URIRef) and str(c).startswith(ADJ.SDG)}
        if ts:
            typed[str(i).split("#")[-1]] = ts

    def refute(a: str, b: str) -> "str | None":
        da, db = desc.get(a, {a}), desc.get(b, {b})
        if b in da or a in db:
            return f"({a},{b}) — one entails the other via the genus lattice (subsumption, not disjointness)"
        for ind, ts in typed.items():
            if ts & da and ts & db:
                return (f"({a},{b}) REFUTED — individual `{ind}` is an inferred co-member "
                        f"(typed {sorted(ts & da)[0]} ⊑ {a} and {sorted(ts & db)[0]} ⊑ {b})")
        return None

    return refute


def _labels_for(members: "list[str]") -> str:
    return "\n".join(f"  - {m}" for m in members)


def _propose(fam_batch: "list[tuple[str, list[str]]]", feedback: "dict[str, str]") -> dict:
    blocks = []
    for genus, members in fam_batch:
        b = f"GENUS: {genus}\nMEMBERS ({len(members)}):\n{_labels_for(members)}"
        if genus in feedback:
            b += f"\nPRIOR ROUND REJECTED NOMINATIONS — {feedback[genus]}  → re-nominate accordingly."
        blocks.append(b)
    out = complete_detailed(
        "Adjudicate these sibling families (nominate ONLY genuinely-disjoint pairs; "
        "everything un-nominated is overlap-permitted by default):\n\n" + "\n\n".join(blocks),
        capability="instruct", system_prompt=_SYS, json_schema=json.dumps(SCHEMA),
        max_tokens=4096)
    try:
        return json.loads(out["text"])
    except Exception:  # noqa: BLE001
        return {"families": []}


def _hermit_check(disjoint_frames: "list[str]") -> "tuple[bool, list[str]]":
    """M-C: append the frames to the realized OMN, reason once. Returns (ok, unsat_locals).

    PARSE GUARD: a degraded parse (OWLAPI's silent multi-parser fallback → empty
    ontology) must be RUN-FATAL, never a vacuous pass — the membrane refuses to
    testify about a document it did not actually load."""
    from aegir.ontology.deeponto_harness import ensure_jvm
    ensure_jvm()   # BEFORE any deeponto import — else click.prompt() aborts headless runs
    from build_realized_ontology import _reason  # the realize boundary's own machinery
    doc = OMN_PATH.read_text().rstrip() + "\n\n" + "\n".join(disjoint_frames) + "\n"
    onto, tmp, consistent, n_classes, unsat, _why = _reason(doc)
    del onto, tmp
    if n_classes < 100:   # the realized theory is hundreds of classes; ~0 = parse fallback
        raise RuntimeError(
            f"M-C PARSE DEGRADATION: the reasoned document loaded only {n_classes} classes "
            f"— OWLAPI fell back on a Manchester parse error (check the appended frames); "
            f"refusing the vacuous verdict")
    return (bool(consistent) and not unsat,
            [str(u).split("#")[-1].split(".")[-1] for u in (unsat or [])])


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--families-per-call", type=int, default=6)
    ap.add_argument("--limit-families", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="don't save adjudications.json")
    ap.add_argument("--no-hermit", dest="hermit", action="store_false",
                    help="skip the batch HermiT membrane (M-A/M-B only)")
    a = ap.parse_args()

    universe = ADJ.sibling_families(OWL_PATH)
    fams = sorted(universe["adjudicable"].items(), key=lambda kv: -len(kv[1]))
    if a.limit_families:
        fams = fams[: a.limit_families]
    refute = _abox_refuter(OWL_PATH)
    print(f"universe: {universe['universe_pairs']} adjudicable pairs across {len(fams)} families "
          f"(grounding debt excluded: {universe['debt_pairs']} pairs) · "
          f"closure-aware ABox refuter armed", flush=True)

    adj = ADJ.load_adjudications()
    accepted: dict[str, dict] = {}      # genus -> record
    feedback: dict[str, str] = {}
    stats = {"nominated": 0, "abox_refuted": 0, "hermit_refuted": 0, "shape_rejected": 0}

    pending = [f for f in fams if f[0] not in adj.get("families", {})
               or set(adj["families"][f[0]].get("members", [])) != set(f[1])]
    print(f"pending (new or family-grew): {len(pending)} of {len(fams)}", flush=True)

    for rnd in range(a.rounds):
        todo = [f for f in pending if f[0] not in accepted]
        if not todo:
            break
        print(f"— round {rnd + 1}: {len(todo)} families", flush=True)
        for i in range(0, len(todo), a.families_per_call):
            batch = todo[i:i + a.families_per_call]
            members_of = dict(batch)
            result = _propose(batch, feedback)
            for fam in result.get("families", []):
                genus = fam.get("genus", "")
                if genus not in members_of or genus in accepted:
                    continue
                members = set(members_of[genus])
                ok_pairs, reasons = [], []
                for p in fam.get("disjoint_pairs", []):
                    x, y, why = p.get("a", ""), p.get("b", ""), (p.get("rationale") or "").strip()
                    stats["nominated"] += 1
                    if x not in members or y not in members or x == y or not why:
                        stats["shape_rejected"] += 1
                        reasons.append(f"M-A: ({x},{y}) not a valid family-internal rationaled pair")
                        continue
                    why_ref = refute(x, y)
                    if why_ref:
                        stats["abox_refuted"] += 1
                        reasons.append(f"M-B: {why_ref}")
                        continue
                    ok_pairs.append({"a": min(x, y), "b": max(x, y), "rationale": why})
                if reasons:
                    feedback[genus] = "; ".join(reasons[:4])
                else:
                    feedback.pop(genus, None)
                accepted[genus] = {
                    "genus": genus, "members": sorted(members), "default": "overlap",
                    "disjoint_pairs": ok_pairs,
                    "provenance": {"date": str(date.today()), "model": "engine/instruct",
                                   "membranes": "adjudicate_siblings M-A/M-B"
                                                + ("/M-C" if a.hermit else "")}}
            print(f"   {min(i + a.families_per_call, len(todo))}/{len(todo)} · "
                  f"families adjudicated {len(accepted)}", flush=True)

        if a.hermit:
            frames = ADJ.disjoint_manchester_frames({"families": accepted})
            if frames:
                ok, unsat = _hermit_check(frames)
                if not ok:
                    hit = set(unsat)
                    for genus, rec in accepted.items():
                        bad = [p for p in rec["disjoint_pairs"]
                               if p["a"] in hit or p["b"] in hit]
                        if bad:
                            rec["disjoint_pairs"] = [p for p in rec["disjoint_pairs"]
                                                     if p not in bad]
                            stats["hermit_refuted"] += len(bad)
                            feedback[genus] = ("M-C: HermiT unsat after asserting "
                                               + ", ".join(f"({p['a']},{p['b']})" for p in bad)
                                               + " — the theory entails co-membership; do not re-nominate")
                    print(f"   HermiT: {stats['hermit_refuted']} nomination(s) refuted "
                          f"(unsat: {sorted(hit)[:6]})", flush=True)
                else:
                    print(f"   HermiT: {len(frames)} DisjointClasses consistent, 0 unsat ✓", flush=True)

    n_disjoint = sum(len(r["disjoint_pairs"]) for r in accepted.values())
    print(f"\nadjudicated {len(accepted)} families · {n_disjoint} disjoint pairs · "
          f"stats {stats}")
    if accepted and not a.dry_run:
        adj.setdefault("families", {}).update(accepted)
        adj["universe_rule"] = ("sibling pairs under a SPECIFIC genus (CCO/sdg or BFO "
                                "realizable); bare-BFO-genus pairs are grounding debt")
        ADJ.save_adjudications(adj)
        cov = ADJ.coverage(universe, adj)
        print(f"saved → {ADJ.ADJUDICATIONS_FILE.name} · coverage now: {cov}")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
