"""reauthor_contamination — the CAS remediation loop for external-namespace contamination.

The boundary (verify_external_refs) DETECTS a fictional/legacy external reference and propagates a
BoundarySignal with LIVE context (candidates queried from the current authority + domain anchors) to
the engine's Remediate capability; the agent REASONS and returns a correction; the membranes (the gate
+ parse + HermiT) DISPOSE it; a rejection re-prompts with its reason. The correction lives in the
agent's adaptation — no static rule here (Holland CAS / Signals & Boundaries). [[bfo_cco_grounding_mandate]]

    just engine-serve                 # the engine must serve Remediate (restart to pick it up)
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/reauthor_contamination.py [--limit N] [--apply]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.engine.client import remediate  # noqa: E402
from aegir.ontology.external_index import get_index, verify_external_refs  # noqa: E402
from aegir.ontology.schema import load_catalog, save_catalog, CATALOG_FILE  # noqa: E402
from grounding_anchors import Retriever  # noqa: E402

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ABBREV = {"ice": "information content entity"}


def _phrase(local: str) -> str:
    words = _CAMEL.sub(" ", local.replace("BFO_", "bfo ")).replace("_", " ").lower().split()
    return " ".join(_ABBREV.get(w, w) for w in words)


def _candidates(idx, retriever, offending: str) -> list:
    """LIVE candidates from the CURRENT authority (exact label) + the semantic anchor retriever, deduped —
    the agent reasons over these; nothing is applied by rule."""
    local = offending.split(":", 1)[1]
    phrase = _phrase(local)
    seen, out = set(), []
    for curie in idx.search(phrase):                       # exact label against the current authority
        if curie not in seen:
            seen.add(curie)
            out.append({"iri": curie, "label": phrase, "kind": idx.kind(curie) or "class"})
    for a in retriever.retrieve(phrase, k=6):              # semantic — real cco:/bfo:/fhir:/sdg: anchors
        if a["curie"].split(":", 1)[0] in ("cco", "bfo", "fhir") and a["curie"] not in seen:
            seen.add(a["curie"])
            out.append({"iri": a["curie"], "label": a["label"], "kind": "class", "score": round(a["sim"], 3)})
    return out[:10]


_RULES = ["external namespaces (cco:/bfo:/fhir:) may NOT be coined — a reference must EXIST in the current "
          "authoritative ontology; use a real IRI from the candidates or coin a term in the sdg: namespace",
          "keep the head class and the intended meaning; change only the offending reference(s)"]


def remediate_template(t, idx, retriever, rounds: int = 3):
    """Run the CAS cycle for one contaminated template. Returns (corrected_manchester, disposition) or None."""
    from evolve_rigor import validate_detailed  # noqa: PLC0415 — JVM membrane, import late
    axiom = t.manchester_template
    tid = t.template_id
    last_reason = ""
    for rnd in range(rounds):
        viol = verify_external_refs(axiom)
        if not viol:  # already clean (a re-prompt fixed it) → dispose through parse+HermiT below
            offending, reason = "", last_reason or "prior correction cleared the namespace violation"
        else:
            offending, reason = viol[0]
        signal = {"kind": "EXTERNAL_NAMESPACE_VIOLATION", "subject": axiom,
                  "offending": offending or "(resolved)", "reason": reason,
                  "authority": f"CCO {idx.version('cco').split('/')[-1]}, BFO {idx.version('bfo').split('/')[-1]}"}
        context = {"candidates": _candidates(idx, retriever, offending) if offending else [],
                   "anchors": [], "rules": _RULES + ([f"the reasoner/parser last rejected your attempt: {last_reason}"]
                                                     if last_reason else [])}
        resp = remediate(signal, context, capability="instruct")  # "reauthor" ability → instruct model (provisional)
        corr = (resp.get("correction") or "").strip()
        if not corr or "Class:" not in corr:
            last_reason = "no valid Manchester axiom returned"
            continue
        # DISPOSE through the membranes: gate (no fictional refs) → parse+HermiT
        ext = verify_external_refs(corr)
        if ext:
            last_reason = "still contains a fictional external ref: " + ext[0][1][:120]
            axiom = corr
            continue
        ok, why = validate_detailed([(tid, corr)], {tid: t}).get(tid, (False, "not validated"))
        if ok:
            return corr, resp.get("disposition", "CORRECTED"), resp.get("rationale", "")
        last_reason = why
        axiom = corr
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default="", help="only templates whose axiom contains this ref (e.g. cco:DirectiveICE)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cat = load_catalog(CATALOG_FILE)
    idx = get_index()
    contaminated = [t for t in cat.templates if verify_external_refs(t.manchester_template or "")
                    and (not args.only or args.only in (t.manchester_template or ""))]
    if args.limit:
        contaminated = contaminated[:args.limit]
    print(f"reauthor_contamination: {len(contaminated)} contaminated templates · CAS loop ≤{args.rounds} rounds "
          f"(authority: CCO {idx.version('cco').split('/')[-1]}, BFO {idx.version('bfo').split('/')[-1]})")
    retriever = Retriever()

    fixed = 0
    corrections: dict[str, str] = {}
    for t in contaminated:
        before = verify_external_refs(t.manchester_template)[0][0]
        res = remediate_template(t, idx, retriever, rounds=args.rounds)
        if res:
            corr, disp, rationale = res
            corrections[t.template_id] = corr
            fixed += 1
            print(f"  ✓ {t.template_id[:34]:34s} [{before} → {disp}]  {rationale[:70]}")
        else:
            print(f"  ✗ {t.template_id[:34]:34s} [{before}] — unresolved after {args.rounds} rounds")

    print(f"\n{fixed}/{len(contaminated)} remediated (agent-reasoned, membrane-disposed)")
    if args.apply and corrections:
        for t in cat.templates:
            if t.template_id in corrections:
                t.manchester_template = corrections[t.template_id]
        save_catalog(cat, CATALOG_FILE)
        print(f"APPLIED → {CATALOG_FILE.name}")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
