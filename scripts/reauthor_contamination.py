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


def _candidates(idx, retriever, offending: str, axiom: str = "") -> list:
    """LIVE candidates from the CURRENT authority — TYPE-AWARE and existence-filtered. An offending ref used
    as an object property (a relation, `ref some/only/…`) is offered OBJECT-PROPERTY candidates (so the agent
    reaches the real BFO/CCO relation, e.g. has-part → bfo:0000178, rather than coining a redundant sdg: one);
    a class slot gets class candidates + semantic anchors. Every candidate is idx.exists()-checked, so malformed
    junk (e.g. double-prefixed cco:BFO_…) never reaches the agent. The agent reasons over these; nothing by rule."""
    local = offending.split(":", 1)[1]
    phrase = _phrase(local)
    is_rel = bool(re.search(rf"(?<![\w]){re.escape(offending)}\s+(?:some|only|min|max|exactly|value)\b", axiom))
    kind = "objectproperty" if is_rel else None
    seen, out = set(), []
    for curie in idx.search(phrase, kind=kind):            # exact + token-partial, type-filtered, all real
        if curie not in seen:
            seen.add(curie)
            out.append({"iri": curie, "label": idx.label(curie) or phrase, "kind": idx.kind(curie) or "class"})
    if is_rel and not out:                                 # numeric-legacy / off-vocab relation → the real BFO family
        for curie, lab in idx.by_kind("objectproperty", "bfo"):
            out.append({"iri": curie, "label": lab, "kind": "objectproperty"})
    if not is_rel:                                         # semantic CLASS anchors only for a class slot
        for a in retriever.retrieve(phrase, k=6):          # real cco:/bfo:/fhir: anchors, existence-filtered
            c = a["curie"]
            if c.split(":", 1)[0] in ("cco", "bfo", "fhir") and c not in seen and idx.exists(c):
                seen.add(c)
                out.append({"iri": c, "label": a["label"], "kind": "class", "score": round(a["sim"], 3)})
    return out[:16] if is_rel else out[:10]


_RULES = ["external namespaces (cco:/bfo:/fhir:) may NOT be coined — a reference must EXIST in the current "
          "authoritative ontology; use a real IRI from the candidates or coin a term in the sdg: namespace",
          "keep the head class and the intended meaning; change only the offending reference(s)",
          "PRESERVE every {Name:Type} slot-DSL wrapper EXACTLY as written (e.g. {Dish:Class}, "
          "{p:ObjectProperty}, {X:Class:bfo:0000002}) — the template is a slot skeleton; NEVER rewrite a "
          "{Name:Class} slot into a bare identifier or a full IRI. Change ONLY the offending external reference."]


def remediate_template(t, idx, retriever, rounds: int = 3):
    """Run the CAS cycle for one contaminated template. Returns (corrected_manchester, disposition) or None."""
    from evolve_rigor import validate_detailed  # noqa: PLC0415 — JVM membrane, import late
    axiom = t.manchester_template
    tid = t.template_id
    last_reason = ""
    deferred = None  # a namespace-clean, parsing, but structurally-UNSAT rewrite → reauthor_unsat's job, not ours
    for rnd in range(rounds):
        viol = verify_external_refs(axiom)  # ALL offending refs — a template may carry several
        refs = [r for r, _ in viol]
        if not viol:  # already clean (a re-prompt fixed it) → dispose through parse+HermiT below
            offending, reason = "(resolved)", last_reason or "prior correction cleared the namespace violation"
            cands = []
        else:
            offending = ", ".join(refs)
            reason = "; ".join(f"{r} — {w}" for r, w in viol[:5])
            cands, seen = [], set()  # union of live candidates across every offending ref
            for ref in refs:
                for c in _candidates(idx, retriever, ref, axiom):
                    if c["iri"] not in seen:
                        seen.add(c["iri"]); cands.append(c)
        signal = {"kind": "EXTERNAL_NAMESPACE_VIOLATION", "subject": axiom,
                  "offending": offending, "reason": reason,
                  "authority": f"CCO {idx.version('cco').split('/')[-1]}, BFO {idx.version('bfo').split('/')[-1]}"}
        rules = _RULES + ([f"fix EVERY one of the {len(refs)} offending references in a single rewrite: {offending}"]
                          if len(refs) > 1 else [])
        rules += [f"the reasoner/parser last rejected your attempt: {last_reason}"] if last_reason else []
        context = {"candidates": cands, "anchors": [], "rules": rules}
        try:
            resp = remediate(signal, context, capability="instruct")  # "reauthor" ability → instruct model (provisional)
        except Exception as e:  # noqa: BLE001 — one template's engine/RPC hiccup must not crash the whole batch
            last_reason = f"engine error: {str(e)[:140]}"
            continue
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
        # Namespace-clean + parses but STRUCTURALLY UNSAT (a domain/range clash) is reauthor_unsat's job, NOT the
        # namespace loop's — the two loops compose. DEFER it: the ref is now real, and a single unsat class is not
        # an inconsistent ontology, so realize emits its signal → reauthor_unsat re-authors the offending conjunct.
        if "unsatisfiable" in why.lower():
            deferred = (corr, resp.get("rationale", ""), why)
        last_reason = why
        axiom = corr
    if deferred:
        return deferred[0], "UNSAT_DEFER", f"namespace-clean; → reauthor_unsat: {deferred[2][:110]}"
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

    fixed = deferred = 0
    corrections: dict[str, str] = {}
    for t in contaminated:
        before = verify_external_refs(t.manchester_template)[0][0]
        res = remediate_template(t, idx, retriever, rounds=args.rounds)
        if res:
            corr, disp, rationale = res
            corrections[t.template_id] = corr
            fixed += 1
            mark = "⚠" if disp == "UNSAT_DEFER" else "✓"
            deferred += disp == "UNSAT_DEFER"
            print(f"  {mark} {t.template_id[:34]:34s} [{before} → {disp}]  {rationale[:70]}")
        else:
            print(f"  ✗ {t.template_id[:34]:34s} [{before}] — unresolved after {args.rounds} rounds")

    print(f"\n{fixed}/{len(contaminated)} namespace-clean ({fixed - deferred} consistent, {deferred} unsat→reauthor_unsat)")
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
