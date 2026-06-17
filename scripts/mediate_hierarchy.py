"""mediate_hierarchy — an autonomous, tool-gated cycle that turns noisy BERTSubs-style
subsumption candidates into a COHERENT, HermiT-CONSISTENT term hierarchy for the lineup.

This is a RASE instance *and* a RASE-pattern increment (Signals Programme): the agent
(Grok, via the ACP effector seam) realizes the objective — a sound subsumption hierarchy —
using a meta-harness of domain tools, with NO human review. Verification is intrinsic:

  candidate generator (mpnet top-k — embedding relatedness, the noisy signal)
        │   raw subsumption is a "minefield" (cross-category edges, cycles, spurious is-a)
        ▼
  Grok proposes a coherent hierarchy from the candidates + glosses + BFO/CCO anchors
        ▼
  TWO-LAYER GATE — both must pass, the agent's contract:
    • FORMAL  : HermiT over the SubClassOf set + BFO disjointness grounding
                (global consistency + no unsatisfiable class + structural acyclicity)
    • DOMAIN  : each edge's child/parent share real domain vocabulary (the empirical anchor)
        │   offending edges → feedback → Grok revises → converge
        ▼
  verified hierarchy (the largest consistent + domain-grounded edge set) → ontology / lineup

The reasoner is the arbiter, not the embedding score; the human is replaced by the tools,
not removed from the verification. Run under the JVM libs:

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/mediate_hierarchy.py \
        [--category 06_belief_structure | --all] [--k 6] [--max-iters 4]
"""
from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.lineup.build import _term_vocab, _verbal  # noqa: E402
from aegir.lineup.sources import load_ontology  # noqa: E402
from aegir.ontology.schema import CatalogTemplate  # noqa: E402
import generate_ontology as go  # noqa: E402  (mpnet _embed, normalized)
from mediate_acp import ACPMintEffector  # noqa: E402  (the frozen Grok executor seam)
from mediate_consistency import _GROUNDING, ensure_jvm  # noqa: E402

log = logging.getLogger("mediate_hierarchy")

_ART = "/raid/checkpoints/aegir-artifacts"
_PREFIXES = (
    "Prefix: ex: <http://aegir.example.org/term#>\n"
    "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
    "Prefix: cco: <http://www.commoncoreontologies.org/>"
)


def _ln(term_id: str) -> str:
    """A Manchester-safe local name for a term id."""
    return re.sub(r"[^A-Za-z0-9_]", "_", term_id)


def _anchor(t: CatalogTemplate) -> str | None:
    return (t.bfo_anchor_path or [None])[-1]


# ── candidate generation (the noisy signal) ────────────────────────────────────

def _term_text(t: CatalogTemplate) -> str:
    return f"{t.template_id.replace('_', ' ')}. {_verbal(t)} {' '.join(_term_vocab(t)[:24])}".strip()


def candidates(terms: list[CatalogTemplate], k: int = 6) -> list[tuple[str, str]]:
    """Per term, its k embedding-nearest siblings → undirected related pairs. mpnet
    embeddings are normalized, so the gram matrix is cosine similarity directly."""
    import numpy as np

    ids = [t.template_id for t in terms]
    V = go._embed([_term_text(t) for t in terms])
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    pairs: set[tuple[str, str]] = set()
    for i in range(len(ids)):
        for j in np.argsort(-S[i])[:k]:
            a, b = sorted((ids[i], ids[int(j)]))
            if a != b:
                pairs.add((a, b))
    return sorted(pairs)


# ── the FORMAL gate (HermiT over the SubClassOf set) ────────────────────────────

def _find_cycles(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Edges (child ⊑ parent) that close a subsumption loop. Detected structurally
    (a cycle makes classes equivalent under OWL, not unsatisfiable — we forbid it)."""
    from collections import defaultdict

    par: dict[str, set[str]] = defaultdict(set)
    for c, p in edges:
        par[c].add(p)
    bad: list[tuple[str, str]] = []
    for c, p in edges:
        seen, stack = set(), [p]
        while stack:
            n = stack.pop()
            if n == c:
                bad.append((c, p))
                break
            for nxt in par.get(n, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return bad


def _render(term_anchor: dict[str, str | None], edges: list[tuple[str, str]]) -> str:
    sup: dict[str, list[str]] = {t: ([a] if a else []) for t, a in term_anchor.items()}
    for c, p in edges:
        sup.setdefault(c, []).append(f"ex:{_ln(p)}")
    lines = [_PREFIXES, "Ontology: <http://aegir.example.org/hierarchy>", _GROUNDING]
    for t, sups in sup.items():
        head = f"Class: ex:{_ln(t)}"
        if sups:
            head += " SubClassOf: " + ", ".join(sups)
        lines.append(head)
    return "\n".join(lines)


def hierarchy_coherence(term_anchor: dict[str, str | None],
                        edges: list[tuple[str, str]]) -> dict:
    """The reasoner verdict over a proposed subsumption set. `consistent` = globally
    consistent AND no named term unsatisfiable AND acyclic. `offending` = the edges to
    drop/revise (cycle-closers + any edge incident to an unsatisfiable class; all edges
    if the TBox is globally inconsistent). Fail-closed on reasoner error."""
    ensure_jvm()
    from deeponto.onto import Ontology

    ln2id = {_ln(t): t for t in term_anchor}
    cycles = _find_cycles(edges)
    safe = [e for e in edges if e not in set(cycles)]  # don't hand cycle-closers to OWL
    omn = _render(term_anchor, safe)
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
            f.write(omn)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        r = onto.reasoner.owl_reasoner
        is_consistent = bool(r.isConsistent())
        unsat: list[str] = []
        if is_consistent:
            for c in r.getUnsatisfiableClasses().getEntities().toArray():
                if not c.isOWLNothing():
                    iri = str(c.getIRI())
                    if "term#" in iri:
                        unsat.append(ln2id.get(iri.split("term#")[-1], iri))
    except Exception as e:  # noqa: BLE001 — an unreasonable hierarchy is not admitted
        return {"consistent": False, "is_consistent": False, "unsat": [], "cycles": cycles,
                "offending": list(edges), "error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)
    uset = set(unsat)
    if not is_consistent:
        offending = list(edges)
    else:
        offending = list(set(cycles) | {e for e in edges if e[0] in uset or e[1] in uset})
    return {"consistent": bool(is_consistent and not unsat and not cycles),
            "is_consistent": is_consistent, "unsat": unsat, "cycles": cycles,
            "offending": offending, "error": ""}


# ── the DOMAIN gate (the empirical anchor) ──────────────────────────────────────

def domain_grounded(c: str, p: str, vocab: dict[str, set[str]], tau: float = 0.1) -> bool:
    """A real subsumption keeps the parent's domain vocabulary and adds to it, so child
    and parent must share more than incidental terms. Catches Grok-invented edges between
    vocabulary-unrelated terms that HermiT would (wrongly) wave through as merely consistent."""
    vc, vp = vocab.get(c, set()), vocab.get(p, set())
    union = len(vc | vp)
    return union > 0 and len(vc & vp) / union >= tau


# ── the agent (Grok proposes a coherent hierarchy) ──────────────────────────────

def propose_prompt(category: str, terms: list[CatalogTemplate],
                   cands: list[tuple[str, str]], feedback: list[str] | None) -> str:
    term_lines = []
    for t in terms:
        d = " ".join(_verbal(t).split())
        if len(d) > 160:
            d = d[:157] + "..."
        term_lines.append(f"- {t.template_id} ⟨{_anchor(t) or '—'}⟩: {d}")
    cand_lines = [f"{a} ~ {b}" for a, b in cands]
    fb = ("\n\nYOUR PREVIOUS PROPOSAL HAD REJECTED EDGES — fix the direction or drop them:\n"
          + "\n".join(feedback) + "\n") if feedback else ""
    return f"""You are an ontology engineer assembling a SUBSUMPTION HIERARCHY (rdfs:subClassOf) for the
'{category}' category of a BFO/CCO-grounded enterprise lexicon. An edge `child ⊑ parent` asserts that
EVERY instance of `child` is necessarily an instance of `parent` — i.e. `child` is strictly MORE SPECIFIC.

WORK SOLELY FROM THE DATA IN THIS MESSAGE. Do NOT read files, search the repository, or call any tools —
every term, gloss, anchor, and candidate pair you need is provided below. Reason here and answer directly.
(A HermiT reasoner and a domain check run on MY side, after you respond; you do not run them.)

TERMS (id ⟨BFO/CCO anchor⟩: gloss):
{chr(10).join(term_lines)}

CANDIDATE related pairs (embedding-nearest neighbours — RELATEDNESS ONLY; you choose the direction, or no edge):
{chr(10).join(cand_lines)}

CONTRACT — a sound reasoner (HermiT) and a domain check VERIFY every edge; propose only what will pass:
1. ACYCLIC — the edges must form a DAG. Multiple parents are fine; cycles are not.
2. COHERENT — NEVER assert an edge across DISJOINT BFO categories. Terms anchored under bfo:Process
   (Occurrents) and terms anchored under cco:Artifact / cco:*ICE (Continuants) are DISJOINT: no edge
   between them holds in either direction. Respect each term's anchor.
3. SPECIFIC — assert `child ⊑ parent` ONLY when the child's gloss genuinely NARROWS the parent's
   (a true is-a, not mere relatedness or sibling resemblance). Prefer few, high-confidence edges.
4. DOMAIN-GROUNDED — child and parent must share real domain vocabulary (a specialization keeps the
   parent's vocabulary and adds to it).
{fb}
Reason about each candidate, then output ONLY a fenced ```json array of objects, each:
{{"child": "<term id>", "parent": "<term id>", "rationale": "<one clause>"}}
Use term ids EXACTLY as listed. Output [] if no sound edges exist."""


def _parse_edges(resp: str) -> list[tuple[str, str, str]]:
    import json

    blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", resp or "", re.DOTALL)
    for block in sorted(blocks or ([resp] if resp else []), key=len, reverse=True):
        data = None
        try:
            data = json.loads(block.strip())
        except Exception:  # noqa: BLE001
            m = re.search(r"\[.*\]", block, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    data = None
        if isinstance(data, list):
            out = [(str(e["child"]).strip(), str(e["parent"]).strip(), str(e.get("rationale", "")).strip())
                   for e in data if isinstance(e, dict) and e.get("child") and e.get("parent")]
            if out:
                return out
    return []


# ── the loop (the meta-harness; tools decide, no human) ─────────────────────────

def _components(ids: set[str], cands: list[tuple[str, str]], cap: int = 12) -> list[list[str]]:
    """Connected components of the candidate graph → natural sub-clusters mediated as
    separate, SMALL prompts. The whole-category prompt overwhelms the agent (it narrates
    tool-use / truncates the combinatorial answer); a component (the ``mass_function_*``
    cluster, the ``belief_interval_*`` cluster, …) is in the regime that works. Oversized
    components are greedily chunked to ``cap`` so every prompt stays small. Singletons drop."""
    from collections import defaultdict, deque

    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in cands:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for tid in sorted(ids):
        if tid in seen:
            continue
        comp: list[str] = []
        q = deque([tid])
        seen.add(tid)
        while q:
            n = q.popleft()
            comp.append(n)
            for m in sorted(adj.get(n, ())):
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        for i in range(0, len(comp), cap):
            chunk = comp[i:i + cap]
            if len(chunk) >= 2:
                comps.append(chunk)
    return comps


def _mediate_cluster(category: str, ci: int, cterms: list[CatalogTemplate],
                     ccands: list[tuple[str, str]], anchor_all: dict[str, str | None],
                     vocab: dict[str, set[str]], eff: ACPMintEffector,
                     max_iters: int) -> tuple[list[tuple[str, str]], list[dict]]:
    """propose → two-layer gate → feedback → converge, over ONE small cluster."""
    cids = {t.template_id for t in cterms}
    canchor = {tid: anchor_all[tid] for tid in cids}
    best: list[tuple[str, str]] = []
    feedback: list[str] | None = None
    trace: list[dict] = []
    for it in range(max_iters):
        resp = eff.complete(propose_prompt(category, cterms, ccands, feedback))
        raw = _parse_edges(resp)
        seen: set[tuple[str, str]] = set()
        edges: list[tuple[str, str]] = []
        for c, p, _ in raw:
            if c in cids and p in cids and c != p and (c, p) not in seen:
                edges.append((c, p))
                seen.add((c, p))
        dom_fail = [e for e in edges if not domain_grounded(*e, vocab)]
        dom_ok = [e for e in edges if e not in set(dom_fail)]
        gate = hierarchy_coherence(canchor, dom_ok)
        off = set(gate["offending"])
        verified = [e for e in dom_ok if e not in off]
        offending = off | set(dom_fail)

        trace.append({"cluster": ci, "iter": it, "proposed": len(raw), "valid": len(edges),
                      "dom_fail": len(dom_fail), "hermit_offending": len(gate["offending"]),
                      "unsat": gate["unsat"], "cycles": gate["cycles"], "verified": len(verified),
                      "consistent": gate["consistent"], "error": gate.get("error", "")})
        log.info("  cluster %d (%d terms) iter %d: proposed=%d valid=%d dom_fail=%d hermit_off=%d → verified=%d%s",
                 ci, len(cterms), it, len(raw), len(edges), len(dom_fail), len(gate["offending"]),
                 len(verified), f"  unsat={gate['unsat']}" if gate["unsat"] else "")

        if len(verified) >= len(best):
            best = verified
        if edges and not offending:
            break
        if not raw:
            break

        rej: list[str] = []
        for c, p in dom_fail:
            rej.append(f"  {c} ⊑ {p} — REJECTED (domain): child & parent share too little vocabulary")
        for c, p in gate["cycles"]:
            rej.append(f"  {c} ⊑ {p} — REJECTED (cycle): closes a subsumption loop")
        uset = set(gate["unsat"])
        for c, p in gate["offending"]:
            if (c, p) not in set(gate["cycles"]) and (c in uset or p in uset):
                rej.append(f"  {c} ⊑ {p} — REJECTED (incoherent): makes a class unsatisfiable "
                           f"(a disjoint-category edge)")
        feedback = rej or None
    return best, trace


def mediate_category(category: str, terms: list[CatalogTemplate], eff: ACPMintEffector,
                     max_iters: int = 4, k: int = 6, cap: int = 12) -> tuple[list[tuple[str, str]], list[dict], dict]:
    ids = {t.template_id for t in terms}
    anchor = {t.template_id: _anchor(t) for t in terms}
    vocab = {t.template_id: set(_term_vocab(t)) for t in terms}
    cands = candidates(terms, k=k)
    comps = _components(ids, cands, cap=cap)
    log.info("category=%s terms=%d candidate-pairs=%d → %d cluster(s) %s",
             category, len(terms), len(cands), len(comps), [len(c) for c in comps])

    by_id = {t.template_id: t for t in terms}
    all_edges: list[tuple[str, str]] = []
    trace: list[dict] = []
    for ci, comp in enumerate(comps):
        cset = set(comp)
        ccands = [(a, b) for a, b in cands if a in cset and b in cset]
        verified, ctrace = _mediate_cluster(category, ci, [by_id[x] for x in comp], ccands,
                                             anchor, vocab, eff, max_iters)
        all_edges.extend(verified)
        trace.extend(ctrace)

    # certify the UNION over the whole category (cross-cluster consistency), drop residuals once
    final = hierarchy_coherence(anchor, all_edges)
    if not final["consistent"]:
        all_edges = [e for e in all_edges if e not in set(final["offending"])]
        final = hierarchy_coherence(anchor, all_edges)
    meta = {"category": category, "n_terms": len(terms), "n_candidates": len(cands),
            "n_clusters": len(comps), "final_consistent": final["consistent"]}
    return all_edges, trace, meta


def promote_to_catalog(hierarchy_dir: Path, catalog_glob: str) -> dict:
    """Promote verified subsumption edges from the evidence artifacts INTO the catalog
    (the SOURCE OF TRUTH): set ``broader`` on each child term. The evidence JSON is the
    mediation *staging*; the catalog is canonical — dependency arrows point inward, so the
    lineup then projects the hierarchy from the catalog, not the evidence side-artifact.
    Idempotent: re-running syncs each term's ``broader`` to the current evidence."""
    import glob
    import json
    from aegir.ontology.schema import load_catalog, save_catalog

    broader: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(Path(hierarchy_dir) / "*.json"))):
        rec = json.loads(Path(f).read_text())
        for e in rec.get("edges", []):
            c, p = e.get("child"), e.get("parent")
            if c and p and p not in broader.setdefault(c, []):
                broader[c].append(p)

    n_terms = n_files = 0
    for cf in sorted(glob.glob(catalog_glob)):
        if "candidate" in cf or "combined" in cf:
            continue
        cat = load_catalog(cf)
        changed = 0
        for t in cat.templates:
            new = sorted(broader.get(t.template_id, []))
            if new != sorted(t.broader):
                t.broader = new
                changed += 1
        if changed:
            save_catalog(cat, cf)
            n_files += 1
            n_terms += changed
            log.info("  promoted %s: broader set on %d child terms", Path(cf).stem, changed)
    res = {"edges": sum(len(v) for v in broader.values()), "child_terms": n_terms, "files": n_files}
    log.info("promoted %d edges over %d child terms into %d catalog files (the ontology SoT)",
             res["edges"], res["child_terms"], res["files"])
    return res


def main() -> None:
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default=None, help="family stem (default: smallest)")
    ap.add_argument("--all", action="store_true", help="run every category, smallest first")
    ap.add_argument("--promote", action="store_true",
                    help="promote verified evidence edges into the catalog `broader` (the SoT); no Grok")
    ap.add_argument("--k", type=int, default=6, help="candidate neighbours per term")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--out", default=f"{_ART}/evidence/hierarchy")
    args = ap.parse_args()

    if args.promote:
        res = promote_to_catalog(Path(args.out), str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))
        print(f"promoted {res['edges']} edges over {res['child_terms']} child terms "
              f"into {res['files']} catalog files — the hierarchy is now ontology SoT")
        return

    by_cat: dict[str, list[CatalogTemplate]] = {}
    for fam, t in load_ontology():
        by_cat.setdefault(fam, []).append(t)
    sizes = {c: len(ts) for c, ts in by_cat.items()}
    print("categories (size):", ", ".join(f"{c}={n}" for c, n in sorted(sizes.items(), key=lambda x: x[1])))

    if args.all:
        cats = sorted(by_cat, key=lambda c: sizes[c])
    elif args.category:
        cats = [args.category]
    else:
        cats = [min(sizes, key=lambda c: sizes[c])]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    eff = ACPMintEffector()
    try:
        for cat in cats:
            best, trace, meta = mediate_category(cat, by_cat[cat], eff,
                                                 max_iters=args.max_iters, k=args.k)
            rec = {**meta, "n_edges": len(best),
                   "edges": [{"child": c, "parent": p} for c, p in best], "trace": trace}
            (outdir / f"{cat}.json").write_text(json.dumps(rec, indent=2))
            print(f"\n=== {cat}: {len(best)} verified subsumption edges "
                  f"(consistent={meta['final_consistent']}) → {outdir / (cat + '.json')} ===")
            for c, p in best[:30]:
                print(f"  {c}  ⊑  {p}")
            if len(best) > 30:
                print(f"  … +{len(best) - 30} more")
    finally:
        eff.close()


if __name__ == "__main__":
    main()
