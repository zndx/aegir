#!/usr/bin/env python
"""optimize_aperture_composites — phase-3 opener: candidate composite anchors, measured.

The Canonical Aperture: each admission anchor is a COMPOSITE of primary constituent
concepts. v1 proposes candidates MECHANICALLY (engine/GEPA proposers slot into the same
harness later — the Atelier >98% top-3 result is the benchmark): each leaf's composite =
its own ``text()`` + the pref labels/definitions of its top-K CONSTITUENT concepts,
retrieved from the FULL vocab collection by MaxSim, capped to the 512-token item grain
(the registered ColBERT limit — composites must remain valid sliding-window items).

PRESERVATION POSTURE (RH 2026-07-23): candidates seed an ISOLATED qdrant collection
(``sdg_aperture_cand_<sha8>``) — the live collection (≡ the Current corpus's topic
identity space) is NEVER touched; all outputs land scratch-side (build/); promotion is
a strategy-release act (lens binding), never an in-place mutation.

Evaluation (same instruments, candidate-pointed):
  * self-retrieval + margin per anchor (does each composite retrieve ITSELF sharply?)
  * the recompose simulation re-scored against the candidate collection over the
    admitted corpus — re-admission rate + margin structure vs the live baseline.

    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs \\
      uv run python scripts/optimize_aperture_composites.py [--constituents 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))                 # scripts.* imports (the recompose instrument)

TOKEN_BUDGET = 500          # headroom under the registered 512-token item grain


def _contrast_terms(anchor_docs: "dict[str, list[str]]", code: str,
                    top_k: int = 24) -> "list[str]":
    """Log-odds-with-Dirichlet-prior (Monroe et al.) contrastive vocabulary: terms
    frequent in THIS anchor's admitted passages and rare in every other anchor's —
    the anti-convergence proposer (v1's similarity enrichment CONVERGED anchors;
    contrast is the measured lever). Fitness carries NO share-uniformity term —
    organic dominance (an anchor aligned with a family's central value proposition)
    is domain shape, never a defect (RH)."""
    import math
    import re as _re
    from collections import Counter
    STOP = set("the a an and or of to in for on with by is are was were be been this "
               "that these those it its as at from which who whom will shall may can "
               "must not no if then than so such per each all any other more most "
               "under over between into about we you they he she i our your their "
               "his her have has had do does did done also both only own same".split())

    def toks(texts: "list[str]") -> Counter:
        c: Counter = Counter()
        for t in texts:
            for w in _re.findall(r"[a-z][a-z\-]{2,}", t.lower()):
                if w not in STOP:
                    c[w] += 1
        return c

    mine = toks(anchor_docs.get(code, []))
    rest: Counter = Counter()
    for k, docs in anchor_docs.items():
        if k != code:
            rest.update(toks(docs))
    if not mine:
        return []
    n1, n2 = sum(mine.values()), sum(rest.values()) or 1
    prior = mine + rest
    a0 = sum(prior.values())
    scored = []
    for w, f1 in mine.items():
        if f1 < 3:
            continue
        f2 = rest.get(w, 0)
        aw = prior[w]
        d1 = math.log((f1 + aw * 0.01) / (n1 + a0 * 0.01 - f1 - aw * 0.01))
        d2 = math.log((f2 + aw * 0.01) / (n2 + a0 * 0.01 - f2 - aw * 0.01))
        var = 1.0 / (f1 + aw * 0.01) + 1.0 / (f2 + aw * 0.01)
        scored.append(((d1 - d2) / math.sqrt(var), w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:top_k]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposer", choices=["constituents", "contrast"], default="contrast",
                    help="constituents = v1 similarity enrichment (measured: CONVERGES "
                         "anchors); contrast = log-odds differentia vocabulary from each "
                         "anchor's own admitted passages vs every other anchor's")
    ap.add_argument("--constituents", type=int, default=8,
                    help="top-K vocab concepts composed into each leaf candidate")
    ap.add_argument("--contrast-terms", type=int, default=24,
                    help="top-K contrastive terms appended per anchor (contrast proposer)")
    ap.add_argument("--min-passages", type=int, default=3,
                    help="contrast proposer: anchors with fewer admitted passages keep "
                         "their base text (no evidence to contrast — never penalized)")
    ap.add_argument("--corpus-eval", action="store_true",
                    help="also re-score the admitted corpus against the candidates "
                         "(the recompose instrument, candidate-pointed — THE fitness)")
    a = ap.parse_args()

    from qdrant_client import models
    from aegir.ontology import domain_index as DI
    from aegir.ontology.colbert_encoder import get_encoder

    enc = get_encoder()
    client = DI._client(DI.DEFAULT_QDRANT_URL)
    af = DI.armed_admission_filter()
    roots = set(af.get("exclude_codes") or [])

    pts, _ = client.scroll(DI.DEFAULT_APERTURE, limit=256, with_payload=True)
    live = {str((p.payload or {}).get("code", "")): (p.payload or {}) for p in pts}
    vocab = DI.load_skos()                     # definitions come from the SoT dataclass

    def n_tokens(t: str) -> int:
        return len(enc._tokenizer(t)["input_ids"])

    # contrast proposer grounding: each anchor's OWN admitted passages (genus rows only)
    anchor_docs: "dict[str, list[str]]" = {}
    if a.proposer == "contrast":
        mpath = REPO / "build/domain_harvest/manifest.jsonl"
        for line in mpath.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            code = str(r.get("code", ""))
            if code in roots or "." not in code:
                continue
            dp = REPO / "build/domain_harvest/docs" / f"{r['hash']}.txt"
            if dp.exists():
                anchor_docs.setdefault(code, []).append(
                    dp.read_text(errors="ignore")[:8000])

    # ── propose: leaf composites (roots ship unchanged) ──
    composites = {}
    for code, pl in sorted(live.items()):
        base = pl.get("retrieval_text") or ""
        if code in roots:
            composites[code] = {"text": base, "changed": False}
            continue
        if a.proposer == "contrast":
            if len(anchor_docs.get(code, [])) < a.min_passages:
                composites[code] = {"text": base, "changed": False,
                                    "note": f"kept base ({len(anchor_docs.get(code, []))} "
                                            "passages < min — no evidence to contrast)"}
                continue
            terms = _contrast_terms(anchor_docs, code, top_k=a.contrast_terms)
            kept_terms = []
            for t_ in terms:
                cand = base + ". Characteristic vocabulary: " + ", ".join(kept_terms + [t_])
                if n_tokens(cand) > TOKEN_BUDGET:
                    break
                kept_terms.append(t_)
            text = (base + ". Characteristic vocabulary: " + ", ".join(kept_terms)) \
                if kept_terms else base
            composites[code] = {"text": text, "changed": text != base,
                                "contrast_terms": kept_terms,
                                "n_passages": len(anchor_docs.get(code, [])),
                                "tokens": n_tokens(text), "base_tokens": n_tokens(base)}
            continue
        hits = DI.classify(base, collection=DI.DEFAULT_COLLECTION,
                           top_k=a.constituents + 4)
        parts = [base]
        used = 0
        for h in hits:
            if used >= a.constituents:
                break
            frag = str(h.get("pref_label") or "").strip()
            vc = vocab.get(str(h.get("iri") or ""))
            defn = str(getattr(vc, "definition", "") or "").strip()
            piece = f"{frag}: {defn}" if defn else frag
            if not piece or piece.lower() in base.lower():
                continue
            cand = ". ".join(parts + [piece])
            if n_tokens(cand) > TOKEN_BUDGET:
                break
            parts.append(piece)
            used += 1
        text = ". ".join(parts)
        composites[code] = {"text": text, "changed": text != base,
                            "n_constituents": used, "tokens": n_tokens(text),
                            "base_tokens": n_tokens(base)}

    cand_id = hashlib.sha256(json.dumps(
        {k: v["text"] for k, v in sorted(composites.items())}).encode()).hexdigest()[:8]
    collection = f"sdg_aperture_cand_{cand_id}"

    # ── seed the ISOLATED candidate collection (payload mirrors the live one) ──
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=enc.dim, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM)))
    items = sorted(live.items())
    vecs = enc.encode([composites[c]["text"] for c, _ in items])
    points = []
    for i, ((code, pl), v) in enumerate(zip(items, vecs)):
        points.append(models.PointStruct(id=i, vector=v.tolist(), payload={
            **{k: pl.get(k) for k in ("iri", "code", "pref_label", "alt_label",
                                      "broader", "ancestor_codes")},
            "retrieval_text": composites[code]["text"],
            "retrieval_tokens": composites[code].get("tokens",
                                                     n_tokens(composites[code]["text"]))}))
    for start in range(0, len(points), 32):
        client.upsert(collection_name=collection, points=points[start:start + 32])

    # ── evaluate 1: self-retrieval + margins, live vs candidate ──
    def self_retrieval(coll: str, texts: "dict[str, str]") -> dict:
        ok = 0
        margins = []
        for code, text in texts.items():
            if code in roots:
                continue
            hits = DI.classify(text, collection=coll, top_k=5,
                               exclude_codes=roots)
            if hits and str(hits[0].get("code")) == code:
                ok += 1
            s = [h["score"] for h in hits[:2]]
            margins.append(((s[0] - s[1]) / s[0]) if len(s) > 1 and s[0] else 1.0)
        n = len(texts) - len(roots & set(texts))
        return {"self_retrieval": f"{ok}/{n}",
                "mean_margin": round(sum(margins) / max(1, len(margins)), 4)}

    live_texts = {c: (pl.get("retrieval_text") or "") for c, pl in live.items()}
    cand_texts = {c: composites[c]["text"] for c in live}
    ev = {"candidate_id": cand_id, "collection": collection,
          "n_changed": sum(1 for v in composites.values() if v.get("changed")),
          "live": self_retrieval(DI.DEFAULT_APERTURE, live_texts),
          "candidate": self_retrieval(collection, cand_texts)}
    print(f"candidate {cand_id}: {ev['n_changed']} leaf composites changed "
          f"(≤{TOKEN_BUDGET} tokens each; roots unchanged)")
    print(f"self-retrieval (leaves-only view): live {ev['live']} → "
          f"candidate {ev['candidate']}")

    out_dir = REPO / "build/aperture_candidates" / cand_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "composites.json").write_text(json.dumps(composites, indent=1))

    # ── evaluate 2 (optional): corpus re-scoring via the recompose instrument ──
    if a.corpus_eval:
        from scripts.measure_admission_selectivity import recompose
        print(f"\ncorpus re-scoring vs candidate collection {collection}:")
        recompose(collection=collection,
                  out_path=out_dir / "corpus_eval.json")

    (out_dir / "evaluation.json").write_text(json.dumps(ev, indent=1))
    print(f"→ {out_dir.relative_to(REPO)}/ (candidate collection ISOLATED: {collection}; "
          "live aperture untouched — promotion is a strategy-release act)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
