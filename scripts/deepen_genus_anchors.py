#!/usr/bin/env python
"""deepen_genus_anchors — the DEEPENING TRACK, increment 1 (RH 2026-07-23).

Every aperture anchor is terminal today (0/23 genus-proper: broader without narrower).
The genus-first directive converges by AUTHORING narrower children — evidence-led,
from each anchor's own admitted-passage pool:

    pool (frozen snapshot) → genus_induction.embed → discriminating_potential →
    frontier_k (the rate-distortion knee: the AUTHENTIC child count — finer than the
    frontier quantizes noise into fabricated differentia) → KMeans at frontier_k →
    per-cluster contrast signature (Monroe log-odds vs the sibling clusters) +
    representative passages → engine-drafted child definitions (the engine is up) →
    PROVISIONAL proposals.

Proposals land in ``build/genus_deepening/<run>/`` for membrane disposal — S2–S9
shape, label/definition presence, dedup vs existing concepts — and NEVER touch the
admission surface: children are SKOS/scheme structure; their aperture membership and
composites remain GEPA's business behind the staged→shadow→armed discipline.

    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=2 \\
      uv run python scripts/deepen_genus_anchors.py [--min-pool 16] [--no-engine]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

STORE = REPO / "build/domain_harvest"
MIN_POOL = 16                # REGISTERED: anchors below this keep waiting for evidence
HEAD_CHARS = 2000            # embed grain per passage (MiniLM truncates anyway)
STOP = set("the a an and or of to in for on with by is are was were be been this that "
           "these those it its as at from which who whom will shall may can must not "
           "no if then than so such per each all any other more most under over "
           "between into about we you they he she i our your their his her have has "
           "had do does did done also both only own same".split())


def contrast_terms(cluster_texts: "list[str]", rest_texts: "list[str]",
                   top_k: int = 12) -> "list[str]":
    def toks(texts):
        c: Counter = Counter()
        for t in texts:
            for w in re.findall(r"[a-z][a-z\-]{2,}", t.lower()):
                if w not in STOP:
                    c[w] += 1
        return c
    mine, rest = toks(cluster_texts), toks(rest_texts)
    n1, n2 = sum(mine.values()) or 1, sum(rest.values()) or 1
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
    ap.add_argument("--min-pool", type=int, default=MIN_POOL)
    ap.add_argument("--no-engine", action="store_true",
                    help="skip engine definition drafting (structured drafts only)")
    a = ap.parse_args()

    import numpy as np
    from sklearn.cluster import KMeans

    from aegir.ontology import domain_index as DI
    from aegir.ontology.genus_induction import discriminating_potential, embed

    concepts = DI.load_skos(str(DI.DEFAULT_OVERLAY))
    by_code = {str(getattr(c, "code", "")): (iri, c) for iri, c in concepts.items()
               if getattr(c, "code", "")}
    excl = set(DI.armed_admission_filter().get("exclude_codes") or [])

    # frozen pool snapshot (the stream advances concurrently — freeze NOW).
    # GRAIN LESSON (measured on increment-1's first run): embedding doc HEADS clusters
    # whole-document topical variety (DATAENG_LINEAGE grew a GENEALOGY child from
    # biblical lineage texts) — the pool must be the ADMITTED ITEM SPANS (the windows
    # that actually matched the anchor). Per-era honest: items-era rows contribute
    # their admitted spans; head-era rows contribute their head (their admission grain).
    item_spans: "dict[str, list]" = {}
    ip = STORE / "manifest_items.jsonl"
    if ip.exists():
        for line in ip.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            item_spans[str(r.get("hash", ""))] = [i for i in (r.get("items") or [])
                                                  if i.get("admitted")]
    pools: "dict[str, list[str]]" = {}
    for line in (STORE / "manifest.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        code = str(r.get("code", ""))
        if "." not in code:
            continue
        p = STORE / "docs" / f"{r['hash']}.txt"
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        spans = item_spans.get(str(r.get("hash", "")))
        if spans:
            for it in spans:
                c0, c1 = it.get("span") or [0, HEAD_CHARS]
                pools.setdefault(str(it.get("code") or code), []).append(text[c0:c1])
        else:
            pools.setdefault(code, []).append(text[:HEAD_CHARS])

    run_dir = REPO / "build/genus_deepening"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"registered": {"min_pool": a.min_pool, "head_chars": HEAD_CHARS},
               "anchors": {}}
    for code, docs in sorted(pools.items(), key=lambda kv: -len(kv[1])):
        if len(docs) < a.min_pool:
            summary["anchors"][code] = {"pool": len(docs), "status": "awaiting evidence "
                                        "(below min-pool — advance the input window)"}
            continue
        iri, con = by_code.get(code, ("", None))
        local = iri.rsplit("#", 1)[-1] if iri else code
        X = embed(docs)
        ks = list(range(2, min(9, max(3, len(docs) // 6) + 1)))
        pot = discriminating_potential(X, ks)
        k = int(pot.get("frontier_k") or 2)
        km = KMeans(n_clusters=k, n_init=10, random_state=0xA119).fit(X)
        children = []
        for ci in range(k):
            idx = [i for i, l_ in enumerate(km.labels_) if l_ == ci]
            if len(idx) < 3:
                continue                     # a child needs its own evidence
            ct = [docs[i] for i in idx]
            rest = [docs[i] for i in range(len(docs)) if km.labels_[i] != ci]
            terms = contrast_terms(ct, rest)
            # representative = nearest to centroid
            centroid = X[idx].mean(axis=0)
            rep_i = min(idx, key=lambda i: float(np.linalg.norm(X[i] - centroid)))
            child_local = f"{local}_" + "_".join(t.upper() for t in terms[:2])
            children.append({"proposed_local": child_local,
                             "n_passages": len(idx),
                             "contrast_terms": terms,
                             "representative_head": docs[rep_i][:280]})
        # engine-drafted definitions (the ACP-propose half; membranes dispose)
        if not a.no_engine and children:
            try:
                from aegir.engine.client import complete_detailed
                _rf = (run_dir / "reasoning.jsonl").open("a")
                for ch in children:
                    prompt = (
                        f"Parent concept: {local} — "
                        f"{(getattr(con, 'definition', '') or '')[:300]}\n"
                        f"A sub-concept is evidenced by {ch['n_passages']} passages; its "
                        f"distinctive vocabulary: {', '.join(ch['contrast_terms'])}.\n"
                        f"Representative passage opening: {ch['representative_head']}\n\n"
                        "Write ONE skos:definition sentence (≤40 words) for this "
                        "sub-concept as a narrower specialization of the parent. State "
                        "the differentia — what distinguishes it from sibling "
                        "sub-concepts — in domain language. Output only the sentence.")
                    r = complete_detailed(prompt, capability="instruct",
                                          max_tokens=2048, temperature=0.5)
                    d = (r.get("text") or "").strip()
                    ch["definition_draft"] = d.split("\n")[0][:400]
                    # REASONING RETENTION (RH 2026-07-24): the thinking trace persists
                    _rf.write(json.dumps({"anchor": local,
                                          "child": ch["proposed_local"],
                                          "reasoning": r.get("reasoning_content", ""),
                                          "definition": ch["definition_draft"]}) + "\n")
                _rf.close()
            except Exception as e:  # noqa: BLE001 — drafts degrade visibly, never block
                for ch in children:
                    ch.setdefault("definition_draft",
                                  f"(engine unavailable: {str(e)[:60]} — structured "
                                  f"draft) A {local} specialization characterized by "
                                  + ", ".join(ch["contrast_terms"][:5]) + ".")
        # provisional TTL block (the candidate-block pattern; NOT applied to the overlay)
        ttl = []
        for ch in children:
            cl = ch["proposed_local"]
            ttl.append(
                f"<https://signals.zndx.org/sdg#{cl}> a skos:Concept ;\n"
                f'    skos:prefLabel "{cl.replace("_", " ").title()}" ;\n'
                f'    skos:altLabel "{cl}" ;\n'
                f'    skos:definition "{(ch.get("definition_draft") or "").replace(chr(34), chr(39))}" ;\n'
                f"    skos:broader <{iri}> ;\n"
                f"    skos:inScheme <https://signals.zndx.org/sdg/scheme> .\n")
        (run_dir / f"{code.replace('.', '_')}_{local}.proposals.ttl").write_text(
            "\n".join(ttl))
        summary["anchors"][code] = {
            "anchor": local, "pool": len(docs), "frontier_k": k,
            "silhouette_at_k": pot.get("by_k", {}).get(k, {}).get("silhouette")
                               if isinstance(pot.get("by_k"), dict) else None,
            "children_proposed": len(children),
            "children": [{k_: v for k_, v in ch.items()
                          if k_ != "representative_head"} for ch in children]}
        print(f"{code} {local}: pool {len(docs)} → frontier_k {k} → "
              f"{len(children)} children proposed "
              f"({', '.join(ch['proposed_local'] for ch in children[:3])}…)")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    n_ch = sum(len(v.get("children", [])) for v in summary["anchors"].values()
               if isinstance(v, dict))
    print(f"\nDEEPENING: {n_ch} provisional children across "
          f"{sum(1 for v in summary['anchors'].values() if v.get('children_proposed'))} "
          f"anchors → build/genus_deepening/ (membranes dispose before the overlay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
