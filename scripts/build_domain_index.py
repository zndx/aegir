#!/usr/bin/env python
"""Build + probe the ontology/SKOS domain index (ColBERT late-interaction over Qdrant).

The input filter for content-first derivation is grounded in OUR SKOS hierarchy: each concept becomes a
ColBERT multi-vector in a Qdrant MAX_SIM collection; streamed documents are scored against it. To aim the
aperture at a new domain you expand the SKOS hierarchy first; ``diagnose`` is the early warning that the
hierarchy actually separates (else its Qdrant entries lack classification power).

    uv run --no-sync python scripts/build_domain_index.py build                       # encode SKOS → Qdrant
    uv run --no-sync python scripts/build_domain_index.py diagnose                     # separation-power report
    uv run --no-sync python scripts/build_domain_index.py classify --text "Eye irritation study of water…"
    uv run --no-sync python scripts/build_domain_index.py subtree --domain "Process"   # list a subtree's codes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import domain_index as DI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "diagnose", "classify", "subtree", "warmup", "corpus-belief"):
        s = sub.add_parser(name)
        s.add_argument("--url", default=DI.DEFAULT_QDRANT_URL)
        s.add_argument("--collection", default=DI.DEFAULT_COLLECTION)
        s.add_argument("--vocab", default=str(DI.DEFAULT_VOCAB))
        if name == "classify":
            s.add_argument("--text", required=True)
            s.add_argument("--top-k", type=int, default=5)
        if name == "diagnose":
            s.add_argument("--sample", type=int, default=0, help="0 = all concepts")
        if name == "subtree":
            s.add_argument("--domain", required=True, help="notation code or prefLabel")
        if name == "corpus-belief":
            s.add_argument("--corpus", default="/raid/datasets/aegir-corpus-v1/finepdfs-lab/finepdfs_lab_train.txt")
            s.add_argument("--n", type=int, default=40, help="real docs to sample")
            s.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "warmup":
        from aegir.ontology.colbert_encoder import warmup
        print(json.dumps(warmup(), indent=2))
        return 0

    if args.cmd == "build":
        rep = DI.build_index(vocab=args.vocab, url=args.url, collection=args.collection)
        print(f"built domain index: {rep['concepts']} SKOS concepts → {rep['collection']} "
              f"(dim {rep['dim']}, {rep['url']})")
        return 0

    if args.cmd == "classify":
        h = DI.classify_hierarchical(args.text, top_k=args.top_k, url=args.url, collection=args.collection)
        top = h["top"] or {}
        print(f"top: {top.get('pref_label')} [{top.get('code')}]  rel_margin={h['rel_margin']} belief={h['belief']} margin={h['margin']}")
        print(f"path: {' › '.join(h.get('path', []))}")
        for hit in h["hits"]:
            print(f"  {hit['score']:.3f}  {hit.get('pref_label')} [{hit.get('code')}]")
        return 0

    if args.cmd == "subtree":
        concepts = DI.load_skos(args.vocab)
        codes = DI.subtree_codes(concepts, args.domain)
        by_code = {c.code: c for c in concepts.values()}
        print(f"subtree '{args.domain}': {len(codes)} concepts")
        for code in sorted(codes, key=lambda c: [int(x) if x.isdigit() else x for x in c.split(".")]):
            print(f"  {code:8s} {by_code[code].pref_label if code in by_code else ''}")
        return 0

    if args.cmd == "corpus-belief":
        # the honest early-warning: how confidently do REAL documents route to the SKOS hierarchy?
        from collections import Counter
        docs, buf, n = [], "", args.n
        with open(args.corpus, encoding="utf-8", errors="ignore") as fh:
            while len(docs) < n:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split("\x03")
                buf = parts.pop()
                docs.extend(p.strip()[:4000] for p in parts if len(p.strip()) > 600)
        docs = docs[args.skip:args.skip + n]
        rels, roots = [], Counter()
        for d in docs:
            h = DI.classify_hierarchical(d, top_k=5, url=args.url, collection=args.collection)
            rels.append(h["rel_margin"])
            roots[(h.get("path") or ["?"])[0]] += 1
        rels.sort()
        pct = lambda p: rels[min(len(rels) - 1, int(p * len(rels)))] if rels else 0  # noqa: E731
        conf = sum(1 for r in rels if r >= 0.10)
        print(f"CORPUS→DOMAIN ROUTING — {len(docs)} real FinePDFs docs")
        print(f"  rel_margin: p10={pct(.1):.3f} median={pct(.5):.3f} p90={pct(.9):.3f}  ·  on-domain(≥0.10): {conf}/{len(docs)}")
        print(f"  top-root distribution: {dict(roots)}")
        print(f"  NOTE: a random corpus SHOULD route mostly off-domain (low rel_margin); the gate keeps only the on-domain tail.")
        return 0

    if args.cmd == "diagnose":
        rep = DI.separation_report(vocab=args.vocab, url=args.url, collection=args.collection, sample=args.sample)
        print(f"SKOS DOMAIN SEPARATION — {rep['concepts']} concepts")
        print(f"  self-retrieval: {rep['self_retrieval']:.1%}   mean rank1−rank2 margin: {rep['mean_margin']}")
        print(f"  VERDICT: {rep['verdict']}")
        if rep["confusable"]:
            print(f"  most-confusable (the augmentation worklist):")
            for c in rep["confusable"][:15]:
                print(f"    '{c['concept']}' [{c['code']}] → '{c['confused_with']}' [{c['with_code']}] (margin {c['margin']})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
