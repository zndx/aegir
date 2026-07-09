"""vocabulary_sufficiency — the publish-side MaxSim sufficiency gate for the SHARE vocabulary.

The published ``corpora/vocabulary/annotations.*`` is a MaxSim ANCHOR SURFACE — Atelier
consumes it as a source taxonomy (its ``just optimize`` sweeps calibrate against it), and
our own aperture/topic machinery embeds the same shapes. A vocabulary row is SUFFICIENT
when its composed anchor text (label · common_names · description — the axiom column is
excluded by construction) retrieves ITSELF at rank-1 with hierarchical margin against the
whole vocabulary. This is the M6 self-retrieval gate applied at the know→share boundary:
definitions that cannot discriminate do not publish silently — they land on a worklist
for the membrane-gated authoring wave.

Verdict convention follows ``domain_index.separation_report``: self-retrieval ≥ 0.85 = OK,
below = "WEAK — the vocabulary needs specification". Report → build/vocabulary_sufficiency.json.

  uv run --no-sync python scripts/vocabulary_sufficiency.py            # gate + report
  uv run --no-sync python scripts/vocabulary_sufficiency.py --limit 50 # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

COLLECTION = "sdg_vocab_sufficiency"   # throwaway check collection, rebuilt per run
SELF_RETRIEVAL_FLOOR = 0.85


def anchor_text(r: dict) -> str:
    """The composable anchor surface — mirrors what a consumer embeds (Atelier's
    compose_annotation_text shape: label | description | names). NEVER the axiom."""
    return ". ".join(p for p in (r.get("label", ""), r.get("common_names", ""),
                                 r.get("description", "")) if p).strip()


def _ancestors(records: "list[dict]") -> "dict[str, set]":
    by_code = {r["code"]: r for r in records}
    out: dict[str, set] = {}
    for r in records:
        chain, cur = set(), r.get("parent_code") or ""
        while cur and cur in by_code and cur not in chain:
            chain.add(cur)
            cur = by_code[cur].get("parent_code") or ""
        out[r["code"]] = chain
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default="http://localhost:6355")
    ap.add_argument("--floor", type=float, default=SELF_RETRIEVAL_FLOOR)
    a = ap.parse_args()

    from build_skos_vocab import build_records
    from qdrant_client import models
    from aegir.ontology.colbert_encoder import get_encoder
    from aegir.ontology.domain_index import _client

    records = [r for r in build_records() if anchor_text(r)]
    if a.limit:
        records = records[: a.limit]
    anc = _ancestors(records)
    enc = get_encoder()
    cl = _client(a.url)
    if cl.collection_exists(COLLECTION):
        cl.delete_collection(COLLECTION)
    cl.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=enc.dim, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM)))
    texts = [anchor_text(r) for r in records]
    vecs = enc.encode(texts)
    points = [models.PointStruct(id=i, vector=v.tolist(),
                                 payload={"code": r["code"], "label": r["label"]})
              for i, (r, v) in enumerate(zip(records, vecs))]
    for s in range(0, len(points), 64):
        cl.upsert(collection_name=COLLECTION, points=points[s:s + 64])

    ok = 0
    margins: list[float] = []
    failures: list[dict] = []
    for r, v in zip(records, vecs):
        hits = cl.query_points(collection_name=COLLECTION, query=v.tolist(),
                               limit=6, with_payload=True).points
        codes = [(h.payload or {}).get("code", "") for h in hits]
        if not hits or codes[0] != r["code"]:
            failures.append({"code": r["code"], "label": r["label"],
                             "rank1": codes[0] if codes else "?", "reason": "not-self-at-rank-1"})
            continue
        s0 = float(hits[0].score)
        # hierarchical forgiveness: a parent/child in rank-2 is lineage, not confusion
        comp = next((h for h in hits[1:]
                     if (h.payload or {}).get("code", "") not in anc[r["code"]]
                     and r["code"] not in anc.get((h.payload or {}).get("code", ""), set())), None)
        m = ((s0 - float(comp.score)) / s0) if (comp and s0) else 1.0
        margins.append(m)
        ok += 1
    margins.sort()
    n = len(records)
    rate = ok / n if n else 0.0
    report = {
        "n_records": n, "self_retrieval_rate": round(rate, 4),
        "rel_margin_h_p10": round(margins[len(margins) // 10], 4) if margins else 0,
        "rel_margin_h_p50": round(margins[len(margins) // 2], 4) if margins else 0,
        "floor": a.floor, "verdict": "OK" if rate >= a.floor else "WEAK — needs specification",
        "failures": failures[:40],
    }
    out = REPO / "build" / "vocabulary_sufficiency.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=1))
    print(f"failures: {len(failures)} (worklist → {out.name})")
    for f in failures[:10]:
        print(f"  {f['code']}: rank1={f['rank1']}")
    return 0 if rate >= a.floor else 1


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
