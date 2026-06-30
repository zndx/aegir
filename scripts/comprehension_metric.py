"""comprehension_metric.py — the comprehension metric, by construction.

The discrete ontology is validated logically by HermiT; HERE it is validated SUBSTANTIVELY — the latent
halo it induces (the prose discoursed FROM it) must correspond to the input's. We measure topic TRANSIT
S0 input → S1 ontology-shadow → S3 prose over the ONTOLOGY CONCEPTS as the basis (not KMeans T_I — that IS
BERTopic), scored by **late-interaction ColBERT MaxSim via Qdrant** — the design directive, and the same
apparatus the aperture uses. (An earlier cut used dense mpnet cosine; that smears the input into a uniform
activation — the exact failure MaxSim exists to prevent — and is corrected here.)

The ontology concepts become a Qdrant MAX_SIM multivector collection (`build_index`'s pattern); each stage
is ColBERT-encoded and scored by native MaxSim against every concept → a per-concept signature. A concept
whose normalized activation transits all stages, net of the genre null, is a latent topic confirmed BY
CONSTRUCTION; S0-only = lost, S3-only = hallucinated/canned; low per-doc concentration = the residual frontier.

FIRST RUN on v0.3 (the only complete input→ontology→output we have), catalog basis. NEXT: the cold-start
content (the 431-harvest + P3's fresh derive). See memory `comprehension_metric_topic_transit`.

  LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=4 \
    uv run --no-sync python scripts/comprehension_metric.py --limit 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from e6a_trace import (  # noqa: E402  — reuse the proven stage extraction + the genre-matched null
    COVERAGE_V0, CORPUS_DIR, RUNS, boot_ci, null_transfer, row_cos, strip_slots,
)
from aegir.ontology import domain_index as DI  # noqa: E402
from aegir.ontology.colbert_encoder import get_encoder  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

BASIS_COLLECTION = "comprehension_basis"


def build_basis(texts: list[str], url: str) -> int:
    """ColBERT-encode the ontology concepts → a Qdrant MAX_SIM multivector collection (build_index's pattern)."""
    from qdrant_client import models
    enc = get_encoder()
    client = DI._client(url)
    if client.collection_exists(BASIS_COLLECTION):
        client.delete_collection(BASIS_COLLECTION)
    client.create_collection(
        collection_name=BASIS_COLLECTION,
        vectors_config=models.VectorParams(
            size=enc.dim, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
    )
    vecs = enc.encode(texts)
    points = [models.PointStruct(id=i, vector=v.tolist(), payload={}) for i, v in enumerate(vecs)]
    for s in range(0, len(points), 64):
        client.upsert(collection_name=BASIS_COLLECTION, points=points[s:s + 64])
    return len(points)


def maxsim_matrix(texts: list[str], n_concepts: int, url: str, max_chars: int = 4000) -> np.ndarray:
    """Each text → ColBERT multivector → native MaxSim vs every concept (Qdrant) → (n_texts × n_concepts)."""
    enc = get_encoder()
    client = DI._client(url)
    vecs = enc.encode([(t or "")[:max_chars] for t in texts])
    R = np.zeros((len(texts), n_concepts), dtype=np.float32)
    for i, v in enumerate(vecs):
        res = client.query_points(collection_name=BASIS_COLLECTION, query=v.tolist(),
                                  limit=n_concepts, with_payload=False).points
        for p in res:
            R[i, int(p.id)] = float(p.score)
    return R


def normalize(R: np.ndarray) -> np.ndarray:
    return R / np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-9, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/combined.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--url", default=DI.DEFAULT_QDRANT_URL)
    ap.add_argument("--seed", type=int, default=20260630)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # === THE BASIS: ontology concepts as a ColBERT MAX_SIM Qdrant collection ===
    cat = load_catalog(a.catalog)
    concepts = [(t.template_id, strip_slots(t.verbal_template))
                for t in cat.templates if (t.verbal_template or "").strip()]
    labels = [c[0] for c in concepts]
    n_c = build_basis([c[1] for c in concepts], a.url)
    print(f"BASIS: {n_c} ontology concepts → Qdrant MAX_SIM collection '{BASIS_COLLECTION}' (late-interaction)")

    # === stages (same extraction as e6a_trace) ===
    verbal = {t.template_id: t.verbal_template for t in cat.templates}
    repr_text = {r["topic_id"]: r["topic_repr_text"] for r in pq.read_table(COVERAGE_V0).to_pylist()}
    chapters = []
    for run in RUNS:
        chapters.extend(pq.read_table(f"{CORPUS_DIR}/{run}/chapters.parquet").to_pylist())
    if a.limit:
        chapters = chapters[: a.limit]
    rows = []
    for c in chapters:
        s0 = repr_text.get(c["target_topic_id"], "")
        s1 = " ".join(strip_slots(verbal.get(tid, "")) for tid in (c["template_ids"] or []))
        s3 = c["response_text"] or ""
        if s0.strip() and s1.strip() and s3.strip():
            rows.append((s0, s1, s3))
    n = len(rows)
    print(f"chapters with all three stages: {n}\nscoring stages by MaxSim (Qdrant)…")

    R0 = maxsim_matrix([r[0] for r in rows], n_c, a.url)
    R1 = maxsim_matrix([r[1] for r in rows], n_c, a.url)
    R3 = maxsim_matrix([r[2] for r in rows], n_c, a.url)
    S0, S1, S3 = normalize(R0), normalize(R1), normalize(R3)   # length-invariant topic shapes
    print("scored.\n")

    def rep(name: str, sa: np.ndarray, sb: np.ndarray):
        obs = row_cos(sa, sb); nul = null_transfer(sa, sb, rng)
        o, _, _ = boot_ci(obs, rng); d_m, d_lo, d_hi = boot_ci(obs - nul, rng)
        print(f"  {name:8s} obs={o:.3f}  Δ-vs-null={d_m:+.3f}[{d_lo:+.3f},{d_hi:+.3f}]  "
              f"{'BEATS-NULL' if d_lo > 0 else 'at-null'}")
    print("=== aggregate transit (ONTOLOGY basis, MaxSim) ===")
    rep("S0->S1", S0, S1); rep("S1->S3", S1, S3); rep("S0->S3", S0, S3)

    # per-concept on the normalized (length-invariant) signatures
    a0, a1, a3 = S0.mean(0), S1.mean(0), S3.mean(0)
    transit = np.minimum(np.minimum(a0, a1), a3)
    print("\n=== top concepts confirmed BY CONSTRUCTION (input ∧ ontology ∧ prose) ===")
    for c in np.argsort(-transit)[:15]:
        print(f"  transit={transit[c]:.4f}  {labels[c]:42s}  [S0 {a0[c]:.3f} · S1 {a1[c]:.3f} · S3 {a3[c]:.3f}]")

    lost, hall = a0 - a3, a3 - a0
    print("\n=== LOST (strong in input, dropped by output) ===")
    for c in np.argsort(-lost)[:6]:
        print(f"  Δ={lost[c]:+.4f}  {labels[c]:42s}  [S0 {a0[c]:.3f} → S3 {a3[c]:.3f}]")
    print("=== HALLUCINATED/CANNED (in output, absent from input) ===")
    for c in np.argsort(-hall)[:6]:
        print(f"  Δ={hall[c]:+.4f}  {labels[c]:42s}  [S0 {a0[c]:.3f} → S3 {a3[c]:.3f}]")

    # residual: per-doc concentration on its best concept (length-invariant signature mass) — low = frontier
    conc = S0.max(1)
    print(f"\n=== residual frontier (input concentration on best ontology concept) ===")
    print(f"  per-doc top-concept mass: mean={conc.mean():.3f}  p10={np.percentile(conc,10):.3f}  p90={np.percentile(conc,90):.3f}")
    print(f"  docs below 0.20 (diffuse → uncovered → SKOS frontier): {int((conc<0.20).sum())}/{n}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
