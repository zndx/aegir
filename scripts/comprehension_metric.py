"""comprehension_metric.py — the comprehension metric, by construction.

The discrete ontology is validated logically by HermiT; HERE it is validated SUBSTANTIVELY — the latent
halo it induces (the prose discoursed FROM it) must correspond to the input's. We measure topic TRANSIT
S0 input → S1 ontology-shadow → S3 prose in ONE basis — but **the basis is the ontology concepts**, not the
200 KMeans centroids of `T_I` (that IS BERTopic). A concept whose activation transits all stages, net of
the genre null, is a latent topic confirmed BY CONSTRUCTION; the residual (input mass off the basis) is the
SKOS frontier; S0-only = lost, S3-only = hallucinated.

FIRST CUT: the basis-swap on the v0.3 corpus (the only complete input→ontology→output we have), reusing
e6a_trace's dense (mpnet) scoring. NEXT: late-interaction MaxSim scoring + the cold-start content (the
431-doc harvest + P3's fresh derive). See memory `comprehension_metric_topic_transit`.

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

from e6a_trace import (  # noqa: E402  — reuse the proven stage machinery
    COVERAGE_V0, CORPUS_DIR, RUNS, boot_ci, doc_embed, null_transfer, row_cos, signatures, strip_slots,
)
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.topic_alignment import encode_sentences, get_encoder, load_topic_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/combined.json"))
    ap.add_argument("--ti-cache", default=str(REPO / "src/aegir/ontology/catalog/T_I_canonical.pkl"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260630)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    t_i = load_topic_model(a.ti_cache)
    encoder = get_encoder(t_i.encoder_model)

    # === THE BASIS SWAP: ontology concepts, not the KMeans T_I centroids ===
    cat = load_catalog(a.catalog)
    concepts = [(t.template_id, strip_slots(t.verbal_template))
                for t in cat.templates if (t.verbal_template or "").strip()]
    labels = [c[0] for c in concepts]
    basis = encode_sentences([c[1] for c in concepts], encoder=encoder).astype(np.float32)
    basis = basis / np.clip(np.linalg.norm(basis, axis=1, keepdims=True), 1e-9, None)
    print(f"BASIS: {len(labels)} ontology concepts (replacing T_I's {len(t_i.centroids)} KMeans centroids)")

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
            rows.append((c["chapter_id"], s0, s1, s3))
    print(f"chapters with all three stages: {len(rows)}")

    def stage(idx: int):
        embs, keep = [], []
        for i, r in enumerate(rows):
            e = doc_embed(r[idx], encoder)
            if e is not None:
                embs.append(e); keep.append(i)
        return np.array(embs, dtype=np.float32), keep

    e0, k0 = stage(1); e1, k1 = stage(2); e3, k3 = stage(3)
    keep = sorted(set(k0) & set(k1) & set(k3))
    m0 = {k: i for i, k in enumerate(k0)}; m1 = {k: i for i, k in enumerate(k1)}; m3 = {k: i for i, k in enumerate(k3)}
    E0 = np.array([e0[m0[k]] for k in keep]); E1 = np.array([e1[m1[k]] for k in keep]); E3 = np.array([e3[m3[k]] for k in keep])
    n = len(keep)
    print(f"encoded {n} complete chapters\n")

    R0, R1, R3 = E0 @ basis.T, E1 @ basis.T, E3 @ basis.T          # raw per-concept activations (n×K)
    S0, S1, S3 = signatures(E0, basis), signatures(E1, basis), signatures(E3, basis)  # normalized signatures

    # === aggregate transit, ONTOLOGY basis, vs genre-matched null ===
    def rep(name: str, sa: np.ndarray, sb: np.ndarray):
        obs = row_cos(sa, sb); nul = null_transfer(sa, sb, rng)
        o, _, _ = boot_ci(obs, rng); d_m, d_lo, d_hi = boot_ci(obs - nul, rng)
        print(f"  {name:8s} obs={o:.3f}  Δ-vs-null={d_m:+.3f}[{d_lo:+.3f},{d_hi:+.3f}]  "
              f"{'BEATS-NULL' if d_lo > 0 else 'at-null'}")
    print("=== aggregate transit (ONTOLOGY basis) ===")
    rep("S0->S1", S0, S1); rep("S1->S3", S1, S3); rep("S0->S3", S0, S3)

    # === per-concept transit — the comprehension content, NAMED ===
    a0, a1, a3 = R0.mean(0), R1.mean(0), R3.mean(0)
    transit = np.minimum(np.minimum(a0, a1), a3)   # a concept TRANSITS iff strongly active in all three stages
    print("\n=== top concepts confirmed BY CONSTRUCTION (input ∧ ontology ∧ prose) ===")
    for c in np.argsort(-transit)[:15]:
        print(f"  transit={transit[c]:.3f}  {labels[c]:42s}  [S0 {a0[c]:.2f} · S1 {a1[c]:.2f} · S3 {a3[c]:.2f}]")

    # === directional: lost (input-only) vs hallucinated (prose-only) ===
    lost = a0 - a3; hall = a3 - a0
    print("\n=== LOST (strong in input, dropped by output) ===")
    for c in np.argsort(-lost)[:6]:
        print(f"  Δ={lost[c]:+.3f}  {labels[c]:42s}  [S0 {a0[c]:.2f} → S3 {a3[c]:.2f}]")
    print("=== HALLUCINATED/CANNED (in output, absent from input) ===")
    for c in np.argsort(-hall)[:6]:
        print(f"  Δ={hall[c]:+.3f}  {labels[c]:42s}  [S0 {a0[c]:.2f} → S3 {a3[c]:.2f}]")

    # === residual frontier: input mass not captured by any concept ===
    cov = R0.max(1)
    print(f"\n=== residual frontier (input coverage by the ontology basis) ===")
    print(f"  per-doc best-concept cosine: mean={cov.mean():.3f}  p10={np.percentile(cov,10):.3f}  p90={np.percentile(cov,90):.3f}")
    print(f"  docs below 0.30 (uncovered → SKOS frontier): {int((cov<0.30).sum())}/{n}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
