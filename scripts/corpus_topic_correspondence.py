#!/usr/bin/env python
"""Corpus-level BERTopic correspondence: input FinePDFs ↔ output prose.

Question: of the 200 audit topics extracted from FinePDFs, how many are
also occupied (and at what density) by the generated chapter corpus?

This is the *corpus*-level analog of scripts/topic_correspondence.py
(which measures per-chapter anchor correspondence). Here we ask: does
the output corpus span the same topic space as the input, or is it
collapsed onto a narrow subset?

Method:
  1. Audit produced 200 KMeans topics over 10K FinePDFs docs.
     - topic_doc_count = number of FinePDFs docs in that topic.
     - topic_repr_text = nearest-to-centroid doc; we re-embed it as a
       proxy for the topic centroid.
  2. Embed each generated chapter, find its nearest topic among the 200.
  3. Compute the topic distribution per arm + compare to FinePDFs.

Outputs:
  - topic_occupancy: count of chapters per topic_id per arm
  - corpus-level metrics:
      coverage_at_K: fraction of FinePDFs topics that received ≥K chapters
      gini: concentration of chapter distribution over topics
      jsd: Jensen-Shannon divergence (chapter dist ‖ FinePDFs dist)
      effective_topics: exp(entropy(chapter_dist))

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/corpus_topic_correspondence.py \\
        --chapters-runs <run1> <run2> <run3> \\
        --audit-run /raid/.../coverage_v0/<audit_id>/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("corpus-topic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-runs", required=True, nargs="+")
    p.add_argument("--audit-run", required=True)
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-chars-per-chapter", type=int, default=8000)
    return p.parse_args()


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions over the same support."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def gini(x: np.ndarray) -> float:
    """Gini coefficient over a non-negative distribution. 0 = uniform, 1 = collapsed."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    audit_run = Path(args.audit_run)

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Topic centroids — re-embed representative texts (same as topic_correspondence.py)
    audit = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    audit = audit.sort_values("topic_id").reset_index(drop=True)
    n_topics = len(audit)
    logger.info(f"audit topics: {n_topics}, FinePDFs sample size: {int(audit['topic_doc_count'].sum())}")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedding_model, device=device)
    topic_embs = embedder.encode(audit["topic_repr_text"].fillna("").tolist(),
                                  normalize_embeddings=True,
                                  show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32)

    # FinePDFs reference distribution = topic_doc_count
    finepdfs_dist = audit["topic_doc_count"].to_numpy().astype(float)
    finepdfs_dist /= finepdfs_dist.sum()

    arm_dists: dict[str, np.ndarray] = {}
    arm_nearest_sims: dict[str, np.ndarray] = {}

    for run_dir in args.chapters_runs:
        run_dir = Path(run_dir)
        chapters = pq.read_table(run_dir / "chapters.parquet").to_pandas()
        arm = chapters["ablation"].iloc[0] if "ablation" in chapters.columns else run_dir.name
        texts = chapters["response_text"].fillna("").apply(
            lambda s: s[:args.max_chars_per_chapter]
        ).tolist()
        chap_embs = embedder.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False,
                                     convert_to_numpy=True).astype(np.float32)
        sims = chap_embs @ topic_embs.T  # (n_chap, n_topics)
        nearest = sims.argmax(axis=1)
        nearest_sim = sims.max(axis=1)
        # Distribution over the same 200-topic support
        dist = np.bincount(nearest, minlength=n_topics).astype(float)
        arm_dists[arm] = dist
        arm_nearest_sims[arm] = nearest_sim
        logger.info(f"{arm}: {len(chapters)} chapters, mean nearest_sim={nearest_sim.mean():.3f}")

    # Print scorecard
    print()
    print(f"{'arm':<14} {'n':>4} {'occ@1':>6} {'occ@5':>6} {'cov%':>6} "
          f"{'gini↓':>7} {'eff_topics':>11} {'JSD(arm‖fpdf)':>14} {'mean_sim':>9}")
    print("─" * 90)
    print(f"{'FinePDFs ref':<14} {int(finepdfs_dist.sum() * audit['topic_doc_count'].sum()):>4d} "
          f"{(finepdfs_dist > 0).sum():>6d} "
          f"{((finepdfs_dist * audit['topic_doc_count'].sum()) >= 5).sum():>6d} "
          f"{100*(finepdfs_dist>0).mean():>5.1f}% "
          f"{gini(finepdfs_dist):>7.3f} "
          f"{np.exp(-np.sum(np.where(finepdfs_dist>0, finepdfs_dist * np.log(finepdfs_dist + 1e-12), 0))):>11.1f} "
          f"{'—':>14} {'—':>9}")
    for arm, dist in arm_dists.items():
        n = int(dist.sum())
        occ1 = int((dist >= 1).sum())
        occ5 = int((dist >= 5).sum())
        cov = 100 * occ1 / n_topics
        dist_norm = dist / max(dist.sum(), 1)
        eff = float(np.exp(-np.sum(np.where(dist_norm > 0, dist_norm * np.log(dist_norm + 1e-12), 0))))
        d_jsd = jsd(dist, finepdfs_dist)
        ms = float(arm_nearest_sims[arm].mean())
        print(f"{arm:<14} {n:>4d} {occ1:>6d} {occ5:>6d} {cov:>5.1f}% "
              f"{gini(dist):>7.3f} {eff:>11.1f} {d_jsd:>14.3f} {ms:>9.3f}")

    # Topic-level top-10 by arm
    print()
    print("Top-10 topics by chapter count per arm:")
    for arm, dist in arm_dists.items():
        order = np.argsort(-dist)[:10]
        print(f"\n  {arm}:")
        for tid in order:
            if dist[tid] == 0:
                break
            fp = int(audit['topic_doc_count'].iloc[tid])
            rep = audit['topic_repr_text'].iloc[tid][:90].replace('\n', ' ')
            print(f"    topic {tid:3d}: {int(dist[tid]):3d} chap  ({fp:3d} FinePDFs)  | {rep}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
