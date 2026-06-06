"""Binding: ``topic_recovery_fn`` → the FROZEN FinePDFs topic model.

Loads the L2-normalized topic centroids (all-mpnet-base-v2, 141 topics) fit on 10K
FinePDFs docs by ``scripts/experiment_topic_recovery.py --phase calibrate``, and
scores a generated chapter's re-grounding to its seeding FinePDFs topic by cosine to
the centroids (the volume-independent per-doc metric from the 2026-05-31 study).

The model is FROZEN and versioned (DOF 3): the same embed+centroid pipeline scores
in-loop and at the downstream generality check. Encoder + centroids are cached.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CALIB = REPO / "build/experiments/topic_recovery_v0/calibration"
ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"


@functools.lru_cache(maxsize=4)
def _load_centroids(calib_dir: str):
    d = Path(calib_dir)
    centroids = np.load(d / "topic_centroids.npy")          # (n_topics, 768), L2-normalized
    topic_ids = json.loads((d / "topic_ids.json").read_text())
    id_to_row = {int(t): i for i, t in enumerate(topic_ids)}
    return centroids, topic_ids, id_to_row


@functools.lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER_NAME)


def score_text(text: str, target_topic_id: int, calib_dir: str | Path = DEFAULT_CALIB) -> dict:
    """Cosine of the text's embedding to every frozen topic centroid; report the
    target topic's cosine and rank (0 = nearest)."""
    centroids, _topic_ids, id_to_row = _load_centroids(str(calib_dir))
    emb = _encoder().encode([text[:8000]], normalize_embeddings=True)[0]   # (768,)
    sims = centroids @ emb                                                  # cosine (centroids normalized)
    order = np.argsort(-sims)
    row = id_to_row.get(int(target_topic_id))
    if row is None:                       # tolerate a raw row index
        row = int(target_topic_id) if 0 <= int(target_topic_id) < len(sims) else int(order[0])
    rank = int(np.where(order == row)[0][0])
    return {
        "cosine_to_target": float(sims[row]),
        "rank": rank,
        "hit_at_1": rank == 0,
        "hit_at_5": rank < 5,
        "top_cosine": float(sims[order[0]]),
        "top_topic_row": int(order[0]),
    }


def make_topic_recovery_fn(calib_dir: str | Path = DEFAULT_CALIB,
                           mode: str = "cosine_to_target"):
    """Return ``topic_recovery_fn(chapter_md, target_topic_id) -> float`` ∈ [0,1].

    ``mode``: ``cosine_to_target`` (graded, default) or ``hit_at_1`` (binary). The
    τ_topic threshold against this score is a DOF-2 calibration knob; the binding's
    job is a frozen, discriminating score.
    """
    def topic_recovery_fn(chapter_md: str, target_topic_id) -> float:
        s = score_text(chapter_md, int(target_topic_id), calib_dir)
        if mode == "hit_at_1":
            return 1.0 if s["hit_at_1"] else 0.0
        return max(0.0, s["cosine_to_target"])

    return topic_recovery_fn
