"""Binding: topic-cluster-level train/holdout partition of FinePDFs (DOF 8).

Partitions the FinePDFs BERTopic *clusters* (not documents) into train / holdout
BEFORE any generation: held-out clusters seed/ground NO generated unit and serve the
downstream generality check; generation seeds draw only from training clusters. This
enforces cross-*region* generality, and (being deterministic in ``seed``) is stable
and leakage-free. The BERTopic noise cluster (-1) is excluded from both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aegir.ontology.bindings.topic_recovery import DEFAULT_CALIB


def _frac(topic_id: int, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{topic_id}".encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def partition_clusters(calib_dir: str | Path = DEFAULT_CALIB,
                       holdout_frac: float = 0.2, seed: int = 4649) -> dict:
    """Deterministically split topic clusters into train / holdout."""
    topic_ids = [int(t) for t in json.loads((Path(calib_dir) / "topic_ids.json").read_text())]
    real = [t for t in topic_ids if t != -1]
    holdout = sorted(t for t in real if _frac(t, seed) < holdout_frac)
    train = sorted(t for t in real if _frac(t, seed) >= holdout_frac)
    return {
        "train": train,
        "holdout": holdout,
        "noise_excluded": [-1] if -1 in topic_ids else [],
        "seed": seed,
        "holdout_frac": holdout_frac,
        "n_train": len(train),
        "n_holdout": len(holdout),
    }


def save_partition(part: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(part, indent=2))


def load_partition(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
