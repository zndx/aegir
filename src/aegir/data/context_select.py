"""
Context column selection via Maximal Marginal Relevance (MMR).

Selects a diverse, relevant subset of context columns for a target column.
Adapted from REVEAL (arXiv:2508.17203).

MMR requires an embedding per column — 50-cell-prefix concatenated and
passed through MPNet. Dominant cost of dataset __init__ on non-trivial
corpora (GitTables full has 1M tables × 10-50 cols). Results are cached
to disk (blake2b(content) → fp16 numpy vector in a sharded SQLite DB)
so re-runs skip the encoder entirely. Set ``AEGIR_MMR_CACHE_DIR`` to
override the default location.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


_DEFAULT_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME = "all-mpnet-base-v2"
_MODEL_DIM = 768  # fixed for all-mpnet-base-v2
_CELL_PREFIX = 50  # cells per column to hash + encode


def _get_embedding_model() -> SentenceTransformer:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = SentenceTransformer(_MODEL_NAME)
    return _DEFAULT_MODEL


class _MMRCache:
    """SQLite-backed blake2b-keyed embedding cache.

    Thread-local connection so dataloader workers each get their own
    handle without serialization through a shared lock. Writes use
    WAL mode so reads never block.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        # Ensure schema exists (one connection handles CREATE TABLE).
        with sqlite3.connect(self.path) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                "CREATE TABLE IF NOT EXISTS emb (key BLOB PRIMARY KEY, vec BLOB NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._tls.conn = c
        return c

    def get_many(self, keys: list[bytes]) -> dict[bytes, np.ndarray]:
        if not keys:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" * len(keys))
        rows = conn.execute(
            f"SELECT key, vec FROM emb WHERE key IN ({placeholders})", keys
        ).fetchall()
        return {
            k: np.frombuffer(v, dtype=np.float16) for k, v in rows
        }

    def put_many(self, items: list[tuple[bytes, np.ndarray]]) -> None:
        if not items:
            return
        conn = self._conn()
        payload = [(k, v.astype(np.float16).tobytes()) for k, v in items]
        conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", payload)


_CACHE: Optional[_MMRCache] = None


def _get_cache() -> Optional[_MMRCache]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if os.environ.get("AEGIR_MMR_CACHE_DISABLE") == "1":
        return None
    cache_dir = Path(
        os.environ.get("AEGIR_MMR_CACHE_DIR")
        or Path.home() / ".cache" / "aegir" / "mmr"
    )
    try:
        _CACHE = _MMRCache(cache_dir / f"{_MODEL_NAME}.sqlite3")
        return _CACHE
    except Exception:
        # Disk full, read-only fs, etc. — fall back to in-memory only.
        return None


def _column_key(col_cells: list[str]) -> bytes:
    """Cache key = blake2b of the first N cells joined with NUL.

    NUL is safe because parquet cells don't contain it. Prefix size matches
    what `embed_columns` actually encodes so we never cache embeddings
    computed from a different amount of context.
    """
    text = "\x00".join(str(v) for v in col_cells[:_CELL_PREFIX])
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).digest()


def embed_columns(
    columns: list[list[str]],
    model: SentenceTransformer = None,
) -> torch.Tensor:
    """Embed column contents, fetching from disk cache where possible.

    Args:
        columns: List of columns, each a list of cell values.
        model: SentenceTransformer model. Uses default if None.

    Returns:
        Embeddings tensor, shape (num_columns, embedding_dim), dtype float32
        on the model's device (CPU by default).
    """
    keys = [_column_key(col) for col in columns]
    cache = _get_cache()
    cached = cache.get_many(keys) if cache is not None else {}

    out = np.zeros((len(columns), _MODEL_DIM), dtype=np.float32)
    missing_idxs: list[int] = []
    for i, key in enumerate(keys):
        vec = cached.get(key)
        if vec is not None and vec.shape[0] == _MODEL_DIM:
            out[i] = vec.astype(np.float32)
        else:
            missing_idxs.append(i)

    if missing_idxs:
        if model is None:
            model = _get_embedding_model()
        texts = [
            " ".join(str(v) for v in columns[i][:_CELL_PREFIX])
            for i in missing_idxs
        ]
        new_embs = model.encode(texts, convert_to_numpy=True).astype(np.float32)
        for j, i in enumerate(missing_idxs):
            out[i] = new_embs[j]
        if cache is not None:
            cache.put_many([(keys[i], out[i]) for i in missing_idxs])

    return torch.from_numpy(out)


def maximal_marginal_relevance(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    top_k: int = 8,
    lambda_param: float = 0.5,
) -> list[int]:
    """Select top_k candidates balancing relevance and diversity.

    Args:
        query_embedding: Target column embedding, shape (D,) or (1, D).
        candidate_embeddings: Context candidate embeddings, shape (N, D).
        top_k: Number of context columns to select.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        List of selected candidate indices.
    """
    if candidate_embeddings.shape[0] == 0:
        return []

    top_k = min(top_k, candidate_embeddings.shape[0])
    selected_indices = []
    remaining_indices = list(range(candidate_embeddings.shape[0]))

    query_similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]

    for _ in range(top_k):
        mmr_scores = []
        for idx in remaining_indices:
            relevance = query_similarities[idx].item()

            if selected_indices:
                selected_embs = candidate_embeddings[selected_indices]
                diversity_scores = util.cos_sim(
                    candidate_embeddings[idx].unsqueeze(0), selected_embs
                )[0]
                max_diversity = diversity_scores.max().item()
            else:
                max_diversity = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_diversity
            mmr_scores.append(mmr)

        best_local_idx = max(range(len(mmr_scores)), key=lambda i: mmr_scores[i])
        best_idx = remaining_indices[best_local_idx]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices
