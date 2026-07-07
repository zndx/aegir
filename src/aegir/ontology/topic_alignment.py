"""Topic-alignment computation for the SDG verifier's R_D component.

.. deprecated:: 2026-07-07
   V0.3-ERA LEGACY. The corpus-fitted topic model (T_I.pkl, KMeans/BERTopic-style) was
   superseded by the INVERTED TOPIC LAYER (``aegir.ontology.topic_layer`` — topics ≡
   ontology-grounded qdrant anchors, margin-gated item associations, RH 2026-07-07
   rulings a-h) and, in the live flow, by ``aegir.ontology.congruence`` (the R_D idea
   reborn over the ColBERT/qdrant substrate). Kept for the v0.3 pipeline's
   reproducibility; do not build new consumers on it.

Implements the topic-alignment piece of the four-component
verifier R(O, I) defined in
``docs/current/src/ontology/concept_brief.md`` v0.5. The brief commits
to BERTopic + Hungarian matching; this module implements the
methodologically-equivalent **sentence-embedding + KMeans
clustering + Hungarian matching** chain that captures the same
semantic principle (embed sentences, cluster into topic
centroids, match centroids across corpora) without a hard
dependency on BERTopic / UMAP / HDBSCAN.

Architecture:

1. **Encoder** — sentence-transformers / all-MiniLM-L6-v2 loaded
   directly via ``transformers`` to avoid the
   sentence-transformers ⇒ datasets ⇒ pyarrow API mismatch path.
2. **Topic model fit** — encode sentences to 384-d embeddings,
   L2-normalize, fit KMeans into ``k`` clusters, take L2-
   normalized centroids as topic representations.
3. **Alignment** — Hungarian-optimal one-to-one matching between
   *T_V* and *T_I* centroids by cosine similarity; mean matched
   similarity is the raw alignment score.
4. **Normalization** — raw score normalized into [0, 1] against
   the structural-shuffle null distribution: ``null_mean``
   maps to 0, ``null_p95`` (or another high-percentile threshold)
   maps to 1. Clipped to [0, 1].

Wall-clock budget on CPU at the current scale (10K-doc *I*,
sub-100-doc *V(O)*): ~8 min one-shot for *T_I* fit (cached);
~3 s per runtime *T_V* fit + alignment.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 100
DEFAULT_RANDOM_STATE = 4649

_ENCODER_CACHE: dict[str, Any] = {}


def get_encoder(model_name: str = DEFAULT_ENCODER_MODEL):
    """Return ``(tokenizer, model)`` from cache or load fresh.

    Loaded via ``transformers`` directly to avoid the
    sentence-transformers transitive ``datasets`` import that
    breaks under the current pyarrow API.
    """
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]

    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    _ENCODER_CACHE[model_name] = (tok, mdl)
    logger.info("loaded encoder %s", model_name)
    return tok, mdl


def encode_sentences(
    sentences: list[str],
    encoder=None,
    batch_size: int = 64,
    max_length: int = 256,
) -> np.ndarray:
    """Mean-pool + L2-normalize embeddings via direct transformers."""
    import torch
    import torch.nn.functional as F

    if encoder is None:
        encoder = get_encoder()
    tok, mdl = encoder

    all_embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.numpy())

    return np.concatenate(all_embs, axis=0)


@dataclass
class TopicModel:
    """A clustering of sentences in a corpus into ``k`` topic centroids.

    ``centroids`` is shape ``(k, embed_dim)``, L2-normalized so
    that pairwise cosine similarity is just an inner product.
    """

    centroids: np.ndarray
    k: int
    n_docs: int
    encoder_model: str = DEFAULT_ENCODER_MODEL
    config: dict[str, Any] = field(default_factory=dict)


def fit_topic_model(
    sentences: list[str],
    k: int = DEFAULT_K,
    encoder=None,
    random_state: int = DEFAULT_RANDOM_STATE,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
) -> TopicModel:
    """Fit a topic model on ``sentences``: encode + KMeans-cluster.

    If fewer sentences than ``k`` are supplied, ``k`` is reduced
    accordingly. The smallest meaningful ``k`` is 2.
    """
    if encoder is None:
        encoder = get_encoder(encoder_model)
    embeddings = encode_sentences(sentences, encoder)

    effective_k = max(2, min(k, len(sentences)))
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=effective_k, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    centroids = km.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-9, None)

    return TopicModel(
        centroids=centroids,
        k=effective_k,
        n_docs=len(sentences),
        encoder_model=encoder_model,
        config={"random_state": random_state, "requested_k": k},
    )


def alignment_score(t_v: TopicModel, t_i: TopicModel) -> float:
    """Hungarian-optimal mean cosine similarity between *T_V* and
    *T_I* centroids.

    Returns the mean matched cosine similarity over the smaller
    of the two topic counts. Both centroid arrays are assumed
    L2-normalized; we still re-normalize defensively.
    """
    from scipy.optimize import linear_sum_assignment

    a = t_v.centroids / np.clip(np.linalg.norm(t_v.centroids, axis=1, keepdims=True), 1e-9, None)
    b = t_i.centroids / np.clip(np.linalg.norm(t_i.centroids, axis=1, keepdims=True), 1e-9, None)
    sim = a @ b.T

    row, col = linear_sum_assignment(-sim)  # negate to maximize
    matched = sim[row, col]
    return float(matched.mean())


def save_topic_model(model: TopicModel, path: str | Path) -> None:
    """Pickle a topic model to disk for caching across runs."""
    payload = {
        "centroids": model.centroids,
        "k": model.k,
        "n_docs": model.n_docs,
        "encoder_model": model.encoder_model,
        "config": model.config,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("saved topic model to %s", path)


def load_topic_model(path: str | Path) -> TopicModel:
    """Load a pickled topic model from disk."""
    with open(path, "rb") as f:
        payload = cast(dict[str, Any], pickle.load(f))
    return TopicModel(
        centroids=payload["centroids"],
        k=payload["k"],
        n_docs=payload["n_docs"],
        encoder_model=payload.get("encoder_model", DEFAULT_ENCODER_MODEL),
        config=payload.get("config", {}),
    )


def normalize_alignment(
    raw: float,
    null_mean: float,
    null_p95: float,
) -> float:
    """Normalize raw alignment to [0, 1] using null statistics.

    The brief commits ``null_mean → 0`` and ``null_p95 → 1`` with
    clipping. A composition that scores at the null mean produces
    R_D = 0; one that exceeds the 95th percentile of random
    compositions produces R_D = 1.
    """
    spread = null_p95 - null_mean
    if spread < 1e-6:
        return 0.5
    return float(np.clip((raw - null_mean) / spread, 0.0, 1.0))
