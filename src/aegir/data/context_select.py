"""
Context column selection via Maximal Marginal Relevance (MMR).

Selects a diverse, relevant subset of context columns for a target column.
Adapted from REVEAL (arXiv:2508.17203).
"""

import torch
from sentence_transformers import SentenceTransformer, util


_DEFAULT_MODEL = None


def _get_embedding_model() -> SentenceTransformer:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _DEFAULT_MODEL


def embed_columns(
    columns: list[list[str]],
    model: SentenceTransformer = None,
) -> torch.Tensor:
    """Embed column contents by concatenating cell values into strings.

    Args:
        columns: List of columns, each a list of cell values.
        model: SentenceTransformer model. Uses default if None.

    Returns:
        Embeddings tensor, shape (num_columns, embedding_dim).
    """
    if model is None:
        model = _get_embedding_model()
    texts = [" ".join(str(v) for v in col[:50]) for col in columns]
    return model.encode(texts, convert_to_tensor=True)


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
