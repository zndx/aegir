"""
Training utilities: loss functions, parameter grouping, evaluation metrics.

Combines:
- Load balancing loss from H-Net (routing regularization)
- Parameter grouping from H-Net (per-stage LR multipliers)
- F1 metrics from REVEAL (CTA/CPA evaluation)
"""

import numpy as np
import torch

from aegir.modules.dc import RoutingModuleOutput
from aegir.modules.utils import apply_optimization_params


def boundary_diagnostics(
    bpred_output: list[RoutingModuleOutput] | None,
    target_N: float,
) -> dict[str, float]:
    """Compute per-stage chunking diagnostics for the current batch.

    Returns a flat dict keyed by ``stage{i}_{metric}``:

      - ``mean_F``     — actual boundary selection rate (fraction of tokens kept).
      - ``mean_G``     — mean predicted boundary probability.
      - ``achieved_N`` — effective downsample factor (1 / mean_F).
      - ``abs_ratio_err`` — |mean_F - 1/target_N| (0 means perfect balance).

    ``bpred_output`` is the list Aegir returns alongside logits. None / empty →
    empty dict, which the training loop accumulator treats as "no chunking stages
    this batch" (e.g. the isotropic-only smoke config).

    Interpreting the numbers: at training start mean_F is usually ~0.5 (random
    boundaries), mean_G near 0.5. As load-balancing loss pulls toward the target
    the two values should converge to ~1/target_N. If they drift apart, the STE
    gradient isn't threading through — a useful early-warning signal.
    """
    if not bpred_output:
        return {}
    out: dict[str, float] = {}
    for i, router in enumerate(bpred_output):
        mask = router.boundary_mask.float()
        prob = router.boundary_prob[..., -1].float()
        mean_F = mask.mean().item()
        mean_G = prob.mean().item()
        achieved_N = (1.0 / mean_F) if mean_F > 1e-6 else float("inf")
        out[f"stage{i}_mean_F"] = mean_F
        out[f"stage{i}_mean_G"] = mean_G
        out[f"stage{i}_achieved_N"] = achieved_N
        out[f"stage{i}_abs_ratio_err"] = abs(mean_F - 1.0 / target_N)
    return out


def format_boundary_diagnostics(diag: dict[str, float]) -> str:
    """One-line string for the per-epoch log. Example:

        chunking: stage0 F=0.33 G=0.31 N=3.0 err=0.010 | stage1 F=0.50 ...
    """
    if not diag:
        return ""
    # Infer stage indices from keys.
    stages = sorted({int(k.split("_", 1)[0][5:]) for k in diag if k.startswith("stage")})
    parts: list[str] = []
    for s in stages:
        F = diag.get(f"stage{s}_mean_F", float("nan"))
        G = diag.get(f"stage{s}_mean_G", float("nan"))
        N = diag.get(f"stage{s}_achieved_N", float("nan"))
        err = diag.get(f"stage{s}_abs_ratio_err", float("nan"))
        parts.append(f"stage{s} F={F:.3f} G={G:.3f} N={N:4.1f} err={err:.3f}")
    return "chunking: " + " | ".join(parts)


def load_balancing_loss(
    router_output: RoutingModuleOutput,
    N: float,
) -> torch.Tensor:
    """Compute load balancing loss for routing regularization.

    Encourages the routing module to maintain a balanced downsampling ratio
    by penalizing deviations from the target boundary selection rate.

    Args:
        router_output: Output from the RoutingModule.
        N: Downsampling factor (e.g. 2.0 for 50% downsampling). Must be > 1.

    Returns:
        Scalar loss tensor.
    """
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask

    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()

    return (
        (1 - true_ratio) * (1 - average_prob)
        + (true_ratio) * (average_prob) * (N - 1)
    ) * N / (N - 1)


def group_params(model) -> list[dict]:
    """Create optimizer parameter groups with per-stage LR multipliers.

    Parameters are grouped by their ``_optim`` annotation (set by
    ``apply_lr_multiplier``). Bias and norm parameters get zero weight decay.

    Args:
        model: Model with parameters annotated via ``apply_optimization_params``.

    Returns:
        List of parameter group dicts for the optimizer.
    """
    for name, param in model.named_parameters():
        if not hasattr(param, "_optim"):
            param._optim = {}
        if name.endswith(".bias") or ".norm." in name:
            apply_optimization_params(param, weight_decay=0.0)

    all_keys = set()
    for _name, param in model.named_parameters():
        all_keys.update(param._optim.keys())

    all_keys = list(all_keys)
    all_tuples = []
    param_groups = []

    for _name, param in model.named_parameters():
        current_tuple = tuple(param._optim.get(key, None) for key in all_keys)
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            param_groups.append({
                "params": [param],
                **param._optim,
            })
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)

    return param_groups


def f1_score_multilabel(
    true_list: list[int],
    pred_list: list[int],
    num_classes: int = None,
) -> tuple[float, float, np.ndarray]:
    """Compute micro and macro F1 scores.

    Args:
        true_list: Ground truth labels.
        pred_list: Predicted labels.
        num_classes: Number of classes (optional).

    Returns:
        Tuple of (micro_f1, macro_f1, per_class_f1).
    """
    from sklearn.metrics import multilabel_confusion_matrix

    conf_mat = multilabel_confusion_matrix(true_list, pred_list)
    agg_conf_mat = conf_mat.sum(axis=0)

    # Micro F1
    tp = agg_conf_mat[1, 1]
    precision = tp / agg_conf_mat[1, :].sum() if agg_conf_mat[1, :].sum() > 0 else 0.0
    recall = tp / agg_conf_mat[:, 1].sum() if agg_conf_mat[:, 1].sum() > 0 else 0.0
    micro_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Macro F1 (per-class average)
    per_class_f1 = []
    for i in range(conf_mat.shape[0]):
        cm = conf_mat[i]
        tp_i = cm[1, 1]
        p_i = tp_i / cm[1, :].sum() if cm[1, :].sum() > 0 else 0.0
        r_i = tp_i / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0.0
        f1_i = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0
        per_class_f1.append(f1_i)

    per_class_f1 = np.array(per_class_f1)
    macro_f1 = per_class_f1.mean()

    return micro_f1, macro_f1, per_class_f1


def supcon_loss(
    embeddings: torch.Tensor,
    cluster_ids: torch.Tensor,
    col_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., NeurIPS 2020).

    Treats every ``(sample_i_col_j, sample_k_col_l)`` pair with matching
    ``cluster_ids`` as a positive — regardless of whether they live in the
    same table — which is precisely the DED objective: columns with the same
    real-world data element cluster together in embedding space.

    Args:
        embeddings: (B, N, D) L2-normalized column embeddings.
        cluster_ids: (B, N) int64 cluster labels. ``-1`` = ignore.
        col_mask: (B, N) bool flagging valid (non-padded) columns.
        temperature: SupCon temperature. 0.1 is standard; lower concentrates
            positives, higher softens the distribution.

    Returns:
        Scalar loss. Returns 0 when fewer than 2 valid columns share any
        cluster (no positives to pull together).
    """
    import torch
    import torch.nn.functional as F

    B, N, D = embeddings.shape
    flat_emb = embeddings.reshape(B * N, D)
    flat_cid = cluster_ids.reshape(B * N)
    flat_mask = col_mask.reshape(B * N) & (flat_cid >= 0)

    valid = flat_mask.nonzero(as_tuple=True)[0]
    if valid.numel() < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    emb = flat_emb[valid]                                    # (M, D)
    cid = flat_cid[valid]                                    # (M,)

    # Similarity matrix — cosine because embeddings are L2-normalized.
    logits = emb @ emb.T / temperature                       # (M, M)
    # Stabilize: subtract per-row max. Positives keep their relative ordering.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Positive mask: same cluster, excluding self.
    pos = cid.unsqueeze(0) == cid.unsqueeze(1)               # (M, M)
    eye = torch.eye(pos.shape[0], device=pos.device, dtype=torch.bool)
    pos = pos & ~eye
    # Denominator: all non-self pairs.
    all_other = ~eye

    # If a row has no positives, exclude it entirely (its loss is ill-defined).
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    exp_logits = torch.exp(logits) * all_other.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-prob over positives, averaged across rows with ≥1 positive.
    pos_count = pos.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (pos * log_prob).sum(dim=1) / pos_count
    loss = -mean_log_prob_pos[has_pos].mean()
    return loss


def bcubed_f1(
    predictions: torch.Tensor,
    cluster_ids: torch.Tensor,
    col_mask: torch.Tensor,
) -> tuple[float, float, float]:
    """B-cubed precision, recall, F1 for clustering evaluation.

    B-cubed (Bagga & Baldwin, 1998) computes per-item precision/recall based
    on the item's own cluster, avoiding the pitfalls of matching-based
    metrics like purity. Standard for cluster-based entity resolution and
    data element discovery evaluation.

    Args:
        predictions: (B, N) predicted cluster IDs. Clustering can come from
            any algorithm — the evaluation is label-free up to a permutation.
        cluster_ids: (B, N) ground-truth cluster IDs. ``-1`` = ignore.
        col_mask: (B, N) bool flagging valid columns.

    Returns:
        (precision, recall, f1) tuple of floats.
    """
    import torch

    flat_pred = predictions.reshape(-1)
    flat_true = cluster_ids.reshape(-1)
    flat_mask = col_mask.reshape(-1) & (flat_true >= 0)

    pred = flat_pred[flat_mask]
    true = flat_true[flat_mask]
    n = pred.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0

    # Per-item B-cubed: precision = |pred_cluster ∩ true_cluster| / |pred_cluster|,
    # recall similarly with denom |true_cluster|.
    same_pred = pred.unsqueeze(0) == pred.unsqueeze(1)       # (n, n)
    same_true = true.unsqueeze(0) == true.unsqueeze(1)
    tp_mask = same_pred & same_true

    tp_per_row = tp_mask.sum(dim=1).float()
    pred_size = same_pred.sum(dim=1).float().clamp(min=1.0)
    true_size = same_true.sum(dim=1).float().clamp(min=1.0)

    precision = (tp_per_row / pred_size).mean().item()
    recall = (tp_per_row / true_size).mean().item()
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
