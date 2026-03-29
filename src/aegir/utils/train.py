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
