"""Held-out evaluation harness for the P5-trained policy.

Combines two sources:

1. The C1 verifier-validation test set at
   ``tests/ontology_test_set/`` (15 good + 15 bad TTL ontologies
   labelled by the C1 sweep). Reused unchanged from P2.
2. A fresh held-out set of 50 ontologies authored after P5
   training begins, so the policy cannot have seen them via
   any catalog-side leakage.

The metric is policy-mean R against the locked verifier on each
set, plus AUC against the labels for the C1 portion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from aegir.ontology.schema import load_catalog
from aegir.ontology.verifier import CompositionEntry, verify

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    catalog_path: str = "src/aegir/ontology/catalog/combined.json"
    t_i_cache_path: str = "src/aegir/ontology/T_I.pkl"
    null_stats_path: str = "src/aegir/ontology/null_stats.json"
    c1_test_set_dir: str = "tests/ontology_test_set"
    held_out_50_path: str = "tests/p5_held_out_50/labels.json"
    n_compositions_per_prompt: int = 4


@dataclass
class EvalResult:
    set_name: str
    n_compositions: int
    mean_r: float
    median_r: float
    std_r: float
    r_a_pass_rate: float
    auc: float | None = None
    per_ontology: list[dict] = field(default_factory=list)


def evaluate(
    cfg: EvalConfig,
    rollout_fn,
    set_name: str,
    prompts: list[dict],
) -> EvalResult:
    """Evaluate the policy on a list of prompts. Each prompt
    is a dict with keys ``ontology_id``, ``prompt_text``, and
    optionally ``label`` (1=good, 0=bad). Returns aggregated
    metrics + per-ontology breakdown.
    """
    catalog = load_catalog(cfg.catalog_path)
    rs: list[float] = []
    pass_count = 0
    per_ontology: list[dict] = []
    labels: list[int] = []
    scores: list[float] = []

    for p in prompts:
        compositions = rollout_fn(p["prompt_text"], cfg.n_compositions_per_prompt)
        per_prompt_rs: list[float] = []
        for comp in compositions:
            entries = [
                CompositionEntry(template_id=e["template_id"], slot_fillers=e["slot_fillers"])
                for e in comp
            ]
            res = verify(
                entries, catalog,
                t_i_cache_path=cfg.t_i_cache_path,
                null_stats_path=cfg.null_stats_path,
            )
            per_prompt_rs.append(res.R)
            rs.append(res.R)
            if res.R_A > 0:
                pass_count += 1
        mean_per_prompt = sum(per_prompt_rs) / max(len(per_prompt_rs), 1)
        per_ontology.append({
            "ontology_id": p["ontology_id"],
            "label": p.get("label"),
            "n_compositions": len(compositions),
            "mean_r": mean_per_prompt,
            "min_r": min(per_prompt_rs) if per_prompt_rs else 0.0,
            "max_r": max(per_prompt_rs) if per_prompt_rs else 0.0,
        })
        if "label" in p:
            labels.append(p["label"])
            scores.append(mean_per_prompt)

    n = max(len(rs), 1)
    sorted_rs = sorted(rs)
    median = sorted_rs[n // 2] if sorted_rs else 0.0
    mean = sum(rs) / n
    std = (sum((r - mean) ** 2 for r in rs) / max(n - 1, 1)) ** 0.5

    auc = None
    if labels and len(set(labels)) >= 2:
        auc = _auc(labels, scores)

    return EvalResult(
        set_name=set_name,
        n_compositions=len(rs),
        mean_r=mean,
        median_r=median,
        std_r=std,
        r_a_pass_rate=pass_count / max(len(rs), 1),
        auc=auc,
        per_ontology=per_ontology,
    )


def _auc(labels: list[int], scores: list[float]) -> float:
    """ROC AUC via the rank-sum identity. No sklearn dep."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = 0.0
    for rank, (_, lbl) in enumerate(pairs, start=1):
        if lbl == 1:
            rank_sum_pos += rank
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_c1_prompts(cfg: EvalConfig) -> list[dict]:
    """Load the C1 test set as evaluation prompts."""
    labels_path = Path(cfg.c1_test_set_dir) / "labels.json"
    if not labels_path.exists():
        return []
    raw = json.loads(labels_path.read_text())
    out: list[dict] = []
    for ttl_name, label in raw.items():
        out.append({
            "ontology_id": ttl_name,
            "prompt_text": f"Generate a domain-aligned ontology composition matching the style of {ttl_name}.",
            "label": int(label),
        })
    return out


def load_held_out_50(cfg: EvalConfig) -> list[dict]:
    """Load the (eventual) 50-ontology held-out set authored
    after P5 begins. Returns ``[]`` if not yet created."""
    p = Path(cfg.held_out_50_path)
    if not p.exists():
        return []
    return json.loads(p.read_text())
