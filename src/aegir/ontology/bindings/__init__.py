"""Live backends that bind the closed loop to real infrastructure for the
calibration run: the frozen FinePDFs topic model (re-grounding score), the
topic-cluster train/holdout partition, and the GLM/Grok generation mix.
"""

from aegir.ontology.bindings.holdout import (
    load_partition,
    partition_clusters,
    save_partition,
)
from aegir.ontology.bindings.llm_generate import make_generate_fn, parse_mix
from aegir.ontology.bindings.topic_recovery import (
    make_topic_recovery_fn,
    score_text,
)

__all__ = [
    "make_topic_recovery_fn", "score_text",
    "partition_clusters", "save_partition", "load_partition",
    "make_generate_fn", "parse_mix",
]
