"""score_ontology — compute R(O, I) directly on a TTL/OWL artifact.

Companion to ``aegir-verify`` for the C1 verifier-validation
workflow. Whereas ``aegir-verify`` scores compositions over the
catalog, ``score_ontology`` scores arbitrary OWL ontology
artifacts, computing each R-component from DeepOnto-derived
properties:

- ``R_A`` — 1.0 iff the ontology loads in DeepOnto, else 0.0
  (hard gate; matches the catalog-side R_A semantics).
- ``R_B`` — ``min(|asserted_complex| / τ_B, 1)``.
- ``R_C`` — ``mean(verbalization_length) / L_target``, clipped to
  [0, 1]. Verbalization is run over named classes + complex
  expressions; failures are silently dropped from the mean.
- ``R_D`` — Hungarian-optimal cosine alignment between *T_V*
  (BERTopic-equivalent on the ontology's verbalizations) and
  the cached *T_I*, normalized against the catalog's null
  statistics.

Output is a single JSON record on stdout with full per-component
breakdown plus diagnostics. Designed to be redirected into the
test-set score file consumed by ``tune_verifier_weights``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_T_I_CACHE = REPO_ROOT / "src" / "aegir" / "ontology" / "T_I.pkl"
DEFAULT_NULL_STATS = REPO_ROOT / "src" / "aegir" / "ontology" / "null_stats.json"

TAU_B = 5.0
L_TARGET = 80.0
DEFAULT_TV_K = 12
W_R_B, W_R_C, W_R_D = 0.50, 0.05, 0.45  # Locked at P2; see aegir-verify.py.


def aggregate(R_A: float, R_B: float, R_C: float, R_D: float,
              w_b: float = W_R_B, w_c: float = W_R_C, w_d: float = W_R_D) -> float:
    return R_A * (w_b * R_B + w_c * R_C + w_d * R_D)


def gather_verbalizations(onto, verbaliser) -> list[str]:
    """Verbalize every named class + complex expression that
    DeepOnto can render. Failures are silently dropped (per
    P1b-α design — no manufactured fallbacks)."""
    out: list[str] = []

    # Named classes via subsumption axioms (covers most ontologies).
    for ax in onto.get_subsumption_axioms("Classes"):
        try:
            v = verbaliser.verbalise_class_subsumption_axiom(ax)
            if isinstance(v, tuple) and len(v) == 2:
                sub_v = v[0].verbal if hasattr(v[0], "verbal") else str(v[0])
                sup_v = v[1].verbal if hasattr(v[1], "verbal") else str(v[1])
                if sub_v and sup_v:
                    out.append(f"{sub_v} is a {sup_v}")
            elif hasattr(v, "verbal"):
                out.append(v.verbal)
        except Exception:
            continue

    # Equivalence axioms.
    for ax in onto.get_equivalence_axioms("Classes"):
        try:
            v = verbaliser.verbalise_class_equivalence_axiom(ax)
            if isinstance(v, tuple) and len(v) == 2:
                a = v[0].verbal if hasattr(v[0], "verbal") else str(v[0])
                b = v[1].verbal if hasattr(v[1], "verbal") else str(v[1])
                if a and b:
                    out.append(f"{a} is equivalent to {b}")
            elif hasattr(v, "verbal"):
                out.append(v.verbal)
        except Exception:
            continue

    # Complex expressions.
    for cc in onto.get_asserted_complex_classes():
        try:
            v = verbaliser.verbalise_class_expression(cc)
            body = v.verbal if hasattr(v, "verbal") else ""
            if body:
                out.append(f"something {body}")
        except Exception:
            continue

    # Filter empties / very-short.
    return [s for s in out if s and len(s) >= 5]


def score_owl_file(
    owl_path: Path,
    t_i_cache_path: Path,
    null_stats_path: Path,
    tv_k: int = DEFAULT_TV_K,
) -> dict:
    """Score a single OWL artifact end-to-end."""
    from aegir.ontology.deeponto_harness import ensure_jvm
    from aegir.ontology.topic_alignment import (
        alignment_score,
        fit_topic_model,
        get_encoder,
        load_topic_model,
        normalize_alignment,
    )
    ensure_jvm()
    from deeponto.onto import Ontology, OntologyVerbaliser

    record: dict = {
        "file": str(owl_path),
        "R": 0.0,
        "R_A": 0.0,
        "R_B": 0.0,
        "R_C": 0.0,
        "R_D": 0.0,
        "raw_R_D": 0.0,
        "n_classes": 0,
        "n_complex": 0,
        "n_verbalizations": 0,
        "mean_verbal_length": 0.0,
        "error": "",
    }

    try:
        onto = Ontology(str(owl_path))
    except Exception as e:
        record["error"] = f"load failure: {e}"
        return record

    record["R_A"] = 1.0
    record["n_classes"] = len(list(onto.owl_classes))

    complex_classes = list(onto.get_asserted_complex_classes())
    record["n_complex"] = len(complex_classes)
    record["R_B"] = float(min(record["n_complex"] / TAU_B, 1.0))

    verbaliser = OntologyVerbaliser(onto)
    verbalizations = gather_verbalizations(onto, verbaliser)
    record["n_verbalizations"] = len(verbalizations)

    if verbalizations:
        mean_len = float(np.mean([len(s) for s in verbalizations]))
        record["mean_verbal_length"] = mean_len
        record["R_C"] = float(np.clip(mean_len / L_TARGET, 0.0, 1.0))

        if t_i_cache_path.exists():
            t_i = load_topic_model(t_i_cache_path)
            encoder = get_encoder(t_i.encoder_model)
            if len(verbalizations) >= 2:
                t_v = fit_topic_model(verbalizations, k=tv_k, encoder=encoder)
                raw = alignment_score(t_v, t_i)
                record["raw_R_D"] = raw
                if null_stats_path.exists():
                    stats = json.loads(null_stats_path.read_text())
                    record["R_D"] = normalize_alignment(
                        raw,
                        null_mean=stats["null_mean"],
                        null_p95=stats["null_p95"],
                    )
                else:
                    record["R_D"] = float(np.clip(raw, 0.0, 1.0))

    record["R"] = aggregate(record["R_A"], record["R_B"], record["R_C"], record["R_D"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("owl", type=str, help="Path to the OWL/TTL ontology to score")
    parser.add_argument("--ti-cache", type=str, default=str(DEFAULT_T_I_CACHE))
    parser.add_argument("--null-stats", type=str, default=str(DEFAULT_NULL_STATS))
    args = parser.parse_args()

    record = score_owl_file(
        Path(args.owl),
        Path(args.ti_cache),
        Path(args.null_stats),
    )
    print(json.dumps(record, indent=2))
    return 0 if not record.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
