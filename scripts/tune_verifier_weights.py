"""tune_verifier_weights — score the C1 test set + AUC weight sweep.

Implements the P2 / claim C1 validation step:

1. Score each ontology in the labeled test set via
   :func:`scripts.score_ontology.score_owl_file`.
2. Grid-sweep over aggregation weights ``{a, b, c}`` (the
   coefficients of R_B, R_C, R_D in the aggregation
   ``R(O) = R_A · (a·R_B + b·R_C + c·R_D)``) under the
   constraint ``a + b + c = 1``.
3. For each weight triple, compute ROC AUC of the resulting R as
   a binary discriminator (good = 1 vs bad = 0).
4. Find the triple maximizing AUC. Report
   ``mean(R | good) - mean(R | bad)`` for the optimal weights.
5. Write per-ontology scores + the optimal-weight summary to
   ``--output``.

The brief's P2 exit gate requires ``AUC ≥ 0.85`` and
``mean(R | good) - mean(R | bad) ≥ 0.30``. This script verifies
both and exits non-zero if either fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_ontology import (  # noqa: E402
    DEFAULT_NULL_STATS,
    DEFAULT_T_I_CACHE,
    aggregate,
    score_owl_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DIR = REPO_ROOT / "tests" / "ontology_test_set"


def score_test_set(test_dir: Path) -> tuple[list[dict], dict]:
    """Score every ``*.ttl`` in ``test_dir`` against the labeled
    set. Returns ``(records, labels)``."""
    labels = json.loads((test_dir / "labels.json").read_text())
    records: list[dict] = []
    for fname, info in sorted(labels.items()):
        ttl_path = test_dir / fname
        logger.info("scoring %s (%s)", fname, info["label"])
        rec = score_owl_file(
            ttl_path,
            DEFAULT_T_I_CACHE,
            DEFAULT_NULL_STATS,
        )
        rec["label"] = info["label"]
        rec["kind"] = info["kind"]
        rec["filename"] = fname
        records.append(rec)
    return records, labels


def auc_from_components(
    records: list[dict],
    a: float, b: float, c: float,
) -> tuple[float, float, float, float]:
    """Re-aggregate R for each record under given weights and
    compute ROC AUC + mean separation between good and bad."""
    from sklearn.metrics import roc_auc_score

    Rs: list[float] = []
    ys: list[int] = []
    for r in records:
        R = aggregate(r["R_A"], r["R_B"], r["R_C"], r["R_D"], w_b=a, w_c=b, w_d=c)
        Rs.append(R)
        ys.append(1 if r["label"] == "good" else 0)
    Rs_arr = np.array(Rs)
    ys_arr = np.array(ys)
    auc = float(roc_auc_score(ys_arr, Rs_arr))
    mean_good = float(Rs_arr[ys_arr == 1].mean())
    mean_bad = float(Rs_arr[ys_arr == 0].mean())
    return auc, mean_good, mean_bad, mean_good - mean_bad


def grid_sweep(records: list[dict], step: float = 0.05) -> dict:
    """Sweep ``{a, b, c}`` on the ``a + b + c = 1`` simplex with
    grid step ``step``. Returns the best triple plus its AUC and
    the full sweep history."""
    best = {"auc": -1.0, "a": 0.0, "b": 0.0, "c": 0.0,
            "mean_good": 0.0, "mean_bad": 0.0, "separation": 0.0}
    history: list[dict] = []
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for a, b in product(grid, grid):
        c = 1.0 - a - b
        if c < -1e-9 or c > 1.0 + 1e-9:
            continue
        c = max(0.0, min(1.0, c))
        auc, mg, mb, sep = auc_from_components(records, a, b, c)
        history.append({
            "a": float(a), "b": float(b), "c": float(c),
            "auc": auc, "mean_good": mg, "mean_bad": mb,
            "separation": sep,
        })
        if auc > best["auc"] or (auc == best["auc"] and sep > best["separation"]):
            best = {
                "auc": auc, "a": float(a), "b": float(b), "c": float(c),
                "mean_good": mg, "mean_bad": mb, "separation": sep,
            }
    return {"best": best, "n_sweep_points": len(history), "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "tests" / "ontology_test_set" / "c1_results.json"))
    parser.add_argument("--auc-threshold", type=float, default=0.85)
    parser.add_argument("--separation-threshold", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not (test_dir / "labels.json").exists():
        print(f"no labels.json at {test_dir}; run scripts/build_test_set.py first", file=sys.stderr)
        return 1

    records, labels = score_test_set(test_dir)

    # Initial baseline at default weights {0.4, 0.3, 0.3}.
    auc0, mg0, mb0, sep0 = auc_from_components(records, 0.4, 0.3, 0.3)
    logger.info(
        "default weights {0.4, 0.3, 0.3}: AUC=%.4f mean_good=%.4f mean_bad=%.4f separation=%.4f",
        auc0, mg0, mb0, sep0,
    )

    # Grid sweep.
    logger.info("running grid sweep over {a, b, c} simplex (step=%.2f)", args.step)
    sweep = grid_sweep(records, step=args.step)
    best = sweep["best"]
    logger.info(
        "best weights: a=%.2f b=%.2f c=%.2f → AUC=%.4f separation=%.4f",
        best["a"], best["b"], best["c"], best["auc"], best["separation"],
    )

    payload = {
        "default_weights": {
            "a": 0.4, "b": 0.3, "c": 0.3,
            "auc": auc0, "mean_good": mg0, "mean_bad": mb0, "separation": sep0,
        },
        "best_weights": best,
        "n_sweep_points": sweep["n_sweep_points"],
        "thresholds": {
            "auc": args.auc_threshold,
            "separation": args.separation_threshold,
        },
        "auc_pass": best["auc"] >= args.auc_threshold,
        "separation_pass": best["separation"] >= args.separation_threshold,
        "records": records,
        "sweep_top_10": sorted(sweep["history"], key=lambda h: -h["auc"])[:10],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote results to %s", args.output)

    print()
    print(f"=== C1 verifier-validation results ===")
    print(f"  test set: {sum(1 for r in records if r['label']=='good')} good, "
          f"{sum(1 for r in records if r['label']=='bad')} bad")
    print(f"  default weights {{0.4, 0.3, 0.3}}: AUC={auc0:.4f}  separation={sep0:.4f}")
    print(f"  best weights: a={best['a']:.2f} b={best['b']:.2f} c={best['c']:.2f}")
    print(f"    AUC={best['auc']:.4f} mean_good={best['mean_good']:.4f} "
          f"mean_bad={best['mean_bad']:.4f} separation={best['separation']:.4f}")
    print(f"  AUC ≥ {args.auc_threshold}: "
          f"{'✓ PASS' if payload['auc_pass'] else '✗ FAIL'}")
    print(f"  Separation ≥ {args.separation_threshold}: "
          f"{'✓ PASS' if payload['separation_pass'] else '✗ FAIL'}")
    print()
    print("=== per-ontology scores (sorted desc) ===")
    for r in sorted(records, key=lambda r: -aggregate(
            r["R_A"], r["R_B"], r["R_C"], r["R_D"],
            w_b=best["a"], w_c=best["b"], w_d=best["c"])):
        R = aggregate(r["R_A"], r["R_B"], r["R_C"], r["R_D"],
                      w_b=best["a"], w_c=best["b"], w_d=best["c"])
        print(f"  [{r['label']}] R={R:.4f}  R_A={r['R_A']:.2f} "
              f"R_B={r['R_B']:.2f} R_C={r['R_C']:.2f} R_D={r['R_D']:.2f}  "
              f"{r['filename']}")

    return 0 if payload["auc_pass"] and payload["separation_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
