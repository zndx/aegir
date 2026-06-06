#!/usr/bin/env python
"""Verify the three calibration-run bindings are 5x5 (live where feasible).

  1. topic_recovery → frozen FinePDFs model: discriminates on-topic vs off-topic
     (real calibration docs).
  2. holdout partition (DOF 8): topic-cluster train/holdout, disjoint + reproducible.
  3. generate_fn → GLM/Grok mix: a LIVE S2 (+ S1) call parses into the type-checkable
     schema and scores r_axiom (the generate→parse→score path).

Usage::  uv run --no-sync python scripts/verify_bindings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from aegir.ontology.bindings.holdout import partition_clusters  # noqa: E402
from aegir.ontology.bindings.llm_generate import make_generate_fn  # noqa: E402
from aegir.ontology.bindings.topic_recovery import DEFAULT_CALIB  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.skills.s2_relational_table import SourceSpan  # noqa: E402
from aegir.ontology.type_check import r_axiom  # noqa: E402


def verify_topic_recovery() -> bool:
    """Reproduce the 2026-05-31 per-doc hit@1 (~0.846) on the existing half_a
    generated corpus: each generated chapter's nearest frozen-topic centroid should
    be one of its seeded topics. This validates the scorer on real generated data
    (not a soft-distr proxy)."""
    import pyarrow.parquet as pq

    from aegir.ontology.bindings.topic_recovery import _encoder, _load_centroids
    centroids, _ids, _ = _load_centroids(str(DEFAULT_CALIB))
    gp = REPO / "build/experiments/topic_recovery_v0/generated/half_a/generated.parquet"
    df = pq.read_table(gp).to_pandas()
    texts = [str(t)[:8000] for t in df["response_text"]]
    seeded = [set(int(x) for x in s) for s in df["seeded_topic_ids"]]
    embs = np.asarray(_encoder().encode(texts, normalize_embeddings=True,
                                        batch_size=32, show_progress_bar=False))
    sims = embs @ centroids.T                       # (n, 141); col == topic id
    order = np.argsort(-sims, axis=1)
    hit1 = float(np.mean([order[i, 0] in seeded[i] for i in range(len(texts))]))
    hit5 = float(np.mean([bool(set(order[i, :5].tolist()) & seeded[i]) for i in range(len(texts))]))
    chance = float(np.mean([len(s) for s in seeded]) / centroids.shape[0])
    print(f"  half_a n={len(texts)}  per-doc hit@1={hit1:.3f}  hit@5={hit5:.3f}  (chance≈{chance:.3f})")
    return hit1 >= 0.60


def verify_holdout() -> bool:
    p = partition_clusters(holdout_frac=0.2, seed=4649)
    p2 = partition_clusters(holdout_frac=0.2, seed=4649)
    tr, ho = set(p["train"]), set(p["holdout"])
    print(f"  train={p['n_train']} holdout={p['n_holdout']} "
          f"disjoint={tr.isdisjoint(ho)} reproducible={p == p2} noise_excluded={p['noise_excluded']}")
    return tr.isdisjoint(ho) and len(ho) > 0 and len(tr) > 0 and p == p2


def verify_generate(catalog) -> bool:
    tmpl = next((t for t in catalog.templates
                 if "ObjectProperty" in t.slot_types.values()
                 and sum(1 for v in t.slot_types.values() if v != "ObjectProperty") >= 2),
                catalog.templates[0])
    ev = [SourceSpan("ev0", "Cells contain organelles such as mitochondria that perform "
                            "cellular respiration to release energy.")]
    gen = make_generate_fn(catalog, max_tokens=8192, temperature=0.4, seed=1)
    try:
        schema = gen("S2", [tmpl.template_id], ev)
        rax, _ = r_axiom(schema, catalog)
        ntab = len(schema.tables)
        ncol = sum(len(t.columns) for t in schema.tables)
        print(f"  S2 live: template={tmpl.template_id}  parsed tables={ntab} cols={ncol}  r_axiom={rax:.3f}")
        text, claims = gen("S1", [tmpl.template_id], ev)
        grounded = sum(1 for c in claims if c.grounded_to)
        print(f"  S1 live: prose_chars={len(text)} claims={len(claims)} grounded={grounded}")
        return ntab >= 1 and ncol >= 1     # the generate→parse→score path works
    except Exception as e:  # network / dspy / key
        print(f"  generate_fn LIVE call failed: {type(e).__name__}: {e}")
        return False


def main() -> int:
    catalog = load_catalog(REPO / "src/aegir/ontology/catalog/combined.json")
    checks = {}
    print("[1] topic_recovery → frozen FinePDFs model")
    checks["topic_recovery (discriminates)"] = verify_topic_recovery()
    print("[2] holdout partition (DOF 8)")
    checks["holdout (disjoint + reproducible)"] = verify_holdout()
    print("[3] generate_fn → GLM/Grok mix (LIVE)")
    checks["generate_fn (live generate→parse→score)"] = verify_generate(catalog)

    print("\n=== 5x5 report ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'5x5 ' if passed else 'FAIL'}] {name}")
        ok &= passed
    print("\nBINDINGS: " + ("ALL 5x5" if ok else "NOT ALL CLEAR"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
