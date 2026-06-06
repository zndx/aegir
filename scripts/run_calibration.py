#!/usr/bin/env python
"""Calibration run — the LIVE closed loop over FinePDFs training clusters.

Seeds episodes from non-holdout (training) BERTopic clusters (DOF 8): evidence is
the cluster's top FinePDFs docs (the re-grounding anchor); refs are structure-bearing
ontology templates. Each episode runs the full skill library + engine against the live
GLM/Grok mix and the frozen FinePDFs topic model, with the HARD gates deciding
admission and the dense per-axis rewards logged for the deferred weight/threshold
calibration (DOF 1/2). Admitted chapters accrue into a hardened-corpus slice.

Bounded by --max-episodes (each episode is several live LLM calls). Start small to
verify the live loop, then scale.

Usage::
    uv run --no-sync python scripts/run_calibration.py --max-episodes 2          # pilot
    uv run --no-sync python scripts/run_calibration.py --max-episodes 100 --out build/calibration_v0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from aegir.ontology.bindings import make_generate_fn, make_topic_recovery_fn, partition_clusters  # noqa: E402
from aegir.ontology.bindings.topic_recovery import (  # noqa: E402
    DEFAULT_CALIB, _encoder, _load_centroids, score_text,
)
from aegir.ontology.engine import Seed, run_episode  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.skills.base import Gates  # noqa: E402
from aegir.ontology.skills.s2_relational_table import SourceSpan  # noqa: E402


def template_embeddings(catalog):
    """mpnet-embed each template's verbalization (or Manchester axiom) once, in the
    SAME space as the topic centroids — giving the topic→template relevance the
    coverage audit provides, computed inline. Swappable to the precomputed coverage_v0
    audit (or to ColBERT-MaxSim relevance when that lands)."""
    texts = [(t.verbal_template or t.manchester_template or t.template_id)
             for t in catalog.templates]
    return np.asarray(_encoder().encode(texts, normalize_embeddings=True,
                                        batch_size=64, show_progress_bar=False))


def build_seeds(n, train_topics, calib_dir, refs_per, catalog, rng):
    centroids, _topic_ids, id_to_row = _load_centroids(str(calib_dir))
    df = pq.read_table(Path(calib_dir) / "finepdfs_sample.parquet").to_pandas()
    distr = np.load(Path(calib_dir) / "topic_distr.npy")
    tmpl_embs = template_embeddings(catalog)          # (n_templates, 768)
    tmpls = catalog.templates
    seeds = []
    for _ in range(n):
        T = int(rng.choice(train_topics))
        row = id_to_row[T]
        top = np.argsort(-distr[:, row])[:3]          # FinePDFs docs strongest in T → evidence
        evidence = [SourceSpan(f"finepdfs_{int(j)}", str(df.iloc[int(j)]["text"])[:1200]) for j in top]
        rel = tmpl_embs @ centroids[row]              # topic→template relevance (cosine)
        refs = [tmpls[int(j)].template_id for j in np.argsort(-rel)[:refs_per]]
        seeds.append(Seed(f"cluster_T{T}", T, refs, evidence))
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-episodes", type=int, default=2)
    ap.add_argument("--refs-per", type=int, default=2)
    ap.add_argument("--mix", default="cerebras/zai-glm-4.7:0.6,xai/grok-4.3:0.4")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-repair", type=int, default=2)
    ap.add_argument("--out", default="build/calibration_v0")
    ap.add_argument("--seed", type=int, default=4649)
    args = ap.parse_args()

    catalog = load_catalog(REPO / "src/aegir/ontology/catalog/combined.json")
    part = partition_clusters(seed=args.seed)
    print(f"holdout partition: {part['n_train']} train / {part['n_holdout']} holdout clusters")
    gen = make_generate_fn(catalog, mix=args.mix, max_tokens=args.max_tokens, seed=args.seed)
    # Gate on hit@1 (nearest frozen centroid == seeded topic); the graded cosine/rank
    # is logged per chapter to calibrate τ_topic later (DOF 2).
    tr_fn = make_topic_recovery_fn(mode="hit_at_1")
    rng = np.random.default_rng(args.seed)
    seeds = build_seeds(args.max_episodes, part["train"], DEFAULT_CALIB, args.refs_per, catalog, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, admitted = [], []
    t_start = time.time()
    for i, seed in enumerate(seeds):
        t0 = time.time()
        try:
            res = run_episode(seed, catalog, gen, tr_fn, Gates(), max_repair=args.max_repair)
        except Exception as e:
            print(f"ep{i} T={seed.target_topic} ERROR {type(e).__name__}: {e}")
            log.append({"episode": i, "target_topic": seed.target_topic, "error": str(e)})
            continue
        dt = time.time() - t0
        sc = score_text(res.chapter_md, seed.target_topic) if res.chapter_md else {}
        rec = {"episode": i, "target_topic": seed.target_topic, "refs": seed.refs,
               "admitted": res.admitted, "topic_recovery": round(res.topic_recovery, 4),
               "r_axiom": round(res.r_axiom, 4),
               "topic_cosine": round(sc.get("cosine_to_target", 0.0), 4),
               "topic_rank": sc.get("rank"),
               "rewards": {k: round(v, 4) for k, v in res.rewards.items()},
               "decisions": res.decisions, "latency_s": round(dt, 1)}
        log.append(rec)
        print(f"ep{i} T={seed.target_topic} admit={res.admitted} "
              f"tr={res.topic_recovery:.3f} r_ax={res.r_axiom:.3f} "
              f"rewards={ {k: round(v, 2) for k, v in res.rewards.items()} } {dt:.0f}s")
        if res.admitted:
            admitted.append({"chapter_id": f"cal_{i}", "target_topic": seed.target_topic,
                             "refs": seed.refs, "content_md": res.chapter_md,
                             "topic_recovery": float(res.topic_recovery), "r_axiom": float(res.r_axiom)})

    (out / "calibration_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    if admitted:
        pq.write_table(pa.Table.from_pylist(admitted), out / "hardened_corpus.parquet", compression="zstd")

    n = len([r for r in log if "error" not in r])
    na = len(admitted)
    axes = {}
    for r in log:
        for k, v in r.get("rewards", {}).items():
            axes.setdefault(k, []).append(v)
    mean_rewards = {k: round(float(np.mean(v)), 4) for k, v in axes.items()}
    summary = {"episodes": len(log), "scored": n, "admitted": na,
               "admission_rate": round(na / n, 3) if n else 0.0,
               "mean_rewards": mean_rewards, "wall_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nadmitted {na}/{n}  mean_rewards={mean_rewards}  ({summary['wall_s']:.0f}s) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
