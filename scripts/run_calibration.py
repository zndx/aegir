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
import re
import sys
import time
from collections import Counter, defaultdict
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


def _primary_family(refs, catalog) -> str:
    fams = []
    for r in refs:
        try:
            f = (catalog.by_id(r).provenance or {}).get("family")
            if f:
                fams.append(f)
        except KeyError:
            pass
    return Counter(fams).most_common(1)[0][0] if fams else "?"


def quality_metrics(log, admitted, wall_s, n_train_topics) -> dict:
    """The metrics that matter for directing the convergence — and the baseline for
    what the UI should surface: admission funnel, per-gate pass rates, hit@k (τ_topic
    calibration), the rejection-reason breakdown (the bottleneck), score distributions,
    per-family re-grounding (where to refine the ontology), repair-by-skill (generation
    quality), coverage, throughput."""
    scored = [r for r in log if "error" not in r]
    n = len(scored)
    if not n:
        return {"episodes": len(log), "scored": 0, "errors": len(log)}

    def dist(key):
        a = np.array([r[key] for r in scored if isinstance(r.get(key), (int, float))], dtype=float)
        if not len(a):
            return {}
        return {"mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
                "p90": round(float(np.percentile(a, 90)), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}

    ranks = [r["topic_rank"] for r in scored if isinstance(r.get("topic_rank"), int)]
    hit_at_k = ({f"hit@{k}": round(sum(1 for x in ranks if x < k) / len(ranks), 3)
                 for k in (1, 5, 10, 20)} if ranks else {})

    rej = [r for r in scored if not r.get("admitted")]
    rejection_reasons = {
        "topic_miss": sum(1 for r in rej if r.get("topic_recovery", 0) < 1.0),
        "r_axiom_low": sum(1 for r in rej if r.get("r_axiom", 0) < 0.45),
        "claim_grounding_low": sum(1 for r in rej
                                   if r.get("rewards", {}).get("claim_grounding", 0) < 0.95),
    }

    fam = defaultdict(lambda: {"n": 0, "admit": 0, "cos": []})
    for r in scored:
        d = fam[r.get("primary_family", "?")]
        d["n"] += 1
        d["admit"] += int(bool(r.get("admitted")))
        d["cos"].append(r.get("topic_cosine", 0.0))
    per_family = {f: {"n": d["n"], "admit_rate": round(d["admit"] / d["n"], 3),
                      "mean_cos": round(sum(d["cos"]) / len(d["cos"]), 3)}
                  for f, d in sorted(fam.items())}

    rep = defaultdict(list)
    for r in scored:
        for dec in r.get("decisions", []):
            m = re.match(r"(S\d)/\w+: \w+ \((\d+) repair", dec)
            if m:
                rep[m.group(1)].append(int(m.group(2)))
    repair_mean_by_skill = {s: round(sum(v) / len(v), 2) for s, v in sorted(rep.items())}

    axes = defaultdict(list)
    for r in scored:
        for k, v in r.get("rewards", {}).items():
            axes[k].append(v)
    mean_rewards = {k: round(float(np.mean(v)), 4) for k, v in axes.items()}

    return {
        "episodes": len(log), "scored": n, "errors": len(log) - n,
        "admitted": len(admitted), "admission_rate": round(len(admitted) / n, 3),
        "rejection_reasons": rejection_reasons,
        "hit_at_k": hit_at_k,
        "mean_rewards": mean_rewards,
        "score_dist": {"topic_cosine": dist("topic_cosine"), "topic_rank": dist("topic_rank"),
                       "r_axiom": dist("r_axiom")},
        "per_family": per_family,
        "repair_mean_by_skill": repair_mean_by_skill,
        "topics_covered": len({r["target_topic"] for r in scored}),
        "train_topics_available": n_train_topics,
        "wall_s": round(wall_s, 1),
        "mean_latency_s": round(sum(r.get("latency_s", 0) for r in scored) / n, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-episodes", type=int, default=2)
    ap.add_argument("--refs-per", type=int, default=2)
    ap.add_argument("--mix", default="cerebras/zai-glm-4.7:0.6,xai/grok-4.3:0.4")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-repair", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent episodes (I/O-bound LLM calls; shared rng/cache/encoder "
                         "are locked). 1 = sequential. Try 6–8 for the batch.")
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

    def run_one(i, seed):
        t0 = time.time()
        try:
            res = run_episode(seed, catalog, gen, tr_fn, Gates(), max_repair=args.max_repair)
        except Exception as e:
            print(f"ep{i} T={seed.target_topic} ERROR {type(e).__name__}: {e}")
            return {"episode": i, "target_topic": seed.target_topic, "error": str(e)}, None
        dt = time.time() - t0
        sc = score_text(res.chapter_md, seed.target_topic) if res.chapter_md else {}
        rec = {"episode": i, "target_topic": seed.target_topic, "refs": seed.refs,
               "primary_family": _primary_family(seed.refs, catalog),
               "admitted": res.admitted, "topic_recovery": round(res.topic_recovery, 4),
               "r_axiom": round(res.r_axiom, 4),
               "topic_cosine": round(sc.get("cosine_to_target", 0.0), 4),
               "topic_rank": sc.get("rank"),
               "rewards": {k: round(v, 4) for k, v in res.rewards.items()},
               "decisions": res.decisions, "latency_s": round(dt, 1)}
        print(f"ep{i} T={seed.target_topic} admit={res.admitted} "
              f"tr={res.topic_recovery:.3f} r_ax={res.r_axiom:.3f} "
              f"rewards={ {k: round(v, 2) for k, v in res.rewards.items()} } {dt:.0f}s")
        adm = ({"chapter_id": f"cal_{i}", "target_topic": seed.target_topic,
                "refs": seed.refs, "content_md": res.chapter_md,
                "topic_recovery": float(res.topic_recovery), "r_axiom": float(res.r_axiom)}
               if res.admitted else None)
        return rec, adm

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, i, s) for i, s in enumerate(seeds)]
            for f in as_completed(futs):
                rec, adm = f.result()
                log.append(rec)
                if adm:
                    admitted.append(adm)
    else:
        for i, s in enumerate(seeds):
            rec, adm = run_one(i, s)
            log.append(rec)
            if adm:
                admitted.append(adm)

    (out / "calibration_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    if admitted:
        pq.write_table(pa.Table.from_pylist(admitted), out / "hardened_corpus.parquet", compression="zstd")

    m = quality_metrics(log, admitted, time.time() - t_start, part["n_train"])
    (out / "summary.json").write_text(json.dumps(m, indent=2))
    print(f"\n=== quality baseline → {out} ===")
    print(f"admitted {m['admitted']}/{m['scored']} (rate {m['admission_rate']})  "
          f"rejections={m['rejection_reasons']}")
    print(f"hit@k={m['hit_at_k']}  topic_cosine={m['score_dist']['topic_cosine']}")
    print(f"mean_rewards={m['mean_rewards']}")
    print(f"repair_by_skill={m['repair_mean_by_skill']}  "
          f"topics={m['topics_covered']}/{m['train_topics_available']}  "
          f"wall={m['wall_s']}s lat={m['mean_latency_s']}s/ep")
    print("per_family:")
    for f, d in m["per_family"].items():
        print(f"  {f:28} n={d['n']:>3} admit={d['admit_rate']:.2f} mean_cos={d['mean_cos']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
