#!/usr/bin/env python
"""Stage: BERTopic corpus-level analysis of the FinePDFs input pool.

Single measurement: topics-per-byte ratio in FinePDFs. The ratio
calibrates downstream generation — once we know FinePDFs's natural
topic density, we know how many bytes of generated prose are needed
per seeded concept.

Plus a topic sampling pass so we can see what BERTopic *imagines* a
topic to be (LDA was abstract on this; BERTopic gives concrete
keywords + representative docs).

If 10K docs is too much for a single BERTopic pass (memory/time),
ratchet down via --n-docs.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/experiment_finepdfs_topic_density.py --n-docs 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

logger = logging.getLogger("finepdfs-density")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-docs", type=int, default=10_000,
                   help="Target sample size. Ratchet down on failure.")
    p.add_argument("--seed", type=int, default=4649,
                   help="Matches the audit's deterministic FinePDFs seed.")
    p.add_argument("--max-chars-per-doc", type=int, default=8000,
                   help="Truncation cap for embedding (all-mpnet-base-v2 "
                        "has a 384-token window anyway).")
    p.add_argument("--output-dir", default="build/experiments/finepdfs_topic_density")
    p.add_argument("--n-sample-topics", type=int, default=20,
                   help="Sample this many topics for inspection (top-N by size)")
    return p.parse_args()


def sample_finepdfs(n: int, seed: int, max_chars: int) -> tuple[list[str], list[str], int]:
    """Stream + deterministic sample. Returns (ids, texts, raw_total_bytes).

    raw_total_bytes is computed BEFORE truncation so the topics-per-byte
    ratio reflects the natural document size, not the embedder's window.
    """
    from datasets import load_dataset
    logger.info(f"streaming HuggingFaceFW/finepdfs eng_Latn train (seed={seed})...")
    ds = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn",
                       split="train", streaming=True)
    rng = np.random.default_rng(seed)
    stream_budget = n * 4
    cand_ids, cand_texts = [], []
    for i, row in enumerate(ds):
        if i >= stream_budget:
            break
        cand_ids.append(str(row.get("id") or f"doc_{i}"))
        cand_texts.append(row.get("text") or "")
    raw_total_bytes = sum(len(t) for t in cand_texts)
    logger.info(f"streamed {len(cand_ids)} candidates, "
                f"raw_total_bytes={raw_total_bytes:,}")
    if len(cand_ids) > n:
        idx = rng.choice(len(cand_ids), size=n, replace=False)
        idx = sorted(idx.tolist())
        cand_ids = [cand_ids[i] for i in idx]
        cand_texts = [cand_texts[i] for i in idx]
    # Recompute raw_total_bytes on the chosen subsample
    raw_total_bytes = sum(len(t) for t in cand_texts)
    # Truncate for embedding only
    trunc_texts = [t[:max_chars] for t in cand_texts]
    return cand_ids, trunc_texts, raw_total_bytes


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _ids, texts, raw_bytes = sample_finepdfs(
        args.n_docs, args.seed, args.max_chars_per_doc
    )
    n_docs = len(texts)
    logger.info(f"sample: n_docs={n_docs}, raw_bytes={raw_bytes:,} "
                f"(mean {raw_bytes/n_docs:.0f} bytes/doc) "
                f"[{time.time()-t0:.1f}s]")

    # Embed
    t1 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"embedding {n_docs} docs on {device}...")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device=device)
    embs = embedder.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=True,
                            batch_size=64).astype(np.float32)
    logger.info(f"embedded shape={embs.shape}  [{time.time()-t1:.1f}s]")

    # BERTopic with defaults — this is the operative measurement
    t2 = time.time()
    from bertopic import BERTopic
    logger.info("running BERTopic with defaults...")
    model = BERTopic(verbose=False)
    topics, _ = model.fit_transform(texts, embeddings=embs)
    topic_info = model.get_topic_info()
    elapsed_bertopic = time.time() - t2
    logger.info(f"BERTopic done  [{elapsed_bertopic:.1f}s]")

    # approximate_distribution — soft, LDA-like per-doc topic mixtures via
    # sliding-window assignment. The right tool for "how many topics does
    # a doc span" rather than the default hard 1-per-doc.
    t3 = time.time()
    logger.info("computing approximate_distribution (soft per-doc mixture)...")
    topic_distr, _ = model.approximate_distribution(
        texts, calculate_tokens=False
    )
    elapsed_distr = time.time() - t3
    logger.info(f"approximate_distribution done "
                f"shape={topic_distr.shape}  [{elapsed_distr:.1f}s]")

    n_clusters = len(topic_info) - (1 if -1 in topic_info["Topic"].values else 0)
    n_noise = int((topic_info[topic_info["Topic"] == -1]["Count"].sum())
                  if -1 in topic_info["Topic"].values else 0)

    # The primary number
    topics_per_mb = (n_clusters / raw_bytes) * 1_000_000
    topics_per_kdoc = (n_clusters / n_docs) * 1000

    # Topic size distribution stats
    cluster_sizes = (topic_info[topic_info["Topic"] != -1]["Count"]
                     .astype(int).to_numpy())
    cluster_sizes_sorted = sorted(cluster_sizes.tolist(), reverse=True)

    # Topic sample — top-N by size, with c-TF-IDF keywords + repr doc excerpt
    sample_rows = []
    for _, row in topic_info.head(args.n_sample_topics + 1).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        kw_pairs = model.get_topic(tid) or []
        keywords = [w for w, _ in kw_pairs[:10]]
        rep_doc_excerpt = (model.representative_docs_.get(tid, [""])[0]
                           if hasattr(model, "representative_docs_")
                           and model.representative_docs_ else "")
        rep_doc_excerpt = " ".join(rep_doc_excerpt.split())[:300]
        sample_rows.append({
            "topic_id": tid,
            "count": int(row["Count"]),
            "keywords": keywords,
            "rep_doc_excerpt": rep_doc_excerpt,
        })

    # Per-doc topic counts from the soft distribution
    # topic_distr is shape (n_docs, n_topics). Values are non-negative
    # mass; not normalized to sum to 1 (BERTopic emits raw aggregated
    # window mass). Use a few thresholds to see the shape.
    per_doc_counts: dict[str, list[float]] = {}
    for tau in (0.01, 0.05, 0.10, 0.20):
        counts = (topic_distr >= tau).sum(axis=1)
        per_doc_counts[f"tau={tau}"] = {
            "mean": float(counts.mean()),
            "p50": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
            "frac_zero": float((counts == 0).mean()),
        }
    # Also: how many topics receive ANY positive mass per doc (no threshold)
    any_mass = (topic_distr > 0).sum(axis=1)
    per_doc_counts["any_positive_mass"] = {
        "mean": float(any_mass.mean()),
        "p50": float(np.median(any_mass)),
        "p90": float(np.percentile(any_mass, 90)),
        "max": int(any_mass.max()),
    }

    # Persist
    summary = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": n_docs,
        "raw_total_bytes": raw_bytes,
        "mean_bytes_per_doc": raw_bytes / n_docs,
        "n_clusters": n_clusters,
        "n_noise_docs": n_noise,
        "noise_frac": n_noise / n_docs if n_docs > 0 else 0.0,
        "topics_per_million_bytes": topics_per_mb,
        "topics_per_1000_docs": topics_per_kdoc,
        "cluster_size_top10": cluster_sizes_sorted[:10],
        "cluster_size_p50": (int(np.median(cluster_sizes))
                              if len(cluster_sizes) else 0),
        "cluster_size_min": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
        "cluster_size_max": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "elapsed_bertopic_s": elapsed_bertopic,
        "elapsed_approximate_distribution_s": elapsed_distr,
        "per_doc_topic_counts": per_doc_counts,
        "topic_samples": sample_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print headline
    print()
    print("=" * 74)
    print("FinePDFs BERTopic — corpus-level density")
    print("=" * 74)
    print(f"  n_docs:                  {n_docs:,}")
    print(f"  raw total bytes:         {raw_bytes:,}")
    print(f"  mean bytes/doc:          {raw_bytes/n_docs:,.0f}")
    print(f"  clusters discovered:     {n_clusters}")
    print(f"  noise docs (-1):         {n_noise:,} ({n_noise/n_docs:.1%})")
    print()
    print(f"  → topics per 1M bytes:   {topics_per_mb:.3f}")
    print(f"  → topics per 1000 docs:  {topics_per_kdoc:.2f}")
    print()
    print(f"  cluster size: max={summary['cluster_size_max']:,}  "
          f"p50={summary['cluster_size_p50']}  "
          f"min={summary['cluster_size_min']}")
    print(f"  top-10 sizes: {cluster_sizes_sorted[:10]}")
    print()
    print("=" * 74)
    print("Per-doc soft topic distribution (approximate_distribution)")
    print("=" * 74)
    print(f"  {'threshold':<24} {'mean':>6} {'p50':>5} {'p90':>5} "
          f"{'p99':>5} {'max':>5} {'%zero':>6}")
    for label, stats in per_doc_counts.items():
        if label == "any_positive_mass":
            continue
        print(f"  topics with mass ≥ {label:<7} "
              f"{stats['mean']:>6.2f} {stats['p50']:>5.1f} "
              f"{stats['p90']:>5.1f} {stats['p99']:>5.1f} "
              f"{stats['max']:>5d} {stats['frac_zero']:>5.0%}")
    p = per_doc_counts["any_positive_mass"]
    print(f"  topics with any mass     "
          f"{p['mean']:>6.2f} {p['p50']:>5.1f} "
          f"{p['p90']:>5.1f} {'   - ':>5} {p['max']:>5d} {'   - ':>5}")
    print()
    print("=" * 74)
    print(f"Sample of top {len(sample_rows)} topics by size")
    print("=" * 74)
    for s in sample_rows:
        print(f"\n  topic {s['topic_id']:3d}  (n={s['count']})")
        print(f"    keywords: {', '.join(s['keywords'][:10])}")
        print(f"    repr doc: {s['rep_doc_excerpt'][:240]}...")
    print()
    print(f"summary written to: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
