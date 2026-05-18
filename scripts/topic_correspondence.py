#!/usr/bin/env python
"""Topic correspondence analyzer.

For each generated chapter, ask: does its embedding land near the
topic centroid it was anchored against? If yes (high correspondence),
the pipeline preserves topic identity from FinePDFs slice through
LLM generation. If no, the LLM has drifted off the seed topic.

Key metric: ``anchor_topic_rank`` — the rank of the anchored topic
among ALL 200 topic centroids when sorted by chapter-to-centroid
similarity. Rank 1 means the anchored topic is the chapter's nearest
neighbor (perfect correspondence). Higher ranks mean drift.

We report:
  - ``mean_anchor_rank`` per arm (lower is better; 1.0 = perfect)
  - ``hit@K`` (% of chapters where anchored topic is in top-K)
  - ``mean_anchor_sim`` (raw cosine — higher is better)
  - ``mean_top1_sim`` (sanity — chapters should be near SOME topic)
  - ``drift`` = mean_top1_sim - mean_anchor_sim (low = on-topic)

Outputs a parquet with per-chapter scores + summary statistics
per ablation arm.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/topic_correspondence.py \\
        --chapters-runs /raid/checkpoints/aegir-artifacts/ablation_v0/72cc17cff35b63f6 \\
                        /raid/checkpoints/aegir-artifacts/ablation_v0/3a81d92e48daaa13 \\
                        /raid/checkpoints/aegir-artifacts/ablation_v0/7434ac168d97acf7 \\
        --audit-run /raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("topic-correspondence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-runs", required=True, nargs="+",
                   help="One or more chapters_v0/<run_id>/ dirs (one per arm)")
    p.add_argument("--audit-run", required=True,
                   help="Path to coverage_v0/<run_id>/ for topic centroids")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-chars-per-chapter", type=int, default=8000)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    audit_run = Path(args.audit_run)

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"device: {device}")

    # Load coverage audit topic data
    audit_topics = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    audit_topics = audit_topics.sort_values("topic_id").reset_index(drop=True)
    logger.info(f"audit topics: {len(audit_topics)}")

    # Recompute topic centroids by re-embedding the topic_repr_text. This is
    # an approximation since the audit ran KMeans on full-doc embeddings,
    # but the repr-text is a doc nearest-to-centroid; its embedding is close
    # to the centroid by construction.
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedding_model, device=device)

    topic_texts = audit_topics["topic_repr_text"].fillna("").tolist()
    logger.info(f"embedding {len(topic_texts)} topic representatives...")
    topic_embs = embedder.encode(topic_texts,
                                  normalize_embeddings=True,
                                  show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32)
    topic_ids = audit_topics["topic_id"].to_numpy()

    all_rows = []
    for run_dir in args.chapters_runs:
        run_dir = Path(run_dir)
        chapters = pq.read_table(run_dir / "chapters.parquet").to_pandas()
        logger.info(f"\n  arm={run_dir.name}  chapters={len(chapters)}")
        if "ablation" in chapters.columns:
            ablation = chapters["ablation"].iloc[0]
        else:
            ablation = "full"  # backward compat for runs predating the field

        chapter_texts = chapters["response_text"].fillna("").apply(
            lambda s: s[:args.max_chars_per_chapter]
        ).tolist()
        chapter_embs = embedder.encode(chapter_texts,
                                        normalize_embeddings=True,
                                        show_progress_bar=False,
                                        convert_to_numpy=True).astype(np.float32)

        # Similarity matrix (chapters × topics)
        sims = chapter_embs @ topic_embs.T  # both unit-normed → cosine

        for i, ch in chapters.iterrows():
            chapter_sims = sims[i]  # (n_topics,)
            # Anchored topic ids: the chapter was generated against these
            anchored = ch.get("style_topic_ids")
            anchored = list(anchored) if anchored is not None else []
            anchored_int = [int(t) for t in anchored]

            # For each anchored topic, find its rank in the chapter's sim ordering
            sort_idx = np.argsort(-chapter_sims)  # descending sim
            rank_of_topic = {int(topic_ids[idx]): rank + 1
                             for rank, idx in enumerate(sort_idx)}

            anchor_ranks = [rank_of_topic.get(t, len(topic_ids))
                            for t in anchored_int]
            best_anchor_rank = min(anchor_ranks) if anchor_ranks else len(topic_ids)
            anchor_sims = [
                float(chapter_sims[np.where(topic_ids == t)[0][0]])
                for t in anchored_int if t in topic_ids
            ]
            mean_anchor_sim = float(np.mean(anchor_sims)) if anchor_sims else 0.0
            top1_sim = float(chapter_sims.max())
            top1_topic = int(topic_ids[int(np.argmax(chapter_sims))])

            all_rows.append({
                "chapter_id": str(ch["chapter_id"]),
                "ablation": str(ablation),
                "family": str(ch.get("family", "")),
                "model": str(ch.get("model", "")),
                "prompt_kind": str(ch.get("prompt_kind", "")),
                "anchored_topic_ids": anchored_int,
                "best_anchor_rank": int(best_anchor_rank),
                "mean_anchor_sim": mean_anchor_sim,
                "top1_topic_id": top1_topic,
                "top1_sim": top1_sim,
                "drift": top1_sim - mean_anchor_sim,
            })

    # Write parquet
    out_path = Path(args.chapters_runs[0]).parent / "topic_correspondence.parquet"
    schema = pa.schema([
        ("chapter_id", pa.string()),
        ("ablation", pa.string()),
        ("family", pa.string()),
        ("model", pa.string()),
        ("prompt_kind", pa.string()),
        ("anchored_topic_ids", pa.list_(pa.int32())),
        ("best_anchor_rank", pa.int32()),
        ("mean_anchor_sim", pa.float32()),
        ("top1_topic_id", pa.int32()),
        ("top1_sim", pa.float32()),
        ("drift", pa.float32()),
    ])
    pq.write_table(pa.Table.from_pylist(all_rows, schema=schema),
                    out_path, compression="zstd")
    logger.info(f"wrote {out_path}")

    # Summary per arm
    from collections import defaultdict
    by_arm = defaultdict(list)
    for r in all_rows:
        by_arm[r["ablation"]].append(r)

    print()
    print(f"{'arm':<14} {'n':>4} {'mean_rank':>10} {'hit@1':>7} {'hit@3':>7} "
          f"{'hit@10':>7} {'mean_anchor_sim':>16} {'mean_top1_sim':>14} "
          f"{'mean_drift':>11}")
    print("─" * 110)
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        n = len(rows)
        ranks = np.array([r["best_anchor_rank"] for r in rows])
        anchor_sims = np.array([r["mean_anchor_sim"] for r in rows])
        top1_sims = np.array([r["top1_sim"] for r in rows])
        drifts = np.array([r["drift"] for r in rows])
        hit1 = float((ranks == 1).mean())
        hit3 = float((ranks <= 3).mean())
        hit10 = float((ranks <= 10).mean())
        print(f"{arm:<14} {n:>4d} {float(ranks.mean()):>10.2f} "
              f"{hit1:>7.1%} {hit3:>7.1%} {hit10:>7.1%} "
              f"{float(anchor_sims.mean()):>16.4f} "
              f"{float(top1_sims.mean()):>14.4f} "
              f"{float(drifts.mean()):>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
