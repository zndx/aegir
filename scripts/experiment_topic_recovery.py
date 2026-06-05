#!/usr/bin/env python
"""Topic recovery experiment: can Grok generate prose whose freshly-
discovered BERTopic structure recovers a meaningful fraction of the
FinePDFs topics that seeded the prompts?

Three phases, runnable independently:

  --phase calibrate
     Stream 10K FinePDFs (seed=4649, same as audit/density run), embed,
     fit BERTopic, run approximate_distribution. Persists:
        finepdfs_sample.parquet  — docs + ids
        topic_info.json          — per-topic keywords + repr-doc excerpt
        topic_centroids.npy      — (n_topics, embed_dim) centroids
        topic_distr.npy          — (n_docs, n_topics) soft mixtures
     One-time setup; reuse for both pilot and full runs.

  --phase generate --n-docs N
     Loads calibration artifacts. Samples N random FinePDFs docs as seeds.
     For each: extracts topics where mass >= tau (default 0.05), builds a
     prompt with those topics' keywords + repr-doc excerpts, calls Grok-4.3
     for one doc. Persists generated.parquet keyed by seed doc index.
     Output goes to a tagged subdir so pilot/full runs don't collide.

  --phase analyze
     Loads generated docs from the latest generation subdir. Embeds them,
     runs BERTopic with defaults, maps each output cluster centroid to its
     nearest input topic by cosine. Computes topic_recovery_rate.

Workflow:
  1. python experiment_topic_recovery.py --phase calibrate
  2. python experiment_topic_recovery.py --phase generate --n-docs 10 --tag pilot
  3. [inspect generated/pilot/generated.parquet — adapt if needed]
  4. python experiment_topic_recovery.py --phase generate --n-docs 500 --tag full
  5. python experiment_topic_recovery.py --phase analyze --tag full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("topic-recovery")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True,
                   choices=("calibrate", "generate", "analyze"))
    p.add_argument("--output-dir", default="build/experiments/topic_recovery_v0")
    p.add_argument("--tag", default="full",
                   help="Subdir tag for generate/analyze phases "
                        "(e.g., 'pilot', 'full'). Lets pilot and full "
                        "runs coexist.")
    # Calibrate phase
    p.add_argument("--finepdfs-n", type=int, default=10_000)
    p.add_argument("--finepdfs-seed", type=int, default=4649)
    p.add_argument("--max-chars-per-doc", type=int, default=8000)
    # Generate phase
    p.add_argument("--n-docs", type=int, default=500)
    p.add_argument("--gen-seed", type=int, default=20000,
                   help="Seed for picking which FinePDFs docs to use "
                        "as prompt seeds. Independent of --finepdfs-seed.")
    p.add_argument("--tau", type=float, default=0.05,
                   help="Threshold on approximate_distribution mass for a "
                        "topic to be included as a seed.")
    p.add_argument("--model", default="xai/grok-4.3")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Grok 4.3 reasoning model: this is the *total* "
                        "completion budget; reasoning tokens count "
                        "invisibly against it. 8192 = ~4x safety margin "
                        "over expected ~2K total per doc.")
    p.add_argument("--repr-doc-chars", type=int, default=350,
                   help="Per-topic representative-doc excerpt length in "
                        "the prompt. Total prompt scales as n_topics * this.")
    return p.parse_args()


def _calib_dir(out_dir: Path) -> Path:
    return out_dir / "calibration"


def _gen_dir(out_dir: Path, tag: str) -> Path:
    return out_dir / "generated" / tag


# ─────────────────────────────────────────────────────────────────────────
# Phase: calibrate
# ─────────────────────────────────────────────────────────────────────────

def phase_calibrate(args: argparse.Namespace, out_dir: Path) -> int:
    calib_dir = _calib_dir(out_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample FinePDFs
    t0 = time.time()
    logger.info(f"sampling {args.finepdfs_n} FinePDFs docs (seed={args.finepdfs_seed})...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn",
                       split="train", streaming=True)
    rng = np.random.default_rng(args.finepdfs_seed)
    stream_budget = args.finepdfs_n * 4
    cand_ids: list[str] = []
    cand_texts: list[str] = []
    for i, row in enumerate(ds):
        if i >= stream_budget:
            break
        cand_ids.append(str(row.get("id") or f"doc_{i}"))
        cand_texts.append(row.get("text") or "")
    if len(cand_ids) > args.finepdfs_n:
        idx = rng.choice(len(cand_ids), size=args.finepdfs_n, replace=False)
        idx = sorted(idx.tolist())
        cand_ids = [cand_ids[i] for i in idx]
        cand_texts = [cand_texts[i] for i in idx]
    n_docs = len(cand_ids)
    raw_bytes = sum(len(t) for t in cand_texts)
    trunc_texts = [t[:args.max_chars_per_doc] for t in cand_texts]
    logger.info(f"sampled n={n_docs} raw_bytes={raw_bytes:,} "
                f"[{time.time()-t0:.1f}s]")

    # 2. Embed
    t1 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device=device)
    logger.info(f"embedding on {device}...")
    embs = embedder.encode(trunc_texts, normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=True,
                            batch_size=64).astype(np.float32)
    logger.info(f"embeddings: {embs.shape}  [{time.time()-t1:.1f}s]")

    # 3. BERTopic with explicit umap seed for reproducibility
    t2 = time.time()
    from bertopic import BERTopic
    from umap import UMAP
    model = BERTopic(
        umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                         metric="cosine", random_state=42),
        verbose=False,
    )
    topics, _ = model.fit_transform(trunc_texts, embeddings=embs)
    topic_info = model.get_topic_info()
    n_clusters = len(topic_info) - (1 if -1 in topic_info["Topic"].values else 0)
    logger.info(f"BERTopic: {n_clusters} clusters  [{time.time()-t2:.1f}s]")

    # 4. approximate_distribution
    t3 = time.time()
    logger.info("approximate_distribution...")
    topic_distr, _ = model.approximate_distribution(trunc_texts,
                                                     calculate_tokens=False)
    logger.info(f"topic_distr: {topic_distr.shape}  [{time.time()-t3:.1f}s]")

    # 5. Persist
    # Save raw FinePDFs sample
    pq.write_table(
        pa.Table.from_pylist([
            {"finepdfs_id": cid, "idx": i, "text": txt}
            for i, (cid, txt) in enumerate(zip(cand_ids, cand_texts))
        ], schema=pa.schema([
            ("finepdfs_id", pa.string()),
            ("idx", pa.int32()),
            ("text", pa.string()),
        ])),
        calib_dir / "finepdfs_sample.parquet",
        compression="zstd",
    )

    # Per-topic info — keywords + representative doc excerpt
    topic_records: list[dict] = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        kw_pairs = model.get_topic(tid) or []
        keywords = [w for w, _ in kw_pairs[:10]] if kw_pairs else []
        rep_docs_list = (model.representative_docs_.get(tid, [""])
                         if hasattr(model, "representative_docs_")
                         and model.representative_docs_ else [""])
        rep_doc = rep_docs_list[0] if rep_docs_list else ""
        rep_doc = " ".join(rep_doc.split())
        topic_records.append({
            "topic_id": tid,
            "count": int(row["Count"]),
            "keywords": keywords,
            "rep_doc_excerpt": rep_doc[:args.repr_doc_chars],
        })
    (calib_dir / "topic_info.json").write_text(json.dumps(topic_records, indent=2))

    # Per-topic centroids derived from embedding means
    # (Each cluster's centroid is the mean of its assigned docs' embeddings,
    # then L2-normalized so cosine = dot.)
    topic_ids_sorted = sorted({int(r["topic_id"]) for r in topic_records})
    centroids = np.zeros((len(topic_ids_sorted), embs.shape[1]),
                          dtype=np.float32)
    topics_arr = np.asarray(topics)
    for i, tid in enumerate(topic_ids_sorted):
        mask = topics_arr == tid
        if mask.sum() == 0:
            continue
        c = embs[mask].mean(axis=0)
        n = np.linalg.norm(c) + 1e-12
        centroids[i] = c / n
    np.save(calib_dir / "topic_centroids.npy", centroids)
    np.save(calib_dir / "topic_distr.npy", topic_distr)
    (calib_dir / "topic_ids.json").write_text(
        json.dumps(topic_ids_sorted)
    )

    # Manifest
    (calib_dir / "manifest.json").write_text(json.dumps({
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finepdfs_n": n_docs,
        "raw_total_bytes": raw_bytes,
        "n_topics": len(topic_ids_sorted),
        "n_noise_docs": int((topics_arr == -1).sum()),
        "umap_seed": 42,
        "finepdfs_seed": args.finepdfs_seed,
    }, indent=2))

    logger.info(f"calibration persisted to {calib_dir}")
    logger.info(f"  n_topics={len(topic_ids_sorted)} "
                f"noise={(topics_arr == -1).sum()} ({(topics_arr == -1).mean():.1%})")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Phase: generate
# ─────────────────────────────────────────────────────────────────────────

LOOSE_PROMPT_TEMPLATE = """Write one ~1500-word document. \
The document should naturally engage with the following concepts as a \
real-world document on this domain would. Treat the concepts as \
scaffolding for your thinking — weave them in as the topic demands; \
do NOT recite them, do NOT cite them by name, do NOT use them as a \
checklist. Use whatever structure (prose, headings, lists, tables) \
fits the subject best. Write the kind of document a domain practitioner \
would produce.

CONCEPTS to engage with:

{concepts_block}
"""


def _build_concepts_block(seeded_topics: list[dict],
                           repr_doc_chars: int) -> str:
    blocks = []
    for i, t in enumerate(seeded_topics, 1):
        kw = ", ".join(t["keywords"])
        excerpt = t.get("rep_doc_excerpt", "")[:repr_doc_chars]
        blocks.append(
            f"  Concept {i}:\n"
            f"    keywords: {kw}\n"
            f"    example excerpt: {excerpt}\n"
        )
    return "\n".join(blocks)


def phase_generate(args: argparse.Namespace, out_dir: Path) -> int:
    calib_dir = _calib_dir(out_dir)
    gen_dir = _gen_dir(out_dir, args.tag)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration artifacts
    if not (calib_dir / "topic_distr.npy").exists():
        raise SystemExit(
            f"calibration not found in {calib_dir}; "
            f"run --phase calibrate first."
        )
    topic_distr = np.load(calib_dir / "topic_distr.npy")  # (n_docs, n_topics)
    topic_ids = json.loads((calib_dir / "topic_ids.json").read_text())
    topic_records = json.loads((calib_dir / "topic_info.json").read_text())
    topic_by_id = {int(r["topic_id"]): r for r in topic_records}
    finepdfs = pq.read_table(calib_dir / "finepdfs_sample.parquet").to_pandas()
    n_finepdfs = len(finepdfs)
    logger.info(f"loaded calibration: {n_finepdfs} FinePDFs docs, "
                f"{len(topic_records)} topics")

    # Sample N seed FinePDFs docs
    rng = np.random.default_rng(args.gen_seed)
    seed_doc_indices = rng.choice(n_finepdfs, size=args.n_docs,
                                    replace=False).tolist()
    seed_doc_indices = sorted(seed_doc_indices)

    # For each seed: extract topics with mass >= tau
    seeded_per_doc: list[dict] = []
    for di in seed_doc_indices:
        distr = topic_distr[di]
        above_tau = []
        for col, tid in enumerate(topic_ids):
            if distr[col] >= args.tau:
                above_tau.append((tid, float(distr[col])))
        above_tau.sort(key=lambda x: -x[1])
        seeded_per_doc.append({
            "seed_doc_idx": int(di),
            "seed_finepdfs_id": str(finepdfs.iloc[di]["finepdfs_id"]),
            "topics": [
                {**topic_by_id[tid], "mass": mass}
                for tid, mass in above_tau
                if tid in topic_by_id
            ],
        })

    # Drop docs that ended up with 0 topics at this tau (rare but possible)
    seeded_per_doc = [s for s in seeded_per_doc if len(s["topics"]) > 0]
    logger.info(f"prepared {len(seeded_per_doc)} prompts "
                f"(mean {np.mean([len(s['topics']) for s in seeded_per_doc]):.1f} topics/prompt)")

    # LLM client
    import dspy
    from gaius.hx.exchange import ExchangeRecord
    from aegir.hx import append_exchange

    provider = args.model.split("/", 1)[0]
    api_key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise SystemExit(f"missing {provider.upper()}_API_KEY")
    lm = dspy.LM(
        model=args.model, api_key=api_key,
        max_tokens=args.max_tokens, temperature=args.temperature,
        cache=False,
    )

    # Generate
    rows: list[dict] = []
    schema = pa.schema([
        ("idx", pa.int32()),
        ("seed_doc_idx", pa.int32()),
        ("seed_finepdfs_id", pa.string()),
        ("seeded_topic_ids", pa.list_(pa.int32())),
        ("prompt_chars", pa.int32()),
        ("response_chars", pa.int32()),
        ("response_text", pa.string()),
        ("reasoning_chars", pa.int32()),
        ("response_reasoning", pa.string()),
        ("latency_ms", pa.int32()),
        ("hx_exchange_id", pa.string()),
        ("model", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
    flush_path = gen_dir / "generated.parquet"

    for i, seed in enumerate(seeded_per_doc):
        prompt = LOOSE_PROMPT_TEMPLATE.format(
            concepts_block=_build_concepts_block(
                seed["topics"], args.repr_doc_chars
            )
        )

        t0 = time.time()
        result = lm(messages=[{"role": "user", "content": prompt}])
        latency_ms = int((time.time() - t0) * 1000)

        if isinstance(result, list):
            item = result[0]
            response_text = item.get("text") or ""
            response_reasoning = item.get("reasoning_content") or ""
        else:
            response_text = str(result)
            response_reasoning = ""

        # HX persist
        record = ExchangeRecord(
            provider=provider,
            request_messages=[{"role": "user", "content": prompt}],
            request_model=args.model,
            request_params={"max_tokens": args.max_tokens,
                            "temperature": args.temperature},
            response_content=response_text,
            response_model=args.model.split("/")[-1],
            response_reasoning=response_reasoning,
            latency_ms=latency_ms,
            source_context={
                "aegir_module": "experiment_topic_recovery",
                "tag": args.tag,
                "seed_doc_idx": str(seed["seed_doc_idx"]),
                "seeded_topic_ids": ",".join(
                    str(t["topic_id"]) for t in seed["topics"]
                ),
            },
        )
        append_exchange(record)

        rows.append({
            "idx": i,
            "seed_doc_idx": int(seed["seed_doc_idx"]),
            "seed_finepdfs_id": seed["seed_finepdfs_id"],
            "seeded_topic_ids": [int(t["topic_id"]) for t in seed["topics"]],
            "prompt_chars": len(prompt),
            "response_chars": len(response_text),
            "response_text": response_text,
            "reasoning_chars": len(response_reasoning),
            "response_reasoning": response_reasoning,
            "latency_ms": latency_ms,
            "hx_exchange_id": record.id,
            "model": args.model,
            "created_at": datetime.now(timezone.utc),
        })

        logger.info(
            f"  [{i+1}/{len(seeded_per_doc)}] seed_doc={seed['seed_doc_idx']} "
            f"topics={[int(t['topic_id']) for t in seed['topics']]} "
            f"resp_chars={len(response_text)} "
            f"reason_chars={len(response_reasoning)} "
            f"lat={latency_ms}ms"
        )

        # Streaming flush every 10 docs for crash insurance
        if (i + 1) % 10 == 0:
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                flush_path, compression="zstd",
            )

    # Final flush
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        flush_path, compression="zstd",
    )

    # Manifest for the generation run
    (gen_dir / "manifest.json").write_text(json.dumps({
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "n_docs_requested": args.n_docs,
        "n_docs_generated": len(rows),
        "model": args.model,
        "tau": args.tau,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "gen_seed": args.gen_seed,
        "unique_seeded_topic_ids": sorted(set(
            int(t)
            for row in rows
            for t in row["seeded_topic_ids"]
        )),
    }, indent=2))

    logger.info(f"DONE — wrote {len(rows)} docs to {flush_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Phase: analyze
# ─────────────────────────────────────────────────────────────────────────

def phase_analyze(args: argparse.Namespace, out_dir: Path) -> int:
    calib_dir = _calib_dir(out_dir)
    gen_dir = _gen_dir(out_dir, args.tag)

    generated = pq.read_table(gen_dir / "generated.parquet").to_pandas()
    logger.info(f"loaded {len(generated)} generated docs from tag={args.tag}")

    topic_centroids = np.load(calib_dir / "topic_centroids.npy")
    topic_ids = json.loads((calib_dir / "topic_ids.json").read_text())
    topic_records = json.loads((calib_dir / "topic_info.json").read_text())
    topic_by_id = {int(r["topic_id"]): r for r in topic_records}
    gen_manifest = json.loads((gen_dir / "manifest.json").read_text())
    unique_seeded = set(int(t) for t in gen_manifest["unique_seeded_topic_ids"])

    # Embed
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device=device)
    texts = generated["response_text"].fillna("").apply(
        lambda s: s[:8000]
    ).tolist()
    logger.info(f"embedding {len(texts)} generated docs on {device}...")
    out_embs = embedder.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True,
                                show_progress_bar=False).astype(np.float32)

    # Fresh BERTopic on output
    from bertopic import BERTopic
    from umap import UMAP
    model = BERTopic(
        umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                         metric="cosine", random_state=42),
        verbose=False,
    )
    out_topics, _ = model.fit_transform(texts, embeddings=out_embs)
    out_info = model.get_topic_info()
    out_n_clusters = len(out_info) - (1 if -1 in out_info["Topic"].values else 0)
    out_noise = int((np.asarray(out_topics) == -1).sum())
    logger.info(f"output BERTopic: {out_n_clusters} clusters, "
                f"{out_noise} noise ({out_noise/len(texts):.1%})")

    # For each output cluster, find nearest input topic by cosine
    cluster_to_input: list[dict] = []
    out_topics_arr = np.asarray(out_topics)
    for _, row in out_info.iterrows():
        out_tid = int(row["Topic"])
        if out_tid == -1:
            continue
        mask = out_topics_arr == out_tid
        if mask.sum() == 0:
            continue
        centroid = out_embs[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = topic_centroids @ centroid
        nearest = int(np.argmax(sims))
        nearest_input_tid = topic_ids[nearest]
        cluster_to_input.append({
            "out_cluster_id": out_tid,
            "out_cluster_n": int(mask.sum()),
            "nearest_input_topic_id": nearest_input_tid,
            "nearest_input_sim": float(sims[nearest]),
            "input_keywords": topic_by_id[nearest_input_tid]["keywords"][:10],
        })

    # Recovery: how many unique seeded topics are the nearest-input-match
    # for at least one output cluster?
    recovered = set()
    for c in cluster_to_input:
        recovered.add(int(c["nearest_input_topic_id"]))
    recovered_seeded = recovered & unique_seeded
    recovery_rate = (len(recovered_seeded) / max(len(unique_seeded), 1))

    summary = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "n_generated_docs": len(texts),
        "n_unique_seeded_topics": len(unique_seeded),
        "n_output_clusters": out_n_clusters,
        "n_output_noise_docs": out_noise,
        "noise_frac": out_noise / len(texts),
        "n_recovered_topics": len(recovered_seeded),
        "n_recovered_outside_seeded": len(recovered - unique_seeded),
        "topic_recovery_rate": recovery_rate,
        "cluster_to_input_top": sorted(
            cluster_to_input, key=lambda c: -c["out_cluster_n"]
        )[:20],
        "unrecovered_seeded_topic_ids": sorted(unique_seeded - recovered),
    }
    (gen_dir / "analysis.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 74)
    print(f"Topic recovery analysis — tag={args.tag}")
    print("=" * 74)
    print(f"  generated docs:            {summary['n_generated_docs']}")
    print(f"  unique seeded topics:      {summary['n_unique_seeded_topics']}")
    print(f"  output BERTopic clusters:  {summary['n_output_clusters']}")
    print(f"  output noise docs:         {summary['n_output_noise_docs']} ({summary['noise_frac']:.1%})")
    print()
    print(f"  recovered seeded topics:   {summary['n_recovered_topics']}")
    print(f"  → recovery_rate:           {recovery_rate:.1%}")
    print()
    print(f"  recovered topics outside seed set: {summary['n_recovered_outside_seeded']}")
    print()
    print("  Output cluster → nearest input topic (top 12 by size):")
    for c in summary["cluster_to_input_top"][:12]:
        print(f"    cluster {c['out_cluster_id']:3d} (n={c['out_cluster_n']:3d}) "
              f"→ input topic {c['nearest_input_topic_id']:3d} "
              f"sim={c['nearest_input_sim']:.3f}  "
              f"[{', '.join(c['input_keywords'][:5])}]")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "calibrate":
        return phase_calibrate(args, out_dir)
    if args.phase == "generate":
        return phase_generate(args, out_dir)
    if args.phase == "analyze":
        return phase_analyze(args, out_dir)
    raise SystemExit(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
