#!/usr/bin/env python
"""Ontology coverage audit — how well does the Aegir ontology cover
topics present in a FinePDFs slice?

Pure-embedding pass: no LLM judgments (those come in a second pass once
gaius.hx is vendored and the Cerebras key is in scope). This script's
output is the *baseline measurement* that the future Qwen-driven
ontology improvement loop will optimize against.

Pipeline:
  1. Sample N FinePDFs documents deterministically (seed-keyed).
  2. Embed documents with ``sentence-transformers/all-mpnet-base-v2``
     (same model as BERTopic / SDG verifier — single embedding model
     across the whole pipeline for compositional consistency).
  3. KMeans-cluster embeddings into K topics. Centroid = cluster mean;
     representative text = nearest-doc to centroid.
  4. Load 540 templates from ``src/aegir/ontology/catalog/0*.json``
     and embed each via its verbal-template + manchester-template +
     family.
  5. For each topic centroid: compute cosine similarity to all
     templates; record top-K matches.
  6. Classify topic status by max similarity: covered / borderline /
     gap (configurable thresholds).
  7. Aggregate density per template, per family.
  8. Write three parquets (Iceberg-ready schemas) and print a summary.

Outputs (under ``--output-dir``):
  - topic_coverage.parquet      — one row per topic
  - template_density.parquet    — one row per template
  - family_density.parquet      — one row per ontology family

Each parquet uses Iceberg-compatible column types and includes
``run_id`` (deterministic hash of inputs) and ``created_at`` (utc
timestamp) for stable partitioning when promoted to Iceberg.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("ontology-coverage-audit")


# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-corpus", nargs="+", default=None,
                   help="Local \\x03-delimited corpus file(s) to sample INSTEAD of streaming "
                        "HF (the canonical ground = the corpus the model actually pretrains on).")
    p.add_argument("--finepdfs-dataset", default="HuggingFaceFW/finepdfs",
                   help="HuggingFace dataset id")
    p.add_argument("--finepdfs-config", default="eng_Latn",
                   help="Dataset config (language code or named subset)")
    p.add_argument("--finepdfs-split", default="train")
    p.add_argument("--sample-size", type=int, default=10_000)
    p.add_argument("--skip-docs", type=int, default=0,
                   help="forward-index cursor: skip the first N docs of the stream/corpus before sampling "
                        "the next window — advances the aperture through the full FinePDFs corpus over runs")
    p.add_argument("--max-chars-per-doc", type=int, default=8000)
    p.add_argument("--seed", type=int, default=4649)
    p.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--n-topics", type=int, default=200,
                   help="K for KMeans clustering (BERTopic-equivalent target)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of nearest templates recorded per topic")
    p.add_argument("--tau-high", type=float, default=0.55,
                   help="Similarity ≥ this counts as 'covered'")
    p.add_argument("--tau-low", type=float, default=0.35,
                   help="Similarity < this counts as 'gap'; "
                        "between thresholds is 'borderline'")
    p.add_argument("--output-dir",
                   default="/raid/checkpoints/aegir-artifacts/coverage_v0/")
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Run-id (deterministic from inputs)
# ─────────────────────────────────────────────────────────────────────────

def compute_run_id(args: argparse.Namespace, catalog_files: list[Path]) -> str:
    """Hash inputs so identical re-runs produce identical run_id."""
    h = hashlib.sha256()
    if getattr(args, "local_corpus", None):
        h.update(f"local:{':'.join(sorted(args.local_corpus))}:{args.sample_size}:"
                 f"{args.seed}:{args.max_chars_per_doc}\n".encode())
    else:
        h.update(f"finepdfs:{args.finepdfs_dataset}:{args.finepdfs_config}:"
                 f"{args.finepdfs_split}:{args.sample_size}:{args.seed}:"
                 f"{args.max_chars_per_doc}\n".encode())
    h.update(f"embedding_model:{args.embedding_model}\n".encode())
    h.update(f"skip_docs:{args.skip_docs}\n".encode())   # forward-index window → distinct run per cursor
    h.update(f"clustering:kmeans:k={args.n_topics}:seed={args.seed}\n".encode())
    h.update(f"thresholds:tau_high={args.tau_high}:tau_low={args.tau_low}\n".encode())
    for p in sorted(catalog_files):
        with open(p, "rb") as f:
            file_h = hashlib.sha256(f.read()).hexdigest()[:16]
        h.update(f"{p.name}:{file_h}\n".encode())
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────
# FinePDFs sampling
# ─────────────────────────────────────────────────────────────────────────

def sample_local_corpus(args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    """Reservoir-sample docs from local corpus file(s) (\\x03-delimited; the ACTUAL
    filtered FinePDFs the model pretrains on) — the canonical ground. ``--skip-docs`` advances the
    forward-index cursor: the first N eligible docs are skipped, then a window is sampled. Returns
    (ids, texts, docs_consumed) where docs_consumed = skip + window-scanned (the next cursor)."""
    rng = np.random.default_rng(args.seed)
    skip = max(0, args.skip_docs)
    cap = max(args.sample_size * 5, 50_000)
    ids: list[str] = []
    texts: list[str] = []
    n_seen = 0          # eligible docs seen in the WINDOW (post-skip)
    skipped = 0         # eligible docs skipped to reach the cursor
    for path in args.local_corpus:
        name = Path(path).name
        buf = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split("\x03")
                buf = parts.pop()
                for doc in parts:
                    doc = doc.strip()
                    if len(doc) < 200:
                        continue
                    if skipped < skip:           # advance the cursor to the window start
                        skipped += 1
                        continue
                    doc = doc[: args.max_chars_per_doc]
                    gidx = skip + n_seen
                    if len(texts) < args.sample_size:
                        ids.append(f"{name}:{gidx}")
                        texts.append(doc)
                    else:
                        j = int(rng.integers(0, n_seen + 1))
                        if j < args.sample_size:
                            ids[j] = f"{name}:{gidx}"
                            texts[j] = doc
                    n_seen += 1
                if n_seen >= cap:
                    break
        if n_seen >= cap:
            break
    logger.info(f"sampled {len(texts):,} docs from {n_seen:,} scanned (local corpus; skipped {skipped:,} to cursor {skip:,})")
    return ids, texts, skip + n_seen


def sample_finepdfs(args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    """Stream the dataset and pick a deterministic sample (forward-index aware).

    ``--skip-docs`` advances the cursor through the (remote/full) FinePDFs stream before sampling the
    next window. Returns (ids, texts, docs_consumed) — docs_consumed = skip + window-scanned (next cursor).
    """
    if getattr(args, "local_corpus", None):
        return sample_local_corpus(args)
    from datasets import load_dataset

    logger.info(f"streaming {args.finepdfs_dataset} "
                f"config={args.finepdfs_config} split={args.finepdfs_split} skip={args.skip_docs:,}")
    ds = load_dataset(
        args.finepdfs_dataset, args.finepdfs_config,
        split=args.finepdfs_split, streaming=True,
    )

    rng = np.random.default_rng(args.seed)
    skip = max(0, args.skip_docs)
    skipped = 0
    # Reservoir sampling so the sample is truly uniform over the WINDOW (post-skip).
    reservoir_ids: list[str] = []
    reservoir_texts: list[str] = []
    n_seen = 0
    log_every = 5000
    t0 = time.time()
    text_field_candidates = ("text", "content", "raw_text")
    text_field = None
    id_field_candidates = ("id", "url", "_id")
    id_field = None

    for row in ds:
        if text_field is None:
            for cand in text_field_candidates:
                if cand in row:
                    text_field = cand
                    break
            if text_field is None:
                logger.error(f"no recognized text field in row keys "
                             f"{list(row.keys())}")
                raise SystemExit(1)
            for cand in id_field_candidates:
                if cand in row:
                    id_field = cand
                    break
            if id_field is None:
                id_field = "_synthetic_id"
            logger.info(f"text field='{text_field}', id field='{id_field}'")

        text = row[text_field] or ""
        if not text.strip():
            continue
        if skipped < skip:                  # advance the forward-index cursor to the window start
            skipped += 1
            continue
        text = text[: args.max_chars_per_doc]
        doc_id = str(row[id_field]) if id_field in row else f"row{skip + n_seen}"

        if len(reservoir_texts) < args.sample_size:
            reservoir_ids.append(doc_id)
            reservoir_texts.append(text)
        else:
            j = int(rng.integers(0, n_seen + 1))
            if j < args.sample_size:
                reservoir_ids[j] = doc_id
                reservoir_texts[j] = text

        n_seen += 1
        if n_seen % log_every == 0:
            elapsed = time.time() - t0
            logger.info(f"  scanned {n_seen:,} docs in {elapsed:.1f}s "
                        f"(reservoir={len(reservoir_texts):,})")
        # Cap WINDOW scan at ~5x sample_size for time bound; reservoir converges fast.
        if n_seen >= max(args.sample_size * 5, 50_000):
            break

    logger.info(f"sampled {len(reservoir_texts):,} docs from {n_seen:,} scanned "
                f"(skipped {skipped:,} to cursor {skip:,})")
    return reservoir_ids, reservoir_texts, skip + n_seen


# ─────────────────────────────────────────────────────────────────────────
# Catalog loading + template embedding text
# ─────────────────────────────────────────────────────────────────────────

def load_all_catalogs(catalog_dir: Path) -> tuple[list, list[Path]]:
    """Read every canonical NN_*.json (skip *.candidate.json + combined.json)."""
    files = sorted(p for p in catalog_dir.glob("0*.json")
                   if "candidate" not in p.name)
    templates = []
    for p in files:
        with open(p) as f:
            data = json.load(f)
        for t in data.get("templates", []):
            t = dict(t)
            t["_source_file"] = p.name
            t["_family"] = p.stem  # e.g., '01_foundation'
            templates.append(t)
    return templates, files


def template_embedding_text(t: dict) -> str:
    """Build a rich representation of a template for embedding.

    Combines: verbal template (LLM-generated), manchester syntax,
    family/branch, slot types. Substitutes slot placeholders with
    a neutral 'concept' token so the text reads naturally.
    """
    verbal = t.get("verbal_template") or ""
    manchester = t.get("manchester_template") or ""
    family = t.get("_family", "")
    branch = t.get("provenance", {}).get("branch", "")
    slot_types = " ".join(f"{k}:{v}" for k, v in (t.get("slot_types") or {}).items())

    # Replace {X}-style placeholders with 'concept' for natural reading.
    import re
    verbal_clean = re.sub(r"\{[A-Za-z]\}", "concept", verbal)
    manchester_clean = re.sub(r"\{[A-Za-z]+:[A-Za-z]+\}", "concept", manchester)

    parts = []
    if verbal_clean:
        parts.append(verbal_clean)
    parts.append(manchester_clean)
    parts.append(f"family: {family.split('_', 1)[-1].replace('_', ' ')}")
    if branch:
        parts.append(f"branch: {branch}")
    parts.append(f"slots: {slot_types}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────
# Audit core
# ─────────────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], model, device: str,
                batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Encode in batches; return (N, D) float32."""
    return model.encode(
        texts, batch_size=batch_size, device=device,
        normalize_embeddings=normalize, show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    args = parse_args()

    # Device resolution
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"device: {device}")

    catalog_dir = REPO / args.catalog_dir
    templates, catalog_files = load_all_catalogs(catalog_dir)
    logger.info(f"loaded {len(templates):,} templates from "
                f"{len(catalog_files)} catalog files")

    run_id = compute_run_id(args, catalog_files)
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run_id={run_id}  output_dir={output_dir}")

    # Save the resolved manifest for this run
    manifest = {
        "run_id": run_id,
        "args": vars(args),
        "catalog_files": [p.name for p in catalog_files],
        "n_templates": len(templates),
        "device": device,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Stage 1: sample FinePDFs ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Stage 1/5: FinePDFs sample")
    logger.info("=" * 70)
    t0 = time.time()
    doc_ids, doc_texts, docs_consumed = sample_finepdfs(args)
    logger.info(f"  {len(doc_texts):,} docs sampled in {time.time()-t0:.1f}s "
                f"(cursor: {args.skip_docs:,} → {docs_consumed:,})")
    # Forward-index cursor: persist where the next window should start. A driver advances the aperture
    # by passing `--skip-docs <next_skip>` on the next audit run, walking the full FinePDFs corpus.
    with open(output_dir / "cursor.json", "w") as _cf:
        json.dump({"start_doc": args.skip_docs, "window_sampled": len(doc_texts),
                   "docs_consumed": docs_consumed, "next_skip": docs_consumed,
                   "sample_size": args.sample_size}, _cf, indent=2)

    # ── Stage 2: embed docs ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Stage 2/5: embed docs")
    logger.info("=" * 70)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(args.embedding_model, device=device)
    logger.info(f"  loaded {args.embedding_model} in {time.time()-t0:.1f}s")

    t0 = time.time()
    doc_embeddings = embed_batch(doc_texts, model, device=device)
    logger.info(f"  embedded {len(doc_texts):,} docs in {time.time()-t0:.1f}s "
                f"(shape={doc_embeddings.shape})")

    # ── Stage 3: KMeans cluster ─────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Stage 3/5: KMeans clustering")
    logger.info("=" * 70)
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=args.n_topics,
        random_state=args.seed,
        batch_size=2048, max_iter=200, n_init=5,
    )
    cluster_labels = km.fit_predict(doc_embeddings)
    logger.info(f"  clustered into {args.n_topics} topics in "
                f"{time.time()-t0:.1f}s")

    # Topic representative: doc whose embedding is nearest to centroid.
    topic_centroids = km.cluster_centers_  # (n_topics, D)
    # Re-normalize centroids since cosine sim assumes unit vectors.
    norms = np.linalg.norm(topic_centroids, axis=1, keepdims=True)
    topic_centroids = topic_centroids / np.clip(norms, 1e-9, None)

    topic_doc_counts = Counter(cluster_labels.tolist())
    topic_repr_idx = np.full(args.n_topics, -1, dtype=np.int64)
    # For each topic, find the doc with max similarity to centroid.
    for tid in range(args.n_topics):
        members = np.where(cluster_labels == tid)[0]
        if len(members) == 0:
            continue
        member_embs = doc_embeddings[members]
        sims = member_embs @ topic_centroids[tid]
        topic_repr_idx[tid] = members[int(np.argmax(sims))]

    # ── Stage 4: embed templates ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Stage 4/5: embed templates")
    logger.info("=" * 70)
    template_texts = [template_embedding_text(t) for t in templates]
    t0 = time.time()
    template_embeddings = embed_batch(template_texts, model, device=device)
    logger.info(f"  embedded {len(templates):,} templates in "
                f"{time.time()-t0:.1f}s")

    # ── Stage 5: similarity + classify ──────────────────────────────────
    logger.info("=" * 70)
    logger.info("Stage 5/5: similarity matrix + classification")
    logger.info("=" * 70)
    # (n_topics, n_templates) similarity
    sims = topic_centroids @ template_embeddings.T  # both unit-norm
    logger.info(f"  similarity matrix: {sims.shape}")

    # Per-topic: top-K templates
    topk_idx = np.argsort(-sims, axis=1)[:, : args.top_k]
    topk_sim = np.take_along_axis(sims, topk_idx, axis=1)

    # Classify
    max_sims = sims.max(axis=1)
    def classify(sim: float) -> str:
        if sim >= args.tau_high: return "covered"
        if sim < args.tau_low: return "gap"
        return "borderline"
    statuses = np.array([classify(s) for s in max_sims])

    # ── Write topic_coverage.parquet ────────────────────────────────────
    created_at = _dt.datetime.now(_dt.timezone.utc)
    topic_rows = []
    for tid in range(args.n_topics):
        repr_idx = int(topic_repr_idx[tid])
        repr_text = (doc_texts[repr_idx][:400] + "..."
                     if repr_idx >= 0 and len(doc_texts[repr_idx]) > 400
                     else (doc_texts[repr_idx] if repr_idx >= 0 else ""))
        top_templates = []
        for k in range(args.top_k):
            t_idx = int(topk_idx[tid, k])
            top_templates.append({
                "template_id": templates[t_idx]["template_id"],
                "family": templates[t_idx]["_family"],
                "similarity": float(topk_sim[tid, k]),
                "manchester_template": templates[t_idx]["manchester_template"],
            })
        topic_rows.append({
            "topic_id": int(tid),
            "topic_doc_count": int(topic_doc_counts.get(tid, 0)),
            "topic_repr_doc_id": str(doc_ids[repr_idx]) if repr_idx >= 0 else "",
            "topic_repr_text": repr_text,
            "top_templates": top_templates,
            "top_template_id": top_templates[0]["template_id"] if top_templates else "",
            "top_family": top_templates[0]["family"] if top_templates else "",
            "coverage_score": float(max_sims[tid]),
            "status": str(statuses[tid]),
            "run_id": run_id,
            "created_at": created_at,
        })
    topic_schema = pa.schema([
        ("topic_id", pa.int32()),
        ("topic_doc_count", pa.int32()),
        ("topic_repr_doc_id", pa.string()),
        ("topic_repr_text", pa.string()),
        ("top_templates", pa.list_(pa.struct([
            ("template_id", pa.string()),
            ("family", pa.string()),
            ("similarity", pa.float32()),
            ("manchester_template", pa.string()),
        ]))),
        ("top_template_id", pa.string()),
        ("top_family", pa.string()),
        ("coverage_score", pa.float32()),
        ("status", pa.string()),
        ("run_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
    pq.write_table(
        pa.Table.from_pylist(topic_rows, schema=topic_schema),
        output_dir / "topic_coverage.parquet",
        compression="zstd",
    )
    logger.info(f"  wrote {output_dir / 'topic_coverage.parquet'}")

    # Persist the (unit-norm) topic centroids, row tid == topic_id. Downstream
    # gates (E5 target-topic) and a lite re-audit need the true centroid, not
    # the single repr-doc, to reproduce the audit's template↔topic similarity.
    np.save(output_dir / "topic_centroids.npy", topic_centroids)
    logger.info(f"  wrote {output_dir / 'topic_centroids.npy'}  {topic_centroids.shape}")

    # ── Write template_density.parquet ──────────────────────────────────
    topk_set_per_topic = topk_idx  # (n_topics, top_k)
    n_topics_top1 = np.bincount(topk_set_per_topic[:, 0], minlength=len(templates))
    in_topk = np.zeros(len(templates), dtype=np.int32)
    for tid in range(args.n_topics):
        for k in range(args.top_k):
            in_topk[topk_set_per_topic[tid, k]] += 1
    template_rows = []
    for ti, t in enumerate(templates):
        template_rows.append({
            "template_id": t["template_id"],
            "manchester_template": t["manchester_template"],
            "family": t["_family"],
            "verbal_template": t.get("verbal_template") or "",
            "is_complex": bool(t.get("is_complex", False)),
            "n_topics_top1": int(n_topics_top1[ti]),
            "n_topics_in_topk": int(in_topk[ti]),
            "max_similarity": float(sims[:, ti].max()),
            "mean_similarity": float(sims[:, ti].mean()),
            "run_id": run_id,
            "created_at": created_at,
        })
    template_schema = pa.schema([
        ("template_id", pa.string()),
        ("manchester_template", pa.string()),
        ("family", pa.string()),
        ("verbal_template", pa.string()),
        ("is_complex", pa.bool_()),
        ("n_topics_top1", pa.int32()),
        ("n_topics_in_topk", pa.int32()),
        ("max_similarity", pa.float32()),
        ("mean_similarity", pa.float32()),
        ("run_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
    pq.write_table(
        pa.Table.from_pylist(template_rows, schema=template_schema),
        output_dir / "template_density.parquet",
        compression="zstd",
    )
    logger.info(f"  wrote {output_dir / 'template_density.parquet'}")

    # ── Write family_density.parquet ────────────────────────────────────
    family_counts = Counter(t["_family"] for t in templates)
    family_topic_assignment = Counter()
    family_topic_in_topk = Counter()
    family_doc_assignment = Counter()
    for tid in range(args.n_topics):
        top_t = templates[int(topk_idx[tid, 0])]
        family_topic_assignment[top_t["_family"]] += 1
        family_doc_assignment[top_t["_family"]] += int(topic_doc_counts.get(tid, 0))
        seen_fams = set()
        for k in range(args.top_k):
            fam = templates[int(topk_idx[tid, k])]["_family"]
            if fam not in seen_fams:
                family_topic_in_topk[fam] += 1
                seen_fams.add(fam)

    family_rows = []
    for fam in sorted(family_counts):
        family_rows.append({
            "family": fam,
            "n_templates": int(family_counts[fam]),
            "n_topics_as_top1": int(family_topic_assignment[fam]),
            "n_topics_in_topk": int(family_topic_in_topk[fam]),
            "n_docs_under_top1": int(family_doc_assignment[fam]),
            "topic_share": float(family_topic_assignment[fam] / args.n_topics),
            "doc_share": float(family_doc_assignment[fam] / max(len(doc_texts), 1)),
            "run_id": run_id,
            "created_at": created_at,
        })
    family_schema = pa.schema([
        ("family", pa.string()),
        ("n_templates", pa.int32()),
        ("n_topics_as_top1", pa.int32()),
        ("n_topics_in_topk", pa.int32()),
        ("n_docs_under_top1", pa.int32()),
        ("topic_share", pa.float32()),
        ("doc_share", pa.float32()),
        ("run_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
    pq.write_table(
        pa.Table.from_pylist(family_rows, schema=family_schema),
        output_dir / "family_density.parquet",
        compression="zstd",
    )
    logger.info(f"  wrote {output_dir / 'family_density.parquet'}")

    # ── Print summary ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    n_covered = int((statuses == "covered").sum())
    n_borderline = int((statuses == "borderline").sum())
    n_gap = int((statuses == "gap").sum())
    total_docs = sum(topic_doc_counts.values())

    n_docs_covered = sum(topic_doc_counts[tid] for tid in range(args.n_topics)
                          if statuses[tid] == "covered")
    n_docs_borderline = sum(topic_doc_counts[tid] for tid in range(args.n_topics)
                             if statuses[tid] == "borderline")
    n_docs_gap = sum(topic_doc_counts[tid] for tid in range(args.n_topics)
                      if statuses[tid] == "gap")

    print(f"\n  run_id: {run_id}")
    print(f"  output: {output_dir}/")
    print(f"\n  topics ({args.n_topics} total): "
          f"{n_covered} covered / {n_borderline} borderline / {n_gap} gap")
    print(f"  docs ({total_docs} total): "
          f"{n_docs_covered} ({100*n_docs_covered/max(total_docs,1):.1f}%) covered / "
          f"{n_docs_borderline} ({100*n_docs_borderline/max(total_docs,1):.1f}%) borderline / "
          f"{n_docs_gap} ({100*n_docs_gap/max(total_docs,1):.1f}%) gap")
    print(f"\n  family distribution (by top-1 topic assignment):")
    for fam in sorted(family_counts):
        share = family_topic_assignment[fam] / args.n_topics
        bar = "█" * max(1, int(share * 40))
        print(f"    {fam:30s} {family_topic_assignment[fam]:4d} topics "
              f"({100*share:5.1f}%) {bar}")
    print(f"\n  top 5 GAP topics by doc count:")
    gap_topics = [(tid, topic_doc_counts[tid]) for tid in range(args.n_topics)
                   if statuses[tid] == "gap"]
    gap_topics.sort(key=lambda x: -x[1])
    for tid, dc in gap_topics[:5]:
        repr_text = ""
        if topic_repr_idx[tid] >= 0:
            repr_text = doc_texts[int(topic_repr_idx[tid])][:120].replace("\n", " ")
        print(f"    topic {tid:3d} ({dc:4d} docs, max_sim={max_sims[tid]:.3f}): "
              f"{repr_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
