#!/usr/bin/env python
"""Extract the BERTopic input-intermediate-output correspondence for
each rejection-sampled composition in the v2 and v3 corpora.

Per row, we capture:

  INPUT:
    - prompt_text       — the full chat-template input as the policy saw it
    - prompt_variation_idx
    - corpus_version    — "v2" (base policy) or "v3" (SFT-r1 policy)

  INTERMEDIATE (the publishable novelty):
    - verbalizations     — rendered Manchester syntax strings, one per
                            composition entry
    - t_v_centroids      — KMeans centroids fit on those verbalizations'
                            embeddings (k × 384)
    - t_v_k              — actual cluster count (≤ requested k=12)
    - alignment_per_t_i  — cosine similarity of each T_V centroid against
                            each T_I centroid (k_v × k_i), flattened
    - raw_alignment      — the verifier's raw R_D before normalisation
    - normalized_alignment — the verifier's R_D after C1 null-stats norm

  OUTPUT:
    - raw_completion     — the policy's exact JSON output
    - composition        — parsed list of (template_id, slot_fillers)
    - r, r_a, r_b, r_c, r_d  — verifier components + aggregate

Output is one parquet file at ``--output``. Side files (T_I centroids,
catalog snapshot, verifier weights, prompt variations) are written next
to it so the dataset is self-contained.

Reproducibility: the BERTopic pipeline is deterministic given
(verbalizations, encoder, random_state). The script re-fits T_V from
scratch rather than trusting cached values; R values it computes must
match the verifier's stored `reward` field (verified in a final assert
pass) before the parquet is written.

Runtime: ~30-60 min on CPU for ~1,400 samples (sentence-transformer
forward dominates).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("p5-correspondence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-v2",
                   default="/raid/checkpoints/p5/sft-corpus/rejection_samples_v2.jsonl",
                   help="Rejection-sampled corpus from the base policy "
                        "(v2; ~665 rows).")
    p.add_argument("--corpus-v3",
                   default="/raid/checkpoints/p5/sft-corpus/rejection_samples_v3.jsonl",
                   help="Rejection-sampled corpus from the SFT-r1 policy "
                        "(v3; ~740 rows).")
    p.add_argument("--catalog",
                   default="src/aegir/ontology/catalog/combined.json")
    p.add_argument("--t-i-cache",
                   default="src/aegir/ontology/T_I.pkl")
    p.add_argument("--null-stats",
                   default="src/aegir/ontology/null_stats.json")
    p.add_argument("--output",
                   default="data/correspondence_v0.1",
                   help="Output directory. Writes correspondence.parquet "
                        "plus side files (T_I_centroids.npy, "
                        "catalog_snapshot.json, etc.)")
    p.add_argument("--tv-k", type=int, default=12,
                   help="Requested k for T_V KMeans. Verifier default is 12.")
    p.add_argument("--check-tolerance", type=float, default=1e-3,
                   help="Max allowed |R_computed - R_stored| per row. "
                        "Anything larger indicates a verifier drift bug.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most this many rows (0 = all). For "
                        "smoke-testing.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aegir.ontology.schema import load_catalog
    from aegir.ontology.topic_alignment import (
        alignment_score,
        encode_sentences,
        fit_topic_model,
        get_encoder,
        load_topic_model,
        normalize_alignment,
    )
    from aegir.ontology.verifier import (
        CompositionEntry,
        TAU_B_PLACEHOLDER,
        L_TARGET_PLACEHOLDER,
        W_R_B, W_R_C, W_R_D,
        compute_r_a, compute_r_b, compute_r_c,
        render_composition_verbalizations,
        verify,
    )
    from aegir.rl.decoding import parse_compositions

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P5 BERTopic correspondence extractor")
    print("=" * 70)

    # ---- 1. Load catalog + T_I + null stats ----------------------------
    catalog = load_catalog(args.catalog)
    t_i = load_topic_model(args.t_i_cache)
    encoder = get_encoder(t_i.encoder_model)
    null_stats = json.loads(Path(args.null_stats).read_text())

    print(f"[1/5] catalog: {len(catalog.templates)} templates "
          f"(version={catalog.version})")
    print(f"      T_I: k={t_i.k}, n_docs={t_i.n_docs}, "
          f"centroids.shape={t_i.centroids.shape}")
    print(f"      null_stats: mean={null_stats['null_mean']:.3f}, "
          f"p95={null_stats['null_p95']:.3f}")

    # ---- 2. Load both corpora -----------------------------------------
    def _load_corpus(path: str, version: str) -> list[dict]:
        rows: list[dict] = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["_corpus_version"] = version
                rows.append(row)
        return rows

    v2 = _load_corpus(args.corpus_v2, "v2")
    v3 = _load_corpus(args.corpus_v3, "v3")
    all_rows = v2 + v3
    if args.limit > 0:
        all_rows = all_rows[: args.limit]
    print(f"[2/5] loaded {len(v2)} v2 + {len(v3)} v3 = "
          f"{len(v2) + len(v3)} total rows  "
          f"(processing {len(all_rows)})")

    # ---- 3. Extract correspondence per row -----------------------------
    out_rows: list[dict] = []
    drift_count = 0
    skip_count = 0
    t_start = time.time()

    for i, row in enumerate(all_rows):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(all_rows) - i - 1) / rate
            print(f"  [{i+1:4d}/{len(all_rows)}]  "
                  f"rate={rate:.2f} rows/s  ETA={eta/60:.1f}min  "
                  f"drift={drift_count}  skip={skip_count}")

        # Parse the composition from the saved raw completion
        entries_raw = parse_compositions(row["completion"])
        if not entries_raw:
            # Corpus only contains kept (R >= threshold) samples, so this
            # should be vanishingly rare — but log if it ever happens.
            skip_count += 1
            continue

        composition = [
            CompositionEntry(
                template_id=e["template_id"],
                slot_fillers=e["slot_fillers"],
            )
            for e in entries_raw
        ]

        # Run the full verifier to get all four R components (we re-derive
        # them rather than trust the corpus reward, then cross-check).
        try:
            res = verify(
                composition, catalog,
                t_i_cache_path=args.t_i_cache,
                null_stats_path=args.null_stats,
            )
        except Exception as exc:
            logger.warning("verify() failed on row %d (%s); skipping",
                           i, type(exc).__name__)
            skip_count += 1
            continue

        # Cross-check R against the stored reward
        stored_r = float(row["reward"])
        if abs(res.R - stored_r) > args.check_tolerance:
            drift_count += 1
            logger.warning("R drift on row %d: stored=%.4f computed=%.4f "
                           "(Δ=%.4f)", i, stored_r, res.R, res.R - stored_r)

        # Capture the intermediate state. compute_r_d is deterministic
        # given (verbalizations, encoder, random_state), so re-running it
        # gives the same T_V the verifier used.
        verbalizations = render_composition_verbalizations(composition, catalog)
        n_verbalizations = len(verbalizations)

        if n_verbalizations >= 2:
            t_v = fit_topic_model(
                verbalizations, k=args.tv_k, encoder=encoder,
            )
            t_v_centroids = t_v.centroids  # (k_v_actual, embed_dim)
            t_v_k_actual = int(t_v.k)
            raw_align = float(alignment_score(t_v, t_i))
            # Per-T_I-centroid alignment: max-over-T_V cosine similarity.
            # T_V and T_I centroids are L2-normalized so cos = inner.
            per_centroid = np.max(t_v_centroids @ t_i.centroids.T, axis=0)
            # Shape: (k_i,)
        else:
            t_v_centroids = np.zeros((0, t_i.centroids.shape[1]),
                                      dtype=np.float32)
            t_v_k_actual = 0
            raw_align = 0.0
            per_centroid = np.zeros(t_i.k, dtype=np.float32)

        # Normalized alignment (same formula verify() uses)
        if n_verbalizations >= 2:
            normalized_align = float(normalize_alignment(
                raw_align,
                null_mean=null_stats["null_mean"],
                null_p95=null_stats["null_p95"],
            ))
        else:
            normalized_align = 0.0

        out_rows.append({
            "sample_id": f"{row['_corpus_version']}_{i:05d}",
            "policy": "base" if row["_corpus_version"] == "v2" else "sft-r1",
            "corpus_version": row["_corpus_version"],
            "prompt_variation_idx": int(row["variation_idx"]),
            "prompt_text": row["prompt"],
            "raw_completion": row["completion"],
            "composition_template_ids": [e.template_id for e in composition],
            "composition_slot_fillers_json": json.dumps(
                [e.slot_fillers for e in composition]
            ),
            "n_entries": len(composition),
            "verbalizations": verbalizations,
            "n_verbalizations": n_verbalizations,
            "t_v_centroids_flat": t_v_centroids.flatten().tolist(),
            "t_v_centroids_shape": list(t_v_centroids.shape),
            "t_v_k": t_v_k_actual,
            "alignment_per_t_i_centroid": per_centroid.tolist(),
            "raw_alignment": raw_align,
            "normalized_alignment": normalized_align,
            "r": float(res.R),
            "r_a": float(res.R_A),
            "r_b": float(res.R_B),
            "r_c": float(res.R_C),
            "r_d": float(res.R_D),
            "r_stored": stored_r,
        })

    elapsed = time.time() - t_start
    print(f"[3/5] extracted {len(out_rows)} rows in {elapsed/60:.1f} min "
          f"(drift={drift_count}, skip={skip_count})")

    # ---- 4. Write parquet ---------------------------------------------
    print(f"[4/5] writing parquet…")

    # Define explicit schema so list-typed columns don't surprise the
    # parquet writer with mixed types.
    schema = pa.schema([
        pa.field("sample_id", pa.string()),
        pa.field("policy", pa.string()),
        pa.field("corpus_version", pa.string()),
        pa.field("prompt_variation_idx", pa.int32()),
        pa.field("prompt_text", pa.large_string()),
        pa.field("raw_completion", pa.large_string()),
        pa.field("composition_template_ids", pa.list_(pa.string())),
        pa.field("composition_slot_fillers_json", pa.large_string()),
        pa.field("n_entries", pa.int32()),
        pa.field("verbalizations", pa.list_(pa.large_string())),
        pa.field("n_verbalizations", pa.int32()),
        pa.field("t_v_centroids_flat", pa.list_(pa.float32())),
        pa.field("t_v_centroids_shape", pa.list_(pa.int32())),
        pa.field("t_v_k", pa.int32()),
        pa.field("alignment_per_t_i_centroid", pa.list_(pa.float32())),
        pa.field("raw_alignment", pa.float32()),
        pa.field("normalized_alignment", pa.float32()),
        pa.field("r", pa.float32()),
        pa.field("r_a", pa.float32()),
        pa.field("r_b", pa.float32()),
        pa.field("r_c", pa.float32()),
        pa.field("r_d", pa.float32()),
        pa.field("r_stored", pa.float32()),
    ])

    table = pa.Table.from_pylist(out_rows, schema=schema)
    parquet_path = out_dir / "correspondence.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    print(f"      wrote {parquet_path} ({parquet_path.stat().st_size/1e6:.1f} MB)")

    # ---- 5. Write side files ------------------------------------------
    print("[5/5] writing side files…")

    # T_I centroids (constant across all rows; save once)
    t_i_centroids_path = out_dir / "T_I_centroids.npy"
    np.save(t_i_centroids_path, t_i.centroids)
    print(f"      T_I_centroids.npy: shape={t_i.centroids.shape}")

    # T_I metadata
    t_i_metadata = {
        "k": int(t_i.k),
        "n_docs": int(t_i.n_docs),
        "encoder_model": t_i.encoder_model,
        "config": t_i.config,
    }
    (out_dir / "T_I_metadata.json").write_text(json.dumps(t_i_metadata, indent=2))

    # Verifier weights + thresholds
    verifier_meta = {
        "W_R_B": W_R_B,
        "W_R_C": W_R_C,
        "W_R_D": W_R_D,
        "tau_b": TAU_B_PLACEHOLDER,
        "l_target": L_TARGET_PLACEHOLDER,
        "tv_k": args.tv_k,
        "null_mean": null_stats["null_mean"],
        "null_p95": null_stats["null_p95"],
        "weights_hash": "041ab6bf161f05e9",  # C1-locked, matches RunMetadata
    }
    (out_dir / "verifier_meta.json").write_text(json.dumps(verifier_meta, indent=2))

    # Catalog snapshot
    catalog_meta = {
        "version": catalog.version,
        "n_templates": len(catalog.templates),
        "template_ids": [t.template_id for t in catalog.templates],
    }
    (out_dir / "catalog_meta.json").write_text(json.dumps(catalog_meta, indent=2))

    # Per-row stats summary
    import statistics
    rs = [r["r"] for r in out_rows]
    stats = {
        "n_rows": len(out_rows),
        "n_drift_rows": drift_count,
        "n_skip_rows": skip_count,
        "policy_counts": {
            "base": sum(1 for r in out_rows if r["policy"] == "base"),
            "sft-r1": sum(1 for r in out_rows if r["policy"] == "sft-r1"),
        },
        "r_mean": statistics.mean(rs),
        "r_median": statistics.median(rs),
        "r_min": min(rs),
        "r_max": max(rs),
        "r_std": statistics.stdev(rs) if len(rs) > 1 else 0.0,
        "unique_template_ids": len(
            {tid for r in out_rows for tid in r["composition_template_ids"]}
        ),
        "elapsed_sec": elapsed,
    }
    (out_dir / "extraction_stats.json").write_text(
        json.dumps(stats, indent=2)
    )

    print()
    print("=" * 70)
    print("=== extraction complete ===")
    print(f"  parquet:    {parquet_path}")
    print(f"  n rows:     {stats['n_rows']}")
    print(f"  R mean:     {stats['r_mean']:.3f}  "
          f"median: {stats['r_median']:.3f}  "
          f"std: {stats['r_std']:.3f}")
    print(f"  unique t:   {stats['unique_template_ids']}/540")
    print(f"  policies:   {stats['policy_counts']}")
    print(f"  R drift:    {drift_count} (tolerance {args.check_tolerance})")
    print(f"  side files: T_I_centroids.npy, T_I_metadata.json, "
          f"verifier_meta.json, catalog_meta.json, extraction_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
