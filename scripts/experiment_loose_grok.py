#!/usr/bin/env python
"""One-off experiment: bare-ontology prompting + BERTopic correspondence.

The thesis under test: our current pipeline wraps ontology guidance in
elaborate schema-rich, template-citing, table-mandating ceremony — built
for an era when LLMs needed structural scaffolding to produce coherent
domain content. Modern frontier models (Grok-4.3, GLM-4.7) may not need
the scaffolding. The question is whether they can take a *handful of
ontology verbalizations* as concept hints and produce documents that
land in the same FinePDFs topic space — without the schema/table/citation
ceremony.

Two-phase, single-script:

  Phase generate:
    1. Pick a FinePDFs topic from the audit (default: topic 111,
       audit/risk committee charter — governance-aligned, n=95
       FinePDFs docs).
    2. Pull its top-5 templates from the audit; extract verbal_template
       fields as the "handful of ontology verbalizations".
    3. Build a deliberately minimal prompt: domain descriptor + 5
       verbalizations as "concepts to weave in, not recite". No mandatory
       structure, no table requirements, no axiom citations.
    4. Generate N documents via Grok-4.3. Save to docs.parquet.

  Phase analyze:
    1. Re-sample FinePDFs deterministically (same seed as audit) to get
       actual document-level reference set.
    2. Embed all (Grok docs + FinePDFs sample + audit topic
       representatives) with sentence-transformers.
    3. For each Grok doc: find nearest audit topic (out of 200) and
       record. Compute distribution.
    4. Run BERTopic on Grok docs alone to discover cluster structure;
       compute centroid → audit-topic correspondence.
    5. Mixed-cluster analysis: cluster the union (Grok + FinePDFs),
       measure per-cluster Grok-FinePDFs ratio.

Success signal: most Grok docs cluster near topic 111 (the seed) and a
few "neighbor" audit topics — not random/uniform distribution across
all 200 audit topics. Failure signal: Grok wanders, docs cluster in
embedding regions disjoint from FinePDFs.

Output: build/experiments/loose_grok_v0/
  manifest.json    — full args + verbalizations used
  docs.parquet     — generated documents
  analysis.json    — BERTopic results + correspondence metrics
  *.md             — per-doc markdown for inspection

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/experiment_loose_grok.py --phase all \\
        --seed-topic 111 --n-docs 80 \\
        --audit-run /raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf
"""

from __future__ import annotations

import argparse
import hashlib
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

logger = logging.getLogger("loose-grok")


LOOSE_PROMPT = """Write a 1500-word document about: {domain}.

As context, here is a representative excerpt from real-world documents \
on this topic:

{topic_excerpt}

The document you write should naturally engage with the following five \
concepts. Weave them into the prose as the topic demands — do NOT recite \
them, do NOT treat them as a checklist, do NOT cite them by name. They \
are scaffolding for your thinking, not deliverables:

{verbalizations}

Use whatever structure suits the subject best (prose, headings, lists, \
tables — whichever fits). Write the kind of document a domain practitioner \
would produce, with the depth and tone you would expect from a published \
technical or governance document. No required format."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("generate", "analyze", "all"),
                   default="all")
    p.add_argument("--audit-run", required=True)
    p.add_argument("--seed-topic", type=int, default=111,
                   help="FinePDFs audit topic_id to seed generation. "
                        "Default 111 (audit/risk committee charter).")
    p.add_argument("--n-docs", type=int, default=80)
    p.add_argument("--n-verbalizations", type=int, default=5)
    p.add_argument("--model", default="xai/grok-4.3")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Bumped vs chapters' 0.4 — bare prompts benefit "
                        "from more variation since they don't have axiom-"
                        "citation guardrails.")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=11000)
    p.add_argument("--output-dir", default="build/experiments/loose_grok_v0")
    p.add_argument("--finepdfs-resample", type=int, default=2000,
                   help="Number of FinePDFs docs to re-sample for BERTopic "
                        "reference set. Same audit seed for determinism.")
    p.add_argument("--audit-finepdfs-seed", type=int, default=4649,
                   help="Original audit FinePDFs sampling seed. Default "
                        "matches ontology_coverage_audit.py default.")
    return p.parse_args()


def hand_crafted_domain(topic_id: int) -> str:
    """Short, human-curated domain descriptors for the seed topics we test.

    The audit's topic_repr_text is one doc near the centroid — useful as
    the prompt's 'topic excerpt' field, but too narrow as a domain name.
    These descriptors give the LLM a domain anchor without leaking
    structure expectations.
    """
    descriptors = {
        111: "corporate audit and risk committee charters, "
             "governance documentation, board oversight processes",
        90:  "management information systems coursework and curricula, "
             "business technology education",
        74:  "FDA regulation of medical devices, premarket notification, "
             "regulatory compliance for clinical equipment",
        28:  "human resources documentation, employment policies, "
             "civil service personnel administration",
        6:   "doctoral program administration, university research "
             "funding, IDEX/Excellence Initiative governance",
    }
    return descriptors.get(topic_id, "general institutional documentation")


# Hand-crafted natural-language verbalizations for the templates we use
# in this experiment. The catalog's `verbal_template` field actually holds
# Manchester syntax (the DeepOnto verbalization step was never run); for
# a "give Grok loose ontology hints" experiment we need actual English.
# Keyed by template_id. TODO: replace with a real DeepOnto pass on the
# catalog when that infrastructure lands.
NATURAL_VERBALIZATIONS: dict[str, str] = {
    # Topic 111 — audit/risk committee charter
    "audit_for_period": (
        "An audit is an event that covers a defined period of time."
    ),
    "audit_conducted_by": (
        "An audit is conducted by an identifiable person who is "
        "accountable for its execution."
    ),
    "audit_subclass": (
        "An audit is an event that examines a control or set of "
        "controls against a directive (policy, regulation, or rule)."
    ),
    "attestation_signed_by": (
        "An attestation is a signed statement, attributable to a "
        "specific person, that affirms a fact or condition."
    ),
    "requirement_equiv_specifies_target": (
        "A requirement is a directive that specifies some target "
        "(what must be done) and applies to some scope (where or to "
        "whom it applies)."
    ),
    "verification_min_one_evidence": (
        "A verification is an event that produces at least one piece "
        "of evidence."
    ),
    "policy_min_one_enforcer": (
        "A policy is a directive that is enforced by at least one "
        "agent (person, role, or system)."
    ),
    "constraint_evaluated_during": (
        "A constraint is evaluated during a specified phase, time "
        "interval, or process."
    ),
    "attestation_with_supporting_evidence": (
        "An attestation should be backed by at least one piece of "
        "supporting evidence."
    ),
    "control_with_evidence_requirement": (
        "A control has an associated requirement that evidence of its "
        "operation be retained."
    ),
}


def build_prompt(domain: str, topic_excerpt: str,
                  verbalizations: list[str]) -> str:
    verb_block = "\n".join(f"  {i}. {v}" for i, v in enumerate(verbalizations, 1))
    excerpt = (topic_excerpt or "")[:700]
    return LOOSE_PROMPT.format(
        domain=domain,
        topic_excerpt=excerpt,
        verbalizations=verb_block,
    )


def compute_doc_id(prompt: str, model: str, idx: int) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(str(idx).encode("utf-8"))
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────
# Phase: generate
# ─────────────────────────────────────────────────────────────────────────

def phase_generate(args: argparse.Namespace, out_dir: Path) -> Path:
    audit_run = Path(args.audit_run)
    topic_cov = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()

    seed_row = topic_cov[topic_cov.topic_id == args.seed_topic]
    if len(seed_row) == 0:
        raise SystemExit(f"seed topic {args.seed_topic} not in audit")
    seed_row = seed_row.iloc[0]
    topic_excerpt = seed_row["topic_repr_text"] or ""
    top_templates = list(seed_row["top_templates"])[:args.n_verbalizations]

    # Look up natural-language verbalizations from the inline table.
    # If a template_id isn't there yet, log it and skip — we keep this
    # experiment honest by only using actually-verbalized concepts as
    # ontology guidance, not OWL DL syntax.
    verbalizations = []
    missing = []
    for t in top_templates:
        tid = t["template_id"]
        if tid in NATURAL_VERBALIZATIONS:
            verbalizations.append(NATURAL_VERBALIZATIONS[tid])
        else:
            missing.append(tid)
    if missing:
        logger.warning(
            f"missing natural verbalizations for: {missing} — "
            f"add them to NATURAL_VERBALIZATIONS to include in the prompt"
        )
    if not verbalizations:
        raise SystemExit(
            f"no verbalizations available for seed topic {args.seed_topic}; "
            f"populate NATURAL_VERBALIZATIONS for at least one of "
            f"{[t['template_id'] for t in top_templates]}"
        )
    domain = hand_crafted_domain(args.seed_topic)
    prompt = build_prompt(domain, topic_excerpt, verbalizations)

    # Persist manifest
    used_template_ids = [t["template_id"] for t in top_templates
                          if t["template_id"] in NATURAL_VERBALIZATIONS]
    manifest = {
        "seed_topic": int(args.seed_topic),
        "domain": domain,
        "topic_excerpt_preview": (topic_excerpt or "")[:200],
        "verbalizations": verbalizations,
        "verbalization_template_ids": used_template_ids,
        "model": args.model,
        "n_docs": int(args.n_docs),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "audit_run": audit_run.name,
        "prompt": prompt,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"manifest written to {out_dir / 'manifest.json'}")
    logger.info(f"verbalizations: {verbalizations}")

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
        cache=False,  # bust per-call cache — we send the SAME prompt N
                      # times and want N independent completions to
                      # observe stochastic variety. Default cache=True
                      # would return one cached completion N times.
    )

    doc_rows = []
    for i in range(args.n_docs):
        t0 = time.time()
        # Each call uses the SAME prompt — variation comes from
        # temperature, not prompt mutation. This is intentional: the
        # experiment tests whether a single bare-ontology prompt
        # produces topic-coherent variety.
        result = lm(messages=[{"role": "user", "content": prompt}])
        latency_ms = int((time.time() - t0) * 1000)

        if isinstance(result, list):
            item = result[0]
            response_text = item.get("text") or ""
            response_reasoning = item.get("reasoning_content") or ""
        else:
            response_text = str(result)
            response_reasoning = ""

        doc_id = compute_doc_id(prompt, args.model, i)
        logger.info(
            f"  doc {i+1}/{args.n_docs} (id={doc_id}) "
            f"chars={len(response_text)} reasoning={len(response_reasoning)} "
            f"latency={latency_ms}ms"
        )

        # HX persist
        record = ExchangeRecord(
            provider=provider,
            request_messages=[{"role": "user", "content": prompt}],
            request_model=args.model,
            request_params={"max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "seed": args.seed + i},
            response_content=response_text,
            response_model=args.model.split("/")[-1],
            response_reasoning=response_reasoning,
            latency_ms=latency_ms,
            source_context={
                "aegir_module": "experiment_loose_grok",
                "doc_id": doc_id,
                "seed_topic": str(args.seed_topic),
                "experiment": "loose_grok_v0",
            },
        )
        append_exchange(record)

        doc_rows.append({
            "doc_id": doc_id,
            "idx": i,
            "model": args.model,
            "response_text": response_text,
            "response_reasoning": response_reasoning,
            "response_chars": len(response_text),
            "latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc),
        })

        # Streaming flush every 20 docs as cheap insurance against process death
        if (i + 1) % 20 == 0:
            _flush_docs(doc_rows, out_dir / "docs.parquet")

    docs_path = out_dir / "docs.parquet"
    _flush_docs(doc_rows, docs_path)
    logger.info(f"DONE — wrote {len(doc_rows)} docs to {docs_path}")
    return docs_path


def _flush_docs(rows: list[dict], path: Path) -> None:
    schema = pa.schema([
        ("doc_id", pa.string()),
        ("idx", pa.int32()),
        ("model", pa.string()),
        ("response_text", pa.string()),
        ("response_reasoning", pa.string()),
        ("response_chars", pa.int32()),
        ("latency_ms", pa.int32()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path, compression="zstd",
    )


# ─────────────────────────────────────────────────────────────────────────
# Phase: analyze
# ─────────────────────────────────────────────────────────────────────────

def sample_finepdfs(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Re-sample FinePDFs deterministically. Mirrors the audit's logic."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn",
                       split="train", streaming=True)
    rng = np.random.default_rng(seed)
    ids: list[str] = []
    texts: list[str] = []
    # Reservoir-ish: walk the stream, accept with probability scaled to
    # need. Audit does a similar deterministic sample at 10K; we want
    # ~2K here for the BERTopic reference. Stream the first ~3-4x of
    # what we need and uniformly subsample.
    stream_budget = n * 4
    candidates_ids = []
    candidates_texts = []
    for i, row in enumerate(ds):
        if i >= stream_budget:
            break
        candidates_ids.append(str(row.get("id") or f"doc_{i}"))
        candidates_texts.append((row.get("text") or "")[:8000])
    if len(candidates_ids) > n:
        idx = rng.choice(len(candidates_ids), size=n, replace=False)
        ids = [candidates_ids[i] for i in idx]
        texts = [candidates_texts[i] for i in idx]
    else:
        ids = candidates_ids
        texts = candidates_texts
    return ids, texts


def phase_analyze(args: argparse.Namespace, out_dir: Path) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"analysis device: {device}")

    docs_path = out_dir / "docs.parquet"
    docs = pq.read_table(docs_path).to_pandas()
    logger.info(f"loaded {len(docs)} Grok docs")

    audit_run = Path(args.audit_run)
    topic_cov = pq.read_table(audit_run / "topic_coverage.parquet").to_pandas()
    topic_cov = topic_cov.sort_values("topic_id").reset_index(drop=True)
    logger.info(f"loaded {len(topic_cov)} audit topic centroids")

    # Re-sample FinePDFs for reference
    logger.info(f"re-sampling {args.finepdfs_resample} FinePDFs docs "
                f"(seed={args.audit_finepdfs_seed})...")
    _fp_ids, fp_texts = sample_finepdfs(args.finepdfs_resample,
                                         args.audit_finepdfs_seed)
    logger.info(f"got {len(fp_texts)} FinePDFs docs")

    # Embed
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device=device)

    grok_texts = docs["response_text"].fillna("").apply(
        lambda s: s[:8000]
    ).tolist()
    logger.info("embedding...")
    grok_embs = embedder.encode(grok_texts, normalize_embeddings=True,
                                 convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
    fp_embs = embedder.encode(fp_texts, normalize_embeddings=True,
                               convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float32)
    audit_embs = embedder.encode(topic_cov["topic_repr_text"].fillna("").tolist(),
                                  normalize_embeddings=True,
                                  convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)

    # Per-Grok-doc nearest audit topic
    sims = grok_embs @ audit_embs.T
    nearest = sims.argmax(axis=1)
    nearest_sim = sims.max(axis=1)
    topic_ids = topic_cov["topic_id"].to_numpy()

    seed_topic_idx = int(np.where(topic_ids == args.seed_topic)[0][0])
    seed_topic_emb = audit_embs[seed_topic_idx]
    grok_seed_sim = grok_embs @ seed_topic_emb

    # BERTopic on Grok docs alone, with our own embedder pre-computed
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    logger.info("running BERTopic on Grok docs...")
    bertopic_model = BERTopic(
        umap_model=UMAP(n_components=5, random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=5),
        calculate_probabilities=False,
        verbose=False,
    )
    grok_topics, _ = bertopic_model.fit_transform(grok_texts, embeddings=grok_embs)
    grok_topic_info = bertopic_model.get_topic_info()
    logger.info(f"BERTopic found {len(grok_topic_info)} clusters in Grok docs")

    # For each Grok BERTopic cluster, find its nearest audit topic
    cluster_to_audit: list[dict] = []
    for tid in sorted(set(grok_topics)):
        if tid == -1:
            continue  # noise
        mask = np.array(grok_topics) == tid
        centroid = grok_embs[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        cluster_sims = audit_embs @ centroid
        nearest_audit = int(np.argmax(cluster_sims))
        cluster_to_audit.append({
            "grok_cluster_id": int(tid),
            "n_docs": int(mask.sum()),
            "nearest_audit_topic": int(topic_ids[nearest_audit]),
            "nearest_audit_sim": float(cluster_sims[nearest_audit]),
        })

    # Mixed-cluster analysis: cluster the union of Grok + FinePDFs
    logger.info("running BERTopic on union (Grok ∪ FinePDFs)...")
    union_texts = grok_texts + fp_texts
    union_embs = np.vstack([grok_embs, fp_embs])
    union_labels = ["grok"] * len(grok_texts) + ["finepdfs"] * len(fp_texts)
    union_model = BERTopic(
        umap_model=UMAP(n_components=5, random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=10),
        calculate_probabilities=False,
        verbose=False,
    )
    union_topics, _ = union_model.fit_transform(union_texts, embeddings=union_embs)

    mixed_clusters: list[dict] = []
    for tid in sorted(set(union_topics)):
        if tid == -1:
            continue
        mask = np.array(union_topics) == tid
        n_grok = sum(1 for j in range(len(union_labels))
                     if mask[j] and union_labels[j] == "grok")
        n_fp = sum(1 for j in range(len(union_labels))
                   if mask[j] and union_labels[j] == "finepdfs")
        if n_grok + n_fp == 0:
            continue
        mixed_clusters.append({
            "union_cluster_id": int(tid),
            "n_grok": n_grok,
            "n_finepdfs": n_fp,
            "grok_frac": n_grok / (n_grok + n_fp),
            "total": n_grok + n_fp,
        })
    mixed_clusters.sort(key=lambda c: -c["n_grok"])

    # Summary statistics
    n_grok_in_seed = int((nearest == seed_topic_idx).sum())
    n_grok_in_seed_top5 = int(
        sum(1 for i in range(len(grok_embs))
            if np.argsort(-sims[i])[:5].tolist().count(seed_topic_idx) > 0)
    )
    # Pure-Grok clusters in union: clusters where grok_frac > 0.9
    pure_grok_clusters = [c for c in mixed_clusters if c["grok_frac"] > 0.9]
    mixed_real = [c for c in mixed_clusters if 0.1 <= c["grok_frac"] <= 0.9]

    summary = {
        "n_grok_docs": int(len(grok_texts)),
        "n_finepdfs_sample": int(len(fp_texts)),
        "seed_topic": int(args.seed_topic),
        "n_grok_nearest_seed_topic": n_grok_in_seed,
        "frac_grok_nearest_seed_topic": float(n_grok_in_seed / len(grok_texts)),
        "n_grok_seed_topic_in_top5": n_grok_in_seed_top5,
        "frac_grok_seed_in_top5": float(n_grok_in_seed_top5 / len(grok_texts)),
        "mean_grok_seed_sim": float(grok_seed_sim.mean()),
        "mean_grok_nearest_sim": float(nearest_sim.mean()),
        "n_grok_bertopic_clusters": len(grok_topic_info),
        "n_union_bertopic_clusters": int(len(set(union_topics)) - (1 if -1 in union_topics else 0)),
        "n_pure_grok_clusters": len(pure_grok_clusters),
        "n_mixed_clusters_0.1_0.9": len(mixed_real),
        "cluster_to_audit": cluster_to_audit,
        "mixed_clusters_top10": mixed_clusters[:10],
        "audit_topic_distribution": {
            int(topic_ids[t]): int((nearest == t).sum())
            for t in range(len(topic_ids))
            if (nearest == t).sum() > 0
        },
    }

    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"analysis written to {out_dir / 'analysis.json'}")
    return summary


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("generate", "all"):
        phase_generate(args, out_dir)

    if args.phase in ("analyze", "all"):
        summary = phase_analyze(args, out_dir)

        print()
        print("=== Loose-Grok experiment summary ===")
        print(f"  seed topic:                          {args.seed_topic}")
        print(f"  Grok docs generated:                 {summary['n_grok_docs']}")
        print(f"  FinePDFs reference docs:             {summary['n_finepdfs_sample']}")
        print()
        print(f"  Grok docs nearest-neighbor on seed:  "
              f"{summary['n_grok_nearest_seed_topic']}/{summary['n_grok_docs']} "
              f"({summary['frac_grok_nearest_seed_topic']:.0%})")
        print(f"  Grok docs with seed in top-5:        "
              f"{summary['n_grok_seed_topic_in_top5']}/{summary['n_grok_docs']} "
              f"({summary['frac_grok_seed_in_top5']:.0%})")
        print(f"  Mean Grok→seed similarity:           "
              f"{summary['mean_grok_seed_sim']:.3f}")
        print(f"  Mean Grok→nearest-topic similarity:  "
              f"{summary['mean_grok_nearest_sim']:.3f}")
        print()
        print(f"  BERTopic clusters in Grok corpus:    {summary['n_grok_bertopic_clusters']}")
        print(f"  BERTopic clusters in union:          {summary['n_union_bertopic_clusters']}")
        print(f"  Pure-Grok clusters (>90% Grok):      {summary['n_pure_grok_clusters']}")
        print(f"  Mixed clusters (10-90% Grok):        {summary['n_mixed_clusters_0.1_0.9']}")
        print()
        print(f"  Audit topic distribution (top 8):")
        dist = sorted(summary["audit_topic_distribution"].items(),
                      key=lambda kv: -kv[1])[:8]
        for tid, n in dist:
            marker = "← seed" if tid == args.seed_topic else ""
            print(f"    audit topic {tid:3d}: {n:3d} docs {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
