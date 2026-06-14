#!/usr/bin/env python
"""E6 Channel A (EVIDENCE.md): pipeline topic-coherence trace.

For each v0.3 chapter, build the NL shadow of each pipeline stage and measure
topic transfer between stages in ONE frozen space — the canonical ground
(`T_I_canonical.pkl`, mpnet/k=200). Every stage is *assigned* into that space
(never refit): a stage's topic signature is its cosine to the 200 canonical
centroids; transfer between two stages = cosine of their signatures.

  S0  input FinePDFs doc   (coverage_v0 topic_repr_text, keyed by target_topic)
  S1  ontology shadow      (cited templates' verbal_templates, slots stripped)
  S3  chapter prose        (response_text)

(S2, the view-verbalization, is logically required but materially absent in
v0.3 — no views were generated; it is a separate synthesis step, not built here.)

Decision rule (pre-registered): S1->S3 transfer beats a shuffled-pairing null
CI-clean  =>  the ontology is on the causal path. S0->S3 passing while S1->S3
sits at the null  =>  the style-anchor bypass, quantified. Per-stage transfers
are the candidate re-derived within-model proxy E1 demands.

S1 uses the REGENERATED (anchor-bearing) verbal_templates — the ontology's true
topical shadow — not the vacuous strings the v0.3 prompts happened to carry.
Deterministic except the null's RNG (seeded).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.topic_alignment import (  # noqa: E402
    get_encoder, encode_sentences, load_topic_model,
)

RUNS = ["811408b392859708", "d7646714bdd5e16f"]
CORPUS_DIR = "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3"
COVERAGE_V0 = "/raid/checkpoints/aegir-artifacts/coverage_v0/232ea5460ce6e0bf/topic_coverage.parquet"
SLOT_RE = re.compile(r"\{([^}]+)\}")


def strip_slots(verbal: str) -> str:
    """Slots carry no topic (generic X/Y); keep the content terms (anchor noun,
    property phrases) by replacing {slot} with its de-underscored name."""
    return SLOT_RE.sub(lambda m: m.group(1).replace("_", " "), verbal or "")


def doc_embed(text: str, encoder, max_windows: int = 12, words_per_window: int = 180) -> np.ndarray | None:
    """Chunk into word windows, encode each (mpnet truncates at 256 tok), mean,
    L2-normalize -> one document embedding. Empty text -> zero vector."""
    words = (text or "").split()
    if not words:
        return None
    windows = [" ".join(words[i:i + words_per_window])
               for i in range(0, len(words), words_per_window)][:max_windows]
    embs = encode_sentences(windows, encoder=encoder)
    v = embs.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def signatures(embs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each doc embedding into the canonical ground: cosine to every
    centroid (both L2-normalized -> dot product). Rows L2-normalized so a
    later cosine compares topic *shape*, not magnitude."""
    sig = embs @ centroids.T                       # (n, k)
    norms = np.linalg.norm(sig, axis=1, keepdims=True)
    return sig / np.clip(norms, 1e-9, None)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1)


def boot_ci(x: np.ndarray, rng, n=2000, lo=2.5, hi=97.5) -> tuple[float, float, float]:
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def null_transfer(sa: np.ndarray, sb: np.ndarray, rng, n_perm=25) -> np.ndarray:
    """Shuffled-pairing null: chapter i's stage-A signature vs chapter j's
    stage-B signature (j != i), averaged over n_perm derangement-ish perms."""
    m = len(sa)
    acc = np.zeros(m)
    for _ in range(n_perm):
        perm = rng.permutation(m)
        # avoid accidental identity pairings
        fix = perm == np.arange(m)
        if fix.any():
            perm[fix] = (perm[fix] + 1) % m
        acc += row_cos(sa, sb[perm])
    return acc / n_perm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ti-cache", default=str(REPO / "src/aegir/ontology/catalog/T_I_canonical.pkl"))
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/combined.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap chapters (0=all)")
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--out", default="/raid/checkpoints/aegir-artifacts/evidence/e6a/trace.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import pyarrow.parquet as pq
    t_i = load_topic_model(args.ti_cache)
    centroids = t_i.centroids
    print(f"canonical ground: {t_i.encoder_model}  k={len(centroids)}")
    encoder = get_encoder(t_i.encoder_model)

    verbal = {t.template_id: t.verbal_template for t in load_catalog(args.catalog).templates}
    repr_text = {r["topic_id"]: r["topic_repr_text"]
                 for r in pq.read_table(COVERAGE_V0).to_pylist()}

    chapters = []
    for run in RUNS:
        chapters.extend(pq.read_table(f"{CORPUS_DIR}/{run}/chapters.parquet").to_pylist())
    if args.limit:
        chapters = chapters[:args.limit]
    print(f"chapters: {len(chapters)}")

    # build stage texts; keep only chapters with all three stages non-empty
    rows = []
    for c in chapters:
        s0 = repr_text.get(c["target_topic_id"], "")
        s1 = " ".join(strip_slots(verbal.get(tid, "")) for tid in (c["template_ids"] or []))
        s3 = c["response_text"] or ""
        if s0.strip() and s1.strip() and s3.strip():
            rows.append({"chapter_id": c["chapter_id"], "model": c["model"],
                         "topic": c["target_topic_id"], "s0": s0, "s1": s1, "s3": s3})
    print(f"usable (all stages present): {len(rows)}")

    # encode each stage -> doc embeddings -> canonical signatures
    def stage_sig(key: str) -> tuple[np.ndarray, list[int]]:
        embs, keep = [], []
        for i, r in enumerate(rows):
            e = doc_embed(r[key], encoder)
            if e is not None:
                embs.append(e); keep.append(i)
        return np.array(embs), keep

    print("encoding stages (S0, S1, S3)...")
    e0, k0 = stage_sig("s0"); e1, k1 = stage_sig("s1"); e3, k3 = stage_sig("s3")
    keep = sorted(set(k0) & set(k1) & set(k3))
    idx = {kk: ii for ii, kk in enumerate(k0)}
    i1 = {kk: ii for ii, kk in enumerate(k1)}
    i3 = {kk: ii for ii, kk in enumerate(k3)}
    S0 = signatures(np.array([e0[idx[k]] for k in keep]), centroids)
    S1 = signatures(np.array([e1[i1[k]] for k in keep]), centroids)
    S3 = signatures(np.array([e3[i3[k]] for k in keep]), centroids)
    models = np.array([rows[k]["model"] for k in keep])
    print(f"encoded {len(keep)} complete chapters")

    def report(name: str, sa: np.ndarray, sb: np.ndarray, mask=None) -> dict:
        if mask is not None:
            sa, sb = sa[mask], sb[mask]
        obs = row_cos(sa, sb)
        nul = null_transfer(sa, sb, rng)
        o_m, o_lo, o_hi = boot_ci(obs, rng)
        n_m, n_lo, n_hi = boot_ci(nul, rng)
        diff = obs - nul
        d_m, d_lo, d_hi = boot_ci(diff, rng)
        clean = d_lo > 0
        print(f"  {name:10s} obs={o_m:.3f}[{o_lo:.3f},{o_hi:.3f}]  "
              f"null={n_m:.3f}[{n_lo:.3f},{n_hi:.3f}]  "
              f"Δ={d_m:+.3f}[{d_lo:+.3f},{d_hi:+.3f}]  {'BEATS-NULL' if clean else 'at-null'}")
        return {"name": name, "n": int(len(sa)), "obs": o_m, "obs_ci": [o_lo, o_hi],
                "null": n_m, "null_ci": [n_lo, n_hi], "delta": d_m, "delta_ci": [d_lo, d_hi],
                "beats_null": bool(clean)}

    out = {"ground": t_i.encoder_model, "k": int(len(centroids)),
           "n_chapters": int(len(keep)), "transfers": {}}
    print("\n=== ALL (n={}) ===".format(len(keep)))
    for nm, a, b in [("S0->S1", S0, S1), ("S1->S3", S1, S3), ("S0->S3", S0, S3), ("S0->S1*", S0, S1)]:
        if nm == "S0->S1*":
            continue
        out["transfers"][nm] = report(nm, a, b)
    for mdl in sorted(set(models)):
        mask = models == mdl
        print(f"\n=== {mdl} (n={int(mask.sum())}) ===")
        for nm, a, b in [("S0->S1", S0, S1), ("S1->S3", S1, S3), ("S0->S3", S0, S3)]:
            out["transfers"][f"{mdl}:{nm}"] = report(nm, a, b, mask)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    # per-chapter transfers (for the E1 re-derivation proxy candidate)
    s0s1 = row_cos(S0, S1); s1s3 = row_cos(S1, S3); s0s3 = row_cos(S0, S3)
    per = [{"chapter_id": rows[k]["chapter_id"], "model": rows[k]["model"],
            "topic": rows[k]["topic"], "s0s1": float(s0s1[i]),
            "s1s3": float(s1s3[i]), "s0s3": float(s0s3[i])}
           for i, k in enumerate(keep)]
    per_path = Path(args.out).with_name("per_chapter.json")
    per_path.write_text(json.dumps(per) + "\n")
    print(f"wrote {per_path} ({len(per)} chapters)")
    # verdict
    s1s3 = out["transfers"]["S1->S3"]
    s0s3 = out["transfers"]["S0->S3"]
    print("\n=== VERDICT ===")
    print(f"  S1->S3 beats null: {s1s3['beats_null']} (Δ={s1s3['delta']:+.3f})  "
          f"=> ontology {'ON-PATH' if s1s3['beats_null'] else 'NOT shown on-path'}")
    print(f"  S0->S3 beats null: {s0s3['beats_null']} (Δ={s0s3['delta']:+.3f})")
    if s0s3["beats_null"] and not s1s3["beats_null"]:
        print("  => STYLE-ANCHOR BYPASS: input transfers but ontology does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
