#!/usr/bin/env python
"""Build the shared ontology-CPA evaluation task from generated chapters.

This is the *downstream-validation* stage of the v0.3 procedure. It turns a
set of ontology-grounded chapters into a fixed, shared column-property-
annotation task — *which ontology templates (typed-slot axioms) does this
chapter instantiate?* — phrased as multi-label classification over the
template-id vocabulary.

Why this is the calibration instrument
--------------------------------------
The convergence loop drives the ontology on PROXY signals (verifier R,
topic-recovery, the family complex). This task is the GROUND-TRUTH signal:
a model pretrained on one ablation arm's corpus is later fine-tuned + scored
on *this* shared task (see ``eval_ontology_cpa.py``), so the arms differ
only in their pretraining. Comparing arms answers "does ontology / schema
grounding in the corpus produce a model that better recognizes ontological
structure?" — and whether that downstream ranking matches the proxy ranking
(``verification.parquet`` r_composite). The proxy half is free; this builds
the half that has never been measured.

v1 granularity: chapter-level multi-label over ``template_ids`` (uses the
column directly; no column->slot alignment needed). Refinable to column-level
slot-type prediction once generation emits column->slot provenance — the
catalog ``slot_types`` are recorded here for that next step.

Labels come from each chapter's ``template_ids``; the vocabulary is the set
of templates that actually appear (floored by ``--min-label-count``). The
train/val/test split is a deterministic hash of ``chapter_id`` so it is
identical for every arm and stable across reruns.

Usage::

    uv run --no-sync python scripts/build_ontology_cpa_eval.py \\
        --chapters /raid/checkpoints/aegir-artifacts/ablation_v1/6e6901e291ef1f87/chapters.parquet \\
        --out-dir  /raid/checkpoints/aegir-artifacts/ontology_cpa_v0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("build-ontology-cpa")


def _as_list(x) -> list[str]:
    """Coerce a parquet list-cell (list / numpy array / scalar / NA) to list[str]."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return [str(t) for t in x.tolist()]
        if isinstance(x, float) and np.isnan(x):
            return []
    except Exception:
        pass
    return [str(x)]


def _as_text(x) -> str:
    if x is None:
        return ""
    try:
        import numpy as np
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def _split_of(chapter_id: str, val_frac: float, test_frac: float) -> str:
    """Deterministic per-chapter split — identical for every arm, stable across reruns."""
    h = int(hashlib.sha256(chapter_id.encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--chapters", nargs="+", required=True,
                   help="One or more chapters.parquet (the labeled source; "
                        "default usage: the full-ontology arm, which carries the "
                        "most faithful template grounding).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--min-label-count", type=int, default=2,
                   help="Drop template_ids appearing in fewer than this many "
                        "chapters (rare labels make F1 noisy/degenerate).")
    p.add_argument("--max-chapters", type=int, default=None,
                   help="Cap chapters loaded (for fast smoke builds).")
    p.add_argument("--catalog",
                   default=str(REPO / "src/aegir/ontology/catalog/combined.json"),
                   help="Catalog for slot-type metadata (recorded for the "
                        "column-level refinement; not required to build).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    frames = []
    for cp in args.chapters:
        df = pq.read_table(cp).to_pandas()
        frames.append(df)
        log.info("loaded %d chapters from %s", len(df), cp)
    chapters = pd.concat(frames, ignore_index=True)
    if args.max_chapters:
        chapters = chapters.head(args.max_chapters)

    rows: list[tuple[str, str, list[str]]] = []
    label_counts: dict[str, int] = {}
    for _, ch in chapters.iterrows():
        cid = str(ch["chapter_id"])
        text = _as_text(ch.get("response_text"))
        tids = _as_list(ch.get("template_ids"))
        if not text or not tids:
            continue
        rows.append((cid, text, tids))
        for t in set(tids):
            label_counts[t] = label_counts.get(t, 0) + 1

    kept = sorted([t for t, c in label_counts.items() if c >= args.min_label_count])
    vocab = {t: i for i, t in enumerate(kept)}
    log.info("label vocab: %d templates (of %d distinct; floor=%d)",
             len(vocab), len(label_counts), args.min_label_count)
    if not vocab:
        raise SystemExit("empty label vocab — lower --min-label-count or check the source")

    # Slot-type metadata for the eventual column-level refinement (recorded, not used in v1).
    slot_types: dict[str, dict] = {}
    try:
        from aegir.ontology.schema import load_catalog
        cat = load_catalog(args.catalog)
        for t in kept:
            try:
                slot_types[t] = cat.by_id(t).slot_types
            except KeyError:
                slot_types[t] = {}
    except Exception as e:  # catalog is optional metadata
        log.warning("catalog slot-type lookup skipped: %s", e)

    out_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    label_split_support = {"train": set(), "val": set(), "test": set()}
    for cid, text, tids in rows:
        lab = sorted({vocab[t] for t in tids if t in vocab})
        if not lab:
            continue  # only rare/dropped templates
        sp = _split_of(cid, args.val_frac, args.test_frac)
        split_counts[sp] += 1
        label_split_support[sp].update(lab)
        out_rows.append({"chapter_id": cid, "response_text": text,
                         "label_idx": lab, "split": sp})

    log.info("dataset rows: %d  splits=%s", len(out_rows), split_counts)
    log.info("label support per split (#labels seen): %s",
             {s: len(v) for s, v in label_split_support.items()})

    schema = pa.schema([
        ("chapter_id", pa.string()),
        ("response_text", pa.string()),
        ("label_idx", pa.list_(pa.int32())),
        ("split", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema),
                   out_dir / "eval_dataset.parquet", compression="zstd")
    (out_dir / "label_vocab.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "slot_types.json").write_text(json.dumps(slot_types, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({
        "chapters_sources": args.chapters,
        "n_rows": len(out_rows),
        "n_labels": len(vocab),
        "split_counts": split_counts,
        "label_support_per_split": {s: len(v) for s, v in label_split_support.items()},
        "min_label_count": args.min_label_count,
        "val_frac": args.val_frac, "test_frac": args.test_frac,
        "granularity": "chapter-level multi-label over template_ids (v1)",
    }, indent=2))
    log.info("wrote ontology-CPA task to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
