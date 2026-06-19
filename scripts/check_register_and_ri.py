#!/usr/bin/env python
"""Two $0 preconditions for the dual next-phase gate (2026-06-15).

R0 (register hypothesis, G-cov): does embedding the VERBALIZATION-ONLY (vs the
full formal `template_embedding_text` with Manchester + family/slots metadata)
raise template↔topic-centroid similarity? If debloating lifts more topics to
borderline (≥0.35), register was the dominant confound in E5's coverage-close.

RI (referential integrity, G-rel / E2(b)): in the corpus's verifiable JSON, do
FK-shaped cell values actually appear as PK (col-0) values of sibling tables?
E2(b)-cells-only hinges on this value-overlap signal existing.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from aegir.ontology.schema import load_catalog  # noqa: E402
from ontology_coverage_audit import template_embedding_text  # noqa: E402

SLOT = re.compile(r"\{[^}]+\}")
JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.S)


def r0_register(centroids_path: str, tau: float = 0.35) -> None:
    cents = np.load(centroids_path)  # (n_topics, D), unit-norm
    tmpls = []
    for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))):
        if "candidate" in f or "combined" in f:
            continue
        fam = Path(f).stem
        for t in load_catalog(f).templates:
            tmpls.append((t, fam))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    full_txt = [template_embedding_text({**asdict(t), "_family": fam}) for t, fam in tmpls]
    vonly_txt = [SLOT.sub("concept", (t.verbal_template or "")) for t, _ in tmpls]
    full = m.encode(full_txt, normalize_embeddings=True, convert_to_numpy=True)
    vonly = m.encode(vonly_txt, normalize_embeddings=True, convert_to_numpy=True)
    max_full = (cents @ full.T).max(axis=1)
    max_v = (cents @ vonly.T).max(axis=1)
    print("=== R0: register hypothesis (per-topic max template↔centroid sim) ===")
    print(f"  templates={len(tmpls)}  topics={len(cents)}")
    print(f"  FULL  (Manchester+meta): mean_max={max_full.mean():.3f}  "
          f"borderline(≥{tau})={int((max_full>=tau).sum())}  covered(≥0.55)={int((max_full>=0.55).sum())}")
    print(f"  VONLY (verbalization)  : mean_max={max_v.mean():.3f}  "
          f"borderline(≥{tau})={int((max_v>=tau).sum())}  covered(≥0.55)={int((max_v>=0.55).sum())}")
    print(f"  Δ mean_max = {max_v.mean()-max_full.mean():+.3f}  "
          f"Δ borderline = {int((max_v>=tau).sum())-int((max_full>=tau).sum()):+d}")


def ri_check(corpus_runs: list[str], sample: int = 300) -> list:
    chapters = []
    import pyarrow.parquet as pq
    for run in corpus_runs:
        p = Path(run) / "chapters.parquet"
        if p.exists():
            chapters.extend(pq.read_table(p, columns=["response_text"]).to_pylist())
    chapters = chapters[:sample]
    n_multi, n_fk_cols, overlaps = 0, 0, []
    for c in chapters:
        m = JSON_FENCE.search(c["response_text"] or "")
        if not m:
            continue
        try:
            tables = json.loads(m.group(1)).get("tables", [])
        except Exception:
            continue
        if len(tables) < 2:
            continue
        n_multi += 1
        pk_pool: dict[str, set] = {}  # table -> set of col-0 (PK) values
        for t in tables:
            pk_pool[t["name"]] = {r[0] for r in t.get("rows", []) if r}
        all_pks: set = set().union(*pk_pool.values()) if pk_pool else set()
        for t in tables:
            rows = t.get("rows", [])
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            for ci in range(1, ncol):  # skip PK col 0
                vals = [r[ci] for r in rows if len(r) > ci and isinstance(r[ci], str)]
                if not vals:
                    continue
                # fraction of this column's values that are a PK of SOME sibling table
                sib_pks = all_pks - pk_pool[t["name"]]
                frac = sum(1 for v in vals if v in sib_pks) / len(vals)
                if frac >= 0.5:  # FK-shaped column
                    n_fk_cols += 1
                    overlaps.append(frac)
    print("\n=== RI: referential integrity of FK-shaped columns (cells-only signal for E2b) ===")
    print(f"  chapters with ≥2 JSON tables: {n_multi}")
    print(f"  FK-shaped columns found (≥50% values ∈ sibling PKs): {n_fk_cols}")
    if overlaps:
        ov = np.array(overlaps)
        print(f"  among those, value-overlap rate: mean={ov.mean():.3f}  "
              f"median={np.median(ov):.3f}  ==1.0: {int((ov>=0.999).sum())}/{len(ov)}")
        print(f"  => E2(b)-cells-only signal {'PRESENT' if ov.mean()>0.7 else 'WEAK — may need a generator fix'}")
    else:
        print("  => NO FK value-overlap found — E2(b)-cells-only NOT viable as-is (generator fix needed).")
    return overlaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centroids", default="/raid/checkpoints/aegir-artifacts/coverage_v1/043d7dcc185245c8/topic_centroids.npy")
    ap.add_argument("--corpus-runs", nargs="+", default=[
        "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3/811408b392859708",
        "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3/d7646714bdd5e16f"])
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--expect-ri-one", action="store_true",
                    help="Track A regression gate: assert FK value-overlap ≈ 1.0 (RI is true by "
                         "construction for chapters written around the fixed materialized tables).")
    args = ap.parse_args()
    r0_register(args.centroids)
    overlaps = ri_check(args.corpus_runs, args.sample)
    if args.expect_ri_one:
        if not overlaps:
            print("  [expect-ri-one] no FK-shaped columns — cannot confirm RI (not a Track A corpus?)")
            return 2
        mean = float(np.mean(overlaps))
        ok = mean >= 0.99
        print(f"  [expect-ri-one] mean FK overlap {mean:.3f} — "
              f"{'PASS (RI≈1.0 by construction)' if ok else 'FAIL (tables corrupted post-generation?)'}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
