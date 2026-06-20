#!/usr/bin/env python
"""inc-2d-full / #43 — realize a generated corpus as CPA via HermiT (the discriminating relational eval).

Per chapter: lower its cited templates to a Domain/Range TBox (``abox.derive_domain_range_tbox``),
lift the chapter's verifiable-JSON tables to an ABox (FK relations asserted, types NOT), run **HermiT
realization**, and map the reasoner-computed subject types back to template ids — the **CPA prediction**,
computed by a sound-&-complete oracle rather than learned. The relational/type skill that floored at
tiny scale (descoped G-rel) is re-homed to the reasoner.

``--control``:
  * ``full``           — Domain/Range TBox: realization recovers a row's template from its relations.
  * ``no-domain-range`` — the **matched-token control**: identical classes+properties, NO Domain/Range,
    so nothing is inferrable from the relations. ``score_realization_cpa.py`` reports the selectivity
    gap (full − control) = the schema's isolated contribution, CI-clean vs the held-out reference.

Run BOTH controls, then score:
  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/realize_corpus_as_cpa.py \
    --chapters-run <sdg_corpus_v0_3/run> --control full           --out evidence/realization_cpa/pred_full.parquet
  ... --control no-domain-range --out .../pred_control.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import abox  # noqa: E402
from aegir.ontology.ddl import template_to_table  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

_JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.S)


def extract_tables(response_text: str) -> dict[str, list]:
    """Chapter verifiable-JSON → {table_name: rows}. Last fence wins (the authoritative footer)."""
    out: dict[str, list] = {}
    for m in _JSON_FENCE.finditer(response_text or ""):
        try:
            for tb in json.loads(m.group(1)).get("tables", []):
                if tb.get("name") and isinstance(tb.get("rows"), list):
                    out[tb["name"]] = tb["rows"]
        except Exception:  # noqa: BLE001
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chapters-run", required=True, help="dir with chapters.parquet")
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--control", choices=["full", "no-domain-range"], default="full")
    ap.add_argument("--reference", default=None,
                    help="reference.parquet — restrict to its chapter set (the eval split)")
    ap.add_argument("--max-chapters", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat = load_catalog(Path(args.catalog))
    by_id = {t.template_id: t for t in cat.templates}
    tindex = {abox._san(t.template_id): i for i, t in enumerate(cat.templates)}

    chs = pq.read_table(Path(args.chapters_run) / "chapters.parquet",
                        columns=["chapter_id", "template_ids", "response_text"]).to_pylist()
    if args.reference:
        keep = {r["chapter_id"] for r in
                pq.read_table(Path(args.reference), columns=["chapter_id"]).to_pylist()}
        chs = [c for c in chs if c["chapter_id"] in keep]
    if args.max_chapters:
        chs = chs[: args.max_chapters]

    with_dr = args.control == "full"
    rows_out = []
    for n, ch in enumerate(chs, 1):
        cited = [by_id[t] for t in (ch["template_ids"] or []) if t in by_id]
        if not cited:
            continue
        cited_idx = sorted({tindex[abox._san(t.template_id)] for t in cited})
        tables = extract_tables(ch["response_text"])
        onto, path = abox.load_ontology(abox.derive_domain_range_tbox(cited, with_domain_range=with_dr))
        subj_iris: list[str] = []
        for name, trows in tables.items():
            tid = name[2:] if name.startswith("t_") else name
            t = by_id.get(tid)
            if t is None:
                continue
            st = template_to_table(t, "")
            subj_iris += abox.rows_to_individuals(onto, st, trows, ch["chapter_id"], t)
        pred_idx: list[int] = []
        if subj_iris:
            abox.refresh(onto)
            inferred = abox.realize_individuals(onto, subj_iris, direct=False)
            all_iris = [iri for v in inferred.values() for iri in v]
            pred_idx = abox.iris_to_label_idx(all_iris, tindex)
        Path(path).unlink(missing_ok=True)
        rows_out.append({"chapter_id": ch["chapter_id"], "control": args.control,
                         "pred_idx": pred_idx, "cited_idx": cited_idx, "n_subjects": len(subj_iris)})
        if n % 10 == 0:
            print(f"  {n}/{len(chs)} chapters realized ({args.control})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_out, schema=pa.schema([
        ("chapter_id", pa.string()), ("control", pa.string()),
        ("pred_idx", pa.list_(pa.int32())), ("cited_idx", pa.list_(pa.int32())),
        ("n_subjects", pa.int32())])), args.out)
    n_pred = sum(1 for r in rows_out if r["pred_idx"])
    tot_labels = sum(len(r["pred_idx"]) for r in rows_out)
    print(f"DONE [{args.control}] {len(rows_out)} chapters, {n_pred} with ≥1 recovered template, "
          f"{tot_labels} labels total → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
