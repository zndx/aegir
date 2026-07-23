#!/usr/bin/env python
"""probe_relational_admission — does item admission have TRACTION on natively
relational data? (RH 2026-07-23 — the first probe of the relational-sources arc.)

FinePDFs items are token-windows of prose; relational data needs a CUSTOM
TABLE-STRUCTURE WINDOWING stage producing meaningful DATA-ELEMENT candidate items:

  SchemaPile (schema side; PERMISSIVE-licensed rows only, clean-room doctrine):
    · one item per TABLE — name + columns (type, key markers) + per-column sample
      values, verbalized, capped at the 512-token item grain.
  GitTables (data side; lineage-retained corpus already feeding the value pools):
    · one HEADER item per table — table stem + column headers;
    · one COLUMN item per column — header + K distinct sampled values
      (the CTA data-element grain).

Every item scores against the LIVE aperture with the armed-filter dual view (one
top-10 MaxSim query; genus admission at τ; root_strong = would-have-root-admitted →
the review-lane analogue). Everything scratch-side; nothing enters the store.

    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=2 \\
      uv run python scripts/probe_relational_admission.py --schemas 60 --tables 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SCHEMAPILE = Path("/raid/datasets/schemapile/schemapile_full.parquet")
GITTABLES = Path("/raid/datasets/gittables")
TAU = 0.10                  # the admission rule of record
TOKEN_BUDGET = 500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemas", type=int, default=60, help="SchemaPile databases sampled")
    ap.add_argument("--tables", type=int, default=60, help="GitTables files sampled")
    ap.add_argument("--seed", type=int, default=0xA119)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from aegir.ontology import domain_index as DI
    from aegir.ontology.colbert_encoder import get_encoder
    enc = get_encoder()
    client = DI._client(DI.DEFAULT_QDRANT_URL)
    excl = set(DI.armed_admission_filter().get("exclude_codes") or [])

    def cap(text: str) -> str:
        ids = enc._tokenizer(text)["input_ids"]
        if len(ids) <= TOKEN_BUDGET:
            return text
        return enc._tokenizer.decode(ids[1:TOKEN_BUDGET - 1])

    items: "list[dict]" = []

    # ── SchemaPile: per-TABLE schema items (PERMISSIVE only) ──
    import pyarrow.parquet as pq
    sp = pq.read_table(SCHEMAPILE, columns=["FILENAME", "PERMISSIVE", "TABLES"]).to_pylist()
    perm = [r for r in sp if r.get("PERMISSIVE")]
    rng.shuffle(perm)
    n_db = 0
    for r in perm:
        if n_db >= a.schemas:
            break
        tables = r.get("TABLES") or []
        if not tables:
            continue
        n_db += 1
        db = str(r.get("FILENAME") or "db").rsplit(".", 1)[0]
        for t in tables[:8]:
            cols = t.get("COLUMNS") or []
            parts = []
            for c in cols[:24]:
                mark = " primary key" if c.get("IS_PRIMARY") else ""
                parts.append(f"{c.get('NAME')} ({c.get('TYPE')}{mark})")
            vals = []
            for c in cols[:8]:
                vv = [str(x) for x in (c.get("VALUES") or [])[:4] if x is not None]
                if vv:
                    vals.append(f"{c.get('NAME')}: {', '.join(vv)}")
            text = (f"Database {db}. Table {t.get('TABLE_NAME')}. "
                    f"Columns: {'; '.join(parts)}."
                    + (f" Sample values — {'; '.join(vals)}." if vals else ""))
            items.append({"source": "schemapile", "kind": "schema_table",
                          "ref": f"{db}.{t.get('TABLE_NAME')}", "text": cap(text)})

    # ── GitTables: header + per-COLUMN data-element items ──
    files = []
    for i, p in enumerate(GITTABLES.iterdir()):
        if p.suffix == ".parquet":
            files.append(p)
        if i > 250_000 and len(files) > 5_000:
            break
    rng.shuffle(files)
    n_t = 0
    for p in files:
        if n_t >= a.tables:
            break
        try:
            import pandas as pd
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001 — GitTables has malformed parquet; skip
            continue
        if df.shape[1] < 2 or df.shape[0] < 3:
            continue
        n_t += 1
        stem = p.stem[:60]
        heads = [str(c) for c in df.columns[:24]]
        items.append({"source": "gittables", "kind": "table_header",
                      "ref": stem, "text": cap(f"Table {stem}. Columns: {', '.join(heads)}.")})
        for col in df.columns[:8]:
            try:
                vals = [str(v)[:48] for v in df[col].dropna().unique()[:12]]
            except Exception:  # noqa: BLE001
                continue
            if not vals:
                continue
            items.append({"source": "gittables", "kind": "column",
                          "ref": f"{stem}.{col}",
                          "text": cap(f"Column {col} of table {stem}. "
                                      f"Values: {', '.join(vals)}.")})

    print(f"windowed items: {len(items)} "
          f"({Counter((i['source'], i['kind']) for i in items).most_common()})")

    # ── score: one top-10 MaxSim per item, dual view ──
    def view(hits: "list[dict]") -> "tuple[dict, float]":
        s = [h["score"] for h in hits[:5]]
        rel = ((s[0] - s[1]) / s[0]) if len(s) > 1 and s[0] else 1.0
        return (hits[0] if hits else {}), rel

    B = 64
    for start in range(0, len(items), B):
        batch = items[start:start + B]
        vecs = enc.encode([it["text"] for it in batch])
        for it, v in zip(batch, vecs):
            res = client.query_points(collection_name=DI.DEFAULT_APERTURE,
                                      query=v.tolist(), limit=10,
                                      with_payload=True).points
            hits = [{"score": float(pt.score), **(pt.payload or {})} for pt in res]
            base_top, base_margin = view(hits)
            genus_top, genus_margin = view([h for h in hits
                                            if str(h.get("code", "")) not in excl])
            it.update({
                "code": str(genus_top.get("code", "")),
                "label": str(genus_top.get("pref_label", "")),
                "genus_margin": round(genus_margin, 4),
                "admitted": bool(genus_top) and genus_margin >= TAU,
                "root_strong": (str(base_top.get("code", "")) in excl
                                and base_margin >= TAU)})
        print(f"  … scored {min(start + B, len(items))}/{len(items)}", flush=True)

    # ── aggregate ──
    out: dict = {"registered": {"tau": TAU, "token_budget": TOKEN_BUDGET,
                                "seed": a.seed},
                 "n_items": len(items), "groups": {}}
    for (src, kind), _ in Counter((i["source"], i["kind"]) for i in items).most_common():
        g = [i for i in items if i["source"] == src and i["kind"] == kind]
        adm = [i for i in g if i["admitted"]]
        out["groups"][f"{src}/{kind}"] = {
            "n": len(g), "admitted": len(adm),
            "admission_rate": round(len(adm) / max(1, len(g)), 4),
            "mean_genus_margin_of_admits": round(
                sum(i["genus_margin"] for i in adm) / max(1, len(adm)), 4),
            "root_strong": sum(i["root_strong"] for i in g),
            "top_anchors": dict(Counter(i["code"] for i in adm).most_common(6)),
            "sample_admits": [{"ref": i["ref"], "code": i["code"],
                               "margin": i["genus_margin"]} for i in adm[:5]]}
    out["items"] = [{k: v for k, v in i.items() if k != "text"} for i in items]
    Path(REPO / "build/relational_admission_probe.json").write_text(
        json.dumps(out, indent=1))
    for k, g in out["groups"].items():
        print(f"{k:26s} n={g['n']:4d} · admit {g['admission_rate']:.1%} · "
              f"margin {g['mean_genus_margin_of_admits']:.3f} · "
              f"root-strong {g['root_strong']} · top {list(g['top_anchors'])[:3]}")
    print("→ build/relational_admission_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
