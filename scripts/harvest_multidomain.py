"""harvest_multidomain.py — single-pass multi-domain FinePDFs harvest (GPU classify, route-all).

The per-domain harvest_domain_docs re-skips the stream cursor for EACH domain (the resume cost compounds
across N domains). This scans the stream ONCE, classifies each doc on the GPU against the domains-only
aperture, and routes it to its best content-domain (the top concept's root, if it clears the rel_margin
gate) — so N domains cost one pass, not N. Writes the same build/domain_harvest store the deriver reads.

  LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=4 \
    uv run --no-sync python scripts/harvest_multidomain.py --max-stream 8000 --tau 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from aegir.ontology import domain_index as DI
from aegir.ontology import admission_items as AI

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "build" / "domain_harvest"
# domain rollup = first code segment of the admitted item (9=LIMS … 14=DATAENG,
# 15=FINTECH, 16=BIOTECH) — membership lives in the aperture collection, not here


def _hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", "ignore")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-stream", type=int, default=8000, help="docs to scan this pass")
    ap.add_argument("--tau", type=float, default=0.10, help="min rank1-rank2 rel_margin to keep")
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--dataset", default="HuggingFaceFW/finepdfs")
    ap.add_argument("--config", default="eng_Latn")
    ap.add_argument("--collection", default=DI.DEFAULT_APERTURE)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--stride-tokens", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=16)
    a = ap.parse_args()
    _af = DI.armed_admission_filter()
    _adm_exclude = _af.get("effective_exclude") or set()
    if _adm_exclude:
        print(f"admission filter ARMED: excluding codes {sorted(_adm_exclude)} "
              f"({', '.join((_af.get('excluded_concepts') or {}).values())})")

    (STORE / "docs").mkdir(parents=True, exist_ok=True)
    manifest = STORE / "manifest.jsonl"
    cursor_f = STORE / "cursor.json"
    seen: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                seen.add(json.loads(line)["hash"])
            except (ValueError, KeyError):
                continue
    cursor = 0
    if not a.no_resume and cursor_f.exists():
        cursor = json.loads(cursor_f.read_text()).get("cursor", 0)
    print(f"multidomain harvest · resume cursor={cursor} · stored={len(seen)} · "
          f"max-stream={a.max_stream} · tau={a.tau} · collection={a.collection}")

    from datasets import load_dataset
    ds = load_dataset(a.dataset, a.config, split="train", streaming=True)
    if cursor:
        ds = ds.skip(cursor)

    t0 = time.time()
    scanned = short = dup = n_review = 0
    by_domain: Counter = Counter()
    mf = manifest.open("a")
    imf = (STORE / "manifest_items.jsonl").open("a")
    rvf = (STORE / "aperture_review.jsonl").open("a")
    for row in ds:
        if scanned >= a.max_stream:
            break
        scanned += 1
        text = (row.get("text") or "").strip()
        if len(text) < a.min_chars:
            short += 1
            continue
        h = _hash(text)
        if h in seen:
            dup += 1
            continue
        # ITEM SEMANTICS (RH 2026-07-23): sliding ≤512-token windows, every window
        # seeing the whole aperture; NON-OVERLAPPING admitted items resolved greedily
        # by genus margin; root-preponderance-without-genus-admission → ACP review
        # worklist (ontology extension/refinement proposals accumulate; never dropped).
        scan = AI.item_scan(text, collection=a.collection, tau=a.tau,
                            exclude_codes=_adm_exclude or None,
                            stride=a.stride_tokens, max_windows=a.max_windows)
        kept = [i for i in scan["items"] if i["admitted"]]
        if kept:
            best = max(kept, key=lambda i: i["genus_margin"])
            root = str(best["code"]).split(".")[0]
            (STORE / "docs" / f"{h}.txt").write_text(text, encoding="utf-8")
            mf.write(json.dumps({"hash": h, "domain": root, "code": best["code"],
                                 "label": best.get("label", ""),
                                 "rel_margin": best["genus_margin"],
                                 "chars": len(text), "window_mode": "items",
                                 "n_windows": len(scan["candidates"]),
                                 "n_admitted_items": len(kept),
                                 "dataset": f"{a.dataset}/{a.config}"}) + "\n")
            mf.flush()
            imf.write(json.dumps({"hash": h, "items": scan["items"]}) + "\n")
            imf.flush()
            seen.add(h)
            by_domain[root] += 1
        elif scan["review"]:
            rvf.write(json.dumps({"hash": h, "chars": len(text),
                                  "head": text[:280], **scan["review"]}) + "\n")
            rvf.flush()
            n_review += 1
        if scanned % 1000 == 0:
            print(f"  …scanned {scanned} · matched {sum(by_domain.values())} · {scanned / max(time.time() - t0, 1):.0f} docs/s",
                  flush=True)
    mf.close()
    imf.close()
    rvf.close()
    cursor_f.write_text(json.dumps({"cursor": cursor + scanned, "dataset": a.dataset, "config": a.config}, indent=1))
    dt = time.time() - t0
    print(f"HARVEST: scanned {scanned} · short {short} · dup {dup} · matched {sum(by_domain.values())} "
          f"[{dt:.0f}s, {scanned / max(dt, 1):.0f} docs/s]")
    print(f"  by domain (root): {dict(by_domain)}  (9=LIMS 10=MFG 11=ENERGY 12=CSG 13=UTILITY 14=DATAENG 15=FINTECH 16=BIOTECH)")
    print(f"  ACP review worklist: +{n_review} root-preponderance docs → aperture_review.jsonl")
    print(f"  store: {len(seen)} unique in-domain docs total · cursor → {cursor + scanned}")
    import os
    os._exit(0)  # avoid the datasets/grpc teardown SIGABRT (exit 134)


if __name__ == "__main__":
    main()
