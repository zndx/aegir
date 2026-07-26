#!/usr/bin/env python
"""Harvest in-domain documents from the FinePDFs stream — the continuous aperture.

Stream UNSEEN FinePDFs docs from the HF hub (``streaming=True``), discard those that don't match our domain
specification (the ColBERT/Qdrant SKOS classifier — see ``domain_index``), and keep the matches in a
**content-addressable** store so repeated sample runs over time never duplicate a document. This is the
front of the content-first pipeline: harvested in-domain docs feed ``derive_ontology`` (which then grounds
its content-extracted primitives in the seeded SKOS subtree).

Dedup is structural: each kept doc is written to ``docs/<sha256>.txt`` (identical content → identical path →
idempotent), and a manifest records metadata. A persistent stream cursor (``ds.skip(cursor)``) advances
through unseen docs across runs; the content hash is the belt-and-suspenders guarantee against near-stream
duplicates.

    just engine-serve   # NOT needed — harvest uses only the ColBERT encoder (CPU) + Qdrant
    uv run --no-sync python scripts/harvest_domain_docs.py --domain "Laboratory Information Management" \
        --target 50 --max-stream 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import domain_index as DI  # noqa: E402

_STORE = REPO / "build" / "domain_harvest"


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _token_windows(text: str, *, stride: int = 256, size: int = 512,
                   max_windows: int = 16) -> "list[tuple[int, int]]":
    """CHAR spans of ≤``size``-token windows at ``stride`` tokens — the 512-token
    encoder limit IS the item grain (RH), so every window is a valid ITEM. Offsets come
    from the encoder's own tokenizer (no second tokenization scheme)."""
    from aegir.ontology.colbert_encoder import get_encoder
    tk = get_encoder()._tokenizer(text, truncation=False, return_offsets_mapping=True,
                                  add_special_tokens=False)
    offs = [o for o in tk["offset_mapping"] if o[1] > o[0]]
    if not offs:
        return [(0, len(text))]
    out = []
    i = 0
    while i < len(offs) and len(out) < max_windows:
        j = min(i + size, len(offs))
        out.append((offs[i][0], offs[j - 1][1]))
        if j >= len(offs):
            break
        i += stride
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceFW/finepdfs")
    ap.add_argument("--config", default="eng_Latn")
    ap.add_argument("--revision", default=None,
                    help="pin the upstream dataset revision (HF commit sha or tag). Omitted = read "
                         "whatever is current AND record the sha it resolved to — attribution must "
                         "be reproducible, not merely declared (ODC-By 1.0 attribution obligation).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--domain", default="Laboratory Information Management", help="SKOS subtree to keep (code or prefLabel)")
    ap.add_argument("--domain-tau", type=float, default=0.10, help="min rank1−rank2 relative margin to count as in-domain")
    ap.add_argument("--target", type=int, default=50, help="stop after this many in-domain matches")
    ap.add_argument("--max-stream", type=int, default=2000, help="cap docs scanned this run")
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=4000, help="chars fed to the classifier")
    ap.add_argument("--store", default=str(_STORE))
    ap.add_argument("--domain-url", default=DI.DEFAULT_QDRANT_URL)
    ap.add_argument("--domain-collection", default=DI.DEFAULT_APERTURE,
                    help="domains-only AIMING collection (the full vocab's abstract catalog absorbs the top concept)")
    ap.add_argument("--no-resume", action="store_true", help="ignore the saved stream cursor (start at 0)")
    ap.add_argument("--window-mode", choices=["head", "items"], default="head",
                    help="head = one decision on the head slice (legacy); items = the ORIGINAL "
                         "design realized (RH 2026-07-23): sliding ≤512-token windows, optimal-"
                         "match windows admitted as fedwiki ITEMS, non-admitted candidate windows "
                         "recorded interstitially (manifest_items.jsonl)")
    ap.add_argument("--stride-tokens", type=int, default=256,
                    help="items mode: window stride (window size = the 512-token encoder limit)")
    ap.add_argument("--max-windows", type=int, default=16,
                    help="items mode: max windows per doc (bounds encode cost; 16 ≈ 8k tokens)")
    args = ap.parse_args()

    store = Path(args.store)
    (store / "docs").mkdir(parents=True, exist_ok=True)
    manifest = store / "manifest.jsonl"
    cursor_f = store / "cursor.json"

    codes = DI.subtree_codes(DI.load_skos(), args.domain)
    if not codes:
        print(f"domain '{args.domain}' not in the SKOS hierarchy — author it first (expand-hierarchy-first)", file=sys.stderr)
        return 1
    seen: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                seen.add(json.loads(line)["hash"])
            except (ValueError, KeyError):
                continue
    cursor = 0
    if not args.no_resume and cursor_f.exists():
        cursor = json.loads(cursor_f.read_text()).get("cursor", 0)
    # S9-settled admission recomposition (RH 2026-07-23): roots retire from MaxSim
    # competition when the strategy-released filter is ARMED (or force-armed for shadow
    # windows via AEGIR_ADMISSION_FILTER=1); disarmed/absent = unchanged behavior.
    _af = DI.armed_admission_filter()
    _adm_exclude = _af.get("effective_exclude") or set()
    if _adm_exclude:
        print(f"admission filter ARMED: excluding codes {sorted(_adm_exclude)} "
              f"({', '.join((_af.get('excluded_concepts') or {}).values())})")
    print(f"harvest '{args.domain}' ({len(codes)} SKOS concepts, tau={args.domain_tau}) · "
          f"resume cursor={cursor} · already stored={len(seen)} · target={args.target}")

    from datasets import load_dataset

    # RESOLVE AND RECORD the upstream revision actually read. FinePDFs is ODC-By 1.0, which imposes a
    # POSITIVE attribution obligation, and an attribution that cannot be checked against a specific
    # upstream state is not verifiable. Recording the resolved sha also gives the cross-reference
    # check its baseline: a later run can tell `unchanged` from `content-changed` from `withdrawn`
    # per document, which is how a takedown gets an actual remedy instead of a silent divergence.
    upstream_revision = args.revision
    try:
        from huggingface_hub import dataset_info
        upstream_revision = dataset_info(args.dataset, revision=args.revision).sha
    except Exception as e:  # noqa: BLE001 — offline/unauthenticated: say so, never fabricate a sha
        print(f"  ⚑ could not resolve {args.dataset} revision ({type(e).__name__}): recording "
              f"{upstream_revision or 'UNPINNED'} — attribution is not reproducible until pinned")
    print(f"  upstream: {args.dataset}/{args.config} @ {upstream_revision or 'UNPINNED'} (ODC-By 1.0)")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True,
                      **({"revision": args.revision} if args.revision else {}))
    if cursor:
        ds = ds.skip(cursor)

    t0 = time.time()
    scanned = short = dup = matched = 0
    mf = manifest.open("a")
    imf = (store / "manifest_items.jsonl").open("a") if args.window_mode == "items" else None
    for row in ds:
        if scanned >= args.max_stream or matched >= args.target:
            break
        scanned += 1
        text = (row.get(args.text_field) or "").strip()
        if len(text) < args.min_chars:
            short += 1
            continue
        h = _hash(text)
        if h in seen:
            dup += 1
            continue
        if args.window_mode == "items":
            # THE ORIGINAL DESIGN (RH 2026-07-23): sliding ≤512-token windows over the
            # WHOLE doc; each window is a candidate ITEM. Admitted items are the optimal-
            # match windows; non-admitted candidates persist interstitially — the fedwiki
            # stream where every item becomes an editable block. The doc admits iff ANY
            # window admits; the doc-grain manifest row (P←F lineage shape) carries the
            # BEST window's anchor so both grains stay auditable.
            windows = _token_windows(text, stride=args.stride_tokens,
                                     max_windows=args.max_windows)
            items = []
            best = None
            for ix, (c0, c1) in enumerate(windows):
                wcls = DI.classify_hierarchical(text[c0:c1], top_k=5,
                                                url=args.domain_url,
                                                collection=args.domain_collection,
                                                exclude_codes=_adm_exclude or None)
                wtop = wcls.get("top") or {}
                w_admit = DI.in_subtree(wtop, codes) and wcls["rel_margin"] >= args.domain_tau
                items.append({"ix": ix, "span": [c0, c1], "code": wtop.get("code"),
                              "rel_margin": wcls["rel_margin"], "admitted": w_admit})
                if w_admit and (best is None or wcls["rel_margin"] > best[1]["rel_margin"]):
                    best = (wtop, wcls)
            if best:
                top, hcls = best
                (store / "docs" / f"{h}.txt").write_text(text, encoding="utf-8")
                mf.write(json.dumps({"hash": h, "domain": args.domain, "code": top.get("code"),
                                     "label": top.get("pref_label"),
                                     "rel_margin": hcls["rel_margin"],
                                     "id": str(row.get("id") or ""), "chars": len(text),
                                     "window_mode": "items",
                                     "n_windows": len(items),
                                     "n_admitted_items": sum(i["admitted"] for i in items),
                                     "dataset": f"{args.dataset}/{args.config}",
                                     "revision": upstream_revision or "UNPINNED"}) + "\n")
                mf.flush()
                imf.write(json.dumps({"hash": h, "items": items}) + "\n")
                imf.flush()
                seen.add(h)
                matched += 1
                print(f"  ✓ [{matched}/{args.target}] {top.get('code')} "
                      f"{(top.get('pref_label') or '')[:30]:30s} "
                      f"rel_margin={hcls['rel_margin']:.3f} "
                      f"items={sum(i['admitted'] for i in items)}/{len(items)}  {h[:12]}")
            continue
        hcls = DI.classify_hierarchical(text[: args.max_chars], top_k=5,
                                        url=args.domain_url, collection=args.domain_collection,
                                        exclude_codes=_adm_exclude or None)
        top = hcls.get("top") or {}
        if DI.in_subtree(top, codes) and hcls["rel_margin"] >= args.domain_tau:
            (store / "docs" / f"{h}.txt").write_text(text, encoding="utf-8")
            mf.write(json.dumps({"hash": h, "domain": args.domain, "code": top.get("code"),
                                 "label": top.get("pref_label"), "rel_margin": hcls["rel_margin"],
                                 "id": str(row.get("id") or ""), "chars": len(text),
                                 "dataset": f"{args.dataset}/{args.config}",
                                     "revision": upstream_revision or "UNPINNED"}) + "\n")
            mf.flush()
            seen.add(h)
            matched += 1
            print(f"  ✓ [{matched}/{args.target}] {top.get('code')} {(top.get('pref_label') or '')[:30]:30s} "
                  f"rel_margin={hcls['rel_margin']:.3f}  {h[:12]}")
    mf.close()
    if imf:
        imf.close()
    cursor_f.write_text(json.dumps({"cursor": cursor + scanned, "dataset": args.dataset, "config": args.config}, indent=1))

    rate = matched / max(1, scanned - short - dup)
    print(f"\nHARVEST: scanned {scanned} · short {short} · dup {dup} · "
          f"classified {scanned - short - dup} · MATCHED {matched} (rate {rate:.1%}) [{time.time()-t0:.0f}s]")
    print(f"  store: {store.relative_to(REPO)}/docs ({len(seen)} unique in-domain docs total) · cursor → {cursor + scanned}")
    print(f"  → feed derive_ontology with these LIMS-matched docs to grow the ontology in-domain.")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
