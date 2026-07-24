#!/usr/bin/env python
"""harvest_web_domain — the Brave web-evidence channel (RH 2026-07-23).

Prose-starved anchors (MFG 9 docs; MBSE side generally) stay ROBUST to in-abstracto
extension — but we have Brave Web Search (the Gaius capability, keys present here), so
evidence can be GATHERED where FinePDFs is starved, through the SAME admission
instrument as every other source:

    per-anchor queries (composite label + contrast vocabulary) → Brave Search API →
    page fetch (polite: bounded, identified UA) → text extraction → item_scan
    (sliding ≤512-token items, armed-filter dual view, non-overlap resolution) →
    admitted docs enter the store with brave/web provenance (url in the manifest row).

Same posture as the stream advance: additive, content-addressed, scratch-side; the
review lane accrues root-strong pages exactly like FinePDFs docs. Wire into
``just metaflow`` via the harvest stage once field-proven (this script = the probe +
the standing manual channel; `just harvest-web`).

Keys: BRAVE_SEARCH_API_KEY preferred (the Search subscription), BRAVE_API_KEY fallback.

    AEGIR_ADMISSION_FILTER=1 LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs \\
      uv run python scripts/harvest_web_domain.py --domain MFG --queries 6 --per-query 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

STORE = REPO / "build" / "domain_harvest"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
UA = "aegir-harvest/0.3 (research corpus construction; contact rch@zndx.org)"


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _extract_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "header", "footer", "aside"]):
            t.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    except Exception:  # noqa: BLE001 — crude fallback
        return re.sub(r"<[^>]+>", " ", html)


def brave_search(q: str, count: int = 10) -> "list[dict]":
    key = os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    if not key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY / BRAVE_API_KEY not set")
    r = requests.get(BRAVE_URL, params={"q": q, "count": count},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"},
                     timeout=20)
    r.raise_for_status()
    return ((r.json().get("web") or {}).get("results")) or []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="MFG", help="root family local name (MFG, ENERGY, …)")
    ap.add_argument("--queries", type=int, default=6, help="queries per anchor family")
    ap.add_argument("--per-query", type=int, default=8, help="results fetched per query")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--min-chars", type=int, default=600)
    a = ap.parse_args()

    from aegir.ontology import admission_items as AI
    from aegir.ontology import domain_index as DI

    concepts = DI.load_skos(str(DI.DEFAULT_OVERLAY))
    af = DI.armed_admission_filter()
    excl = af.get("effective_exclude") or set()
    if excl:
        print(f"admission filter ARMED: excluding codes {sorted(excl)}")

    # per-genus-anchor queries: pref label + definition keywords (+ MBSE vocabulary
    # for the MFG family — its evidence home is the SysML/MBSE literature)
    fam_anchors = [(str(getattr(c, "code", "")), iri.rsplit("#", 1)[-1], c)
                   for iri, c in concepts.items()
                   if iri.rsplit("#", 1)[-1].startswith(a.domain + "_")
                   and not iri.rsplit("#", 1)[-1].startswith("SDG_")]
    queries: "list[tuple[str, str]]" = []
    for code, local, c in sorted(fam_anchors):
        label = str(getattr(c, "pref_label", "") or local.replace("_", " "))
        defn = str(getattr(c, "definition", "") or "")
        kw = " ".join(re.findall(r"[a-zA-Z]{5,}", defn)[:6])
        queries.append((code, f"{label} {kw}"))
        if a.domain == "MFG":
            queries.append((code, f"{label} model-based systems engineering SysML"))
    queries = queries[: a.queries * max(1, len(fam_anchors))]

    seen: set = set()
    if (STORE / "manifest.jsonl").exists():
        for line in (STORE / "manifest.jsonl").read_text().splitlines():
            try:
                seen.add(json.loads(line)["hash"])
            except (ValueError, KeyError):
                continue
    mf = (STORE / "manifest.jsonl").open("a")
    imf = (STORE / "manifest_items.jsonl").open("a")
    rvf = (STORE / "aperture_review.jsonl").open("a")
    fetched = admitted = reviewed = 0
    for code, q in queries:
        try:
            results = brave_search(q, count=a.per_query)
        except Exception as e:  # noqa: BLE001 — a failed query is visible, never fatal
            print(f"  query failed ({q[:40]}…): {str(e)[:80]}")
            continue
        for res in results:
            url = res.get("url") or ""
            if not url:
                continue
            try:
                page = requests.get(url, headers={"User-Agent": UA}, timeout=15)
                if page.status_code != 200 or "text/html" not in page.headers.get(
                        "content-type", "text/html"):
                    continue
                text = _extract_text(page.text)
            except Exception:  # noqa: BLE001
                continue
            fetched += 1
            if len(text) < a.min_chars:
                continue
            h = _hash(text)
            if h in seen:
                continue
            seen.add(h)
            scan = AI.item_scan(text, tau=a.tau, exclude_codes=excl or None)
            kept = [i for i in scan["items"] if i["admitted"]]
            if kept:
                best = max(kept, key=lambda i: i["genus_margin"])
                (STORE / "docs" / f"{h}.txt").write_text(text, encoding="utf-8")
                mf.write(json.dumps({"hash": h, "domain": str(best["code"]).split(".")[0],
                                     "code": best["code"], "label": best.get("label", ""),
                                     "rel_margin": best["genus_margin"],
                                     "chars": len(text), "window_mode": "items",
                                     "n_windows": len(scan["candidates"]),
                                     "n_admitted_items": len(kept),
                                     "dataset": "brave/web", "url": url,
                                     "query": q}) + "\n")
                mf.flush()
                imf.write(json.dumps({"hash": h, "items": scan["items"]}) + "\n")
                imf.flush()
                admitted += 1
                print(f"  ✓ {best['code']} m={best['genus_margin']:.3f} "
                      f"items={len(kept)}  {url[:70]}")
            elif scan["review"]:
                rvf.write(json.dumps({"hash": h, "chars": len(text), "url": url,
                                      "head": text[:280], **scan["review"]}) + "\n")
                rvf.flush()
                reviewed += 1
            time.sleep(0.5)               # polite pacing
        time.sleep(1.0)
    mf.close(); imf.close(); rvf.close()
    print(f"\nWEB HARVEST ({a.domain}): {len(queries)} queries · {fetched} pages fetched "
          f"· {admitted} admitted · {reviewed} → review lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
