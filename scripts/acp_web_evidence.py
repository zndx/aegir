#!/usr/bin/env python
"""acp_web_evidence — the AGENT-MEDIATED web-evidence channel (RH 2026-07-23:
"employ ACP and provide a proper tmp scratchpad" — the naive query→fetch→admit loop
was lazy).

The closed-loop shape (agent-mediated-feedback doctrine: the agent PROPOSES, the
membranes DISPOSE, every refusal carries its reason):

  WORKSPACE (the proper scratchpad, one per session):
      build/acp_web/<ts>_<domain>/
        materials/     — harness-gathered: Brave results index + fetched page texts
        deliverable/   — the agent's curated evidence documents (its ACP fs_root)
        session.json   — queries, exchange metadata, disposal ledger

  HARNESS (deterministic): per-anchor Brave queries → fetch → materials/ with
      index.json (file → url/query/title). BRAVE_SEARCH_API_KEY preferred.
  AGENT (ACP, vibe-acp fork → the live engine): reads materials/, curates REAL
      domain evidence into deliverable/evidence_*.md (each: `Source: <url>` first
      line + a self-contained passage), rejects boilerplate with reasons in
      deliverable/notes.md. The deliverable is a FILE — never bounded by context.
  MEMBRANE: each deliverable → item_scan admission (armed filter, item semantics) —
      admitted evidence enters the store with brave/acp provenance (url + session);
      refusals are LEDGERED with margins (the reason returns to the next prompt).

    AEGIR_ADMISSION_FILTER=1 LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs \\
      uv run python scripts/acp_web_evidence.py --domain MFG
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

STORE = REPO / "build" / "domain_harvest"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="MFG")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--per-query", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()

    import requests
    from scripts.harvest_web_domain import _extract_text, _hash, brave_search, UA
    from aegir.ontology import admission_items as AI
    from aegir.ontology import domain_index as DI

    ws = REPO / "build" / "acp_web" / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{a.domain}"
    mat = ws / "materials"
    dlv = ws / "deliverable"
    mat.mkdir(parents=True)
    dlv.mkdir(parents=True)

    concepts = DI.load_skos(str(DI.DEFAULT_OVERLAY))
    excl = DI.armed_admission_filter().get("effective_exclude") or set()
    fam = [(str(getattr(c, "code", "")), iri.rsplit("#", 1)[-1], c)
           for iri, c in concepts.items()
           if iri.rsplit("#", 1)[-1].startswith(a.domain + "_")
           and not iri.rsplit("#", 1)[-1].startswith("SDG_")]

    # ── harness: gather materials ──
    queries = []
    for code, local, c in sorted(fam):
        label = str(getattr(c, "pref_label", "") or local.replace("_", " "))
        kw = " ".join(re.findall(r"[a-zA-Z]{5,}",
                                 str(getattr(c, "definition", "") or ""))[:6])
        queries.append((code, local, f"{label} {kw}"))
        if a.domain == "MFG":
            queries.append((code, local,
                            f"{label} model-based systems engineering SysML practice"))
    queries = queries[: a.queries]
    index = []
    n_file = 0
    for code, local, q in queries:
        try:
            results = brave_search(q, count=a.per_query)
        except Exception as e:  # noqa: BLE001
            print(f"  query failed ({q[:40]}…): {str(e)[:80]}")
            continue
        for res in results:
            url = res.get("url") or ""
            try:
                page = requests.get(url, headers={"User-Agent": UA}, timeout=15)
                if page.status_code != 200:
                    continue
                text = _extract_text(page.text)
            except Exception:  # noqa: BLE001
                continue
            if len(text) < 600:
                continue
            n_file += 1
            fn = f"{n_file:03d}_{re.sub(r'[^a-z0-9]+', '-', (res.get('title') or 'page').lower())[:48]}.txt"
            (mat / fn).write_text(text[:12000], encoding="utf-8")  # agent reading copy: ~3k tokens
            index.append({"file": fn, "url": url, "query": q, "anchor": local,
                          "title": res.get("title", "")})
            time.sleep(0.4)
        time.sleep(0.8)
    (mat / "index.json").write_text(json.dumps(index, indent=1))
    print(f"materials: {len(index)} pages for {len(queries)} queries → {mat}")
    if not index:
        print("no materials gathered — session abandoned")
        return 1

    # ── agent: curate over ACP (fs_root = the workspace; deliverable = files) ──
    anchor_briefs = "\n".join(
        f"- {local} ({str(getattr(c, 'pref_label', ''))}): "
        f"{str(getattr(c, 'definition', ''))[:220]}"
        for _, local, c in sorted(fam))
    prompt = f"""You are curating WEB EVIDENCE for these {a.domain}-family concepts:

{anchor_briefs}

materials/ holds {len(index)} fetched pages; materials/index.json maps each file to
its source url, query, and target anchor. Your deliverable is FILES in deliverable/:

1. For each page holding REAL domain evidence (substantive prose about the concept —
   never navigation, marketing, cookie banners, or thin listicles), write
   deliverable/evidence_<NN>.md containing:
   - first line exactly:  Source: <the url from index.json>
   - second line exactly: Anchor: <the target anchor local name>
   - then the curated passage: the page's substantive content on that concept,
     lightly cleaned, 300–2000 words, self-contained. Prefer the page's own words.
2. Write deliverable/notes.md: one line per REJECTED page — file, url, and WHY.

Work through every file in materials/. Quality over quantity — a page with no real
evidence is rejected with a reason, never padded into a deliverable."""
    from aegir.generate.harness import backend_spec
    from aegir.refine.acp import BaseACPClient
    home = ws / "vibe_home"
    home.mkdir()
    spec, cfg_toml, model, provider = backend_spec("local", str(home))
    if cfg_toml:
        (home / "config.toml").write_text(cfg_toml)

    # BATCHED fresh-context sessions (measured: one session over 45 materials blew the
    # engine's 32k context) — each batch gets its own ACP session; the workspace files
    # carry state across batches, never the context.
    BATCH = 3
    agent_notes = []
    for b0 in range(0, len(index), BATCH):
        batch = index[b0:b0 + BATCH]
        listing = "\n".join(f"- materials/{r['file']}  (url: {r['url']}; target anchor: "
                             f"{r['anchor']})" for r in batch)
        bprompt = (prompt
                   + f"\n\nTHIS SESSION covers ONLY these {len(batch)} pages "
                     f"(evidence file numbers may start at {b0 + 1:02d}):\n" + listing)

        async def _run(msg=bprompt):
            client = BaseACPClient(spec, fs_root=str(ws), allow_write=True)
            async with client:
                return await client.prompt(msg, timeout=a.timeout)

        try:
            res = asyncio.run(_run())
            agent_notes.append(getattr(res, "text", "") or "")
            # REASONING RETENTION (RH 2026-07-24): the trace is a corpus value-add —
            # every batch's thought chunks persist beside the deliverables
            with (ws / "reasoning.jsonl").open("a") as rf:
                rf.write(json.dumps({"batch": b0 // BATCH + 1,
                                     "thoughts": list(getattr(res, "thoughts", []) or []),
                                     "text": (getattr(res, "text", "") or "")[:2000]}) + "\n")
            print(f"  batch {b0 // BATCH + 1}/{(len(index) + BATCH - 1) // BATCH}: "
                  f"{len(list(dlv.glob('evidence_*.md')))} deliverables so far", flush=True)
        except Exception as e:  # noqa: BLE001 — the ledger records; the next batch proceeds
            agent_notes.append(f"batch {b0 // BATCH + 1} failed: {e}")
            print(f"  batch {b0 // BATCH + 1} FAILED: {str(e)[:120]}", flush=True)
    agent_note = " | ".join(n[:160] for n in agent_notes if n)

    # ── membrane: item_scan disposes each deliverable ──
    idx_by_url = {r["url"]: r for r in index}
    ledger = []
    admitted = 0
    seen: set = set()
    if (STORE / "manifest.jsonl").exists():
        for line in (STORE / "manifest.jsonl").read_text().splitlines():
            try:
                seen.add(json.loads(line)["hash"])
            except (ValueError, KeyError):
                continue
    mf = (STORE / "manifest.jsonl").open("a")
    imf = (STORE / "manifest_items.jsonl").open("a")
    for f in sorted(dlv.glob("evidence_*.md")):
        raw = f.read_text(errors="ignore")
        m = re.match(r"Source:\s*(\S+)\s*\nAnchor:\s*(\S+)\s*\n(.*)", raw, re.S)
        if not m:
            ledger.append({"file": f.name, "disposed": "malformed (missing Source/Anchor "
                                                       "header)"})
            continue
        url, anchor, body = m.group(1), m.group(2), m.group(3).strip()
        h = _hash(body)
        if h in seen or len(body) < 400:
            ledger.append({"file": f.name, "disposed": "duplicate or too short"})
            continue
        scan = AI.item_scan(body, tau=a.tau, exclude_codes=excl or None)
        kept = [i for i in scan["items"] if i["admitted"]]
        if kept:
            best = max(kept, key=lambda i: i["genus_margin"])
            (STORE / "docs" / f"{h}.txt").write_text(body, encoding="utf-8")
            mf.write(json.dumps({"hash": h, "domain": str(best["code"]).split(".")[0],
                                 "code": best["code"], "label": best.get("label", ""),
                                 "rel_margin": best["genus_margin"], "chars": len(body),
                                 "window_mode": "items",
                                 "n_windows": len(scan["candidates"]),
                                 "n_admitted_items": len(kept),
                                 "dataset": "brave/acp", "url": url,
                                 "session": ws.name, "curated_for": anchor}) + "\n")
            mf.flush()
            imf.write(json.dumps({"hash": h, "items": scan["items"]}) + "\n")
            imf.flush()
            seen.add(h)
            admitted += 1
            ledger.append({"file": f.name, "disposed": "ADMITTED",
                           "code": best["code"], "genus_margin": best["genus_margin"],
                           "n_items": len(kept), "url": url})
        else:
            best_m = max((c["genus_margin"] for c in scan["candidates"]), default=0.0)
            ledger.append({"file": f.name,
                           "disposed": f"refused — best genus margin {best_m:.3f} < "
                                       f"tau {a.tau} (the reason returns to the agent)",
                           "url": url})
    mf.close()
    imf.close()
    (ws / "session.json").write_text(json.dumps(
        {"domain": a.domain, "provider": provider, "model": model,
         "n_materials": len(index), "queries": [q for _, _, q in queries],
         "agent_note": agent_note[:800],
         "n_deliverables": len(list(dlv.glob('evidence_*.md'))),
         "n_admitted": admitted, "ledger": ledger}, indent=1))
    print(f"\nACP WEB EVIDENCE ({a.domain}): {len(index)} materials → "
          f"{len(list(dlv.glob('evidence_*.md')))} deliverables → {admitted} admitted "
          f"(ledger in {ws.relative_to(REPO)}/session.json)")
    n_urls_unused = len(idx_by_url) - len({l.get('url') for l in ledger if l.get('url')})
    print(f"  materials without deliverables: {n_urls_unused} "
          f"(see deliverable/notes.md for the agent's rejections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
