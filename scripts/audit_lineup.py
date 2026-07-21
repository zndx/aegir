#!/usr/bin/env python
"""audit_lineup — the mechanical panel-quality audit (RH 2026-07-20: quality-drift instrument).

Velocity adds surfaces in thin increments; each iteration reads fine and the cumulative product
drifts. This makes drift MEASURABLE: a deterministic sweep of the built projection producing
per-kind defect counts + samples, runnable after every kb-build (and gate-able once the floor
is established). Checks — all computable from the projection alone:

  * hash-titled notes         — titles that are bare hex ids (unusable for browsing)
  * stub bodies               — release/content panels under a floor length (excerpt-only stubs)
  * dangling wikilinks        — [[targets]] resolving in NO root (roots are refs — cross-root ok)
  * pipe-unsafe table rows    — raw '|' inside table-cell wikilink labels the renderer can't split
  * orphan panels             — notes with no inbound links from any root (lens/index kinds exempt)
  * unclosed code fences      — odd ``` count (swallows everything after it in the renderer)

    uv run python scripts/audit_lineup.py [--kb build/dev] [--json build/lineup_audit.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_WL = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_HEX_TITLE = re.compile(r"^[0-9a-f]{8,16}(\s*·.*)?(\s*\(\d+/\d+\))?$")
_ENTRY_KINDS_EXEMPT_ORPHAN = {"lens", "collection-index", "content-index", "archive-snapshot",
                              "release-note", "corpus", "provenance", "lexicon-construct",
                              "training", "item"}
STUB_FLOOR = 400          # chars of body below which a content panel counts as a stub


def _parse(p: Path) -> "tuple[dict, str]":
    text = p.read_text()
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    meta: dict = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                meta[k.strip()] = json.loads(v.strip())
            except Exception:  # noqa: BLE001
                meta[k.strip()] = v.strip().strip('"')
    return meta, text[end + 4:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kb", default=str(REPO / "build/dev"))
    ap.add_argument("--json", default=str(REPO / "build/lineup_audit.json"))
    ap.add_argument("--max-samples", type=int, default=6)
    a = ap.parse_args()
    kb = Path(a.kb)

    notes: "dict[str, dict]" = {}          # id → {kind, root, title, body, path}
    ids_by_root: "dict[str, set]" = defaultdict(set)
    for root in ("current", "scratch", "archive"):
        rdir = kb / root
        if not rdir.exists():
            continue
        for p in rdir.rglob("*.md"):
            meta, body = _parse(p)
            nid = str(meta.get("name") or p.relative_to(rdir).as_posix()[:-3])
            notes[f"{root}:{nid}"] = {"id": nid, "kind": str(meta.get("kind", "?")),
                                      "root": root, "title": str(meta.get("title", "")),
                                      "body": body, "path": str(p.relative_to(kb))}
            ids_by_root[root].add(nid)
    all_ids = set().union(*ids_by_root.values()) if ids_by_root else set()

    D: "dict[str, list]" = defaultdict(list)
    inbound: "dict[str, int]" = defaultdict(int)
    for key, n in notes.items():
        body, kind = n["body"], n["kind"]
        # links + dangling
        for m in _WL.finditer(body):
            tgt = m.group(1).strip()
            if tgt in all_ids:
                inbound[tgt] += 1
            elif re.match(r"(?:ontology/(?:foreign/\w+|self-census)/class/|path/|provenance/\d)", tgt):
                pass    # SYNTHETIC ids — resolved live by the gateway (class panels, path
                        # runs, provenance ego graphs); projected files never exist for them
            elif n["root"] == "archive":
                D["dangling_archive_era"].append(f"{n['id']} → [[{tgt}]]")
                # frozen kastens are historical documents — era-dangling is reported,
                # never floored (the live-projection floor stays zero)
            else:
                D["dangling_wikilink"].append(f"{n['root']}:{n['id']} → [[{tgt}]]")
        # hash titles
        if _HEX_TITLE.match(n["title"].strip()):
            D["hash_title"].append(f"{n['root']}:{n['id']} — {n['title']!r}")
        # stubs (content panels only; archive exempt — frozen eras keep their era's shape)
        if (kind == "content-chapter" and n["root"] != "archive" and len(body) < STUB_FLOOR
                and "__w" not in n["id"]):     # continuation tails are legitimately short
            D["stub_body"].append(f"{n['root']}:{n['id']} ({len(body)} chars)")
        # pipe-unsafe rows: a table row whose wikilink label contains '|' is fine (renderer
        # handles), but a RAW '|' inside backticks in a cell is not detectable — check instead
        # for rows whose pipe-split (wikilink-aware) yields ragged column counts
        rows = [ln for ln in body.splitlines() if ln.startswith("|") and not set(ln) <= {"|", "-", " ", ":"}]
        widths = set()
        for ln in rows:
            depth = 0
            cols = 1
            i = 0
            while i < len(ln):
                if ln[i:i + 2] == "[[":
                    depth += 1
                    i += 2
                    continue
                if ln[i:i + 2] == "]]":
                    depth -= 1
                    i += 2
                    continue
                if ln[i] == "|" and depth == 0:
                    cols += 1
                i += 1
            widths.add(cols)
        if len(widths) > 1 and rows:
            D["ragged_table"].append(f"{n['root']}:{n['id']} (row widths {sorted(widths)})")
        # unclosed fences
        if body.count("```") % 2 == 1:
            D["unclosed_fence"].append(f"{n['root']}:{n['id']}")
        # broken references (RH 2026-07-21: interface says a calm '(unresolved)'; the
        # STRUCTURE is a hard defect — ERROR-logged at build, floored to zero here)
        if "_(unresolved)_" in body:
            D["broken_reference"].append(f"{n['root']}:{n['id']}")
    # orphans (inbound counted across ALL roots — roots are refs)
    for key, n in notes.items():
        if n["kind"] in _ENTRY_KINDS_EXEMPT_ORPHAN or "/" in n["id"] and n["id"].endswith("/terms"):
            continue
        if inbound.get(n["id"], 0) == 0 and not n["id"].startswith(("lens/", "topic/")):
            D["orphan"].append(f"{n['root']}:{n['id']} ({n['kind']})")

    by_kind_stub = defaultdict(int)
    for x in D["stub_body"]:
        by_kind_stub[x.split(":")[0]] += 1
    report = {"n_notes": len(notes),
              "defects": {k: {"n": len(v), "sample": v[:a.max_samples]} for k, v in sorted(D.items())}}
    Path(a.json).write_text(json.dumps(report, indent=1))
    print(f"{len(notes)} notes audited")
    for k, v in sorted(D.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<20} {len(v):>6}")
        for x in v[:3]:
            print(f"      · {x[:110]}")
    print(f"→ {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
