#!/usr/bin/env python
"""Render the lineup KB projection → a browsable mdbook (SHARE-Docs Phase A).

Reads the materialized projection (``build/dev/current`` notes + ``index.json``) and emits a
self-contained mdbook: ``book.toml`` + ``src/SUMMARY.md`` + one flat page per note (id
slugified, ``/`` → ``__``), with the lineup ``[[id|label]]`` wikilinks lowered to mdbook
relative page links. The SUMMARY leads with the collections × lens pivot (the landing), then
the collections, the lenses, and the ontology / relational / content products — so the static
book is the lineup's panel-trail flattened into a navigable document. Deterministic.

  uv run --no-sync python scripts/render_lineup_mdbook.py [--out build/dev/lineup-book] [--build]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.lineup import sources as S  # noqa: E402

WL = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")


def _slug(note_id: str) -> str:
    return note_id.replace("/", "__")


def _lower_links(body: str, present: set[str]) -> str:
    """[[id|label]] → [label](id__slug.md) when the target page exists, else just the label."""
    def repl(m: re.Match) -> str:
        tid, label = m.group(1), m.group(2)
        lbl = label or tid.split("/")[-1]
        return f"[{lbl}]({_slug(tid)}.md)" if tid in present else lbl
    return WL.sub(repl, body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "build" / "dev" / "lineup-book"))
    ap.add_argument("--build", action="store_true", help="run `mdbook build` after rendering")
    args = ap.parse_args()

    kb = S.kb_dir()
    idx_p = kb / "index.json"
    if not idx_p.exists():
        print(f"no projection at {idx_p} — run `just kb-build` first.")
        return 1
    idx = json.loads(idx_p.read_text())
    notes = idx["notes"]
    present = {n["id"] for n in notes}
    by_id = {n["id"]: n for n in notes}

    out = Path(args.out)
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    for stale in src.glob("*.md"):
        stale.unlink()

    # one flat page per note (full body via the gateway's read_note, links lowered)
    from aegir.lineup import notes as N
    for n in notes:
        if n.get("root") != "current":
            continue
        full = N.read_note(kb, n["relpath"])
        page = f"# {n['title']}\n\n_{n['kind']}_\n\n" + _lower_links(full.get("body", ""), present)
        (src / f"{_slug(n['id'])}.md").write_text(page)

    def link(nid: str, label: str | None = None) -> str:
        n = by_id.get(nid)
        return f"[{label or (n['title'] if n else nid)}]({_slug(nid)}.md)"

    def kind(k: str) -> list[str]:
        return sorted((n["id"] for n in notes if n.get("kind") == k and n.get("root") == "current"))

    # Introduction page = the default landing pivot (lens/terms).
    intro = N.read_note(kb, by_id["lens/terms"]["relpath"]) if "lens/terms" in by_id else {"body": ""}
    (src / "README.md").write_text("# The Lineup — ontology-grounded corpus\n\n"
                                   "A browsable rendering of the collections × lens pivot. Start below, "
                                   "or jump to a collection.\n\n" + _lower_links(intro.get("body", ""), present))

    # mdbook renders ONLY pages listed in SUMMARY (and rejects duplicates), so every note
    # appears exactly once, grouped by section, for all cross-links to resolve.
    seen: set[str] = set()
    S_ = ["# Summary", "", "[Introduction](README.md)", ""]

    def section(title: str, ids: list[str], indent: str = "- ") -> None:
        S_.append(f"# {title}")
        S_.append("")
        for nid in ids:
            if nid in by_id and nid not in seen:
                seen.add(nid)
                S_.append(f"{indent}{link(nid)}")
        S_.append("")

    section("Lenses", ["lens/terms", "lens/schema", "lens/content"])
    section("Collections", ["collection/index", *kind("collection")])
    section("Ontology", [*kind("ontology-anchor"), *kind("ontology-category"), *kind("ontology-term")])
    section("Relational", [*kind("relational-category"), *kind("relational-table")])
    section("Content", ["topic/index", "content/index", *kind("topic"), *kind("content-chapter")])

    (src / "SUMMARY.md").write_text("\n".join(S_) + "\n")
    (out / "book.toml").write_text(
        '[book]\ntitle = "Aegir Lineup — ontology-grounded corpus"\nlanguage = "en"\nsrc = "src"\n'
        '[output.html]\ndefault-theme = "rust"\n')

    n_pages = len(list(src.glob("*.md")))
    print(f"rendered {n_pages} pages → {out}  (SUMMARY: lenses · {len(kind('collection'))} collections · "
          f"ontology · relational · content)")

    if args.build:
        r = subprocess.run(["mdbook", "build", str(out)], capture_output=True, text=True)
        ok = r.returncode == 0
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
        print(f"  mdbook build → {'OK' if ok else 'FAILED'}: {tail[0]}")
        if ok:
            print(f"  book at {out}/book/index.html")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
