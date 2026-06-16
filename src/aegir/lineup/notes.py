"""KB note schema + emitter — the normalization contract for the lineup projection.

A KB note is a markdown file with YAML frontmatter + body, addressable by an id of
the form ``<data_product>/<kind>/<slug>`` (path-like, so the id IS the relpath under a
root). Outbound lineup edges are ``[[id]]`` wikilinks inline in the body, mirrored in
frontmatter ``links`` so the gateway can extract the navigation graph without parsing
markdown. Conventions ported from the Gaius KB (``build/dev/{current,scratch,archive}``;
frontmatter + ``[[wikilinks]]``).

Frontmatter values are emitted with ``json.dumps`` per value — valid YAML (JSON is a
YAML subset for scalars/flow collections) AND trivially parseable by any YAML reader.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WL_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_./:-]+")


def slug(s: str) -> str:
    """Filesystem/id-safe slug, preserving ``:`` and ``/`` (kept for ids/anchors)."""
    return _SLUG_RE.sub("_", s).strip("_")


def wl(target_id: str, label: str | None = None) -> str:
    """A lineup edge: ``[[id]]`` or ``[[id|label]]``."""
    return f"[[{target_id}|{label}]]" if label else f"[[{target_id}]]"


def extract_links(body: str) -> list[str]:
    """Outbound edge ids referenced as ``[[id]]`` / ``[[id|label]]`` (dedup, in order)."""
    return list(dict.fromkeys(_WL_RE.findall(body)))


@dataclass
class Note:
    id: str                                  # "<data_product>/<kind>/<slug>"
    title: str
    kind: str                                # ontology-template | ...-family | ...-anchor |
                                             # relational-table | content-chapter | topic
    data_product: str                        # ontology | relational | content
    body: str = ""
    frontmatter: dict = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    root: str = "current"                    # current | scratch | archive


def relpath(note: Note) -> str:
    return f"{note.root}/{note.id}.md"


def to_markdown(note: Note) -> str:
    fm = {"name": note.id, "title": note.title, "kind": note.kind,
          "data_product": note.data_product, "root": note.root,
          "links": note.links, **note.frontmatter}
    head = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fm.items())
    return f"---\n{head}\n---\n\n{note.body.rstrip()}\n"


def write_note(kb_dir: Path, note: Note) -> Path:
    """Write one note; auto-derive ``links`` from the body if not set explicitly."""
    if not note.links:
        note.links = extract_links(note.body)
    p = Path(kb_dir) / relpath(note)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_markdown(note))
    return p


def parse_markdown(text: str) -> tuple[dict, str]:
    """Parse a note file → (frontmatter dict, body). Frontmatter values are JSON-encoded."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta: dict = {}
            for line in parts[1].strip().splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    try:
                        meta[k.strip()] = json.loads(v)
                    except json.JSONDecodeError:
                        meta[k.strip()] = v.strip()
            return meta, parts[2].strip()
    return {}, text


def read_note(kb_dir: Path, relpath: str) -> dict:
    """Read one projected note → a JSON-serializable dict (frontmatter + body + links)."""
    meta, body = parse_markdown((Path(kb_dir) / relpath).read_text())
    return {**meta, "body": body, "links": meta.get("links") or extract_links(body)}


def scan_notes(kb_dir: Path, root: str) -> list[dict]:
    """Index entries for every ``.md`` under ``build/dev/<root>/`` (recursive), so the
    projection (``current/``) AND authored notes (``scratch/``, ``archive/``) are all
    navigable — ``current`` lives alongside the others. Projected notes carry their
    root-less id in frontmatter ``name`` (matching the wikilinks); authored notes use
    their path as the id and infer ``data_product``/lens from frontmatter or the
    ``<utc>_<lens>.md`` zettelkasten filename convention."""
    base = Path(kb_dir) / root
    out: list[dict] = []
    if not base.exists():
        return out
    for p in sorted(base.rglob("*.md")):
        rel = p.relative_to(kb_dir).as_posix()
        try:
            fm, body = parse_markdown(p.read_text())
        except OSError:
            continue
        lens = fm.get("lens")
        if not lens and "_" in p.stem:
            lens = p.stem.rsplit("_", 1)[-1]          # gaius <iso-date>/<utc>_<lens>.md convention
        out.append({
            "id": fm.get("name") or rel[:-3],
            "title": fm.get("title") or p.stem,
            "kind": fm.get("kind") or f"{root}-note",
            "data_product": fm.get("data_product") or lens or root,
            "root": root,
            "relpath": rel,
            "links": fm.get("links") or extract_links(body),
        })
    return out


def write_index(kb_dir: Path, entries: list[dict]) -> Path:
    """Emit ``index.json`` from scanned entries — the gateway's listing + id→relpath
    resolution + edge graph, across all three roots (current / scratch / archive)."""
    by_dp: dict[str, int] = {}
    by_root: dict[str, int] = {}
    for e in entries:
        by_dp[e["data_product"]] = by_dp.get(e["data_product"], 0) + 1
        by_root[e["root"]] = by_root.get(e["root"], 0) + 1
    payload = {"counts": {"total": len(entries), "by_data_product": by_dp, "by_root": by_root},
               "notes": entries}
    p = Path(kb_dir) / "index.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p
