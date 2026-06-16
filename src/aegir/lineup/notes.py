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


def write_index(kb_dir: Path, notes: list[Note]) -> Path:
    """Emit ``index.json`` — the gateway's listing + id→relpath resolution + edge graph."""
    by_dp: dict[str, int] = {}
    for n in notes:
        by_dp[n.data_product] = by_dp.get(n.data_product, 0) + 1
    payload = {
        "counts": {"total": len(notes), "by_data_product": by_dp},
        "notes": [
            {"id": n.id, "title": n.title, "kind": n.kind, "data_product": n.data_product,
             "root": n.root, "relpath": relpath(n), "links": n.links or extract_links(n.body)}
            for n in notes
        ],
    }
    p = Path(kb_dir) / "index.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p
