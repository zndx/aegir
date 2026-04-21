#!/usr/bin/env python3
"""Corpus item 4: render ontology OWL/TTL files as natural-language prose.

Consumes the OWL/JSONLD/TTL files downloaded by item 1
(/raid/datasets/aegir-corpus-v1/ontology/), walks each ontology's
class and property definitions, and emits plain English sentences
suitable for byte-level pretraining. One triple per line format:

    "{ClassName} is a subclass of {ParentName}. A {ClassName} is
     described as '{Description}'. Known properties: {p1}, {p2}, ..."

This converts the ~5 MB of highly-structured RDF/OWL into ~1 GB of
prose that a byte-level LM can actually learn from. The ontology's
hierarchy and definitions become text the model can read the same
way it reads any other text.

Three ontologies handled:
  - Schema.org (JSON-LD format)
  - DBpedia (OWL/RDF-XML)
  - BFO + CCO (OWL-XML, Turtle)

Output: /raid/datasets/aegir-corpus-v1/ontology-prose/{source}.txt,
UTF-8, \\x03 doc separator.

Usage:
    uv run --no-sync python scripts/gen_corpus_item4.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger("corpus_item4")

DEFAULT_ONTOLOGY_DIR = Path("/raid/datasets/aegir-corpus-v1/ontology")
DEFAULT_PROSE_DIR = Path("/raid/datasets/aegir-corpus-v1/ontology-prose")
DOC_DELIM = b"\x03"


# ── Schema.org (JSON-LD) ────────────────────────────────────────


def _render_schemaorg(jsonld_path: Path, out: Path) -> int:
    """Parse Schema.org JSON-LD, emit prose triples per class + property."""
    data = json.loads(jsonld_path.read_text())
    graph = data.get("@graph", [])
    classes: dict[str, dict] = {}
    properties: dict[str, dict] = {}
    for node in graph:
        if not isinstance(node, dict):
            continue
        nid = node.get("@id", "")
        ntype = node.get("@type")
        if ntype == "rdfs:Class":
            classes[nid] = node
        elif ntype == "rdf:Property":
            properties[nid] = node

    def _label(node_or_id) -> str:
        if isinstance(node_or_id, str):
            return node_or_id.split(":")[-1]
        if isinstance(node_or_id, dict):
            lab = node_or_id.get("rdfs:label") or node_or_id.get("@id", "")
            if isinstance(lab, dict):
                lab = lab.get("@value", "")
            return str(lab)
        return ""

    def _comment(node: dict) -> str:
        c = node.get("rdfs:comment", "")
        if isinstance(c, dict):
            c = c.get("@value", "")
        if isinstance(c, list):
            c = c[0] if c else ""
            if isinstance(c, dict):
                c = c.get("@value", "")
        return re.sub(r"\s+", " ", str(c)).strip()

    def _parents(node: dict) -> list[str]:
        sub = node.get("rdfs:subClassOf")
        if sub is None:
            return []
        if isinstance(sub, dict):
            return [_label(sub.get("@id", ""))]
        if isinstance(sub, list):
            out: list[str] = []
            for item in sub:
                if isinstance(item, dict):
                    out.append(_label(item.get("@id", "")))
                elif isinstance(item, str):
                    out.append(_label(item))
            return out
        return [_label(sub)]

    out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    n_bytes = 0
    with open(out, "wb") as fh:
        # Classes: hierarchy + definition
        for cid, cnode in classes.items():
            name = _label(cnode) or _label(cid)
            parents = _parents(cnode)
            desc = _comment(cnode)
            lines = []
            if parents:
                for p in parents:
                    lines.append(f"In the Schema.org vocabulary, {name} is a subclass of {p}.")
            else:
                lines.append(f"In the Schema.org vocabulary, {name} is a top-level type.")
            if desc:
                lines.append(f"A {name} is described as: {desc}")
            # Emit its properties (those with schema:domainIncludes pointing at cid)
            matching_props = []
            for pid, pnode in properties.items():
                dom = pnode.get("schema:domainIncludes")
                if dom is None:
                    continue
                if isinstance(dom, dict):
                    dom = [dom]
                if any(
                    isinstance(d, dict) and d.get("@id", "") == cid
                    for d in dom
                ):
                    matching_props.append(_label(pnode) or _label(pid))
            if matching_props:
                joined = ", ".join(sorted(set(matching_props))[:30])
                lines.append(f"Known properties of {name} include: {joined}.")
            doc = "\n".join(lines).encode("utf-8", errors="replace")
            fh.write(doc)
            fh.write(DOC_DELIM)
            n_lines += 1
            n_bytes += len(doc) + 1

        # Properties: range + description
        for pid, pnode in properties.items():
            name = _label(pnode) or _label(pid)
            desc = _comment(pnode)
            rng = pnode.get("schema:rangeIncludes")
            ranges: list[str] = []
            if isinstance(rng, dict):
                ranges = [_label(rng.get("@id", ""))]
            elif isinstance(rng, list):
                ranges = [
                    _label(r.get("@id", "")) if isinstance(r, dict) else _label(r)
                    for r in rng
                ]
            lines = [f"The Schema.org property {name} relates entities."]
            if ranges:
                lines.append(f"Values of {name} are of type: {', '.join(ranges[:10])}.")
            if desc:
                lines.append(f"{name} is described as: {desc}")
            doc = "\n".join(lines).encode("utf-8", errors="replace")
            fh.write(doc)
            fh.write(DOC_DELIM)
            n_lines += 1
            n_bytes += len(doc) + 1

    log.info("  schemaorg: %d definitions, %.2f MB", n_lines, n_bytes / 2**20)
    return n_bytes


# ── OWL/RDF-XML (DBpedia, BFO) ──────────────────────────────────


RDF_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _iri_label(iri: str) -> str:
    """Derive a readable label from an IRI."""
    s = iri.rstrip("/")
    if "#" in s:
        s = s.rsplit("#", 1)[1]
    elif "/" in s:
        s = s.rsplit("/", 1)[1]
    return s


def _render_owl_xml(owl_path: Path, source_name: str, out: Path) -> int:
    """Parse an OWL/RDF-XML file, emit prose per owl:Class definition."""
    try:
        tree = ET.parse(owl_path)
    except ET.ParseError as e:
        log.warning("  %s: parse failed (%s) — skipping", source_name, e)
        return 0
    root = tree.getroot()

    # Collect classes: {iri: {"parents": [...], "label": "...", "comment": "..."}}
    classes: dict[str, dict] = {}
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag != "Class":
            continue
        about = (
            elem.get(f"{{{RDF_NS['rdf']}}}about")
            or elem.get(f"{{{RDF_NS['rdf']}}}ID")
        )
        if not about:
            continue
        cls = classes.setdefault(about, {"parents": [], "label": "", "comment": ""})
        for child in elem:
            ctag = _strip_ns(child.tag)
            if ctag == "subClassOf":
                res = child.get(f"{{{RDF_NS['rdf']}}}resource")
                if res:
                    cls["parents"].append(res)
                else:
                    # nested Class reference
                    inner = child.find(f"{{{RDF_NS['owl']}}}Class")
                    if inner is not None:
                        inner_res = inner.get(f"{{{RDF_NS['rdf']}}}about")
                        if inner_res:
                            cls["parents"].append(inner_res)
            elif ctag == "label":
                cls["label"] = (child.text or "").strip()
            elif ctag == "comment":
                cls["comment"] = (child.text or "").strip()

    out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    n_bytes = 0
    with open(out, "wb") as fh:
        for cid, cls in classes.items():
            name = cls["label"] or _iri_label(cid)
            parents = [p for p in cls["parents"] if p]
            lines: list[str] = []
            if parents:
                parent_names = ", ".join(_iri_label(p) for p in parents[:5])
                lines.append(
                    f"In the {source_name} ontology, {name} is a subclass of "
                    f"{parent_names}."
                )
            else:
                lines.append(
                    f"In the {source_name} ontology, {name} is a top-level class."
                )
            if cls["comment"]:
                lines.append(
                    f"{name}: {re.sub(chr(10), ' ', cls['comment'])[:400]}"
                )
            lines.append(f"The full IRI for {name} is <{cid}>.")
            doc = "\n".join(lines).encode("utf-8", errors="replace")
            fh.write(doc)
            fh.write(DOC_DELIM)
            n_lines += 1
            n_bytes += len(doc) + 1
    log.info("  %s: %d definitions, %.2f MB", source_name, n_lines, n_bytes / 2**20)
    return n_bytes


# ── Turtle (CCO) ────────────────────────────────────────────────


_TTL_PREFIX_RE = re.compile(r"@prefix\s+(\S+)\s*:\s*<([^>]+)>\s*\.")
_TTL_STATEMENT_RE = re.compile(
    r"([<\S][^\s]*?)\s+(?:a|rdf:type)\s+(?:owl:Class|rdfs:Class)\s*[;\.]",
    re.MULTILINE,
)


def _render_turtle(ttl_path: Path, source_name: str, out: Path) -> int:
    """Light-weight Turtle parser — pulls class definitions without a full RDF library.

    Heuristic: split on blank lines into paragraph-blocks, then inside
    each block look for `rdf:type owl:Class` and collect the subject
    IRI + rdfs:label / rdfs:comment / rdfs:subClassOf. Good enough for
    CCO's well-formatted ``.ttl`` files without pulling in rdflib.
    """
    text = ttl_path.read_text(encoding="utf-8", errors="replace")

    # Split on BLANK lines (paragraph breaks). Each CCO class definition
    # is one such paragraph.
    blocks = re.split(r"\n\s*\n", text)
    prefixes = dict(_TTL_PREFIX_RE.findall(text))

    def _expand(curie: str) -> str:
        if ":" in curie and not curie.startswith("<"):
            pfx, local = curie.split(":", 1)
            if pfx in prefixes:
                return prefixes[pfx] + local
        return curie.strip("<>")

    def _extract_field(block: str, key: str) -> str:
        m = re.search(rf"{re.escape(key)}\s+\"([^\"]+)\"", block)
        return m.group(1).strip() if m else ""

    def _extract_iri_field(block: str, key: str) -> list[str]:
        results: list[str] = []
        for m in re.finditer(rf"{re.escape(key)}\s+(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_-]*:\S+)", block):
            results.append(_expand(m.group(1)))
        return results

    out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    n_bytes = 0
    with open(out, "wb") as fh:
        for blk in blocks:
            # Strip leading "### ..." comment lines to simplify parsing.
            blk_clean = re.sub(r"(?m)^\s*###.*$", "", blk).strip()
            if not blk_clean:
                continue
            # Only take blocks whose FIRST statement declares a Class
            # (this rejects ObjectProperty blocks that contain nested
            # anonymous owl:Class references).
            head = re.match(
                r"\s*(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_-]*:\S+)\s+(?:a|rdf:type)\s+(?:owl:Class|rdfs:Class)\b",
                blk_clean,
            )
            if not head:
                continue
            subj = _expand(head.group(1))
            label = _extract_field(blk_clean, "rdfs:label") or _extract_field(blk_clean, "skos:prefLabel")
            comment = (
                _extract_field(blk_clean, "rdfs:comment")
                or _extract_field(blk_clean, "skos:definition")
                or _extract_field(blk_clean, "skos:scopeNote")
            )
            parents = _extract_iri_field(blk_clean, "rdfs:subClassOf")
            name = label or _iri_label(subj)

            lines: list[str] = []
            if parents:
                parent_names = ", ".join(_iri_label(p) for p in parents[:5])
                lines.append(
                    f"In the {source_name} ontology, {name} is a subclass of {parent_names}."
                )
            else:
                lines.append(
                    f"In the {source_name} ontology, {name} is a top-level class."
                )
            if comment:
                lines.append(f"{name}: {comment[:400]}")
            lines.append(f"The full IRI for {name} is <{subj}>.")
            doc = "\n".join(lines).encode("utf-8", errors="replace")
            fh.write(doc)
            fh.write(DOC_DELIM)
            n_lines += 1
            n_bytes += len(doc) + 1
    log.info("  %s: %d definitions, %.2f MB", source_name, n_lines, n_bytes / 2**20)
    return n_bytes


# ── Orchestrator ────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ontology-dir", type=Path, default=DEFAULT_ONTOLOGY_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_PROSE_DIR)
    args = ap.parse_args()

    total = 0
    jobs = [
        (args.ontology_dir / "schemaorg.jsonld", args.out_dir / "schemaorg.txt",
         lambda i, o: _render_schemaorg(i, o)),
        (args.ontology_dir / "dbpedia.owl", args.out_dir / "dbpedia.txt",
         lambda i, o: _render_owl_xml(i, "DBpedia", o)),
        (args.ontology_dir / "bfo.owl", args.out_dir / "bfo.txt",
         lambda i, o: _render_owl_xml(i, "BFO", o)),
        (args.ontology_dir / "cco.ttl", args.out_dir / "cco.txt",
         lambda i, o: _render_turtle(i, "CCO", o)),
    ]

    for inp, outp, fn in jobs:
        if not inp.exists():
            log.warning("SKIP (missing): %s", inp)
            continue
        if outp.exists() and outp.stat().st_size > 0:
            log.info("SKIP (already generated): %s (%.2f MB)",
                     outp.name, outp.stat().st_size / 2**20)
            total += outp.stat().st_size
            continue
        log.info("Generating %s from %s ...", outp.name, inp.name)
        total += fn(inp, outp)

    log.info("=" * 60)
    log.info("Done. Total prose: %.2f MB", total / 2**20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
