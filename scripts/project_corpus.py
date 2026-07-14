#!/usr/bin/env python
"""Project a corpus run into ``corpora/corpus/collections/`` — the CURRENT construct-based release form.

This REPLACES the vestigial topic-era projection (``build_collections.py`` — topic-NNN-family bundles keyed to
a coverage audit, template-era ``ch_live_*`` chapters, raw FinePDFs excerpts in the card). The live pipeline
(``SdgCorporaFlow``) is PASSAGE-based: each *construct* is derived from one harvested passage and written into
a natural + a semantic chapter. So a COLLECTION here = one construct:

    corpus/collections/<slug>/
      README.md              collection card (entities · tables · chapters · grounding)
      chapters/natural.md    prose + embedded real-value tables & views (the training text)
      chapters/semantic.md
      tables/<name>.sql      CREATE TABLE for each underlying table (types inferred from the real cells)
      tables/<name>.csv      the table's real-value rows (the relational Data Product, explicit)
      terms.md               the ontology terms grounding the chapter (entity · BFO/CCO genus · definition)
      manifest.json          cross-links (construct → passage(basename) → entities/terms → tables → chapters)
    corpus/collections/INDEX.md

Values are real GitTables particulars with retained lineage (provenance ≻ exclusion; see the
``provenance/gittables_value_sampling`` strategy component). The source passage is referenced by content-hash
basename only — never its text (that is licensed FinePDFs material, not ours to redistribute).

Deterministic; no LLM / GPU / network.

    uv run --no-sync python scripts/project_corpus.py --run <run_dir> --out corpora/corpus/collections
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _kebab(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "-", name)          # camelCase → words
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "collection"


def _infer_sql_type(values: "list[str]") -> str:
    """A faithful column type from the REAL cell values (the embedded data is the source of truth)."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return "VARCHAR(255)"
    def all_match(pat):
        return all(re.fullmatch(pat, str(v)) for v in vals)
    if all(str(v).lower() in ("true", "false") for v in vals):
        return "BOOLEAN"
    if all_match(r"-?\d+"):
        return "INTEGER"
    if all_match(r"-?\d+\.\d+"):
        return "DECIMAL"
    if all_match(r"\d{4}-\d{2}-\d{2}"):
        return "DATE"
    if all_match(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?"):
        return "TIMESTAMP"
    w = max((len(str(v)) for v in vals), default=32)
    return f"VARCHAR({min(512, max(32, w + 8))})"


def _table_columns(t: dict) -> "list[tuple[str, list[str]]]":
    return [(c["name"], [x.get("value", "") for x in c.get("cells", [])]) for c in t.get("columns", [])]


def _render_create(t: dict) -> str:
    cols = _table_columns(t)
    pk = set(t.get("pk", []) if isinstance(t.get("pk"), list) else ([t["pk"]] if t.get("pk") else []))
    fk_by_col = {f["col"]: f for f in t.get("fks", [])}
    lines = []
    for name, vals in cols:
        typ = _infer_sql_type(vals)
        parts = [f"  {name} {typ}"]
        if name in pk:
            parts.append("NOT NULL")
        lines.append(" ".join(parts))
    if pk:
        lines.append(f"  PRIMARY KEY ({', '.join(c for c, _ in cols if c in pk)})")
    for col, f in fk_by_col.items():
        lines.append(f"  FOREIGN KEY ({col}) REFERENCES {f['ref_table']} ({f.get('ref_col', 'id')})")
    return f"CREATE TABLE {t['name']} (\n" + ",\n".join(lines) + "\n);\n"


def _write_table_csv(t: dict, path: Path) -> int:
    cols = _table_columns(t)
    names = [n for n, _ in cols]
    nrows = max((len(v) for _, v in cols), default=0)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(nrows):
            w.writerow([(v[i] if i < len(v) else "") for _, v in cols])
    return nrows


# DEFINITION-HYGIENE MEMBRANE — genericize real-COMPANY particulars an LLM-authored definition over-specified
# with (a Server is not specifically 'IBM i'; a database is not specifically 'Oracle'). Real UNIVERSALS stay
# (Linux, HTTP, ISO — nomenclature). This is the release-form analog of the sensitive gate: values are real
# WITH GitTables lineage, but a *definition* naming a non-provenanced company is a quality + policy defect.
# Ordered longest-match-first so multi-word products resolve before the bare brand token.
_DEF_GENERIC = [
    (re.compile(r"\bIBM\s+i\b", re.I), "enterprise"), (re.compile(r"\bIBM\b", re.I), "enterprise"),
    (re.compile(r"\bOracle\s+database\b", re.I), "relational database"),
    (re.compile(r"\bOracle\b", re.I), "relational-database"),
    (re.compile(r"\bMicrosoft\s+Word\b", re.I), "word-processing software"),
    (re.compile(r"\bMicrosoft\s+Excel\b", re.I), "spreadsheet software"),
    (re.compile(r"\bMicrosoft\b", re.I), "commercial"),
]


def genericize_definition(text: str) -> str:
    """Strip non-provenanced company brands from a generated definition, leaving the generic sense intact."""
    for pat, repl in _DEF_GENERIC:
        text = pat.sub(repl, text)
    return re.sub(r"\s{2,}", " ", text).replace(" ,", ",").strip()


def _terms_md(entities: "list[dict]") -> str:
    """The ontology terms grounding the collection — entity, its BFO/CCO genus, and definition."""
    lines = ["# Grounding terms", "",
             "The ontology universals this collection's chapters are written around "
             "(each an sdg: class, grounded in a BFO/CCO genus).", ""]
    for e in entities:
        lines.append(f"## {genericize_definition(e.get('label') or e.get('name'))}")
        lines.append(f"- **genus**: `{e.get('genus', '')}`")
        if e.get("definition"):
            lines.append(f"- **definition**: {genericize_definition(e['definition'])}")
        attrs = e.get("attributes") or []
        if attrs:
            lines.append(f"- **attributes**: {', '.join(a['name'] for a in attrs)}")
        rels = e.get("relations") or []
        if rels:
            lines.append(f"- **relations**: {', '.join(r['prop'] + '→' + r['target'] for r in rels)}")
        lines.append("")
    return "\n".join(lines)


def _slug_for(cid: str, entities: "list[dict]", taken: "set[str]") -> str:
    base = _kebab(entities[0]["name"]) if entities else "collection"
    slug = f"{base}-{cid[:8]}"
    n = 2
    while slug in taken:
        slug = f"{base}-{cid[:8]}-{n}"; n += 1
    taken.add(slug)
    return slug


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="corpus run dir (constructs/, chapters/, entities/)")
    ap.add_argument("--out", default=str(REPO / "corpora" / "corpus" / "collections"))
    ap.add_argument("--registers", default="natural,semantic")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out)
    registers = [r.strip() for r in a.registers.split(",") if r.strip()]

    cons_dir, chap_dir, ent_dir = run / "constructs", run / "chapters", run / "entities"
    if not cons_dir.is_dir():
        raise SystemExit(f"no constructs/ under {run}")

    # WHOLESALE replace the vestigial topic-era collections (topic-NNN-family + ch_live_* + FinePDFs excerpts).
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    taken: "set[str]" = set()
    index: "list[dict]" = []
    n_tables = n_chapters = 0
    for cj in sorted(cons_dir.glob("*.json")):
        cid = cj.stem
        con = json.loads(cj.read_text())
        entities = []
        passage_ref = ""
        ej = ent_dir / f"{cid}.json"
        if ej.exists():
            edoc = json.loads(ej.read_text())
            entities = edoc.get("entities", [])
            passage_ref = Path(edoc.get("passage", "")).name       # content-hash basename ONLY, never the text
        slug = _slug_for(cid, entities, taken)
        cdir = out / slug
        (cdir / "chapters").mkdir(parents=True)
        (cdir / "tables").mkdir(parents=True)

        chapters = []
        for reg in registers:
            src = chap_dir / cid / f"{reg}.md"
            if src.exists():
                shutil.copy2(src, cdir / "chapters" / f"{reg}.md")
                chapters.append(f"{reg}.md"); n_chapters += 1

        tables = []
        for t in con.get("tables", []):
            (cdir / "tables" / f"{t['name']}.sql").write_text(_render_create(t))
            rows = _write_table_csv(t, cdir / "tables" / f"{t['name']}.csv")
            tables.append({"name": t["name"], "rows": rows, "columns": [c["name"] for c in t.get("columns", [])]})
            n_tables += 1

        (cdir / "terms.md").write_text(_terms_md(entities))
        views = [m.split(":", 1)[1] for m in con.get("payload_markers", []) if m.startswith("VIEW:")]
        manifest = {"construct_id": cid, "slug": slug, "passage": passage_ref,
                    "registers": chapters, "n_tables": len(tables), "n_views": len(views),
                    "entities": [e.get("name") for e in entities],
                    "terms": [{"name": e.get("name"), "genus": e.get("genus")} for e in entities],
                    "tables": tables, "views": views}
        (cdir / "manifest.json").write_text(json.dumps(manifest, indent=1))

        title = genericize_definition(str((entities[0].get("label") or entities[0].get("name") or slug)
                                          if entities else slug))
        readme = [f"# Collection — {title}", "",
                  f"Derived from one harvested passage (`{passage_ref or 'n/a'}`). "
                  f"**{len(chapters)} chapters** · **{len(tables)} tables** · **{len(views)} views** · "
                  f"**{len(entities)} ontology terms**.", "",
                  "Table values are real GitTables particulars with retained lineage "
                  "(provenance ≻ exclusion). The source passage is referenced by content-hash only.", "",
                  "## Chapters", ""]
        readme += [f"- [{c}](chapters/{c})" for c in chapters]
        readme += ["", "## Underlying tables", ""]
        readme += [f"- `{t['name']}` ({t['rows']} rows) — "
                   f"[SQL](tables/{t['name']}.sql) · [CSV](tables/{t['name']}.csv)" for t in tables]
        readme += ["", "## Grounding", "", "See [terms.md](terms.md).", ""]
        (cdir / "README.md").write_text("\n".join(readme))

        index.append({"slug": slug, "title": title, "chapters": len(chapters),
                      "tables": len(tables), "terms": len(entities)})

    idx = ["# sdg-corpora — collections", "",
           f"{len(index)} collections (one per derived construct) · {n_chapters} chapters · {n_tables} tables. "
           "Each collection bundles a construct's natural + semantic chapters (prose with embedded real-value "
           "tables and join views), the underlying relational tables (DDL + real-value rows), and the ontology "
           "terms grounding them. Values are real GitTables particulars with retained lineage.", ""]
    for e in sorted(index, key=lambda r: r["slug"]):
        idx.append(f"- [{e['title']}]({e['slug']}/README.md) — {e['chapters']} ch · "
                   f"{e['tables']} tables · {e['terms']} terms")
    (out / "INDEX.md").write_text("\n".join(idx))

    print(f"  projected {len(index)} collections · {n_chapters} chapters · {n_tables} tables → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
