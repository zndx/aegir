#!/usr/bin/env python
"""Emit the relational CTA/SKOS answer key for the published sdg-corpora collections.

One row per column of every collection table. ``reference_code`` is a recorded
link into ``vocabulary/annotations.csv`` — the construct's column concept, or
the entity's genus when the column identifies that entity — and is empty when
no such link exists. Codes are never invented and never taken from a fuzzy
name match against the vocabulary.

    uv run python scripts/emit_column_answer_key.py
    uv run python scripts/emit_column_answer_key.py --check
    uv run python scripts/emit_column_answer_key.py --check --run <run_dir>

Writes ``corpora/corpus/tagging/answer_key.csv``, ``answer_key.parquet``, and
``summary.json``. ``--check`` validates the committed key against collection
manifests and the vocabulary. Pass ``--run`` with ``--check`` to also require
a fresh derivation to match the file.

The default run is the construct snapshot the published collections were
projected from. Published columns are a subset of the construct (the
projection drops some columns); every published column must still be recorded
there. A blind classifier must not read ``corpus/tagging/``. Downstream
training and scoring join on ``(collection, table, column)``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPORA = REPO / "corpora"
COLLECTIONS = CORPORA / "corpus" / "collections"
VOCAB_CSV = CORPORA / "vocabulary" / "annotations.csv"
OUT_DIR = CORPORA / "corpus" / "tagging"
DEFAULT_RUN = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06")

# BFO/CCO genus IRI → SKOS upper anchor. Same bridge Atelier uses when it
# loads a collection. Coined ``sdg:`` genera are not in this map; they resolve
# only when the local name is itself a vocabulary abbrev or label.
GENUS_TO_ANCHOR = {
    "bfo:0000015": "SDG.PROCESS",
    "bfo:0000004": "SDG.INDEPENDENT_CONTINUANT",
    "bfo:0000040": "SDG.MATERIAL_ENTITY",
    "bfo:0000031": "SDG.GDC",
    "bfo:0000019": "SDG.QUALITY",
    "bfo:0000023": "SDG.ROLE",
    "bfo:0000016": "SDG.DISPOSITION",
    "bfo:0000002": "SDG.INDEPENDENT_CONTINUANT",
    "cco:ont00000995": "SDG.ARTIFACT",
    "cco:ont00000958": "SDG.ICE",
    "cco:ont00000853": "SDG.ICE.DESCRIPTIVE",
    "cco:ont00000965": "SDG.ICE.DIRECTIVE",
    "cco:ont00000686": "SDG.ICE.DESIGNATIVE",
}

COLUMNS = [
    "collection", "construct_id", "table", "column", "ordinal",
    "role", "concept", "xsd", "references_table", "references_column",
    "entity", "genus", "reference_code", "reference_label", "ref_basis",
]
# Generator lifecycle plumbing. Domain attributes such as creation_date stay
# attribute columns; only these recorded names are audit timestamps.
AUDIT_NAMES = {"created", "updated", "created_at", "updated_at", "createdAt", "updatedAt"}


def snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")


def scream(name: str) -> str:
    return snake(name).upper()


def table_aliases(entity_name: str) -> set[str]:
    s = snake(entity_name)
    aliases = {s, s + "s", s + "es"}
    if len(s) > 1 and s.endswith("y") and s[-2] not in "aeiou":
        aliases.add(s[:-1] + "ies")
    return {a for a in aliases if a}


def load_vocab(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            if code:
                rows[code] = row
    if not rows:
        raise SystemExit(f"no vocabulary codes in {path}")
    return rows


def vocab_indexes(vocab: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    by_abbrev: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for code, row in vocab.items():
        abbrev = (row.get("abbrev") or "").strip().upper()
        if abbrev:
            by_abbrev.setdefault(abbrev, code)
        label = scream(row.get("label") or "")
        if label:
            by_label.setdefault(label, code)
    return by_abbrev, by_label


def lookup(name: str, vocab: dict[str, dict], by_abbrev: dict[str, str],
           by_label: dict[str, str]) -> str:
    """Exact vocabulary hit for a recorded concept. Empty when there is none.

    Domain-member leaves (``SDG.DOM.<CONCEPT>``) win over a same-named upper
    abbrev. No common-name scan: those strings are surface forms, not column keys.
    """
    if not name:
        return ""
    key = scream(name)
    if not key:
        return ""
    dom = f"SDG.DOM.{key}"
    row = vocab.get(dom)
    if row and (row.get("common_names") or "") == "domain_member":
        return dom
    return by_abbrev.get(key) or by_label.get(key) or ""


def _pk_names(table: dict) -> list[str]:
    pk = table.get("pk")
    if isinstance(pk, list):
        return [str(x) for x in pk if x]
    if pk:
        return [str(pk)]
    return []


def _entity_for_table(table: dict, entities: dict[str, dict]) -> str:
    cols = {c.get("name"): c for c in table.get("columns") or []}
    pks = _pk_names(table)
    if len(pks) == 1:
        concept = (cols.get(pks[0]) or {}).get("concept") or ""
        if concept in entities:
            return concept
    hits = [name for name in entities if table.get("name") in table_aliases(name)]
    if len(hits) == 1:
        return hits[0]
    return ""


def _resolve_entity(entity_name: str, entity: dict | None, concept: str,
                    vocab: dict, by_abbrev: dict, by_label: dict,
                    *, via_fk: bool) -> tuple[str, str]:
    """(code, basis) for a column that designates an entity."""
    tried: set[str] = set()
    for candidate in (entity_name, (entity or {}).get("label") or "", concept):
        if not candidate or candidate in tried:
            continue
        tried.add(candidate)
        code = lookup(candidate, vocab, by_abbrev, by_label)
        if code:
            return code, ("fk-target" if via_fk else "entity-leaf")
    genus = (entity or {}).get("genus") or ""
    anchor = GENUS_TO_ANCHOR.get(genus, "")
    if anchor and anchor in vocab:
        return anchor, "genus-anchor"
    local = genus.split(":")[-1] if ":" in genus else ""
    if local and local not in tried:
        code = lookup(local, vocab, by_abbrev, by_label)
        if code:
            return code, "genus-name"
    return "", "unresolved"


def build_rows(collections: Path, run: Path, vocab: dict[str, dict]) -> list[dict]:
    by_abbrev, by_label = vocab_indexes(vocab)
    rows: list[dict] = []
    for cdir in sorted(p for p in collections.iterdir() if p.is_dir()):
        manifest_path = cdir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cid = manifest.get("construct_id") or ""
        construct_path = run / "constructs" / f"{cid}.json"
        entity_path = run / "entities" / f"{cid}.json"
        if not construct_path.is_file():
            raise SystemExit(f"{cdir.name}: no construct {construct_path.name} under {run}")
        construct = json.loads(construct_path.read_text(encoding="utf-8"))
        entities: dict[str, dict] = {}
        if entity_path.is_file():
            for ent in json.loads(entity_path.read_text(encoding="utf-8")).get("entities") or []:
                if ent.get("name"):
                    entities[ent["name"]] = ent
        published = {t["name"]: t.get("columns") or [] for t in manifest.get("tables") or []}
        constructed = {t.get("name"): t for t in construct.get("tables") or []}
        # The projection may drop construct columns (enum-ish extras, degenerate
        # FKs). The key covers the published table, and every published column
        # must still be a recorded construct column.
        missing_tables = set(published) - set(constructed)
        if missing_tables:
            raise SystemExit(f"{cdir.name}: published tables absent from the construct: {sorted(missing_tables)}")
        table_entity = {name: _entity_for_table(t, entities) for name, t in constructed.items()}
        for spec in manifest.get("tables") or []:
            tname = spec["name"]
            table = constructed[tname]
            by_name = {c.get("name"): c for c in table.get("columns") or []}
            missing_cols = [c for c in spec.get("columns") or [] if c not in by_name]
            if missing_cols:
                raise SystemExit(f"{cdir.name}.{tname}: published columns absent from the construct: {missing_cols}")
            fks = {f.get("col"): f for f in table.get("fks") or []}
            pks = set(_pk_names(table))
            owner_name = table_entity.get(tname) or ""
            owner = entities.get(owner_name) or {}
            attr_xsd = {a.get("name"): a.get("xsd") or "" for a in owner.get("attributes") or []}
            for ordinal, cname in enumerate(spec.get("columns") or []):
                col = by_name[cname]
                concept = col.get("concept") or ""
                fk = fks.get(cname)
                if fk:
                    role = "sdg:ForeignKeyColumn"
                    target = fk.get("ref_table") or ""
                    ent_name = table_entity.get(target) or ""
                    ent = entities.get(ent_name) or {}
                    code, basis = _resolve_entity(
                        ent_name, ent, "", vocab, by_abbrev, by_label, via_fk=True)
                    genus = ent.get("genus") or ""
                    xsd = ""
                    ref_table, ref_col = target, fk.get("ref_col") or ""
                else:
                    ent_name, ent, genus = owner_name, owner, owner.get("genus") or ""
                    ref_table = ref_col = ""
                    if cname in AUDIT_NAMES or concept in AUDIT_NAMES:
                        role = "sdg:AuditTimestampColumn"
                    elif cname in pks:
                        role = "sdg:SurrogateIdentifierColumn"
                    else:
                        role = "sdg:AttributeColumn"
                    if role == "sdg:SurrogateIdentifierColumn":
                        code, basis = _resolve_entity(
                            ent_name, ent, concept, vocab, by_abbrev, by_label, via_fk=False)
                        xsd = ""
                    else:
                        code = lookup(concept, vocab, by_abbrev, by_label) or lookup(
                            cname, vocab, by_abbrev, by_label)
                        basis = "leaf" if code else "unresolved"
                        xsd = attr_xsd.get(concept) or ""
                rows.append({
                    "collection": cdir.name,
                    "construct_id": cid,
                    "table": tname,
                    "column": cname,
                    "ordinal": ordinal,
                    "role": role,
                    "concept": concept,
                    "xsd": xsd,
                    "references_table": ref_table,
                    "references_column": ref_col,
                    "entity": ent_name,
                    "genus": genus,
                    "reference_code": code,
                    "reference_label": (vocab.get(code) or {}).get("label") or "",
                    "ref_basis": basis,
                })
    if not rows:
        raise SystemExit(f"no columns under {collections}")
    return rows


def validate(rows: list[dict], collections: Path, vocab: dict[str, dict]) -> None:
    """Every published column is keyed once, and every code is in the vocabulary."""
    seen: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (row["collection"], row["table"], row["column"])
        if key in seen:
            raise SystemExit(f"duplicate answer-key row {key}")
        code = row.get("reference_code") or ""
        if code and code not in vocab:
            raise SystemExit(f"{key}: reference_code {code} is not in the vocabulary")
        if code and row.get("reference_label") != (vocab[code].get("label") or ""):
            raise SystemExit(f"{key}: reference_label does not match {code}")
        if not code and row.get("ref_basis") != "unresolved":
            raise SystemExit(f"{key}: empty code with basis {row.get('ref_basis')}")
        if code and row.get("ref_basis") == "unresolved":
            raise SystemExit(f"{key}: code set with basis unresolved")
        seen[key] = row
    n_manifest = 0
    for cdir in sorted(p for p in collections.iterdir() if p.is_dir()):
        manifest_path = cdir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for spec in manifest.get("tables") or []:
            for ordinal, cname in enumerate(spec.get("columns") or []):
                n_manifest += 1
                key = (cdir.name, spec["name"], cname)
                row = seen.get(key)
                if row is None:
                    raise SystemExit(f"missing answer-key row {key}")
                if int(row["ordinal"]) != ordinal:
                    raise SystemExit(f"{key}: ordinal {row['ordinal']} != {ordinal}")
    extras = len(seen) - n_manifest
    if extras:
        raise SystemExit(f"{extras} answer-key rows are not in any collection manifest")


def summarize(rows: list[dict], vocab: dict[str, dict], run_label: str) -> dict:
    by_role = Counter(r["role"] for r in rows)
    by_basis = Counter(r["ref_basis"] for r in rows)
    gaps = Counter(r["concept"] or r["column"] for r in rows if r["ref_basis"] == "unresolved")
    return {
        "n_collections": len({r["collection"] for r in rows}),
        "n_columns": len(rows),
        "n_with_code": sum(1 for r in rows if r["reference_code"]),
        "n_unresolved": sum(1 for r in rows if not r["reference_code"]),
        "by_role": dict(sorted(by_role.items())),
        "by_basis": dict(sorted(by_basis.items())),
        "n_unresolved_concepts": len(gaps),
        "unresolved_concepts_top": [
            {"concept": c, "n": n} for c, n in gaps.most_common(40)
        ],
        "source_run": run_label,
        "vocab_codes": len(vocab),
    }


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "answer_key.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in COLUMNS})
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        k: pa.array(
            [int(r[k]) if k == "ordinal" else r[k] for r in rows],
            type=pa.int32() if k == "ordinal" else pa.string(),
        )
        for k in COLUMNS
    })
    pq.write_table(table, out_dir / "answer_key.parquet")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def check_parquet(path: Path, n_rows: int) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    import pyarrow.parquet as pq
    got = pq.read_table(path).num_rows
    if got != n_rows:
        raise SystemExit(f"{path} has {got} rows; csv has {n_rows}")


def read_key(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty answer key {path}")
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path} missing columns {missing}")
    return rows


def _same(got: list[dict], fresh: list[dict]) -> None:
    if len(got) != len(fresh):
        raise SystemExit(f"committed key has {len(got)} rows; derivation has {len(fresh)}")
    for i, (a, b) in enumerate(zip(got, fresh)):
        for col in COLUMNS:
            if str(a[col]) != str(b[col]):
                raise SystemExit(
                    f"row {i} {a['collection']}.{a['table']}.{a['column']} "
                    f"{col}: committed {a[col]!r} != derived {b[col]!r}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="corpus run dir with constructs/ and entities/ (default corpus-v06)")
    ap.add_argument("--collections", type=Path, default=COLLECTIONS)
    ap.add_argument("--vocab", type=Path, default=VOCAB_CSV)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="validate the committed key; with --run, also require it to match a fresh derivation")
    args = ap.parse_args()
    # Distinguish "default run" from "caller passed --run" so --check alone
    # does not require the construct snapshot to be mounted.
    run_passed = any(a == "--run" or a.startswith("--run=") for a in sys.argv[1:])

    vocab = load_vocab(args.vocab)
    if args.check and not run_passed:
        rows = read_key(args.out / "answer_key.csv")
        validate(rows, args.collections, vocab)
        check_parquet(args.out / "answer_key.parquet", len(rows))
        print(f"answer key OK — {len(rows)} columns, "
              f"{sum(1 for r in rows if r['reference_code'])} coded")
        return 0

    if not (args.run / "constructs").is_dir():
        raise SystemExit(f"no constructs/ under {args.run}")
    fresh = build_rows(args.collections, args.run, vocab)
    validate(fresh, args.collections, vocab)
    if args.check:
        committed = read_key(args.out / "answer_key.csv")
        _same(committed, fresh)
        check_parquet(args.out / "answer_key.parquet", len(fresh))
        print(f"answer key matches derivation — {len(fresh)} columns")
        return 0
    summary = summarize(fresh, vocab, args.run.name)
    write_outputs(fresh, summary, args.out)
    print(json.dumps({k: summary[k] for k in
                      ("n_collections", "n_columns", "n_with_code", "n_unresolved", "by_basis")},
                     indent=2))
    print(f"wrote {args.out / 'answer_key.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
