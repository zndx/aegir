#!/usr/bin/env python
"""Emit the relational CTA/SKOS answer key for the published sdg-corpora collections.

One row per column of every collection table. The tables were realized from a
per-construct OWL document, so each column's association is that record:

* ``class_iri`` — the OWL class the column is grounded in (the table's realized
  class, or the referenced class for a foreign key)
* ``property_iri`` — the OWL property the generator assigned (empty for a
  surrogate identifier, and for an audit timestamp, which asserts no property)
* ``range_iri`` — the Manchester restriction filler (``xsd:*`` or a class); for
  a foreign key or a surrogate identifier, the class the values denote
* ``reference_code`` — the SKOS concept that grounding entails, and only a code
  already in ``vocabulary/annotations.csv``

A foreign key or surrogate identifier denotes its class, so the code is that
class's vocabulary concept, else the SKOS anchor entailed by its recorded
genus. A coined ``sdg:`` genus is itself a class in the published ontology;
the code follows that class's recorded ``SubClassOf`` parent until a BFO/CCO
anchor (``genus-walk`` / ``domain-walk``). An attribute uses the same chain
for its domain class when the property is not itself a vocabulary concept.
An audit timestamp asserts no property, and it carries the domain class's
code when that class has an anchor. ``SDG.GENERIC`` is the vocabulary's
anchorless bucket — its children are not one structural kind — so it is
never written as a stand-in. A column receives a ``SDG.GENERIC.*`` code only
when its recorded name is that concept and the genus already entails nothing
richer.

A genus the published ontology does not place under a specific anchor is
tagged with an authored column kind (``kind-authored``) when
``column_kinds.omn`` covers that class. The kind is an OWL subclass of a
BFO or CCO class, and the code is the kind, so an unseen entity of the same
kind aligns to it. The tag is applied only when the recorded genus itself
has no anchor. A leaf concept such as jurisdiction stays. A class whose
genus already entails an anchor is left on that anchor. A published class
with no anchor and no authored kind is an error.

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
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.column_kinds import unanchored_kind  # noqa: E402
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
    "role", "concept", "xsd",
    "class_iri", "property_iri", "range_iri",
    "references_table", "references_column",
    "entity", "genus", "reference_code", "reference_label", "ref_basis",
]
# Generator lifecycle plumbing. Domain attributes such as creation_date stay
# attribute columns; only these recorded names are audit timestamps.
AUDIT_NAMES = {"created", "updated", "created_at", "updated_at", "createdAt", "updatedAt"}
# Bases that carry no SKOS code. The OWL fields are still filled.
NO_CODE_BASES = {"unresolved", "genus-unmapped", "audit-unclaimed"}
_RESTRICTION = re.compile(
    r"(sdg:\w+)\s+(?:some|only|exactly\s+\d+|min\s+\d+|max\s+\d+)\s+(\S+)"
)


def camel(name: str) -> str:
    """Match ``aegir.ontology.entities.camel`` so class IRIs agree with the omn."""
    parts = re.split(r"[^A-Za-z0-9]+", (name or "").strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def prop_name(name: str) -> str:
    """Match ``aegir.ontology.entities.prop_name`` so property IRIs agree with the omn."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", (name or "").strip()) if p]
    if not parts:
        return ""
    head = parts[0][:1].lower() + parts[0][1:]
    rest = "".join(p[:1].upper() + p[1:] for p in parts[1:])
    return head + rest


def snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")


def scream(name: str) -> str:
    return snake(name).upper()


def table_aliases(entity_name: str) -> set[str]:
    s = snake(entity_name)
    aliases = {s, s + "s", s + "es", entity_name, camel(entity_name)}
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
        # Authored kinds are assigned by class IRI once the genus walk has
        # no anchor. Indexing their abbrevs would retag a column that shares
        # a name, such as Facility, and whose genus already entails an anchor.
        if (row.get("family") or "") == "column-kinds":
            continue
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


def _entity_named(local: str, entities: dict[str, dict]) -> str:
    if local in entities:
        return local
    for name in entities:
        if camel(name) == local:
            return name
    return ""


def _plan_entities(construct: dict, entities: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """Table name → (entity name, class IRI) from the construct key plan.

    The plan is the generator's table←class record, including prefixed names
    (``clie_client_engagements`` → ``sdg:ClientEngagement``) that a snake-plural
    alias of the entity does not reach. Junction tables are not in the plan.
    """
    out: dict[str, tuple[str, str]] = {}
    plans = (construct.get("key_plan") or {}).get("plans") or {}
    for iri, plan in plans.items():
        table = plan.get("table") or ""
        local = str(iri).split(":")[-1]
        name = _entity_named(local, entities)
        if table and name:
            out[table] = (name, str(iri))
    return out


def _entity_for_table(table: dict, entities: dict[str, dict]) -> str:
    """Fallback when a published table is absent from the key plan."""
    cols = {c.get("name"): c for c in table.get("columns") or []}
    pks = _pk_names(table)
    if len(pks) == 1:
        concept = (cols.get(pks[0]) or {}).get("concept") or ""
        named = _entity_named(camel(concept), entities) if concept else ""
        if named:
            return named
        if concept in entities:
            return concept
    hits = [name for name in entities if table.get("name") in table_aliases(name)]
    if len(hits) == 1:
        return hits[0]
    return ""


def _parse_omn(omn: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Class IRI → ({property IRI: filler}, genus) from each SubClassOf frame."""
    restrictions: dict[str, dict[str, str]] = {}
    genera: dict[str, str] = {}
    for part in re.split(r"(?m)^Class:\s+", omn or "")[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        head = lines[0].strip()
        restr: dict[str, str] = {}
        genus = ""
        for i, line in enumerate(lines[1:], start=1):
            if "SubClassOf:" not in line:
                continue
            text = line.split("SubClassOf:", 1)[1]
            for nxt in lines[i + 1:]:
                if not nxt.startswith((" ", "\t")):
                    break
                if re.match(r"\s*(Annotations|SubClassOf|EquivalentTo|HasKey|Characteristics):", nxt):
                    break
                text += " " + nxt.strip()
            for bit in text.split(","):
                bit = bit.strip().rstrip(".")
                matched = _RESTRICTION.match(bit)
                if matched:
                    restr[matched.group(1)] = matched.group(2).rstrip(".")
                elif bit and " " not in bit and not genus:
                    genus = bit
            break
        restrictions[head] = restr
        genera[head] = genus
    return restrictions, genera


_GENUS_PARENTS: dict[str, str] | None = None


def _first_bare_parent(header: str, body: list[str]) -> str:
    """The first class conjunct of a SubClassOf frame, including a same-line header."""
    chunks: list[str] = []
    if "SubClassOf:" in header:
        chunks.append(header.split("SubClassOf:", 1)[1])
    for i, line in enumerate(body):
        if "SubClassOf:" not in line:
            continue
        text = line.split("SubClassOf:", 1)[1]
        for nxt in body[i + 1:]:
            if not nxt.startswith((" ", "\t")):
                break
            if re.match(r"\s*(Annotations|EquivalentTo|HasKey|Characteristics):", nxt):
                break
            text += " " + nxt.strip()
        chunks.append(text)
        break
    for chunk in chunks:
        for bit in chunk.split(","):
            bit = bit.strip().rstrip(".")
            if bit and " " not in bit and not _RESTRICTION.match(bit):
                return bit
    return ""


def genus_parents() -> dict[str, str]:
    """Class IRI → recorded parent class in the published ontology.

    Coined genera such as ``sdg:EducationalEntity`` are classes there, with a
    ``SubClassOf`` BFO/CCO anchor. A later frame that restates the class without
    a parent does not erase one already read.
    """
    global _GENUS_PARENTS
    if _GENUS_PARENTS is not None:
        return _GENUS_PARENTS
    found: dict[str, str] = {}
    ont = CORPORA / "ontology"
    for name in ("sdg-ontology.omn", "sdg-ontology-comprehensive.omn"):
        path = ont / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for part in re.split(r"(?m)^Class:\s+", text)[1:]:
            lines = part.splitlines()
            if not lines:
                continue
            header = lines[0].strip()
            head = header.split()[0]
            if not head or head.startswith("<"):
                continue
            parent = _first_bare_parent(header, lines[1:])
            if parent and not found.get(head):
                found[head] = parent
    _GENUS_PARENTS = found
    return found


def _genus_skos(genus: str, vocab: dict, by_abbrev: dict, by_label: dict,
                *, domain: bool) -> tuple[str, str]:
    """SKOS concept entailed by a recorded genus. No code is coined."""
    anchor = GENUS_TO_ANCHOR.get(genus, "")
    if anchor and anchor in vocab:
        return anchor, ("domain-genus" if domain else "genus-anchor")
    local = genus.split(":")[-1] if ":" in (genus or "") else ""
    if local and genus.startswith("sdg:"):
        code = lookup(local, vocab, by_abbrev, by_label)
        if code:
            return code, ("domain-name" if domain else "genus-name")
    seen = {genus} if genus else set()
    current = genus
    parents = genus_parents()
    for _ in range(8):
        nxt = parents.get(current, "")
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        anchor = GENUS_TO_ANCHOR.get(nxt, "")
        if anchor and anchor in vocab:
            return anchor, ("domain-walk" if domain else "genus-walk")
        if nxt.startswith("sdg:"):
            hopped = nxt.split(":")[-1]
            code = lookup(hopped, vocab, by_abbrev, by_label)
            if code:
                return code, ("domain-name" if domain else "genus-name")
        current = nxt
    if genus:
        return "", "genus-unmapped"
    return "", "unresolved"


def _apply_authored_kind(row: dict, vocab: dict, by_abbrev: dict, by_label: dict) -> None:
    """Tag an unanchored class with its authored kind.

    The recorded genus is what decides. A genus that already entails an
    anchor stays on that anchor, including when the class name matches an
    example. A leaf concept stays. The kind code is the structural genus.
    """
    _code, genus_basis = _genus_skos(
        row.get("genus") or "", vocab, by_abbrev, by_label, domain=False)
    if genus_basis != "genus-unmapped":
        return
    if row.get("ref_basis") == "leaf":
        return
    kind = unanchored_kind(row.get("class_iri") or "")
    if not kind:
        return
    code = row.get("reference_code") or ""
    if row.get("ref_basis") in {"genus-unmapped", "entity-leaf", "fk-target"} or code.startswith("SDG.GENERIC"):
        row["reference_code"] = kind
        row["ref_basis"] = "kind-authored"


def _skos_for_class(entity_name: str, entity: dict | None,
                    vocab: dict, by_abbrev: dict, by_label: dict,
                    *, via_fk: bool) -> tuple[str, str]:
    """(code, basis) for a column that denotes a class."""
    tried: set[str] = set()
    for candidate in (entity_name, (entity or {}).get("label") or ""):
        if not candidate or candidate in tried:
            continue
        tried.add(candidate)
        code = lookup(candidate, vocab, by_abbrev, by_label)
        if code:
            return code, ("fk-target" if via_fk else "entity-leaf")
    return _genus_skos((entity or {}).get("genus") or "", vocab, by_abbrev, by_label, domain=False)


def _relation_iri(entities: dict[str, dict], owner: str, concept: str,
                  target_name: str) -> str:
    """Object-property IRI for a foreign key, when one relation records it.

    The non-property side of a junction stores the referenced table name as its
    concept and has no property of its own; that returns empty rather than a guess.
    """
    def candidates(rels: list) -> list[str]:
        iris = []
        for rel in rels:
            prop = prop_name(str(rel.get("prop") or ""))
            if concept and prop == concept:
                target = camel(str(rel.get("target") or ""))
                if not target_name or not target or target == camel(target_name):
                    iris.append("sdg:" + prop)
        return iris

    if owner and owner in entities:
        iris = candidates(entities[owner].get("relations") or [])
        if len(set(iris)) == 1:
            return iris[0]
        if len(set(iris)) > 1:
            return ""
    if not concept:
        return ""
    found: list[str] = []
    for ent in entities.values():
        found.extend(candidates(ent.get("relations") or []))
    uniq = set(found)
    if len(uniq) == 1:
        return next(iter(uniq))
    return ""


def _attribute(entity: dict | None, concept: str) -> dict | None:
    for attr in (entity or {}).get("attributes") or []:
        recorded = str(attr.get("name") or "")
        if recorded == concept or prop_name(recorded) == concept:
            return attr
    return None


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
        restrictions, omn_genus = _parse_omn(construct.get("ontology_omn") or "")
        planned = _plan_entities(construct, entities)
        published = {t["name"]: t.get("columns") or [] for t in manifest.get("tables") or []}
        constructed = {t.get("name"): t for t in construct.get("tables") or []}
        # The projection may drop construct columns (enum-ish extras, degenerate
        # FKs). The key covers the published table, and every published column
        # must still be a recorded construct column.
        missing_tables = set(published) - set(constructed)
        if missing_tables:
            raise SystemExit(f"{cdir.name}: published tables absent from the construct: {sorted(missing_tables)}")

        def resolve_table(tname: str) -> tuple[str, str]:
            if tname in planned:
                return planned[tname]
            table = constructed.get(tname) or {}
            name = _entity_for_table(table, entities)
            iri = f"sdg:{camel(name)}" if name else ""
            return name, iri

        for spec in manifest.get("tables") or []:
            tname = spec["name"]
            table = constructed[tname]
            by_name = {c.get("name"): c for c in table.get("columns") or []}
            missing_cols = [c for c in spec.get("columns") or [] if c not in by_name]
            if missing_cols:
                raise SystemExit(
                    f"{cdir.name}.{tname}: published columns absent from the construct: {missing_cols}")
            fks = {f.get("col"): f for f in table.get("fks") or []}
            pks = set(_pk_names(table))
            owner_name, owner_iri = resolve_table(tname)
            owner = entities.get(owner_name) or {}
            if owner_name and not owner.get("genus"):
                owner = {**owner, "genus": omn_genus.get(owner_iri, "")}
            for ordinal, cname in enumerate(spec.get("columns") or []):
                col = by_name[cname]
                concept = col.get("concept") or ""
                fk = fks.get(cname)
                if fk:
                    role = "sdg:ForeignKeyColumn"
                    target_table = fk.get("ref_table") or ""
                    target_name, target_iri = resolve_table(target_table)
                    target = entities.get(target_name) or {}
                    if target_name and not target.get("genus"):
                        target = {**target, "genus": omn_genus.get(target_iri, "")}
                    code, basis = _skos_for_class(
                        target_name, target, vocab, by_abbrev, by_label, via_fk=True)
                    prop = _relation_iri(entities, owner_name, concept, target_name)
                    # A stored property_iri (normalized constructs) wins over the
                    # reconstructed relation when the concept no longer matches.
                    stored = col.get("property_iri") or ""
                    if stored:
                        prop = stored
                    class_iri = col.get("references_class") or target_iri
                    row = {
                        "role": role, "concept": concept, "xsd": "",
                        "class_iri": class_iri, "property_iri": prop, "range_iri": class_iri,
                        "references_table": target_table,
                        "references_column": fk.get("ref_col") or "",
                        "entity": target_name, "genus": target.get("genus") or "",
                        "reference_code": code, "ref_basis": basis,
                    }
                elif cname in AUDIT_NAMES or concept in AUDIT_NAMES:
                    code, basis = _genus_skos(
                        owner.get("genus") or "", vocab, by_abbrev, by_label, domain=True)
                    row = {
                        "role": "sdg:AuditTimestampColumn", "concept": concept, "xsd": "",
                        "class_iri": owner_iri, "property_iri": "", "range_iri": "",
                        "references_table": "", "references_column": "",
                        "entity": owner_name, "genus": owner.get("genus") or "",
                        "reference_code": code, "ref_basis": basis,
                    }
                elif (len(pks) == 1 and cname in pks and owner_name
                      and (concept in entities or camel(concept) == camel(owner_name))):
                    class_iri = col.get("of_class") or owner_iri
                    denoted = owner_name if camel(concept) == camel(owner_name) else (
                        concept if concept in entities else owner_name)
                    denoted_ent = entities.get(denoted) or owner
                    code, basis = _skos_for_class(
                        denoted, denoted_ent, vocab, by_abbrev, by_label, via_fk=False)
                    row = {
                        "role": "sdg:SurrogateIdentifierColumn", "concept": concept, "xsd": "",
                        "class_iri": class_iri, "property_iri": "", "range_iri": class_iri,
                        "references_table": "", "references_column": "",
                        "entity": denoted, "genus": denoted_ent.get("genus") or "",
                        "reference_code": code, "ref_basis": basis,
                    }
                else:
                    attr = _attribute(owner, concept)
                    prop = ""
                    xsd = ""
                    if attr:
                        prop = "sdg:" + prop_name(str(attr.get("name") or ""))
                        xsd = str(attr.get("xsd") or "")
                    stored = col.get("property_iri") or ""
                    if stored:
                        prop = stored
                    restr = restrictions.get(owner_iri) or {}
                    if prop and prop not in restr and (prop + "Detail") in restr:
                        prop = prop + "Detail"
                    range_iri = restr.get(prop, "")
                    if not range_iri and xsd:
                        range_iri = f"xsd:{xsd}"
                    leaf = lookup(concept, vocab, by_abbrev, by_label) or lookup(
                        cname, vocab, by_abbrev, by_label)
                    if leaf:
                        code, basis = leaf, "leaf"
                    else:
                        code, basis = _genus_skos(
                            owner.get("genus") or "", vocab, by_abbrev, by_label, domain=True)
                        # An object restriction onto another entity denotes that
                        # entity (the way a foreign key does). Enumeration Kind
                        # classes are not entities and keep the domain anchor.
                        filler = ""
                        if range_iri.startswith("sdg:"):
                            filler = _entity_named(range_iri.split(":", 1)[-1], entities)
                        if filler:
                            fcode, fbasis = _skos_for_class(
                                filler, entities.get(filler), vocab, by_abbrev, by_label,
                                via_fk=True)
                            remap = {
                                "fk-target": "range-class",
                                "entity-leaf": "range-class",
                                "genus-anchor": "range-genus",
                                "genus-name": "range-name",
                            }
                            if fcode and fbasis in remap:
                                code, basis = fcode, remap[fbasis]
                    row = {
                        "role": "sdg:AttributeColumn", "concept": concept, "xsd": xsd,
                        "class_iri": owner_iri, "property_iri": prop, "range_iri": range_iri,
                        "references_table": "", "references_column": "",
                        "entity": owner_name, "genus": owner.get("genus") or "",
                        "reference_code": code, "ref_basis": basis,
                    }
                _apply_authored_kind(row, vocab, by_abbrev, by_label)
                code = row["reference_code"]
                rows.append({
                    "collection": cdir.name,
                    "construct_id": cid,
                    "table": tname,
                    "column": cname,
                    "ordinal": ordinal,
                    **row,
                    "reference_label": (vocab.get(code) or {}).get("label") or "",
                })
    if not rows:
        raise SystemExit(f"no columns under {collections}")
    open_genus = [r for r in rows if r["ref_basis"] == "genus-unmapped"]
    if open_genus:
        sample = open_genus[0]
        raise SystemExit(
            f"{len(open_genus)} columns have a genus with no anchor and no authored kind, "
            f"e.g. {sample['collection']}.{sample['table']}.{sample['column']} "
            f"class {sample['class_iri']} genus {sample['genus']}")
    return rows


def validate(rows: list[dict], collections: Path, vocab: dict[str, dict]) -> None:
    """Every published column is keyed once, and every code is in the vocabulary."""
    seen: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (row["collection"], row["table"], row["column"])
        if key in seen:
            raise SystemExit(f"duplicate answer-key row {key}")
        code = row.get("reference_code") or ""
        basis = row.get("ref_basis") or ""
        if code and code not in vocab:
            raise SystemExit(f"{key}: reference_code {code} is not in the vocabulary")
        if code and row.get("reference_label") != (vocab[code].get("label") or ""):
            raise SystemExit(f"{key}: reference_label does not match {code}")
        if (not code) != (basis in NO_CODE_BASES):
            raise SystemExit(f"{key}: basis {basis} disagrees with code {code!r}")
        if not row.get("class_iri"):
            raise SystemExit(f"{key}: no class_iri — the construct records no OWL class")
        role = row.get("role") or ""
        if role == "sdg:AttributeColumn" and not row.get("property_iri"):
            raise SystemExit(f"{key}: attribute without a recorded property")
        if role == "sdg:SurrogateIdentifierColumn" and row.get("property_iri"):
            raise SystemExit(f"{key}: surrogate identifier carries a property")
        if role == "sdg:AuditTimestampColumn" and row.get("property_iri"):
            raise SystemExit(f"{key}: audit timestamp carries a property")
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
    gaps = Counter(
        (r["genus"] or r["concept"] or r["column"])
        for r in rows if r["ref_basis"] in NO_CODE_BASES and r["ref_basis"] != "audit-unclaimed"
    )
    return {
        "n_collections": len({r["collection"] for r in rows}),
        "n_columns": len(rows),
        "n_with_code": sum(1 for r in rows if r["reference_code"]),
        "n_unresolved": sum(1 for r in rows if not r["reference_code"]),
        "n_with_class_iri": sum(1 for r in rows if r["class_iri"]),
        "n_with_property_iri": sum(1 for r in rows if r["property_iri"]),
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
                      ("n_collections", "n_columns", "n_with_code", "n_unresolved",
                       "n_with_class_iri", "n_with_property_iri", "by_basis")},
                     indent=2))
    print(f"wrote {args.out / 'answer_key.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
