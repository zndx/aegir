"""Project column-kind SKOS concepts from ``corpora/ontology/column_kinds.omn``.

Each kind is an OWL class in the SDG namespace, subclass of one BFO or CCO
class. The SKOS parent is that parent's anchor. Each kind has at least one
example class so the kind is a structural genus: a column tagged with the
kind stays on the kind, and an unseen entity of the same kind aligns there.

``unanchored_kind`` returns that kind code for a published class IRI.
``sdg:Client`` is not a subclass of the service-recipient kind. The coaching
example covers that IRI, and only a column whose recorded genus has no
anchor is retagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Bare OWL parent → SKOS anchor. A skosParent annotation that disagrees fails the load.
OWL_TO_SKOS = {
    "cco:ont00001180": "SDG.MATERIAL_ENTITY",
    "cco:ont00000300": "SDG.MATERIAL_ENTITY",
    "cco:ont00001262": "SDG.MATERIAL_ENTITY",
    "bfo:0000024": "SDG.MATERIAL_ENTITY",
    "bfo:0000023": "SDG.ROLE",
    "cco:ont00000192": "SDG.ARTIFACT",
    "cco:ont00000995": "SDG.ARTIFACT",
    "cco:ont00000798": "SDG.ARTIFACT",
    "cco:ont00000965": "SDG.ICE.DIRECTIVE",
    "cco:ont00000853": "SDG.ICE.DESCRIPTIVE",
    "cco:ont00000686": "SDG.ICE.DESIGNATIVE",
    "cco:ont00000228": "SDG.PROCESS",
    "bfo:0000015": "SDG.PROCESS",
}

_BARE = re.compile(r"^[A-Za-z_][\w]*:[A-Za-z0-9_]+$")
_SOME = re.compile(r"^(\S+)\s+some\s+(\S+)$")
_ANN = re.compile(r'^\s*([A-Za-z_][\w]*:[A-Za-z0-9_]+)\s+"([^"]*)"\s*,?\s*$')
_SUB = re.compile(r"^\s*SubClassOf:\s*(.*?)\s*$")

FAMILY = "column-kinds"


class ColumnKindError(ValueError):
    """The column-kind module is not a valid OWL-to-SKOS projection."""


def module_path() -> Path:
    return Path(__file__).resolve().parents[3] / "corpora" / "ontology" / "column_kinds.omn"


def _scream(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_").upper()


@dataclass
class _Class:
    iri: str
    local: str
    annotations: dict[str, list[str]]
    conjuncts: list[str]


@dataclass
class Kind:
    iri: str
    local: str
    code: str
    label: str
    parent_code: str
    notation: str
    definition: str
    alt_labels: tuple[str, ...]
    owl_parent: str
    axiom: str
    examples: tuple[str, ...] = ()


@dataclass
class Example:
    iri: str
    local: str
    code: str
    label: str
    kind_iri: str
    notation: str
    definition: str
    covers: tuple[str, ...]
    axiom: str


@dataclass
class ColumnKinds:
    kinds: tuple[Kind, ...]
    examples: tuple[Example, ...]
    by_class_iri: dict[str, str] = field(default_factory=dict)

    def kind_code(self, class_iri: str) -> str:
        return self.by_class_iri.get((class_iri or "").strip(), "")


def _one(ann: dict[str, list[str]], prop: str, iri: str) -> str:
    vals = ann.get(prop) or []
    if len(vals) != 1 or not vals[0].strip():
        raise ColumnKindError(f"{iri} needs exactly one {prop}")
    return vals[0].strip()


def _parse(text: str) -> list[_Class]:
    found: list[_Class] = []
    for part in re.split(r"(?m)^Class:\s+", text)[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        iri = lines[0].strip().split()[0].rstrip(",")
        if not iri or iri.startswith("<"):
            continue
        local = iri.split(":")[-1]
        ann: dict[str, list[str]] = {}
        conjuncts: list[str] = []
        in_sub = False
        for line in lines[1:]:
            sub = _SUB.match(line)
            if sub:
                in_sub = True
                rest = sub.group(1).strip().rstrip(",")
                if rest:
                    conjuncts.append(rest)
                continue
            if in_sub:
                if line.startswith((" ", "\t")) and line.strip():
                    conjuncts.append(line.strip().rstrip(","))
                    continue
                in_sub = False
            matched = _ANN.match(line)
            if matched:
                ann.setdefault(matched.group(1), []).append(matched.group(2))
        found.append(_Class(iri=iri, local=local, annotations=ann, conjuncts=conjuncts))
    return found


def _axiom(iri: str, conjuncts: list[str]) -> str:
    return f"Class: {iri} SubClassOf: {', '.join(conjuncts)}"


def load(path: Path | None = None) -> ColumnKinds:
    """Load and check the column-kind module."""
    src = path or module_path()
    if not src.is_file():
        raise ColumnKindError(f"missing column-kind module {src}")
    classes = _parse(src.read_text(encoding="utf-8"))
    by_iri = {c.iri: c for c in classes}
    if len(by_iri) != len(classes):
        raise ColumnKindError("duplicate class IRI in column-kind module")

    kinds: list[Kind] = []
    examples: list[Example] = []
    for cls in classes:
        ann = cls.annotations
        if "sdg:skosParent" in ann and "sdg:exampleOf" in ann:
            raise ColumnKindError(f"{cls.iri} is both a kind and an example")
        if "sdg:skosParent" in ann:
            kinds.append(_kind(cls))
        elif "sdg:exampleOf" in ann:
            examples.append(_example(cls))
        elif cls.iri.startswith("sdg:"):
            raise ColumnKindError(f"{cls.iri} has no skosParent and no exampleOf")
        elif ann or cls.conjuncts:
            raise ColumnKindError(f"{cls.iri} is not an SDG class and is not a bare declaration")

    kind_by_iri = {k.iri: k for k in kinds}
    if len(kind_by_iri) != len(kinds):
        raise ColumnKindError("duplicate kind IRI")
    codes: set[str] = set()
    notations: set[str] = set()
    abbrevs: set[str] = set()
    for kind in kinds:
        abbrev = _scream(kind.local)
        if not kind.code.endswith("." + abbrev):
            raise ColumnKindError(f"{kind.iri} code {kind.code} does not end with {abbrev}")
        _claim(codes, notations, abbrevs, kind.code, kind.notation, abbrev, kind.iri)
        if not kind.definition or len(kind.definition) < 80:
            raise ColumnKindError(f"{kind.iri} definition is too thin to separate unseen entities")
    seen_examples: dict[str, list[Example]] = {}
    for ex in examples:
        if ex.kind_iri not in kind_by_iri:
            raise ColumnKindError(f"{ex.iri} exampleOf {ex.kind_iri} is not a kind")
        kind = kind_by_iri[ex.kind_iri]
        abbrev = _scream(ex.local)
        if ex.code != f"{kind.code}.{abbrev}":
            raise ColumnKindError(f"{ex.iri} code {ex.code} != {kind.code}.{abbrev}")
        _claim(codes, notations, abbrevs, ex.code, ex.notation, abbrev, ex.iri)
        seen_examples.setdefault(ex.kind_iri, []).append(ex)
        if ex.axiom != f"Class: {ex.iri} SubClassOf: {ex.kind_iri}":
            raise ColumnKindError(f"{ex.iri} must be a direct subclass of its kind")

    bound: list[Kind] = []
    for kind in kinds:
        group = seen_examples.get(kind.iri) or []
        if not group:
            raise ColumnKindError(f"{kind.iri} has no example, so it is not a structural genus")
        for i, ex in enumerate(group, 1):
            expect = f"{kind.notation}.{i}"
            if ex.notation != expect:
                raise ColumnKindError(f"{ex.iri} notation {ex.notation} != {expect}")
        bound.append(Kind(
            iri=kind.iri, local=kind.local, code=kind.code, label=kind.label,
            parent_code=kind.parent_code, notation=kind.notation,
            definition=kind.definition, alt_labels=kind.alt_labels,
            owl_parent=kind.owl_parent, axiom=kind.axiom,
            examples=tuple(ex.local for ex in group),
        ))

    by_class: dict[str, str] = {}
    for ex in examples:
        code = kind_by_iri[ex.kind_iri].code
        if ex.iri in by_class and by_class[ex.iri] != code:
            raise ColumnKindError(f"{ex.iri} maps to two kinds")
        by_class[ex.iri] = code
        for covered in ex.covers:
            if covered == ex.iri:
                raise ColumnKindError(f"{ex.iri} covers itself")
            if covered in by_iri and "sdg:skosParent" in by_iri[covered].annotations:
                raise ColumnKindError(f"{covered} is a kind and cannot be covered")
            # A covered IRI is not itself subclassed in this module.
            if covered in by_iri and by_iri[covered].conjuncts:
                raise ColumnKindError(
                    f"{covered} is subclassed here; cover an unanchored published name instead")
            if covered in by_class and by_class[covered] != code:
                raise ColumnKindError(f"{covered} maps to two kinds")
            by_class[covered] = code

    if "sdg:Client" in by_iri:
        raise ColumnKindError("sdg:Client must not be subclassed; cover it from the coaching example")
    return ColumnKinds(kinds=tuple(bound), examples=tuple(examples), by_class_iri=by_class)


def _claim(codes: set[str], notations: set[str], abbrevs: set[str],
           code: str, notation: str, abbrev: str, iri: str) -> None:
    if code in codes:
        raise ColumnKindError(f"duplicate SKOS code {code}")
    if notation in notations:
        raise ColumnKindError(f"duplicate notation {notation}")
    if abbrev in abbrevs:
        raise ColumnKindError(f"duplicate abbrev {abbrev} on {iri}")
    if not code.startswith("SDG."):
        raise ColumnKindError(f"{iri} code {code} is outside SDG")
    codes.add(code)
    notations.add(notation)
    abbrevs.add(abbrev)


def _kind(cls: _Class) -> Kind:
    ann = cls.annotations
    parent_code = _one(ann, "sdg:skosParent", cls.iri)
    bare = [c for c in cls.conjuncts if _BARE.match(c)]
    restrictions = [c for c in cls.conjuncts if not _BARE.match(c)]
    if len(bare) != 1:
        raise ColumnKindError(f"{cls.iri} needs exactly one bare SubClassOf parent")
    for item in restrictions:
        if not _SOME.match(item):
            raise ColumnKindError(f"{cls.iri} has an unsupported conjunct {item}")
    owl_parent = bare[0]
    entailed = OWL_TO_SKOS.get(owl_parent)
    if entailed is None:
        raise ColumnKindError(f"{cls.iri} parent {owl_parent} has no SKOS anchor")
    if parent_code != entailed:
        raise ColumnKindError(
            f"{cls.iri} skosParent {parent_code} disagrees with {owl_parent} → {entailed}")
    return Kind(
        iri=cls.iri,
        local=cls.local,
        code=_one(ann, "sdg:skosCode", cls.iri),
        label=_one(ann, "rdfs:label", cls.iri),
        parent_code=parent_code,
        notation=_one(ann, "sdg:notation", cls.iri),
        definition=_one(ann, "iao:0000115", cls.iri),
        alt_labels=tuple(ann.get("skos:altLabel") or []),
        owl_parent=owl_parent,
        axiom=_axiom(cls.iri, cls.conjuncts),
    )


def _example(cls: _Class) -> Example:
    ann = cls.annotations
    if "sdg:skosParent" in ann:
        raise ColumnKindError(f"{cls.iri} example must not set skosParent")
    kind_iri = _one(ann, "sdg:exampleOf", cls.iri)
    if cls.conjuncts != [kind_iri]:
        raise ColumnKindError(f"{cls.iri} must subclass only {kind_iri}")
    return Example(
        iri=cls.iri,
        local=cls.local,
        code=_one(ann, "sdg:skosCode", cls.iri),
        label=_one(ann, "rdfs:label", cls.iri),
        kind_iri=kind_iri,
        notation=_one(ann, "sdg:notation", cls.iri),
        definition=_one(ann, "iao:0000115", cls.iri),
        covers=tuple(ann.get("sdg:coversUnanchored") or []),
        axiom=_axiom(cls.iri, cls.conjuncts),
    )


_CACHED: ColumnKinds | None = None


def column_kinds() -> ColumnKinds:
    global _CACHED
    if _CACHED is None:
        _CACHED = load()
    return _CACHED


def unanchored_kind(class_iri: str) -> str:
    """Kind code for a published class, or empty when this module does not cover it."""
    return column_kinds().kind_code(class_iri)


def vocab_records() -> list[dict]:
    """SKOS rows for the vocabulary build. Kinds, then each kind's examples."""
    mod = column_kinds()
    rows: list[dict] = []
    examples_of = {}
    for ex in mod.examples:
        examples_of.setdefault(ex.kind_iri, []).append(ex)
    for kind in mod.kinds:
        rows.append({
            "code": kind.code,
            "label": kind.label,
            "abbrev": _scream(kind.local),
            "notation": kind.notation,
            "parent_code": kind.parent_code,
            "taxonomy": "sdg",
            "description": kind.definition,
            "common_names": ", ".join(kind.alt_labels),
            "example_values": " | ".join(kind.examples),
            "axiom": kind.axiom,
            "family": FAMILY,
        })
        for ex in examples_of.get(kind.iri, []):
            rows.append({
                "code": ex.code,
                "label": ex.label,
                "abbrev": _scream(ex.local),
                "notation": ex.notation,
                "parent_code": kind.code,
                "taxonomy": "sdg",
                "description": ex.definition,
                "common_names": ex.local,
                "example_values": ex.local,
                "axiom": ex.axiom,
                "family": FAMILY,
            })
    return rows
