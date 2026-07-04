"""Class profiles — the per-class accretion record (#140 phase A).

The class-centric model (docs/scratch/2026-07-04/145424_class_centric_reevaluation.md):
an entity's table should be derived from EVERYTHING the ontology knows about that class —
all its data properties, its object-property restrictions across ALL axioms that mention
it, its enumerations, its roles — not from the arity of the single axiom that introduced
it. A :class:`ClassProfile` is that record. Profiles ACCRETE (the registries' pattern);
templates demote to one *source* of profile facts plus provenance.

Projection sources, in authority order (kvasir-ddl handoff warning 4):

1. the CERTIFIED omn (``corpora/ontology/sdg-ontology.omn``) — structural truth. Parsed
   with the same frame machinery as :mod:`aegir.ontology.kvasir_bridge` (parser skew is a
   soundness hazard; share the code). Restrictions keep their TRUE cardinality here —
   the reasoning lowering weakens ``exactly n`` to ``some``; profiles are the DDL source
   and must not.
2. the catalog templates — PROVENANCE accretion (SKOS domain, grounds_ddl, template_id),
   joined first-``{Name:Class}``-slot → ``sdg:Name`` (the ``_head_map`` rule).
3. ``sdg-vocab.ttl`` DataProperty pools (:func:`aegir.ontology.ddl._data_properties`) —
   anchor-keyed attribute accretion along the grounding walk. NO attribute budget here:
   a profile holds everything the ontology knows; budgets/de-canning are realization-time
   decisions and stay in the worldly layer.

``census()`` is the width-potential instrument: what ``class_to_table`` could emit TODAY
from the ontology's actual knowledge — the honest gap that phase B (engine property
accretion, Convert 2 redefined) and R1's re-derive must close, measured per class.

This module is phase-A scaffolding for that loop, not the destination: until phase B
runs, most profiles are exactly as poor as the axioms that seeded them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aegir.ontology.ddl import (_ANCHOR_PARENTS, _data_properties, _dataprop_col,
                                dataprop_meta, parse_restrictions)
from aegir.ontology.kvasir_bridge import (_FRAME, _INLINE, _SECTION, _clean,
                                          _split_conjuncts, _split_items)
from aegir.ontology.rows import parse_enum_from_definition
from aegir.ontology.schema import Catalog

# restriction parse that RETAINS kind (the reasoning lowering may weaken; the DDL source must not)
_RESTR = re.compile(
    r"^\(?\s*(?P<prop>\S+)\s+(?P<kind>some|only|value|min\s+\d+|max\s+\d+|exactly\s+\d+)\s+"
    r"(?P<filler>[^()\s]+)\s*\)?$")
_LABEL = re.compile(r'^\s*Annotations:.*?rdfs:label\s+"([^"]+)"')
_PREFIX = re.compile(r"^Prefix:\s*(\w*):\s*<([^>]+)>", re.M)
_HEAD_SLOT = re.compile(r"\{(\w+):Class\}")

# numeric ↔ named BFO bridge — only the skeleton pairs the omn + vocab pools actually use
_BFO_NAMED = {
    "bfo:0000001": "bfo:Entity", "bfo:0000002": "bfo:Continuant",
    "bfo:0000003": "bfo:Occurrent", "bfo:0000004": "bfo:IndependentContinuant",
    "bfo:0000015": "bfo:Process", "bfo:0000040": "bfo:MaterialEntity",
}


@dataclass
class DataAttribute:
    """A typed data attribute the ontology asserts for a class (a column candidate)."""
    prop: str                     # prefixed property IRI (sdg:hasEncoding)
    column: str                   # semantic-register column stem (encoding)
    xsd: str                      # xsd range (xsd:string …)
    enum: "list[str] | None" = None   # closed value set from skos:definition, if any
    source: str = "omn"           # omn | vocab | template | engine (phase B)
    cite: str = ""                # evidencing record (omn:<line> | vocab:<prop> | template:<id>)


@dataclass
class ObjectRelation:
    """An object-property restriction on the class (an FK/junction candidate)."""
    prop: str                     # prefixed property IRI
    target: str                   # prefixed target class IRI
    kind: str = "some"            # some | only | value | min n | max n | exactly n (TRUE bounds)
    source: str = "omn"
    cite: str = ""


@dataclass
class ClassProfile:
    """Everything the ontology knows about one class — the accretion target."""
    iri: str                                  # prefixed (sdg:X); the canonical key
    label: str = ""
    genus: list[str] = field(default_factory=list)      # told named superclasses
    defined: bool = False                                # has an EquivalentTo definition
    attributes: list[DataAttribute] = field(default_factory=list)
    relations: list[ObjectRelation] = field(default_factory=list)
    grounding: list[str] = field(default_factory=list)  # bfo:/cco: ancestors, BFS order
    domain: dict = field(default_factory=dict)          # SKOS domain {code, label}
    sources: list[dict] = field(default_factory=list)   # provenance events

    def width_potential(self) -> int:
        """Columns ``class_to_table`` could emit today: subject id + attributes + relation FKs."""
        return 1 + len(self.attributes) + len(self.relations)


# ── omn projection ─────────────────────────────────────────────────────────────
def _prefix_map(doc: str) -> "dict[str, str]":
    """{namespace-uri → prefix:} from the document's own Prefix declarations (never a
    hardcoded table — the cco http/https drift already bit once)."""
    return {m.group(2): (m.group(1) + ":") for m in _PREFIX.finditer(doc) if m.group(1)}


def _norm(tok: str, ns: "dict[str, str]") -> str:
    """Normalize an entity token to prefixed form using the doc's declared namespaces."""
    tok = _clean(tok)
    for uri, p in ns.items():
        if tok.startswith(uri):
            return p + tok[len(uri):]
    return tok


def project_from_omn(doc: str) -> "tuple[dict[str, ClassProfile], dict]":
    """Project profiles from the certified Manchester document (authority source 1).

    Profiles every ``sdg:`` class (the realization surface); the told-subsumption graph is
    built over ALL classes so grounding walks reach the BFO skeleton. Frames repeat per
    IRI in the emitted doc (one axiom per line) — facts MERGE by normalized IRI. Returns
    ``(profiles, report)``.
    """
    ns = _prefix_map(doc)
    profiles: "dict[str, ClassProfile]" = {}
    told: "dict[str, set[str]]" = {}
    labels: "dict[str, str]" = {}
    skipped: "dict[str, int]" = {}

    frames = list(_FRAME.finditer(doc))
    for i, fm in enumerate(frames):
        kind, name = fm.group(1), _norm(fm.group(2), ns)
        body = doc[fm.end(): frames[i + 1].start() if i + 1 < len(frames) else len(doc)]
        for line in body.splitlines():
            lm = _LABEL.match(line)
            if lm:
                labels.setdefault(name, lm.group(1))
        if kind != "Class":
            continue
        line_no = doc[:fm.start()].count("\n") + 1
        sections: "list[tuple[str, str]]" = []
        im = _INLINE.match(fm.group(3) or "")
        if im:
            sections.append((im.group(1), im.group(2)))
        for line in body.splitlines():
            sm = _SECTION.match(line)
            if sm:
                sections.append((sm.group(1), sm.group(2)))

        is_sdg = name.startswith("sdg:")
        if is_sdg and name not in profiles:
            profiles[name] = ClassProfile(iri=name)
        p = profiles.get(name)

        for section, rhs in sections:
            if section not in ("SubClassOf", "EquivalentTo"):
                continue
            if p is not None and section == "EquivalentTo":
                p.defined = True
            for item in _split_items(rhs):
                for conj in _split_conjuncts(item):
                    conj = conj.strip().rstrip(",")
                    if not conj:
                        continue
                    m = _RESTR.match(conj)
                    if m:
                        prop = _norm(m.group("prop"), ns)
                        filler = _norm(m.group("filler"), ns)
                        kind_txt = re.sub(r"\s+", " ", m.group("kind")).strip()
                        cite = f"omn:{line_no}"
                        if p is None:
                            continue  # restrictions on non-sdg classes aren't profiled
                        if filler.startswith("xsd:"):
                            col = _dataprop_col(prop)
                            if all(a.prop != prop for a in p.attributes):
                                p.attributes.append(DataAttribute(
                                    prop=prop, column=col, xsd=filler, cite=cite))
                        else:
                            if all(not (r.prop == prop and r.target == filler)
                                   for r in p.relations):
                                p.relations.append(ObjectRelation(
                                    prop=prop, target=filler, kind=kind_txt, cite=cite))
                    elif " " not in conj:
                        sup = _norm(conj, ns)
                        told.setdefault(name, set()).add(sup)
                        if p is not None and sup not in p.genus:
                            p.genus.append(sup)
                    else:
                        skipped[f"conj:{'nested' if '(' in conj else 'other'}"] = \
                            skipped.get(f"conj:{'nested' if '(' in conj else 'other'}", 0) + 1

    # grounding walk: BFS up the told graph; keep the bfo:/cco: ancestors in visit order
    for p in profiles.values():
        p.label = labels.get(p.iri, "")
        seen: "set[str]" = set()
        queue = list(p.genus)
        anchors: "list[str]" = []
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            if cur.startswith(("bfo:", "cco:")) and cur not in anchors:
                anchors.append(cur)
            queue.extend(sorted(told.get(cur, ())))
        p.grounding = anchors

    report = {"n_frames": len(frames), "n_profiles": len(profiles),
              "n_labeled": sum(1 for p in profiles.values() if p.label),
              "skipped": skipped}
    return profiles, report


# ── catalog accretion (provenance + cardinality refinement) ────────────────────
def accrete_from_catalog(profiles: "dict[str, ClassProfile]", catalog: Catalog) -> dict:
    """Join templates onto profiles (head-slot rule) and accrete PROVENANCE: SKOS domain,
    grounds_ddl/pattern/tier, template_id. Where a template asserts a tighter bound
    (``exactly 1``) than the profiled relation carries, the bound upgrades with
    ``source=template`` — the omn keeps most bounds, so upgrades should be rare and are
    reported. Returns the join report."""
    joined = missed = upgraded = 0
    for t in catalog.templates:
        m = _HEAD_SLOT.search(t.manchester_template)
        if not m:
            missed += 1
            continue
        p = profiles.get(f"sdg:{m.group(1)}")
        if p is None:
            missed += 1
            continue
        joined += 1
        prov = t.provenance or {}
        p.sources.append({"template_id": t.template_id,
                          **{k: prov[k] for k in ("pattern", "tier", "grounds_ddl")
                             if prov.get(k) is not None}})
        dom = prov.get("domain")
        if not p.domain and isinstance(dom, dict) and dom.get("code"):
            p.domain = {"code": dom["code"], "label": dom.get("label", "")}
        for r in parse_restrictions(t):
            if r.is_slot or r.cardinality == "some":
                continue
            for rel in p.relations:
                if rel.prop == r.prop and rel.kind == "some":
                    rel.kind = r.cardinality
                    rel.source += "+template"
                    rel.cite += f" template:{t.template_id}"
                    upgraded += 1
    return {"templates": len(catalog.templates), "joined": joined,
            "unjoined": missed, "bounds_upgraded": upgraded}


# ── vocab accretion (anchor-keyed DataProperty pools) ──────────────────────────
def accrete_from_vocab(profiles: "dict[str, ClassProfile]") -> dict:
    """Accrete ``sdg-vocab.ttl`` typed attributes along each profile's grounding walk.
    Pools key on NAMED anchors (bfo:Process, cco:Artifact); the omn grounds through the
    NUMERIC skeleton — bridged via ``_BFO_NAMED``, then closed through ``_ANCHOR_PARENTS``.
    Full pool, no budget (see module docstring). Enumerations attach from the property's
    ``skos:definition``."""
    pools = _data_properties()
    meta = dataprop_meta()
    added = 0
    hit_profiles = 0
    for p in profiles.values():
        keys: "list[str]" = []
        for g in [p.iri, *p.genus, *p.grounding]:
            for k in (g, _BFO_NAMED.get(g, "")):
                while k and k not in keys:
                    keys.append(k)
                    k = _ANCHOR_PARENTS.get(k, "")
        have_cols = {a.column for a in p.attributes}
        have_props = {a.prop for a in p.attributes}
        hit = False
        for k in keys:
            for col, rng, prop_iri in pools.get(k, ()):
                if col in have_cols or prop_iri in have_props:
                    continue
                have_cols.add(col)
                have_props.add(prop_iri)
                defn = (meta.get(prop_iri) or ("", "", ""))[2]
                p.attributes.append(DataAttribute(
                    prop=prop_iri, column=col, xsd=rng,
                    enum=parse_enum_from_definition(defn),
                    source="vocab", cite=f"vocab:{k}"))
                added += 1
                hit = True
        hit_profiles += hit
    return {"attributes_added": added, "profiles_reached": hit_profiles}


# ── the projection entry point ─────────────────────────────────────────────────
def project_profiles(omn_path: "str | Path",
                     catalog: "Catalog | None" = None) -> "tuple[dict[str, ClassProfile], dict]":
    """omn (+ optional catalog) → profiles, with the per-source accretion report."""
    doc = Path(omn_path).read_text()
    profiles, report = project_from_omn(doc)
    if catalog is not None:
        report["catalog"] = accrete_from_catalog(profiles, catalog)
    report["vocab"] = accrete_from_vocab(profiles)
    return profiles, report


# ── persistence (the schema.py pattern) ────────────────────────────────────────
def save_profiles(profiles: "dict[str, ClassProfile]", path: "str | Path",
                  report: "dict | None" = None) -> None:
    payload = {"version": "0.1", "report": report or {},
               "profiles": [asdict(p) for _, p in sorted(profiles.items())]}
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def load_profiles(path: "str | Path") -> "dict[str, ClassProfile]":
    data = json.loads(Path(path).read_text())
    out: "dict[str, ClassProfile]" = {}
    for row in data["profiles"]:
        row["attributes"] = [DataAttribute(**a) for a in row.get("attributes", [])]
        row["relations"] = [ObjectRelation(**r) for r in row.get("relations", [])]
        p = ClassProfile(**row)
        out[p.iri] = p
    return out


# ── the width-potential instrument ─────────────────────────────────────────────
def census(profiles: "dict[str, ClassProfile]") -> dict:
    """What class_to_table could emit today, per class — the phase-B / R1 target metric.
    Compare against the realized spine census (manifest.json) and SchemaPile norms."""
    ps = list(profiles.values())
    if not ps:
        return {"n_classes": 0}
    widths = sorted(p.width_potential() for p in ps)

    def pct(q: float) -> int:
        return widths[min(len(widths) - 1, int(q * len(widths)))]

    n_attr = [len(p.attributes) for p in ps]
    n_rel = [len(p.relations) for p in ps]
    src = {}
    for p in ps:
        for a in p.attributes:
            src[a.source] = src.get(a.source, 0) + 1
    return {
        "n_classes": len(ps),
        "defined_ratio": round(sum(p.defined for p in ps) / len(ps), 3),
        "width_potential": {"median": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                            "max": widths[-1], "ge5_ratio": round(
                                sum(w >= 5 for w in widths) / len(widths), 3)},
        "attributes": {"total": sum(n_attr), "median": sorted(n_attr)[len(n_attr) // 2],
                       "max": max(n_attr), "zero_ratio": round(
                           sum(n == 0 for n in n_attr) / len(n_attr), 3)},
        "relations": {"total": sum(n_rel), "median": sorted(n_rel)[len(n_rel) // 2],
                      "max": max(n_rel)},
        "attr_by_source": src,
        "enum_attrs": sum(1 for p in ps for a in p.attributes if a.enum),
        "domain_ratio": round(sum(bool(p.domain) for p in ps) / len(ps), 3),
        "grounded_ratio": round(sum(bool(p.grounding) for p in ps) / len(ps), 3),
    }


def main() -> None:  # pragma: no cover — CLI shell
    import argparse
    ap = argparse.ArgumentParser(description="Project class profiles (#140 phase A)")
    ap.add_argument("omn", nargs="?", default="corpora/ontology/sdg-ontology.omn")
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--out", default="build/profiles.json")
    args = ap.parse_args()
    from aegir.ontology.schema import load_catalog
    cat = load_catalog(args.catalog) if args.catalog else None
    profiles, report = project_profiles(args.omn, cat)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_profiles(profiles, args.out, report)
    print(json.dumps({"report": report, "census": census(profiles)}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
