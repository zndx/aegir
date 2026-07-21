"""verify_path — the path-aware VERIFICATION step (RH 2026-07-21: the proof increment).

Not a Mathematica-like calculator: the lineup's first computation is scoped to our verified
core — HermiT (+ the armed signature backbone) over "axioms found before it in the lineup."
A trail's panels resolve to ontology fragments (native terms/groups → catalog axioms with
grounding; foreign groups → axioms extracted from the censused source), the union is
assembled ISOLATED (never production), and HermiT disposes under the budget watchdog. The
result is memoized by path_id, emitted as an OL run (the lineup instance is visible lineage),
and materialized as a PANEL — the computation's result joins the lineup rightmost, exactly
as Ward's Method results become pages.

The adoption use-case (user-provided ontologies; incremental PRODML-style spec adaptation in
scratch): browse your group beside ours → verify the union ON THE FLY, before any release
gate — logical verification at the speed of browsing.

    uv run python -m aegir.lineup.verify_path lens/terms ontology/foreign/fibo/group/equity-instrument
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "build" / "path_runs"
FRAG_CACHE = REPO / "build" / "foreign_fragments"

_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
             "Ontology: <https://signals.zndx.org/sdg/path-verify>\n")


def _mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── foreign fragment extraction (once per tag, cached) ────────────────────────

def _extract_foreign(tag: str) -> dict:
    """class-iri → OMN frame lines, for every named class of the censused source. One sweep,
    cached; the verify step then composes per-group fragments in O(1)."""
    cache = FRAG_CACHE / f"{tag}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("v") == 2:                      # v2: + equivalentClass frames
            return d
    cov = json.loads((REPO / "build" / f"{tag}_coverage.json").read_text())
    root = Path(cov["root"]).expanduser()
    from rdflib import BNode, Graph, URIRef
    from rdflib.namespace import OWL, RDF, RDFS

    def _first(g, s, p):
        for o in g.objects(s, p):
            return o
        return None

    def _members(g, lst):
        out = []
        while lst and lst != RDF.nil:
            out.append(_first(g, lst, RDF.first))
            lst = _first(g, lst, RDF.rest)
        return [x for x in out if x is not None]

    def _render(g, e, depth=0) -> "str | None":
        if isinstance(e, URIRef):
            return f"<{e}>"
        if depth > 5 or not isinstance(e, BNode):
            return None
        lst = _first(g, e, OWL.intersectionOf)
        if lst is not None:
            ms = [_render(g, m, depth + 1) for m in _members(g, lst)]
            ms = [m for m in ms if m]
            return "(" + " and ".join(ms) + ")" if ms else None
        lst = _first(g, e, OWL.unionOf)
        if lst is not None:
            ms = [_render(g, m, depth + 1) for m in _members(g, lst)]
            ms = [m for m in ms if m]
            return "(" + " or ".join(ms) + ")" if ms else None
        on_p = _first(g, e, OWL.onProperty)
        if on_p is None or not isinstance(on_p, URIRef):
            return None                                  # hasSelf/complement/etc → degrade
        for pred, kw in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only")):
            f = _first(g, e, pred)
            if f is not None:
                fr = _render(g, f, depth + 1)
                return f"<{on_p}> {kw} {fr}" if fr else None
        oc = _first(g, e, OWL.onClass)
        for pred, kw in ((OWL.qualifiedCardinality, "exactly"),
                         (OWL.minQualifiedCardinality, "min"),
                         (OWL.maxQualifiedCardinality, "max")):
            n = _first(g, e, pred)
            if n is not None and oc is not None:
                fr = _render(g, oc, depth + 1)
                return f"<{on_p}> {kw} {n} {fr}" if fr else None
        return None                                      # hasValue / data / card-u → degrade

    frames: "dict[str, list]" = {}
    equiv: "dict[str, list]" = {}
    props: set = set()
    files = ([root] if root.is_file() else
             sorted(list(root.rglob("*.rdf")) + list(root.rglob("*.ttl"))))
    for f in files:
        if re.search(r"^(About|All|Metadata)", f.name):
            continue
        g = Graph()
        try:
            g.parse(str(f))
        except Exception:  # noqa: BLE001
            continue
        for s in set(g.subjects(RDF.type, OWL.Class)):
            if not isinstance(s, URIRef):
                continue
            for o in g.objects(s, RDFS.subClassOf):
                r = _render(g, o)
                if r:
                    frames.setdefault(str(s), []).append(r)
                    props |= {m.group(1) for m in re.finditer(r"<([^>]+)> (?:some|only|exactly|min|max)", r)}
            for o in g.objects(s, OWL.equivalentClass):
                r = _render(g, o)
                if r:
                    equiv.setdefault(str(s), []).append(r)
                    props |= {m.group(1) for m in re.finditer(r"<([^>]+)> (?:some|only|exactly|min|max)", r)}
    FRAG_CACHE.mkdir(parents=True, exist_ok=True)
    out = {"v": 2, "frames": frames, "equiv": equiv, "props": sorted(props)}
    cache.write_text(json.dumps(out))
    return out


# ── fragment collection per panel ─────────────────────────────────────────────

def _native_frames(template_ids: "list[str]") -> "tuple[list[str], set]":
    from aegir.ontology.relation_signatures import grounding_frames
    from aegir.ontology.schema import load_catalog
    cat = load_catalog(REPO / "src/aegir/ontology/catalog/catalog.json")
    by_id = {t.template_id: t for t in cat.templates}
    frames, props = [], set()
    for tid in template_ids:
        t = by_id.get(tid)
        if not t:
            continue
        mt = re.sub(r"\{(\w+):Class\}", r"sdg:\1", t.manchester_template or "")
        frames.append(mt)
        props |= {m.group(1) for m in re.finditer(r"sdg:(\w+)\s+(?:some|only|exactly|min|max)", mt)}
    g_omn, _loose = grounding_frames(props)
    return frames + ([g_omn] if g_omn else []), props


def collect(trail: "list[str]") -> dict:
    """Trail → the LEFT-OF-ME axiom universe: per-panel fragments + provenance."""
    from aegir.lineup.path import decode_path, encode_trail
    enc = encode_trail(trail)
    kb = REPO / "build" / "dev"
    panels = decode_path(enc["ranks"])
    native_tids: "list[str]" = []
    foreign: "dict[str, set]" = {}
    for p, nid in zip(panels, trail):
        m = re.match(r"ontology/(?:foreign/(\w+)|self-census)/group/(.+)$", nid)
        if m:
            tag = m.group(1) or "sdg"
            cov = json.loads((REPO / "build" / f"{tag}_coverage.json").read_text())
            grp = next((g for g in cov["pathways"]["groups"] if g["slug"] == m.group(2)), None)
            if grp:
                if tag == "sdg":
                    native_tids += [re.sub(r"(?<!^)(?=[A-Z])", "_", x["local"]).lower()
                                    for x in grp.get("members", [])]
                else:
                    foreign.setdefault(tag, set()).update(
                        x["local"] for x in grp.get("members", []))
        elif nid.startswith("ontology/term/"):
            native_tids.append(nid.rsplit("/", 1)[1])
    return {"encoded": enc, "panels": panels, "native_tids": sorted(set(native_tids)),
            "foreign": {k: sorted(v) for k, v in foreign.items()}}


# ── the step ──────────────────────────────────────────────────────────────────

def verify(trail: "list[str]", budget_s: int = 300) -> dict:
    """The path-aware HermiT step: assemble the trail's fragment union (armed) → reason →
    memoized, OL-emitted, panel-materializable verdict."""
    c = collect(trail)
    path_id = c["encoded"]["path_id"]
    RUNS.mkdir(parents=True, exist_ok=True)
    memo = RUNS / f"{path_id}.json"
    if memo.exists():
        return json.loads(memo.read_text())

    bro = _mod(REPO / "scripts/build_realized_ontology.py", "bro_pv")
    et = _mod(REPO / "scripts/emit_taxonomy.py", "et_pv")
    from aegir.ontology.relation_signatures import SIGNATURES_OMN

    nat_frames, _props = _native_frames(c["native_tids"])
    foreign_frames: "list[str]" = []
    n_foreign_ax = 0
    skipped = 0
    for tag, locals_ in c["foreign"].items():
        frag = _extract_foreign(tag)
        want = set(locals_)
        for iri, rs in frag["frames"].items():
            loc = iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            if loc in want:
                for r in rs:
                    foreign_frames.append(f"Class: <{iri}>\n    SubClassOf: {r}")
                    n_foreign_ax += 1
        skipped = frag.get("skipped", 0)
        for p in frag["props"]:
            foreign_frames.append(f"ObjectProperty: <{p}>")
    # declare every referenced foreign IRI (fillers included) so the doc is closed
    refs = {m.group(1) for f in foreign_frames for m in re.finditer(r"<([^>]+)>", f)}
    decls = [f"Class: <{r}>" for r in sorted(refs)
             if not any(x in r for x in ("BFO_", "owl#", "XMLSchema"))
             and f"Class: <{r}>" not in foreign_frames]

    omn = (_PREFIXES + bro.NUMERIC_BFO + "\n" + SIGNATURES_OMN + "\n"
           + "\n".join(nat_frames) + "\n" + "\n".join(decls) + "\n"
           + "\n".join(foreign_frames))
    v = et.hermit(omn, budget_s=budget_s)
    result = {"path_id": path_id, "version": c["encoded"]["version"], "trail": trail,
              "when": datetime.now(timezone.utc).isoformat(),
              "inputs": {"native_templates": len(c["native_tids"]),
                         "foreign": {k: len(v_) for k, v_ in c["foreign"].items()},
                         "foreign_axioms": n_foreign_ax, "degraded_constructs": skipped},
              "verdict": {"consistent": v.get("consistent"),
                          "n_classes": v.get("n_classes"),
                          "unsat": (v.get("unsat") or [])[:24]}}
    memo.write_text(json.dumps(result, indent=1))
    try:
        from aegir.governance import ol
        ol.ingest_run_event({
            "eventType": "COMPLETE", "eventTime": result["when"], "producer": ol.PRODUCER,
            "run": {"runId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ol.PRODUCER}/path_verify/{path_id}"))},
            "job": {"namespace": "aegir", "name": "path_verify"},
            "inputs": [{"namespace": "lineup", "name": f"path/{path_id}",
                        "facets": {"path": {"version": result["version"], "n_panels": len(trail)}}}],
            "outputs": [{"namespace": "lineup", "name": f"path-run/{path_id}",
                         "facets": {"verdict": result["verdict"]}}]})
        result["lineage"] = "emitted"
    except Exception:  # noqa: BLE001
        result["lineage"] = "unavailable"
    memo.write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    trail = sys.argv[1:]
    if not trail:
        print("usage: python -m aegir.lineup.verify_path <note-id> [<note-id>...]")
        return 2
    r = verify(trail)
    print(json.dumps(r, indent=1))
    return 0 if r["verdict"]["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
