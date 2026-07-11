"""external_index — the authoritative index of externally-defined ontology entities (BFO / CCO / FHIR).

External namespaces are AUTHORITIES, not sandboxes. A `cco:`/`bfo:`/`fhir:` reference is a CLAIM that the
IRI exists in the current external ontology. LLMs (especially local Qwen/Nemotron) confidently coin
plausible-but-fictional external terms (`cco:DirectiveICE`, `cco:BusinessEntity`, `cco:has_part`); those
are tampering with an externally-maintained namespace and must be REJECTED, not silently resolved — the
hardcoded alias dict that resolved them was an anti-pattern that *reinforced* the hallucination
(RH 2026-07-11, [[bfo_cco_grounding_mandate]]). An LLM may coin ONLY in our own `sdg:` namespace.

This index is the ground truth for the hard verification gate: `exists(ref)` is an unambiguous LOGICAL
existence check against the loaded authoritative ontology — the IRI must be a declared entity.

External ontologies EVOLVE: "CCO" means the CURRENT CCO. Each source records its `versionIRI`; BFO 2020 is
stable but verified all the same. `freshness()` can compare the loaded version against upstream (network).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

_REPO = Path(__file__).resolve().parents[3]
_GROUNDING = _REPO / "build" / "grounding"
_CACHE = _GROUNDING / "external_index.pkl"

# namespace prefix → (curie base IRI, source file, upstream URL for freshness)
SOURCES: "dict[str, tuple[str, Path, str]]" = {
    "bfo": ("http://purl.obolibrary.org/obo/BFO_", _GROUNDING / "bfo.owl",
            "http://purl.obolibrary.org/obo/bfo.owl"),
    "cco": ("https://www.commoncoreontologies.org/", _GROUNDING / "cco-merged.ttl",
            "https://raw.githubusercontent.com/CommonCoreOntology/CommonCoreOntologies/develop/"
            "src/cco-merged/CommonCoreOntologiesMerged.ttl"),
    # fhir: added when the FHIR index is staged — until then fhir: refs are handled by the bridge, not gated here.
}
_TYPE_KIND = [(OWL.Class, "class"), (OWL.ObjectProperty, "objectproperty"),
              (OWL.DatatypeProperty, "dataproperty"), (OWL.NamedIndividual, "individual")]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class ExternalIndex:
    """Loaded once, cached. `exists`/`kind` verify; `search` assists (never silently substitutes)."""

    def __init__(self) -> None:
        self.iris: dict[str, str] = {}          # full IRI → kind
        self.labels: dict[str, list] = {}       # normalized label → [full IRIs]
        self.versions: dict[str, str] = {}      # ns → versionIRI
        self._load()

    def _fresh_cache(self) -> bool:
        if not _CACHE.exists():
            return False
        cm = _CACHE.stat().st_mtime
        return all(not f.exists() or f.stat().st_mtime <= cm for _b, f, _u in SOURCES.values())

    def _load(self) -> None:
        if self._fresh_cache():
            try:
                d = pickle.loads(_CACHE.read_bytes())
                self.iris, self.labels, self.versions = d["iris"], d["labels"], d["versions"]
                return
            except Exception:  # noqa: BLE001
                pass
        for ns, (base, path, _u) in SOURCES.items():
            if not path.exists():
                continue
            g = rdflib.Graph().parse(str(path))
            ver = next((str(o) for o in g.objects(None, OWL.versionIRI)), "unknown")
            self.versions[ns] = ver
            for typ, kind in _TYPE_KIND:
                for s in g.subjects(RDF.type, typ):
                    iri = str(s)
                    if not iri.startswith(base):
                        continue
                    self.iris[iri] = kind
                    for lab in g.objects(s, RDFS.label):
                        self.labels.setdefault(_norm(lab), []).append(iri)
        try:
            _CACHE.write_bytes(pickle.dumps({"iris": self.iris, "labels": self.labels, "versions": self.versions}))
        except Exception:  # noqa: BLE001
            pass

    def expand(self, ref: str) -> "str | None":
        """`prefix:local` → full IRI for an indexed namespace; passes a full http IRI through."""
        if ref.startswith("http"):
            return ref
        if ":" not in ref:
            return None
        pfx, local = ref.split(":", 1)
        base = SOURCES.get(pfx, (None,))[0]
        return base + local if base else None

    def indexed_ns(self, ref: str) -> bool:
        return ref.split(":", 1)[0] in SOURCES if ":" in ref and not ref.startswith("http") else False

    def exists(self, ref: str) -> bool:
        iri = self.expand(ref)
        return bool(iri and iri in self.iris)

    def kind(self, ref: str) -> "str | None":
        iri = self.expand(ref)
        return self.iris.get(iri) if iri else None

    def search(self, text: str, prefix: "str | None" = None) -> list:
        """label → [curies], for AGENT ASSISTANCE (suggesting the real IRI), never silent substitution."""
        out = []
        for iri in self.labels.get(_norm(text), []):
            for pfx, (base, _f, _u) in SOURCES.items():
                if iri.startswith(base) and (prefix is None or pfx == prefix):
                    out.append(f"{pfx}:{iri[len(base):]}")
        return out

    def version(self, ns: str) -> "str | None":
        return self.versions.get(ns)


_INSTANCE: "ExternalIndex | None" = None


def get_index() -> ExternalIndex:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ExternalIndex()
    return _INSTANCE


_REF = re.compile(r"\b(bfo|cco|fhir):([A-Za-z0-9_]+)")
# the valid BFO/CCO expression keywords + our own namespaces are NOT external claims
_EXPR_KW = {"some", "only", "value", "min", "max", "exactly", "and", "or", "not", "that",
            "SubClassOf", "EquivalentTo", "Class", "Individual", "Types", "DisjointClasses"}


def verify_external_refs(manchester: str, index: "ExternalIndex | None" = None) -> list:
    """Every bfo:/cco: reference must EXIST in the current authoritative ontology, else it is tampering.
    Returns ``[(ref, reason)]`` — empty ⇒ clean. (fhir: is deferred to the bridge until a FHIR index is staged.)
    A type note is added when the referent exists but is used in the wrong position (a property as a genus, etc.)
    is left to the reasoner; this gate is purely EXISTENCE — hard, unambiguous, logical."""
    idx = index or get_index()
    seen, violations = set(), []
    for m in _REF.finditer(manchester or ""):
        ref = m.group(0)
        if m.group(1) == "fhir" or ref in seen:
            continue
        seen.add(ref)
        if not idx.exists(ref):
            ns = m.group(1)
            ver = idx.version(ns) or "?"
            hint = idx.search(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", m.group(2)).replace("_", " "), prefix=ns)
            tail = f" — did you mean {hint[0]}?" if hint else " — use a real IRI from the anchors or coin an sdg: term"
            violations.append((ref, f"{ref} does NOT exist in {ns.upper()} ({ver}); "
                                    f"external namespaces may not be coined{tail}"))
    return violations
