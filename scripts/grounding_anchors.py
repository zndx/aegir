"""Grounding-anchor retrieval — the BFO-grounding boundary emits domain vocabulary as a signal.

The agent grounds fillers to GENERIC bfo: categories (or coins ungrounded sdg: terms) because it cannot SEE
the mid-level genera it reaches for — Organism, Vehicle, Patient. A boolean "ungrounded" can't repair that
(missing reference, not inattention). This index gives the boundary a richer signal: seed grounded anchors
from CC0 imports — CCO (1431 BFO-aligned classes) + FHIR R5 types (clinical/record types, bridged to
cco:ont00000958) — and ACCRETE our own grounded classes over time ("build on what we define";
each grounded class becomes a reusable building block). define_fillers retrieves the top-k nearest anchors per
concept and injects them, so the agent grounds to cco:ont... / fhir:Patient instead of bfo:0000040. Retrieval  # coined-ok: prose/placeholder, not a real ref
is the stochastic SIGNAL; the grounding gate (chain-to-BFO at realize) still deterministically DISPOSES.

    uv run --no-sync python scripts/grounding_anchors.py build              # (re)build the accreting index
    uv run --no-sync python scripts/grounding_anchors.py query "parasitic plant"

Sources (CC0, fetched into build/grounding/ — regenerable, gitignored):
  CCO:  https://raw.githubusercontent.com/CommonCoreOntology/CommonCoreOntologies/develop/src/cco-iris/CommonCoreOntologiesMerged.ttl
  FHIR: https://hl7.org/fhir/R5/codesystem-fhir-types.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GROUND = REPO / "build" / "grounding"
CCO_TTL = GROUND / "cco-merged.ttl"
FHIR_JSON = GROUND / "fhir-codesystem.json"
OURS_OWL = REPO / "corpora" / "ontology" / "sdg-ontology.owl"
INDEX_PKL = GROUND / "anchors.pkl"
FHIR_NS = "http://hl7.org/fhir/"
SYSML_JSON = REPO / "build" / "sysml" / "sysml_foundation.json"
FOUNDATION_DIR = REPO / "build" / "foundation"
SYSML_NS = "https://signals.zndx.org/sdg/sysml#"
IAO_DEF = "http://purl.obolibrary.org/obo/IAO_0000115"

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _curie(iri: str, pfx: str) -> str:
    return f"{pfx}:{iri.rstrip('/').split('/')[-1].split('#')[-1]}"


def load_cco(path=CCO_TTL):
    """CCO classes → (iri, label, def, 'cco'); also returns the Information Content Entity IRI (FHIR bridge)."""
    import rdflib
    from rdflib import OWL, RDF, RDFS, URIRef
    g = rdflib.Graph().parse(str(path), format="turtle")
    ice, out = None, []
    for c in set(g.subjects(RDF.type, OWL.Class)):
        if not isinstance(c, URIRef):
            continue
        lbl = g.value(c, RDFS.label)
        if not lbl:
            continue
        if str(lbl).strip().lower() == "information content entity":
            ice = str(c)
        d = g.value(c, URIRef(IAO_DEF)) or g.value(c, RDFS.comment) or ""
        out.append((str(c), str(lbl), str(d), "cco"))
    return out, ice


def load_fhir(path=FHIR_JSON):
    """FHIR R5 types (capitalized + defined) → (iri, label, def, 'fhir'). Flattens the nested type tree."""
    if not Path(path).exists():
        return []
    out = []

    def walk(cs):
        for c in cs:
            code, defn = c.get("code", ""), c.get("definition", "")
            if code[:1].isupper() and defn:
                out.append((FHIR_NS + code, code, defn, "fhir"))
            if c.get("concept"):
                walk(c["concept"])
    walk(json.load(open(path)).get("concept", []))
    return out


def load_ours(path=OURS_OWL):
    """Our own grounded classes → (iri, label, def, 'sdg') — the accreting reuse corpus."""
    if not Path(path).exists():
        return []
    import rdflib
    from rdflib import OWL, RDF, RDFS, URIRef
    g = rdflib.Graph().parse(str(path))
    ns = None
    for c in g.subjects(RDF.type, OWL.Class):
        s = str(c)
        if "signals360" in s or "/sdg#" in s or s.endswith("/sdg"):
            ns = s.split("#")[0] + "#" if "#" in s else s.rsplit("/", 1)[0] + "/"
            break
    out = []
    for c in set(g.subjects(RDF.type, OWL.Class)):
        if not isinstance(c, URIRef) or (ns and not str(c).startswith(ns)):
            continue
        lbl = g.value(c, RDFS.label) or str(c).split("#")[-1].split("/")[-1]
        d = g.value(c, URIRef(IAO_DEF)) or g.value(c, RDFS.comment) or ""
        out.append((str(c), str(lbl), str(d), "sdg"))
    return out


def _foundation_terms(terms, default_source):
    """Domain-standard foundation terms (SysMLv2/WITSML/BRL-CAD…) → (iri, label, def, source-prefix).

    Each standard EXTENDS the BFO/CCO/FHIR foundation with domain vocabulary so the deriver can ground domain
    columns onto it (an enclosure → sysml:Part → cco:ont00000995; a wellbore → witsml:Wellbore → bfo:0000029) AND so
    the qdrant classifier can DISCRIMINATE domains. EPL/Apache/US-gov-clean: our glosses only, names drawn+cited."""
    import re
    out = []
    for t in terms:
        name, src = t["name"], t.get("source", default_source)
        gloss = t.get("gloss") or (
            f"{re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).lower()} — "
            f"a {t.get('domain', 'core')} concept grounded as {t['genus']}")
        out.append((f"https://signals.zndx.org/sdg/{src}#{name}", name, gloss, src))
    return out


def load_sysml(path=SYSML_JSON):
    """SysMLv2 foundation seed (build/sysml/sysml_foundation.json, from sysml_foundation.py)."""
    return _foundation_terms(json.load(open(path)), "sysml") if Path(path).exists() else []


def load_domain_foundations(d=FOUNDATION_DIR):
    """Curated domain-standard foundations (WITSML, BRL-CAD…) from build/foundation/*_foundation.json
    (domain_foundations.py). Each file's terms carry a 'source' prefix (witsml / brlcad / …)."""
    out = []
    if Path(d).exists():
        for f in sorted(Path(d).glob("*_foundation.json")):
            out += _foundation_terms(json.load(open(f)), f.stem.split("_")[0])
    return out


def build_index(sources=("cco", "fhir", "ours", "sysml", "domains")):
    anchors, meta = [], {}
    if "cco" in sources:
        cco, ice = load_cco()
        anchors += cco
        meta["cco_ice"] = ice
    if "fhir" in sources:
        anchors += load_fhir()
    if "ours" in sources:
        anchors += load_ours()
    if "sysml" in sources:
        anchors += load_sysml()
    if "domains" in sources:
        anchors += load_domain_foundations()
    seen, uniq = set(), []
    for a in anchors:
        if a[0] not in seen:
            seen.add(a[0])
            uniq.append(a)
    iris = [a[0] for a in uniq]
    labels = [a[1] for a in uniq]
    defs = [a[2] for a in uniq]
    prefixes = [a[3] for a in uniq]
    texts = [f"{l}. {d}"[:300] for l, d in zip(labels, defs)]
    emb = _model().encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    INDEX_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump({"iris": iris, "labels": labels, "defs": defs, "prefixes": prefixes,
                     "curies": [_curie(i, p) for i, p in zip(iris, prefixes)],
                     "emb": np.asarray(emb, dtype=np.float32), "meta": meta}, f)
    from collections import Counter
    return len(iris), dict(Counter(prefixes))


class Retriever:
    """Top-k grounded anchors for a concept — the boundary's signal carrier. In-memory cosine over the pickled
    matrix; fine at ~2k anchors. SCALE PATH (as our ontology accretes): swap this for a dedicated qdrant
    collection behind the same ``retrieve()`` contract — index CCO+FHIR once, upsert each newly-grounded sdg
    class so the corpus grows server-side without a full re-embed (qdrant @6355 already runs for the SKOS index)."""

    def __init__(self, path=INDEX_PKL):
        with open(path, "rb") as f:
            self.ix = pickle.load(f)
        self.emb = self.ix["emb"]

    def retrieve(self, text, k=6, prefixes=None, exclude=()):
        ex = {e.lower() for e in exclude}  # skip self-matches — never offer the filler its own name as a genus
        q = _model().encode([text], normalize_embeddings=True)[0]
        sims = self.emb @ q
        out = []
        for i in np.argsort(-sims):
            if prefixes and self.ix["prefixes"][i] not in prefixes:
                continue
            if self.ix["labels"][i].lower() in ex or self.ix["curies"][i].lower() in ex:
                continue
            out.append({"label": self.ix["labels"][i], "curie": self.ix["curies"][i],
                        "prefix": self.ix["prefixes"][i], "sim": float(sims[i]), "def": self.ix["defs"][i]})
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "query"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("-k", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "build":
        n, by = build_index()
        print(f"built {n} grounding anchors {by} → {INDEX_PKL}")
    else:
        for r in Retriever().retrieve(a.text, a.k):
            print(f"  {r['sim']:.3f} [{r['prefix']}] {r['label']:32s} {r['curie']}   {r['def'][:60]}")
