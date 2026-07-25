"""Ontology/SKOS domain index — ColBERT late-interaction over Qdrant for content-first input filtering.

The domain hierarchy is OUR SKOS ConceptScheme (``corpora/vocabulary/vocabulary.ttl``: ~549 concepts with
``skos:broader`` + dotted ``skos:notation``). Each concept is a domain/subdomain node; its ColBERT
multi-vector encodes prefLabel + altLabel + definition. A streamed document is encoded the same way and
scored against the index via Qdrant **native MaxSim**, yielding the nearest concept(s) → a hierarchical
roll-up (the notation path) + a belief. So the input filter is GROUNDED IN THE ONTOLOGY:

  * to expand coverage into a new domain you FIRST expand the SKOS hierarchy — which seeds derivation
    (the deriver grounds its content-extracted primitives in the seeded subtree); and
  * the classifier's **separation power is an early warning**: if concepts don't separate, the SKOS
    hierarchy needs specification/augmentation before its Qdrant entries have real classification power.

Mirrors the Atelier classify pattern (shared owner): ColBERTv2 (``colbert_encoder``) + Qdrant multivector
MaxSim. Qdrant runs on the aegir devenv ports (HTTP 6355).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6355"
DEFAULT_COLLECTION = "sdg_domains"      # full vocab (catalog + domains) — for diagnose + the ontology SKOS
DEFAULT_APERTURE = "sdg_aperture"       # domains-only AIMING collection — the harvest classifies against THIS
# (the abstract foundation catalog in the full vocab otherwise absorbs the top concept — see the
#  manufacturing/CSG routing finding: catalog roots 0-3 took 136/250 raw docs; removing them sharpened margins ~7x)
_REPO = Path(__file__).resolve().parents[3]
DEFAULT_VOCAB = _REPO / "corpora" / "vocabulary" / "vocabulary.ttl"
# locally-authored rich DOMAIN concepts (the aperture targets); loaded ON TOP of the generated vocab.
DEFAULT_OVERLAY = Path(__file__).resolve().parent / "domain_concepts.ttl"


@dataclass
class SkosConcept:
    iri: str
    code: str = ""              # skos:notation, dotted (e.g. "2.1.3") — encodes the hierarchy
    pref_label: str = ""
    alt_label: str = ""
    definition: str = ""
    scope_note: str = ""
    comment: str = ""
    broader: str = ""           # parent IRI
    deprecated: bool = False    # renamed-id bridge: owl:deprecated true
    replaced_by: str = ""       # dct:isReplacedBy target IRI (the CURRENT concept)
    top_concept_of: str = ""    # skos:topConceptOf — S7/S8: declared scheme roots
    in_scheme: str = ""

    def text(self) -> str:
        """The text encoded into the concept's ColBERT multi-vector. ALL descriptive content is included —
        MaxSim scores documents against this text, so richer definitions = more classification power (it is
        the content, not the label, that classifies)."""
        return ". ".join(p for p in (self.pref_label, self.alt_label, self.definition,
                                     self.scope_note, self.comment) if p).strip()

    def ancestor_codes(self) -> "list[str]":
        """Dotted-notation prefixes that are this concept's ancestors (e.g. '2.1.3' → ['2','2.1'])."""
        if not self.code:
            return []
        parts = self.code.split(".")
        return [".".join(parts[:i]) for i in range(1, len(parts))]


# ── SKOS loading ────────────────────────────────────────────────────────────────

_BLOCK = re.compile(r"<([^>]+)>\s+a\s+skos:Concept\s*;(.*?)(?:\.\s*\n)", re.S)
_PREF = re.compile(r'skos:prefLabel\s+"((?:[^"\\]|\\.)*)"')
_ALT = re.compile(r'skos:altLabel\s+"((?:[^"\\]|\\.)*)"')
_DEF = re.compile(r'skos:definition\s+"((?:[^"\\]|\\.)*)"')
_SCOPE = re.compile(r'skos:scopeNote\s+"((?:[^"\\]|\\.)*)"')
_COMMENT = re.compile(r'rdfs:comment\s+"((?:[^"\\]|\\.)*)"')
_NOTE = re.compile(r'skos:notation\s+"((?:[^"\\]|\\.)*)"')
_REPL = re.compile(r'dct:isReplacedBy <([^>]+)>')
_TOPOF = re.compile(r'skos:topConceptOf <([^>]+)>')
_INSCH = re.compile(r'skos:inScheme <([^>]+)>')
_BROADER = re.compile(r"skos:broader\s+<([^>]+)>")


def load_skos(path: "str | Path" = DEFAULT_VOCAB,
              *, overlays: "tuple | list | None" = (DEFAULT_OVERLAY,)) -> "dict[str, SkosConcept]":
    """Parse the SKOS ConceptScheme TTL(s) → {iri: SkosConcept}, the generated vocab plus locally-authored
    rich DOMAIN overlays (later files win on IRI collision). Regex-based (avoids an rdflib dep); the
    generated file's shape is stable and the overlay follows the same block form."""
    out: dict[str, SkosConcept] = {}
    for p in [path, *(overlays or ())]:
        fp = Path(p)
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        for m in _BLOCK.finditer(text):
            iri, body = m.group(1), m.group(2)
            def _g(rx, _b=body):  # noqa: E306
                mm = rx.search(_b)
                return mm.group(1) if mm else ""
            out[iri] = SkosConcept(iri=iri, code=_g(_NOTE), pref_label=_g(_PREF), alt_label=_g(_ALT),
                                   definition=_g(_DEF), scope_note=_g(_SCOPE), comment=_g(_COMMENT),
                                   broader=_g(_BROADER),
                                   deprecated="owl:deprecated true" in body,
                                   replaced_by=_g(_REPL),
                                   top_concept_of=_g(_TOPOF), in_scheme=_g(_INSCH))
    return out


def subtree_codes(concepts: "dict[str, SkosConcept]", root: str) -> "set[str]":
    """All notation codes in the subtree rooted at ``root`` — an IDENTITY selector only:
    a notation code, an IRI, or an IRI-local. Labels are display, never structure
    (RH 2026-07-21); a label passed here raises with the matching identities named,
    so callers migrate instead of silently depending on name resolution."""
    root_code = root
    if not re.fullmatch(r"[0-9.]+", root):
        hit = concepts.get(root) or next(
            (c for c in concepts.values()
             if c.iri == root or c.iri.rsplit("#", 1)[-1] == root), None)
        if hit is None:
            rl = root.strip().lower()
            named = [c for c in concepts.values()
                     if c.pref_label.lower() == rl or c.alt_label.lower() == rl]
            raise KeyError(
                f"subtree_codes takes an identity (notation code / IRI / IRI-local), got {root!r}"
                + (f" — did you mean: " + ", ".join(
                    f"{c.iri.rsplit('#', 1)[-1]} (code {c.code})" for c in named[:3])
                   if named else ""))
        root_code = hit.code
    return {c.code for c in concepts.values()
            if c.code == root_code or c.code.startswith(root_code + ".")}


# ── integration overlays (in-scope-external contributor schemes) ─────────────────

def _sot_admission_filter() -> dict:
    """The SoT admission_filter.json, unresolved — AUTHORING-time membership.
    (armed_admission_filter resolves the RELEASED strategy copy for classify-time
    policy; seeding and contract measurement must see membership edits immediately.)"""
    import json as _json
    try:
        return _json.loads((Path(__file__).resolve().parent
                            / "admission_filter.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent SoT = no includes, never a crash
        return {}


def integration_overlay_files() -> "list[Path]":
    """The canonical in-scope-external integration overlays: ``*_integration.ttl``.

    File discovery is DECOUPLED from point enumeration (a file may contribute
    schemes and chord structure with zero enumerated points — energistics); which
    concepts become aperture points remains solely aperture_include's business."""
    return sorted(Path(__file__).resolve().parent.glob("*_integration.ttl"))


def integration_overlay_concepts() -> "dict[str, SkosConcept]":
    """ALL skos:Concept subjects across the integration overlays, rdflib-read.

    The integration TTLs (fibo/fhir) use prefixed subjects + multi-typing
    (``sdg:FINTECH a skos:Concept, sdg:FIBOConcept``) — outside load_skos's regex
    block form, so they get a real parser. alt_label is the fragment-equal
    skos:altLabel when present (the chord convention: the fragment IS the chord)."""
    out: "dict[str, SkosConcept]" = {}
    try:
        import rdflib
    except Exception:  # noqa: BLE001 — no rdflib = no integration overlays
        return out
    SK = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    for f in integration_overlay_files():
        g = rdflib.Graph()
        try:
            g.parse(str(f))
        except Exception as e:  # noqa: BLE001 — an unreadable overlay is VISIBLE
            print(f"integration overlay unreadable: {f.name}: {e}")
            continue
        for node in set(g.subjects(rdflib.RDF.type, SK.Concept)):
            iri = str(node)
            frag = iri.rsplit("#", 1)[-1]

            def _v(pred) -> str:
                vals = [str(o) for o in g.objects(node, pred)]
                return vals[0] if vals else ""

            alts = [str(o) for o in g.objects(node, SK.altLabel)]
            out[iri] = SkosConcept(
                iri=iri, code=_v(SK.notation), pref_label=_v(SK.prefLabel),
                alt_label=next((a for a in alts if a == frag), frag),
                definition=_v(SK.definition), scope_note=_v(SK.scopeNote),
                broader=_v(SK.broader))
    return out


# ── Qdrant multivector index ─────────────────────────────────────────────────────

def _client(url: str = DEFAULT_QDRANT_URL):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, timeout=60)


def build_index(*, vocab: "str | Path" = DEFAULT_VOCAB, url: str = DEFAULT_QDRANT_URL,
                collection: str = DEFAULT_COLLECTION, recreate: bool = True) -> dict:
    """Encode every SKOS concept's text → ColBERT multi-vectors → a Qdrant MAX_SIM collection. The ontology
    IS the index.

    When seeding the AIMING aperture: (a) the overlay base is NOTATION-GATED — only
    concepts in the aperture code space seed points (no-notation scheme organization
    like PRODML/SYSML never enters MaxSim competition); (b) the SoT admission_filter's
    ``aperture_include`` members (ENUMERATED — §4.6.3 discipline: membership is never
    inferred from scheme structure) are appended AFTER the overlay concepts, so
    existing point ids stay stable and the includes take the tail ids."""
    from qdrant_client import models

    from aegir.ontology.colbert_encoder import get_encoder

    concepts = load_skos(vocab)
    if collection == DEFAULT_APERTURE:
        # NOTATION-GATED base: only concepts in the aperture CODE SPACE (skos:notation)
        # are point candidates — confirmed schemes' tops/children without notations
        # (PRODML/SYSML) are S9 organization, NOT admission points (measured 2026-07-25:
        # an ungated reseed silently admitted 10 of them as MaxSim competitors).
        concepts = {iri: c for iri, c in concepts.items() if getattr(c, "code", "")}
        # seeding is AUTHORING: membership comes from the SoT file, not the released
        # strategy (armed_admission_filter pins CLASSIFY-time policy and lags edits)
        inc = integration_overlay_concepts()
        for m in _sot_admission_filter().get("aperture_include", {}).get("members", []):
            iri = str(m.get("iri", ""))
            if iri in inc and iri not in concepts:
                concepts[iri] = inc[iri]
            elif iri not in inc:
                print(f"aperture include NOT FOUND in integration overlays: {iri}")
    enc = get_encoder()
    client = _client(url)
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=enc.dim, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
        )
    items = [c for c in concepts.values() if c.text() and not c.deprecated]
    # deprecated bridge concepts NEVER enter the admission surface: same-label twins would
    # duplicate MaxSim anchors; bridges exist for REFERENCE RESOLUTION, not retrieval
    texts = [c.text() for c in items]
    vecs = enc.encode(texts)
    points = []
    for i, (c, v) in enumerate(zip(items, vecs)):
        points.append(models.PointStruct(id=i, vector=v.tolist(), payload={
            "iri": c.iri, "code": c.code, "pref_label": c.pref_label, "alt_label": c.alt_label,
            "broader": c.broader, "ancestor_codes": c.ancestor_codes(),
            # the EXACT encoded text + its token count (RH 2026-07-22): the panel's
            # "In the vector" section renders from the released snapshot, never recompute
            "retrieval_text": texts[i],
            "retrieval_tokens": int(len(enc._tokenizer(texts[i])["input_ids"]))}))
    # batch the upsert — all 548 multivectors at once exceeds Qdrant's 32 MB JSON payload limit
    for start in range(0, len(points), 64):
        client.upsert(collection_name=collection, points=points[start:start + 64])
    return {"collection": collection, "concepts": len(items), "dim": enc.dim, "url": url}


def classify(text: str, *, url: str = DEFAULT_QDRANT_URL, collection: str = DEFAULT_COLLECTION,
             top_k: int = 5, exclude_codes: "set | list | None" = None) -> "list[dict]":
    """Encode a document and return the top-k nearest SKOS concepts (Qdrant native MaxSim), each as
    {code, pref_label, score, ancestor_codes, iri}. ``exclude_codes`` applies the S9-settled
    admission recomposition (RH 2026-07-23) as a QUERY-TIME payload filter — the collection
    (which is also the topic identity space) is never rebuilt for admission policy."""
    from aegir.ontology.colbert_encoder import get_encoder
    enc = get_encoder()
    client = _client(url)
    q = enc.encode_single(text).tolist()
    flt = None
    if exclude_codes:
        from qdrant_client import models as _m
        flt = _m.Filter(must_not=[_m.FieldCondition(key="code",
                                                    match=_m.MatchAny(any=sorted(exclude_codes)))])
    res = client.query_points(collection_name=collection, query=q, limit=top_k,
                              query_filter=flt, with_payload=True).points
    return [{"score": float(p.score), **(p.payload or {})} for p in res]


def armed_admission_filter(ref: "str | None" = None) -> dict:
    """The admission-surface recomposition component (RH 2026-07-23), strategy-resolved.

    Reads ``lens/admission_filter.json`` from the RELEASED strategy (ref-aware:
    AEGIR_STRATEGY_REF routes shadow runs) — falling back to the aegir SoT copy before
    the first release. Returns the component dict plus ``effective_exclude``: the
    exclude set WHEN armed (component ``armed`` flag, or ``AEGIR_ADMISSION_FILTER=1``
    force-arming for shadow windows); empty when disarmed or absent. Absent component
    = no filter — admission behaves exactly as before staging."""
    import json as _json
    import os as _os
    rec: dict = {}
    try:
        from aegir.strategy.manifest import read_component
        rec = _json.loads(read_component("lens/admission_filter.json",
                                         ref or _os.environ.get("AEGIR_STRATEGY_REF") or None))
    except Exception:  # noqa: BLE001 — pre-release: the SoT copy
        try:
            rec = _json.loads((Path(__file__).resolve().parent
                               / "admission_filter.json").read_text())
        except Exception:  # noqa: BLE001
            return {"effective_exclude": set()}
    armed = bool(rec.get("armed")) or _os.environ.get("AEGIR_ADMISSION_FILTER") == "1"
    rec["effective_exclude"] = set(rec.get("exclude_codes") or []) if armed else set()
    return rec


def classify_hierarchical(text: str, *, top_k: int = 5, **kw) -> dict:
    """Top concept + belief + the hierarchical roll-up. ``belief`` = top-1 share of the top-k MaxSim mass;
    a flat distribution (low belief / low margin) means the SKOS hierarchy lacks separating power here."""
    hits = classify(text, top_k=top_k, **kw)
    if not hits:
        return {"hits": [], "top": None, "belief": 0.0, "margin": 0.0}
    scores = [h["score"] for h in hits]
    total = sum(scores) or 1.0
    margin = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]
    # rel_margin = rank1−rank2 as a fraction of rank1 — length-robust (raw MaxSim scales with doc length,
    # belief stays flat when several siblings score high), so it is the gate's confidence signal.
    rel_margin = (margin / scores[0]) if (len(scores) > 1 and scores[0]) else 1.0
    top = hits[0]
    return {"hits": hits, "top": top, "belief": round(scores[0] / total, 3), "margin": round(margin, 3),
            "rel_margin": round(rel_margin, 3),
            "path": top.get("ancestor_codes", []) + [top.get("code", "")]}


def in_subtree(hit: dict, root_codes: "set[str]") -> bool:
    code = hit.get("code", "")
    return code in root_codes or any(a in root_codes for a in hit.get("ancestor_codes", []))


# ── early-warning diagnostic: does the SKOS hierarchy have classification power? ──

def separation_report(*, vocab: "str | Path" = DEFAULT_VOCAB, url: str = DEFAULT_QDRANT_URL,
                      collection: str = DEFAULT_COLLECTION, sample: int = 0) -> dict:
    """Self-retrieval + margin diagnostic. For each concept, classify its OWN text and check whether it
    retrieves itself at rank-1, and the rank1−rank2 margin. Low self-retrieval / low margin = concepts that
    are not separable → the SKOS hierarchy needs specification/augmentation before its entries can classify
    real input. Returns aggregate metrics + the most-confusable concept pairs (the augmentation worklist)."""
    concepts = [c for c in load_skos(vocab).values() if c.text()]
    if sample:
        concepts = concepts[:sample]
    self_hits = margins = 0
    margin_sum = 0.0
    confusions: list[dict] = []
    for c in concepts:
        h = classify_hierarchical(c.text(), top_k=3, url=url, collection=collection)
        top = h.get("top") or {}
        is_self = top.get("code") == c.code or top.get("pref_label") == c.pref_label
        self_hits += int(is_self)
        margin_sum += h["margin"]
        margins += 1
        if not is_self:
            confusions.append({"concept": c.pref_label, "code": c.code,
                               "confused_with": top.get("pref_label"), "with_code": top.get("code"),
                               "margin": h["margin"]})
    n = len(concepts) or 1
    confusions.sort(key=lambda x: x["margin"])
    return {"concepts": len(concepts), "self_retrieval": round(self_hits / n, 3),
            "mean_margin": round(margin_sum / max(1, margins), 4),
            "confusable": confusions[:25],
            "verdict": ("hierarchy has classification power" if self_hits / n >= 0.85
                        else "WEAK — SKOS hierarchy needs specification/augmentation (see confusable)")}
