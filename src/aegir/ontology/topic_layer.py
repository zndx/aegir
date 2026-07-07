"""The inverted topic layer — one passage, one topic, full lineage (RH 2026-07-07, a–h).

Topics ≡ ontology-grounded concept anchors in a qdrant collection (first-order fungible —
the collection IS the topic registry; no fitted topic-model artifact). A passage associates
with EXACTLY ONE topic by late-interaction MaxSim against the anchor surfaces, gated on
``rel_margin`` — or it is UNASSIGNED, and that ambiguous mass is the ERROR SIGNAL that
drives definitional-rigor iteration on the LEXICON side (the inverted-LDA direction: the
input is never the problem; the topics iterate). Every association is content-addressed to
the COLLECTION STATE (``collection_sha`` over the anchors' ids, codes, and vector hashes)
so input ↔ topic ↔ output lineage survives collection evolution — the association record
is the FedWiki-journal provenance for the item (passage).

Design note: ``docs/scratch/2026-07-07/033241_inverted_topic_layer_design.md``.
Composes on :mod:`aegir.ontology.domain_index` (the ColBERT/qdrant MaxSim substrate).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aegir.ontology.domain_index import DEFAULT_APERTURE, DEFAULT_QDRANT_URL, _client, classify

DEFAULT_TAU = 0.10           # rel_margin gate — matches the harvest/derive aperture default
REPO = Path(__file__).resolve().parents[3]


def passage_hash(text: str) -> str:
    """The canonical content address for a passage (whitespace-normalized sha256 —
    mirrors the harvest store, so records join `build/domain_harvest/docs/<hash>.txt`)."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def collection_state(*, url: str = DEFAULT_QDRANT_URL,
                     collection: str = DEFAULT_APERTURE) -> dict:
    """Content-address the topic registry: the qdrant collection's anchors as sorted
    ``(id, code, iri, label, vector_sha)`` identity rows + a 16-hex digest over them.

    This closes the lens-snapshot gap (which hashed ``(id, label, vector_sha)`` only):
    the ontology NOTATION rides along, so an association record names its topic in both
    the ontology's terms and the vector state that adjudicated it.

    Each row also carries ``n_tokens`` — the anchor's ColBERT multivector length IS its
    token count, so the collection DECLARES its own granularity; ``anchor_tokens_p50``
    is the windowing basis (RH: the intra-document window token-count is proportional
    to the collection-item token-count). ``n_tokens`` annotates but does not enter the
    identity digest (the vector hash already changes when the anchor text changes)."""
    import numpy as np
    cl = _client(url)
    rows: list[dict] = []
    offset = None
    while True:
        points, offset = cl.scroll(collection_name=collection, limit=512,
                                   with_payload=True, with_vectors=True, offset=offset)
        for p in points:
            pl = p.payload or {}
            vec = np.asarray(p.vector, dtype=np.float32)
            rows.append({"id": str(p.id), "code": pl.get("code", ""), "iri": pl.get("iri", ""),
                         "label": pl.get("pref_label") or pl.get("label") or "",
                         "vector_sha": hashlib.sha256(vec.tobytes()).hexdigest()[:16],
                         "n_tokens": int(vec.shape[0])})
        if offset is None:
            break
    rows.sort(key=lambda r: r["id"])
    identity = [{k: r[k] for k in ("id", "code", "iri", "label", "vector_sha")} for r in rows]
    sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    tok = sorted(r["n_tokens"] for r in rows) or [0]
    return {"collection": collection, "url": url, "n_topics": len(rows), "sha": sha,
            "encoder": ENCODER_ID, "projection": "-",   # lens identity (L1); vectors already pin it
            "anchor_tokens_p50": tok[len(tok) // 2],
            "anchor_tokens_min": tok[0], "anchor_tokens_max": tok[-1], "rows": rows}


# ── intra-document sliding windows — the FedWiki ITEM granularity ────────────────
def window_plan(state: dict, *, ratio: float = 2.0, stride_frac: float = 0.5) -> dict:
    """Window sizing from the collection's own declared granularity: the item
    (query window) token-count is PROPORTIONAL to the anchor token-count (RH ruling) —
    late interaction is well-conditioned when both sides speak at comparable length,
    and raw MaxSim otherwise scales with document length."""
    w = max(32, round(ratio * state["anchor_tokens_p50"]))
    return {"window_tokens": w, "stride_tokens": max(1, round(w * stride_frac)),
            "ratio": ratio, "anchor_tokens_p50": state["anchor_tokens_p50"]}


def windows(text: str, *, window_tokens: int, stride_tokens: int) -> "list[dict]":
    """Slice a document into token-windowed ITEMS using the SAME tokenizer MaxSim sees;
    char spans index the ORIGINAL text (no decode round-trip). The tail is always
    covered. A short document yields one item (itself)."""
    from aegir.ontology.colbert_encoder import get_encoder
    tok = get_encoder()._tokenizer
    encd = tok(text, add_special_tokens=False, return_offsets_mapping=True,
               truncation=False, verbose=False)
    offs = encd["offset_mapping"]
    n = len(offs)
    if n == 0:
        return []
    starts = list(range(0, max(1, n - window_tokens + 1), stride_tokens))
    if starts[-1] + window_tokens < n:                     # tail coverage
        starts.append(max(0, n - window_tokens))
    out: list[dict] = []
    seen: set[int] = set()
    for idx, s in enumerate(starts):
        if s in seen:
            continue
        seen.add(s)
        e = min(n, s + window_tokens)
        c0, c1 = offs[s][0], offs[e - 1][1]
        out.append({"text": text[c0:c1], "span": [int(c0), int(c1)],
                    "index": idx, "n_tokens": e - s})
    return out


@dataclass
class Association:
    """One item (passage window) ↔ one topic, with the full adjudication context —
    assigned or not. An UNASSIGNED record still names the top candidate and every
    margin: the ambiguous mass must be inspectable, because it is what the
    definitional-rigor loop consumes."""
    passage_hash: str                    # the ITEM's content address (window text)
    assigned: bool
    topic_code: str
    topic_iri: str
    topic_label: str
    topic_path: list                     # ancestor roll-up + code
    score: float
    margin: float                        # flat: rank1 − rank2
    rel_margin: float
    competitor_code: str                 # nearest NON-ancestor competitor (hierarchical)
    competitor_score: float
    rel_margin_h: float                  # THE gate signal — subtree-aware unambiguity
    collection: str
    collection_sha: str
    tau: float
    encoder: str = ""                    # lens identity (L1): the encoder that adjudicated
    projection: str = "-"                # "-" = identity/ColBERT lens (no learned projection)
    unit: str = "item"                   # item (window) | doc (whole passage)
    doc_hash: str = ""                   # the containing document's content address
    item_index: int = 0
    item_span: list = field(default_factory=list)
    n_tokens: int = 0
    n_chars: int = 0
    source: str = ""                     # provenance hint (e.g. the harvest doc path)
    extra: dict = field(default_factory=dict)


def _same_lineage(a: dict, b: dict) -> bool:
    """True when one hit is an ancestor of the other (or the same concept)."""
    ac = {a.get("code", "")} | set(a.get("ancestor_codes") or [])
    bc = {b.get("code", "")} | set(b.get("ancestor_codes") or [])
    return b.get("code", "") in ac or a.get("code", "") in bc


def _adjudicate(hits: "list[dict]", *, tau: float, collection: str, collection_sha: str,
                text: str, source: str = "", unit: str = "doc", doc_hash: str = "",
                item_index: int = 0, item_span: "list | None" = None,
                n_tokens: int = 0) -> Association:
    """Shared adjudication: flat margin (rank1−rank2) for the record, HIERARCHICAL
    margin (rank1 − nearest NON-ancestor competitor) for the gate — a parent/child
    near-miss is not ambiguity, the subtree is the unambiguous assignment (increment 2;
    the harvest gate's ``in_subtree`` implied this all along). No non-ancestor
    competitor in the top-k ⇒ rel_margin_h = 1.0 (maximally unambiguous at this k)."""
    from typing import Any
    ph = passage_hash(text)
    base: "dict[str, Any]" = dict(
        passage_hash=ph, collection=collection, collection_sha=collection_sha,
        tau=tau, encoder=ENCODER_ID, projection="-", unit=unit, doc_hash=doc_hash or ph,
        item_index=item_index, item_span=item_span or [], n_tokens=n_tokens,
        n_chars=len(text), source=source)
    if not hits:
        return Association(assigned=False, topic_code="", topic_iri="", topic_label="",
                           topic_path=[], score=0.0, margin=0.0, rel_margin=0.0,
                           competitor_code="", competitor_score=0.0, rel_margin_h=0.0, **base)
    top = hits[0]
    s0 = float(top["score"])
    s1 = float(hits[1]["score"]) if len(hits) > 1 else 0.0
    comp = next((h for h in hits[1:] if not _same_lineage(top, h)), None)
    sc = float(comp["score"]) if comp else 0.0
    rel_margin = ((s0 - s1) / s0) if s0 else 0.0
    rel_margin_h = ((s0 - sc) / s0) if s0 else 0.0
    return Association(
        assigned=bool(rel_margin_h >= tau),
        topic_code=top.get("code", ""), topic_iri=top.get("iri", ""),
        topic_label=top.get("pref_label", ""),
        topic_path=(top.get("ancestor_codes") or []) + [top.get("code", "")],
        score=round(s0, 4), margin=round(s0 - s1, 4), rel_margin=round(rel_margin, 4),
        competitor_code=(comp.get("code", "") if comp else ""),
        competitor_score=round(sc, 4), rel_margin_h=round(rel_margin_h, 4), **base)


def associate(text: str, *, url: str = DEFAULT_QDRANT_URL,
              collection: str = DEFAULT_APERTURE, collection_sha: str = "",
              tau: float = DEFAULT_TAU, top_k: int = 5, source: str = "") -> Association:
    """Associate one whole passage with exactly one topic — or refuse.

    MaxSim adjudicates (qdrant-native late interaction); the hierarchical margin gate
    makes the assignment UNAMBIGUOUS-or-nothing, per ruling (c). Prefer
    :func:`associate_items` — the item (window) granularity is the design's unit."""
    hits = classify(text, url=url, collection=collection, top_k=top_k)
    return _adjudicate(hits, tau=tau, collection=collection, collection_sha=collection_sha,
                       text=text, source=source, unit="doc")


def associate_items(text: str, *, url: str = DEFAULT_QDRANT_URL,
                    collection: str = DEFAULT_APERTURE, state: "dict | None" = None,
                    tau: float = DEFAULT_TAU, top_k: int = 5, ratio: float = 2.0,
                    stride_frac: float = 0.5, source: str = "",
                    window_tokens: int = 0) -> "list[Association]":
    """Window a document into anchor-proportional ITEMS and associate each — the
    ratified granularity (items ≡ topic-windows). Windows batch-encode through the
    shared ColBERT encoder; each item gets its own qdrant MaxSim adjudication.
    ``window_tokens`` pins the window explicitly (A/B runs against an evolved registry
    must pin to the baseline plan, or the per-item span join breaks)."""
    from aegir.ontology.colbert_encoder import get_encoder
    state = state or collection_state(url=url, collection=collection)
    plan = window_plan(state, ratio=ratio, stride_frac=stride_frac)
    if window_tokens:
        plan = {"window_tokens": window_tokens,
                "stride_tokens": max(1, round(window_tokens * stride_frac))}
    items = windows(text, window_tokens=plan["window_tokens"],
                    stride_tokens=plan["stride_tokens"])
    if not items:
        return []
    enc = get_encoder()
    cl = _client(url)
    vecs = enc.encode([it["text"] for it in items])
    dh = passage_hash(text)
    out: list[Association] = []
    for it, v in zip(items, vecs):
        res = cl.query_points(collection_name=collection, query=v.tolist(),
                              limit=top_k, with_payload=True).points
        hits = [{"score": float(p.score), **(p.payload or {})} for p in res]
        out.append(_adjudicate(hits, tau=tau, collection=collection,
                               collection_sha=state["sha"], text=it["text"], source=source,
                               unit="item", doc_hash=dh, item_index=it["index"],
                               item_span=it["span"], n_tokens=it["n_tokens"]))
    return out


def alignment_report(records: "list[Association] | list[dict]") -> dict:
    """The topic-quality gate's numbers: unambiguous-alignment rate + margin distribution
    + per-topic mass. Low alignment / thin margins indict the LEXICON (ruling f), so the
    per-topic and ambiguous views are what the rigor loop reads."""
    rows = [asdict(r) if isinstance(r, Association) else r for r in records]
    n = len(rows)
    if not n:
        return {"n": 0}
    assigned = [r for r in rows if r["assigned"]]
    margins = sorted(r["rel_margin_h"] for r in rows)
    pct = lambda q: margins[min(n - 1, int(q * n))]  # noqa: E731
    by_topic: dict[str, int] = {}
    for r in assigned:
        key = f"{r['topic_code']} {r['topic_label']}".strip()
        by_topic[key] = by_topic.get(key, 0) + 1
    ambiguous = sorted((r for r in rows if not r["assigned"]),
                       key=lambda r: -r["rel_margin_h"])[:10]
    docs = {r["doc_hash"] for r in rows if r.get("doc_hash")}
    return {
        "n": n, "n_docs": len(docs), "n_assigned": len(assigned),
        "alignment_rate": round(len(assigned) / n, 4),
        "rel_margin_h_p10": round(pct(0.10), 4), "rel_margin_h_p50": round(pct(0.50), 4),
        "rel_margin_h_p90": round(pct(0.90), 4),
        "topics_hit": len(by_topic),
        "by_topic": dict(sorted(by_topic.items(), key=lambda kv: -kv[1])),
        "near_misses": [{"item": r["passage_hash"][:16], "doc": (r.get("doc_hash") or "")[:16],
                         "top": r["topic_code"], "competitor": r["competitor_code"],
                         "rel_margin_h": r["rel_margin_h"]}
                        for r in ambiguous],
    }


def associate_store(docs_dir: "str | Path", *, url: str = DEFAULT_QDRANT_URL,
                    collection: str = DEFAULT_APERTURE, tau: float = DEFAULT_TAU,
                    limit: int = 0, out_dir: "str | Path | None" = None,
                    unit: str = "item", ratio: float = 2.0,
                    stride_frac: float = 0.5,
                    window_tokens: int = 0) -> dict:
    """Associate every passage in a content-addressed store (``<hash>.txt`` files, e.g.
    the harvest's ``build/domain_harvest/docs``) and persist the lineage:
    ``build/topic_associations/<collection>-<collection_sha>/associations.jsonl`` +
    ``report.json``. The out-dir is keyed by the collection STATE, so re-running against
    an evolved registry lands beside — never over — the old adjudications.
    ``unit="item"`` (default) windows each document to the anchor-proportional item
    granularity; ``unit="doc"`` adjudicates whole passages (the increment-1 behavior)."""
    docs = sorted(Path(docs_dir).glob("*.txt"))
    if limit:
        docs = docs[:limit]
    state = collection_state(url=url, collection=collection)
    plan = window_plan(state, ratio=ratio, stride_frac=stride_frac)
    if window_tokens:
        plan = {"window_tokens": window_tokens,
                "stride_tokens": max(1, round(window_tokens * stride_frac)),
                "ratio": None, "pinned": True,
                "anchor_tokens_p50": state["anchor_tokens_p50"]}
    out = Path(out_dir) if out_dir else (REPO / "build" / "topic_associations"
                                         / f"{collection}-{state['sha']}")
    out.mkdir(parents=True, exist_ok=True)
    records: list[Association] = []
    with (out / "associations.jsonl").open("w") as fh:
        for p in docs:
            text = p.read_text(errors="ignore")
            if unit == "item":
                recs = associate_items(text, url=url, collection=collection, state=state,
                                       tau=tau, ratio=ratio, stride_frac=stride_frac,
                                       source=p.name, window_tokens=window_tokens)
            else:
                recs = [associate(text, url=url, collection=collection,
                                  collection_sha=state["sha"], tau=tau, source=p.name)]
            records.extend(recs)
            for rec in recs:
                fh.write(json.dumps(asdict(rec)) + "\n")
    report = {"collection": collection, "collection_sha": state["sha"],
              "n_topics": state["n_topics"], "tau": tau, "unit": unit,
              **({"window": plan} if unit == "item" else {}),
              **alignment_report(records)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "collection_state.json").write_text(json.dumps(state, indent=1))
    return report


# ── τ re-derivation (pre-registered) + the M7 basin gate ─────────────────────────
def derive_tau(docs_dir: "str | Path", *, url: str = DEFAULT_QDRANT_URL,
               collection: str = DEFAULT_APERTURE, alpha: float = 0.05,
               seed: int = 44641, window_tokens: int = 170, stride_frac: float = 0.5,
               limit: int = 0) -> dict:
    """PRE-REGISTERED τ derivation (docs/scratch/2026-07-07/165051_tau_prereg…md):
    τ* = the (1−α) quantile of ``rel_margin_h`` under the SHUFFLED-WINDOW NULL — the
    reference items with their topical coherence destroyed (uniform token shuffle,
    fixed seed) but vocabulary and length preserved. A shuffled window's surviving
    margin is stylistic/vocabulary leakage; τ* is the gate that excludes ≥(1−α) of it.
    Applies per lens identity; report-not-tune after the number is known."""
    import random
    from aegir.ontology.colbert_encoder import get_encoder
    rng = random.Random(seed)
    docs = sorted(Path(docs_dir).glob("*.txt"))
    if limit:
        docs = docs[:limit]
    state = collection_state(url=url, collection=collection)
    enc = get_encoder()
    cl = _client(url)
    stride = max(1, round(window_tokens * stride_frac))
    null_margins: list[float] = []
    for p in docs:
        text = p.read_text(errors="ignore")
        for it in windows(text, window_tokens=window_tokens, stride_tokens=stride):
            toks = it["text"].split()
            rng.shuffle(toks)
            null_margins.append(_adjudicate(
                [{"score": float(q.score), **(q.payload or {})} for q in
                 cl.query_points(collection_name=collection,
                                 query=enc.encode_single(" ".join(toks)).tolist(),
                                 limit=5, with_payload=True).points],
                tau=0.0, collection=collection, collection_sha=state["sha"],
                text=" ".join(toks), unit="null").rel_margin_h)
    null_margins.sort()
    n = len(null_margins)
    tau_star = null_margins[min(n - 1, int((1 - alpha) * n))] if n else 0.0
    return {"tau_star": round(tau_star, 4), "alpha": alpha, "seed": seed, "n_null": n,
            "null_p50": round(null_margins[n // 2], 4) if n else 0.0,
            "null_p95": round(null_margins[int(0.95 * n)], 4) if n else 0.0,
            "collection": collection, "collection_sha": state["sha"],
            "window_tokens": window_tokens}


def rescore_at_tau(assoc_dir: "str | Path", tau: float) -> dict:
    """Re-score an existing association run's records at a different τ (offline —
    records carry ``rel_margin_h``). Returns the alignment report at that τ."""
    rows = [json.loads(ln) for ln in
            (Path(assoc_dir) / "associations.jsonl").read_text().splitlines() if ln]
    for r in rows:
        r["assigned"] = bool(r.get("rel_margin_h", 0.0) >= tau)
    rep = alignment_report(rows)
    rep["tau"] = tau
    return rep


def basin_calibration(base_dir: "str | Path", cand_dir: "str | Path", *,
                      min_items: int = 5, max_ratio: float = 3.0,
                      tau: "float | None" = None) -> dict:
    """The M7 BASIN GATE (the taxi lesson, mechanized): compare two association runs
    over the SAME items (pinned window) and FLAG topics whose assigned-item count
    explodes (``new ≥ min_items`` AND ``new ≥ max_ratio × max(base, 1)``). The gate
    reports evidence (counts + the offending items with margins); disposition is the
    loop's — it never silently mutates. Optional ``tau`` re-scores both sides first."""
    def _load(d):
        rows = [json.loads(ln) for ln in
                (Path(d) / "associations.jsonl").read_text().splitlines() if ln]
        if tau is not None:
            for r in rows:
                r["assigned"] = bool(r.get("rel_margin_h", 0.0) >= tau)
        by: dict[str, list[dict]] = {}
        for r in rows:
            if r.get("assigned"):
                by.setdefault(r.get("topic_code", ""), []).append(r)
        return by
    base, cand = _load(base_dir), _load(cand_dir)
    offenders = []
    for code, rows in sorted(cand.items(), key=lambda kv: -len(kv[1])):
        b = len(base.get(code, []))
        if len(rows) >= min_items and len(rows) >= max_ratio * max(b, 1):
            # doc-concentration = the secondary evidence that resolved both live flags
            # (taxi, ancestor-veneration): N items from ONE long doc is topical
            # concentration, N items corpus-wide is a basin problem.
            docs: dict[str, int] = {}
            for r in rows:
                docs[(r.get("doc_hash") or "?")[:16]] = docs.get((r.get("doc_hash") or "?")[:16], 0) + 1
            top_doc, top_n = max(docs.items(), key=lambda kv: kv[1])
            offenders.append({
                "topic": code, "base": b, "new": len(rows),
                "n_docs": len(docs), "top_doc": top_doc,
                "top_doc_share": round(top_n / len(rows), 3),
                "reading": ("single-source concentration (inspect the doc, likely legitimate)"
                            if top_n / len(rows) >= 0.7 else
                            "corpus-wide attraction (inspect the anchor — basin suspect)"),
                "items": [{"item": r["passage_hash"][:16], "doc": (r.get("doc_hash") or "")[:16],
                           "rel_margin_h": r.get("rel_margin_h")} for r in rows[:12]]})
    return {"ok": not offenders, "min_items": min_items, "max_ratio": max_ratio,
            "tau": tau, "n_topics_base": len(base), "n_topics_cand": len(cand),
            "offenders": offenders}


# ── the TERM-grounded topic registry (increment 3) ───────────────────────────────
DEFAULT_REGISTRY = "sdg_topics"
_VOCAB_NS = "https://signals.zndx.org/sdg#"     # matches build_skos_vocab's scheme
ENCODER_ID = "colbert-ir/colbertv2.0"           # the lens's encoder identity (L1)

# Compact glosses for the BFO classes/relations the catalog's axioms reference by
# numeric IRI — declared genus knowledge that otherwise contributes ZERO anchor text.
_BFO_GLOSS = {
    "bfo:0000002": "continuant", "bfo:0000003": "occurrent",
    "bfo:0000004": "independent continuant", "bfo:0000015": "process",
    "bfo:0000016": "disposition", "bfo:0000017": "realizable entity",
    "bfo:0000019": "quality", "bfo:0000020": "specifically dependent continuant",
    "bfo:0000023": "role", "bfo:0000027": "object aggregate", "bfo:0000030": "object",
    "bfo:0000031": "generically dependent continuant", "bfo:0000040": "material entity",
    "bfo:0000050": "part of", "bfo:0000051": "has part", "bfo:0000052": "inheres in",
    "bfo:0000054": "realized in", "bfo:0000055": "realizes",
    "bfo:0000056": "participates in", "bfo:0000057": "has participant",
    "bfo:0000066": "occurs in",
}


def _curie_glosses() -> "dict[str, tuple[str, str]]":
    """curie → (label, definition) from the grounding-anchors index (CCO/FHIR/SysML/…),
    used to surface the vocabulary a term's axiom DECLARES by numeric reference.
    Graceful {} when the index hasn't been built."""
    import pickle
    p = REPO / "build" / "grounding" / "anchors.pkl"
    if not p.exists():
        return {}
    try:
        d = pickle.load(p.open("rb"))
        return {c: (l, df or "") for c, l, df in zip(d["curies"], d["labels"], d["defs"])}
    except Exception:  # noqa: BLE001
        return {}


def _axiom_vocabulary(manchester: str, glosses: "dict[str, tuple[str, str]]") -> str:
    """The human vocabulary a Manchester axiom references: resolved CCO/FHIR labels
    (+ short definitions when present), BFO glosses, and humanized sdg: properties.
    This is DECLARED content surfaced — no new claims are authored here."""
    import re as _re
    parts: list[str] = []
    seen: set[str] = set()
    for cu in _re.findall(r"(?:cco|fhir|sysml|witsml):[A-Za-z0-9_]+|bfo:\d{7}", manchester):
        if cu in seen:
            continue
        seen.add(cu)
        if cu in _BFO_GLOSS:
            parts.append(_BFO_GLOSS[cu])
        elif cu in glosses:
            label, df = glosses[cu]
            parts.append(f"{label}. {df[:140]}" if df else label)
    for prop in _re.findall(r"sdg:([a-z][A-Za-z0-9_]*)", manchester):
        h = prop.replace("_", " ")
        if h not in seen:
            seen.add(h)
            parts.append(h)
    return "; ".join(parts)


def term_anchor_text(t, glosses: "dict[str, tuple[str, str]] | None" = None) -> str:
    """THE anchor-text composition for a catalog term — the single source both the
    registry build and the authoring membrane use (the membrane must judge exactly
    what MaxSim will see). Full DECLARED surface + any membrane-admitted AUTHORED
    surface (``alt_labels`` / ``scope_note`` / elaborated frames — 4b)."""
    glosses = glosses if glosses is not None else _curie_glosses()
    prov = t.provenance or {}
    dom = prov.get("domain") if isinstance(prov.get("domain"), dict) else {}
    frames = (t.frames() if hasattr(t, "frames") else []) or []
    definition = " ".join(dict.fromkeys(frames[:3])) or (t.verbal_template or "")
    pref = t.template_id.replace("_", " ").removeprefix("filler ")
    alts = ", ".join([s.replace("_", " ") for s in (t.slot_types or {})]
                     + list(getattr(t, "alt_labels", None) or []))
    vocab = _axiom_vocabulary(t.manchester_template or "", glosses)
    span = prov.get("source_span") or ""
    grounded = f"Grounded in: {span[:300]}" if isinstance(span, str) and span else ""
    scope = ". ".join(p for p in (
        f"In the '{(dom or {}).get('label')}' domain" if dom else "",
        f"Pattern {prov.get('pattern')}" if prov.get("pattern") else "",
        f"grounds {prov.get('grounds_ddl')}" if prov.get("grounds_ddl") else "") if p)
    scope = ". ".join(p for p in (getattr(t, "scope_note", "") or "", scope) if p)
    return ". ".join(p for p in (pref, alts, definition,
                                 f"Involves: {vocab}" if vocab else "",
                                 grounded, scope) if p).strip()[:1200]


def build_term_registry(*, url: str = DEFAULT_QDRANT_URL,
                        collection: str = DEFAULT_REGISTRY,
                        include_domains: bool = True, recreate: bool = True) -> dict:
    """Materialize the topic registry from the LIVE catalog: every term is a topic anchor
    (explicit ontology grounding per topic — ruling e), nested under its SKOS domain via
    ``ancestor_codes`` so the hierarchical gate treats domain↔member-term as one lineage
    while sibling terms COMPETE (that competition is the definitional-rigor signal).

    The anchor text is the term's SKOS surface (prefLabel · slots-as-altLabels ·
    verbalization-as-definition · a provenance scopeNote). Thin surfaces are EXPECTED —
    the registry's own token counts are the rigor loop's first worklist (ruling f)."""
    from qdrant_client import models

    from aegir.ontology.colbert_encoder import get_encoder
    from aegir.ontology.domain_index import DEFAULT_OVERLAY, load_skos
    from aegir.ontology.schema import CATALOG_FILE, load_catalog

    anchors: list[dict] = []
    if include_domains:  # the rich aperture domains ride along as roll-up parents
        for c in load_skos(DEFAULT_OVERLAY, overlays=None).values():
            if not c.text():
                continue
            anchors.append({"text": c.text(), "payload": {
                "iri": c.iri, "code": c.code, "pref_label": c.pref_label,
                "alt_label": c.alt_label, "broader": c.broader,
                "ancestor_codes": c.ancestor_codes(), "kind": "domain"}})
    n_domains = len(anchors)
    glosses = _curie_glosses()
    for t in load_catalog(CATALOG_FILE).templates:
        prov = t.provenance or {}
        dom = prov.get("domain") if isinstance(prov.get("domain"), dict) else {}
        dom_code = str((dom or {}).get("code") or "")
        ancestors = ([".".join(dom_code.split(".")[:i]) for i in range(1, dom_code.count(".") + 2)]
                     if dom_code else [])
        pref = t.template_id.replace("_", " ").removeprefix("filler ")
        alts = ", ".join([s.replace("_", " ") for s in (t.slot_types or {})]
                         + list(getattr(t, "alt_labels", None) or []))
        text = term_anchor_text(t, glosses)
        if not text:
            continue
        anchors.append({"text": text, "payload": {
            "iri": f"{_VOCAB_NS}{t.template_id}", "code": t.template_id,
            "pref_label": pref, "alt_label": alts, "broader": dom_code,
            "ancestor_codes": ancestors, "kind": "term"}})

    enc = get_encoder()
    cl = _client(url)
    if recreate and cl.collection_exists(collection):
        cl.delete_collection(collection)
    if not cl.collection_exists(collection):
        cl.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=enc.dim, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM)))
    vecs = enc.encode([a["text"] for a in anchors])
    points = [models.PointStruct(id=i, vector=v.tolist(), payload=a["payload"])
              for i, (a, v) in enumerate(zip(anchors, vecs))]
    for start in range(0, len(points), 64):
        cl.upsert(collection_name=collection, points=points[start:start + 64])

    # the registry's own granularity report + the thin-anchor WORKLIST (rigor loop food)
    state = collection_state(url=url, collection=collection)
    terms = [(a, v.shape[0]) for a, v in zip(anchors, vecs) if a["payload"]["kind"] == "term"]
    terms.sort(key=lambda av: av[1])
    tok = sorted(nt for _, nt in terms) or [0]
    worklist = [{"term": a["payload"]["code"], "n_tokens": int(nt),
                 "domain": a["payload"]["broader"] or None}
                for a, nt in terms if nt < 48]
    out = {"collection": collection, "sha": state["sha"], "n_topics": state["n_topics"],
           "n_domains": n_domains, "n_terms": len(terms),
           "term_tokens_p10": tok[len(tok) // 10], "term_tokens_p50": tok[len(tok) // 2],
           "term_tokens_p90": tok[(len(tok) * 9) // 10],
           "anchor_tokens_p50": state["anchor_tokens_p50"],
           "n_thin_terms": len(worklist)}
    wdir = REPO / "build" / "topic_registry"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "worklist.json").write_text(json.dumps(
        {"collection": collection, "sha": state["sha"], "floor_tokens": 48,
         "thin_terms": worklist}, indent=1))
    (wdir / "registry_report.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--docs", default=str(REPO / "build" / "domain_harvest" / "docs"))
    ap.add_argument("--collection", default="")
    ap.add_argument("--url", default=DEFAULT_QDRANT_URL)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--limit", type=int, default=0, help="associate only the first N passages")
    ap.add_argument("--out", default="", help="override the state-keyed output dir")
    ap.add_argument("--whole-doc", action="store_true",
                    help="adjudicate whole passages (no item windows — increment-1 mode)")
    ap.add_argument("--build-registry", action="store_true",
                    help="(re)build the term-grounded topic registry, then exit")
    ap.add_argument("--window-ratio", type=float, default=2.0,
                    help="item window tokens = ratio × the anchors' median token count")
    ap.add_argument("--window-tokens", type=int, default=0,
                    help="PIN the window size (A/B runs vs an evolved registry)")
    ap.add_argument("--stride", type=float, default=0.5, help="stride as a fraction of the window")
    a = ap.parse_args(argv)
    if a.build_registry:
        print(json.dumps(build_term_registry(
            url=a.url, collection=a.collection or DEFAULT_REGISTRY), indent=1))
        return 0
    collection = a.collection
    if not collection:
        try:  # the strategy declares the aiming collection — honor it when present
            from aegir.strategy.manifest import lens_binding
            collection = lens_binding().get("aiming_collection") or DEFAULT_APERTURE
        except Exception:  # noqa: BLE001
            collection = DEFAULT_APERTURE
    report = associate_store(a.docs, url=a.url, collection=collection, tau=a.tau,
                             limit=a.limit, out_dir=a.out or None,
                             unit="doc" if a.whole_doc else "item",
                             ratio=a.window_ratio, stride_frac=a.stride,
                             window_tokens=a.window_tokens)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
