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
    ``(id, code, iri, label, vector_sha)`` rows + a 16-hex digest over them.

    This closes the lens-snapshot gap (which hashed ``(id, label, vector_sha)`` only):
    the ontology NOTATION rides along, so an association record names its topic in both
    the ontology's terms and the vector state that adjudicated it."""
    cl = _client(url)
    rows: list[dict] = []
    offset = None
    while True:
        points, offset = cl.scroll(collection_name=collection, limit=512,
                                   with_payload=True, with_vectors=True, offset=offset)
        for p in points:
            pl = p.payload or {}
            vec = p.vector
            import numpy as np
            v_sha = hashlib.sha256(np.asarray(vec, dtype=np.float32).tobytes()).hexdigest()[:16]
            rows.append({"id": str(p.id), "code": pl.get("code", ""), "iri": pl.get("iri", ""),
                         "label": pl.get("pref_label") or pl.get("label") or "", "vector_sha": v_sha})
        if offset is None:
            break
    rows.sort(key=lambda r: r["id"])
    sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    return {"collection": collection, "url": url, "n_topics": len(rows), "sha": sha, "rows": rows}


@dataclass
class Association:
    """One passage ↔ one topic, with the full adjudication context — assigned or not.

    An UNASSIGNED record still names the top candidate and both margins: the ambiguous
    mass must be inspectable, because it is what the definitional-rigor loop consumes."""
    passage_hash: str
    assigned: bool
    topic_code: str
    topic_iri: str
    topic_label: str
    score: float
    margin: float
    rel_margin: float
    runner_up_code: str
    runner_up_score: float
    collection: str
    collection_sha: str
    tau: float
    n_chars: int = 0
    source: str = ""                     # provenance hint (e.g. the harvest doc path)
    extra: dict = field(default_factory=dict)


def associate(text: str, *, url: str = DEFAULT_QDRANT_URL,
              collection: str = DEFAULT_APERTURE, collection_sha: str = "",
              tau: float = DEFAULT_TAU, top_k: int = 5, source: str = "") -> Association:
    """Associate one passage with exactly one topic — or refuse (rel_margin < tau).

    MaxSim adjudicates (qdrant-native late interaction); the margin gate makes the
    assignment UNAMBIGUOUS-or-nothing, per ruling (c)."""
    hits = classify(text, url=url, collection=collection, top_k=top_k)
    if not hits:
        return Association(passage_hash=passage_hash(text), assigned=False, topic_code="",
                           topic_iri="", topic_label="", score=0.0, margin=0.0, rel_margin=0.0,
                           runner_up_code="", runner_up_score=0.0, collection=collection,
                           collection_sha=collection_sha, tau=tau, n_chars=len(text), source=source)
    top = hits[0]
    s0 = float(top["score"])
    s1 = float(hits[1]["score"]) if len(hits) > 1 else 0.0
    margin = s0 - s1
    rel_margin = (margin / s0) if s0 else 0.0
    return Association(
        passage_hash=passage_hash(text), assigned=bool(rel_margin >= tau),
        topic_code=top.get("code", ""), topic_iri=top.get("iri", ""),
        topic_label=top.get("pref_label", ""), score=round(s0, 4),
        margin=round(margin, 4), rel_margin=round(rel_margin, 4),
        runner_up_code=(hits[1].get("code", "") if len(hits) > 1 else ""),
        runner_up_score=round(s1, 4), collection=collection,
        collection_sha=collection_sha, tau=tau, n_chars=len(text), source=source)


def alignment_report(records: "list[Association] | list[dict]") -> dict:
    """The topic-quality gate's numbers: unambiguous-alignment rate + margin distribution
    + per-topic mass. Low alignment / thin margins indict the LEXICON (ruling f), so the
    per-topic and ambiguous views are what the rigor loop reads."""
    rows = [asdict(r) if isinstance(r, Association) else r for r in records]
    n = len(rows)
    if not n:
        return {"n": 0}
    assigned = [r for r in rows if r["assigned"]]
    margins = sorted(r["rel_margin"] for r in rows)
    pct = lambda q: margins[min(n - 1, int(q * n))]  # noqa: E731
    by_topic: dict[str, int] = {}
    for r in assigned:
        key = f"{r['topic_code']} {r['topic_label']}".strip()
        by_topic[key] = by_topic.get(key, 0) + 1
    ambiguous = sorted((r for r in rows if not r["assigned"]),
                       key=lambda r: -r["rel_margin"])[:10]
    return {
        "n": n, "n_assigned": len(assigned),
        "alignment_rate": round(len(assigned) / n, 4),
        "rel_margin_p10": round(pct(0.10), 4), "rel_margin_p50": round(pct(0.50), 4),
        "rel_margin_p90": round(pct(0.90), 4),
        "topics_hit": len(by_topic),
        "by_topic": dict(sorted(by_topic.items(), key=lambda kv: -kv[1])),
        "near_misses": [{"passage": r["passage_hash"][:16], "top": r["topic_code"],
                         "runner_up": r["runner_up_code"], "rel_margin": r["rel_margin"]}
                        for r in ambiguous],
    }


def associate_store(docs_dir: "str | Path", *, url: str = DEFAULT_QDRANT_URL,
                    collection: str = DEFAULT_APERTURE, tau: float = DEFAULT_TAU,
                    limit: int = 0, out_dir: "str | Path | None" = None) -> dict:
    """Associate every passage in a content-addressed store (``<hash>.txt`` files, e.g.
    the harvest's ``build/domain_harvest/docs``) and persist the lineage:
    ``build/topic_associations/<collection>-<collection_sha>/associations.jsonl`` +
    ``report.json``. The out-dir is keyed by the collection STATE, so re-running against
    an evolved registry lands beside — never over — the old adjudications."""
    docs = sorted(Path(docs_dir).glob("*.txt"))
    if limit:
        docs = docs[:limit]
    state = collection_state(url=url, collection=collection)
    out = Path(out_dir) if out_dir else (REPO / "build" / "topic_associations"
                                         / f"{collection}-{state['sha']}")
    out.mkdir(parents=True, exist_ok=True)
    records: list[Association] = []
    with (out / "associations.jsonl").open("w") as fh:
        for p in docs:
            text = p.read_text(errors="ignore")
            rec = associate(text, url=url, collection=collection,
                            collection_sha=state["sha"], tau=tau, source=p.name)
            records.append(rec)
            fh.write(json.dumps(asdict(rec)) + "\n")
    report = {"collection": collection, "collection_sha": state["sha"],
              "n_topics": state["n_topics"], "tau": tau, **alignment_report(records)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "collection_state.json").write_text(json.dumps(state, indent=1))
    return report


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--docs", default=str(REPO / "build" / "domain_harvest" / "docs"))
    ap.add_argument("--collection", default="")
    ap.add_argument("--url", default=DEFAULT_QDRANT_URL)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--limit", type=int, default=0, help="associate only the first N passages")
    ap.add_argument("--out", default="", help="override the state-keyed output dir")
    a = ap.parse_args(argv)
    collection = a.collection
    if not collection:
        try:  # the strategy declares the aiming collection — honor it when present
            from aegir.strategy.manifest import lens_binding
            collection = lens_binding().get("aiming_collection") or DEFAULT_APERTURE
        except Exception:  # noqa: BLE001
            collection = DEFAULT_APERTURE
    report = associate_store(a.docs, url=a.url, collection=collection, tau=a.tau,
                             limit=a.limit, out_dir=a.out or None)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
