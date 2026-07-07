"""Author SKOS retrieval surfaces for thin topic anchors — the (f)-loop's AUTHORED half (4b).

The inverted topic layer's worklist (``build/topic_registry/worklist.json``) names the
terms whose DECLARED surface cannot discriminate (mostly intermediates: a bare genus
axiom + one short verbal — many share IDENTICAL axiom bodies, so the authored surface
is the only separator available). This script runs the aegir closed loop over them:

  agent PROPOSES {alt_labels, scope_note, definition} per term (engine gRPC, guided
  JSON) → the ANNOTATION MEMBRANE disposes with a REASON (M1-M6, incl. the
  self-retrieval margin gate against the live registry) → rejects re-prompt WITH the
  reason → accepted surfaces land on the CatalogTemplate (alt_labels / scope_note /
  verbal_templates) + a provenance ``skos_authoring`` record → save_catalog.

Annotations are NEVER ontological claims (M3 enforces it) — the axiom stays untouched,
so HermiT has no stake; the membranes are the whole gate. After a run: rebuild the
registry + re-associate with the window PINNED to the baseline plan to measure the
delta (see the 4b scoping note).

  uv run --no-sync python scripts/author_skos_surfaces.py --limit 10 --rounds 2   # smoke
  uv run --no-sync python scripts/author_skos_surfaces.py                          # full worklist
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology.annotation_membrane import Proposal, validate_annotation  # noqa: E402
from aegir.ontology.schema import CATALOG_FILE, load_catalog, save_catalog  # noqa: E402
from aegir.ontology import topic_layer as TL  # noqa: E402

_SYS = """You author SKOS retrieval annotations for ontology terms. For each term you get its
name, its OWL axiom (Manchester), its current one-line verbalization, the vocabulary its axiom
references, and the NEAREST SIBLING surfaces it must be distinguished from.

Return, per term: alt_labels (2-6 alternative surface forms, 1-5 words each — how practitioners
would actually name this in documents), scope_note (1-2 sentences: WHEN this term applies and
when it does NOT — contrast with the siblings), definition (2-3 sentences of practitioner prose
elaborating the verbalization with concrete, domain-real detail).

HARD RULES (a deterministic membrane rejects violations, with reasons you must fix):
- NO ontological claims: no 'is a kind/type/subclass of', no axiom syntax, no ontology IRIs or
  curies beyond those the axiom already references. Annotations describe usage, never assert
  taxonomy.
- Stay grounded in the term's own vocabulary and domain; no proprietary terminology codes.
- Differentiate: your text must make THIS term retrievable against its siblings."""

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "alt_labels": {"type": "array", "items": {"type": "string"},
                                   "minItems": 2, "maxItems": 6},
                    "scope_note": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["term", "alt_labels", "scope_note", "definition"],
            },
        }
    },
    "required": ["proposals"],
}


def _allowed_vocab(t, glosses) -> set:
    """The grounding set for M4: the term's own name + axiom vocabulary + glosses."""
    import re
    words = set()
    for chunk in (t.template_id, TL._axiom_vocabulary(t.manchester_template or "", glosses),
                  t.verbal_template or "", " ".join(t.slot_types or {})):
        words |= set(re.findall(r"[a-z]{4,}", chunk.lower().replace("_", " ")))
    # generic connective/practitioner words the membrane shouldn't punish
    words |= {"process", "record", "activity", "role", "entity", "involves", "applies",
              "term", "used", "describes", "captures", "tracking", "management", "when",
              "within", "documents", "instances", "specific", "particular", "general",
              "context", "business", "information", "system", "data", "which", "where",
              "typically", "often", "such", "these", "this", "that", "with", "from"}
    return words


def _sibling_surfaces(t, glosses, k: int = 3) -> "list[str]":
    """The k nearest sibling anchor surfaces (by MaxSim of the term's CURRENT surface)
    — what the authored text must be distinguished FROM."""
    from aegir.ontology.domain_index import _client
    from aegir.ontology.colbert_encoder import get_encoder
    enc = get_encoder()
    cl = _client()
    v = enc.encode_single(TL.term_anchor_text(t, glosses))
    res = cl.query_points(collection_name=TL.DEFAULT_REGISTRY, query=v.tolist(),
                          limit=k + 1, with_payload=True).points
    sibs = []
    for p in res:
        pl = p.payload or {}
        if pl.get("code") != t.template_id:
            sibs.append(f"{pl.get('code')}: {pl.get('pref_label')}")
    return sibs[:k]


def _self_retrieval(t, proposal: Proposal, glosses) -> "tuple[str, float]":
    """MaxSim the COMPOSED surface (proposal applied) against the live registry —
    (rank1_code, rel_margin_h vs nearest non-ancestor). The M6-iii signal."""
    import copy
    from aegir.ontology.domain_index import _client
    from aegir.ontology.colbert_encoder import get_encoder
    t2 = copy.deepcopy(t)
    t2.alt_labels = list(proposal.alt_labels)
    t2.scope_note = proposal.scope_note
    t2.verbal_templates = list(t2.verbal_templates or []) + [proposal.definition]
    text = TL.term_anchor_text(t2, glosses)
    enc = get_encoder()
    cl = _client()
    v = enc.encode_single(text)
    res = cl.query_points(collection_name=TL.DEFAULT_REGISTRY, query=v.tolist(),
                          limit=6, with_payload=True).points
    hits = [{"score": float(p.score), **(p.payload or {})} for p in res]
    # STRICT rank-1: the composed surface must retrieve its own (stale) anchor first —
    # drifting past a sibling means the authored text is off-term.
    if not hits or hits[0].get("code") != t.template_id:
        return ((hits[0].get("code", "?") if hits else "?"), 0.0)
    comp = next((h for h in hits[1:] if not TL._same_lineage(hits[0], h)), None)
    s_self = float(hits[0]["score"])
    sc = float(comp["score"]) if comp else 0.0
    return (t.template_id, (s_self - sc) / s_self if s_self else 0.0)


def _n_tokens(t, proposal: Proposal, glosses) -> int:
    import copy
    from aegir.ontology.colbert_encoder import get_encoder
    t2 = copy.deepcopy(t)
    t2.alt_labels = list(proposal.alt_labels)
    t2.scope_note = proposal.scope_note
    t2.verbal_templates = list(t2.verbal_templates or []) + [proposal.definition]
    tok = get_encoder()._tokenizer
    return len(tok(TL.term_anchor_text(t2, glosses), add_special_tokens=False,
                   truncation=False, verbose=False)["input_ids"])


def _propose(batch, cat_by_id, glosses, feedback) -> "list[Proposal]":
    blocks = []
    for tid in batch:
        t = cat_by_id[tid]
        sibs = _sibling_surfaces(t, glosses)
        b = (f"TERM: {tid}\nAXIOM: {t.manchester_template}\n"
             f"CURRENT VERBAL: {t.verbal_template}\n"
             f"AXIOM VOCABULARY: {TL._axiom_vocabulary(t.manchester_template or '', glosses)}\n"
             f"DISTINGUISH FROM: {'; '.join(sibs) or '—'}")
        if tid in feedback:
            b += f"\nPRIOR ATTEMPT REJECTED — {feedback[tid]}  → fix and re-emit."
        blocks.append(b)
    out = complete_detailed(
        "Author the SKOS surfaces for these terms:\n\n" + "\n\n".join(blocks),
        capability="instruct", system_prompt=_SYS, json_schema=json.dumps(PROPOSAL_SCHEMA),
        max_tokens=4096)
    try:
        data = json.loads(out["text"])
    except Exception:  # noqa: BLE001
        return []
    return [Proposal(term=p.get("term", ""), alt_labels=p.get("alt_labels") or [],
                     scope_note=p.get("scope_note", ""), definition=p.get("definition", ""))
            for p in data.get("proposals", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="author only the first N worklist terms")
    ap.add_argument("--margin-floor", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true", help="don't save the catalog")
    a = ap.parse_args()

    wl = json.loads((REPO / "build" / "topic_registry" / "worklist.json").read_text())
    terms = [w["term"] for w in wl["thin_terms"]]
    if a.limit:
        terms = terms[: a.limit]
    cat = load_catalog(CATALOG_FILE)
    by_id = {t.template_id: t for t in cat.templates}
    glosses = TL._curie_glosses()
    state = TL.collection_state(collection=TL.DEFAULT_REGISTRY)
    print(f"authoring {len(terms)} thin terms vs registry "
          f"{TL.DEFAULT_REGISTRY}@{state['sha']} (margin floor {a.margin_floor})", flush=True)

    accepted: dict[str, Proposal] = {}
    feedback: dict[str, str] = {}
    reject_counts: dict[str, int] = {}
    for rnd in range(a.rounds):
        pending = [t for t in terms if t not in accepted]
        if not pending:
            break
        print(f"— round {rnd + 1}: {len(pending)} pending", flush=True)
        for i in range(0, len(pending), a.batch):
            batch = pending[i:i + a.batch]
            for prop in _propose(batch, by_id, glosses, feedback):
                t = by_id.get(prop.term)
                if t is None or prop.term in accepted:
                    continue
                ok, reason = validate_annotation(
                    t, prop, allowed_vocab=_allowed_vocab(t, glosses),
                    n_tokens=_n_tokens(t, prop, glosses),
                    self_retrieval=_self_retrieval(t, prop, glosses),
                    margin_floor=a.margin_floor)
                if ok:
                    accepted[prop.term] = prop
                    feedback.pop(prop.term, None)
                else:
                    feedback[prop.term] = reason
                    reject_counts[reason.split(":")[0]] = reject_counts.get(reason.split(":")[0], 0) + 1
            print(f"   {min(i + a.batch, len(pending))}/{len(pending)} · "
                  f"accepted {len(accepted)}", flush=True)

    print(f"\naccepted {len(accepted)}/{len(terms)} · membrane rejections by gate: {reject_counts}")
    if accepted and not a.dry_run:
        from datetime import date
        for tid, prop in accepted.items():
            t = by_id[tid]
            t.alt_labels = list(prop.alt_labels)
            t.scope_note = prop.scope_note
            t.verbal_templates = list(t.verbal_templates or []) + [prop.definition]
            t.provenance = dict(t.provenance or {})
            t.provenance["skos_authoring"] = {
                "date": str(date.today()), "model": "engine/instruct",
                "membranes": "annotation_membrane M1-M6", "registry": state["sha"]}
        save_catalog(cat, CATALOG_FILE)
        print(f"saved {len(accepted)} authored surfaces → {CATALOG_FILE.name}")
    remaining = [t for t in terms if t not in accepted]
    if remaining:
        print(f"unresolved ({len(remaining)}): {', '.join(remaining[:10])}"
              + (" …" if len(remaining) > 10 else ""))
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
