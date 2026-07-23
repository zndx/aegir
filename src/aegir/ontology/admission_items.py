"""admission_items — sliding-window ITEM admission (the fedwiki grain), RESOLVED.

RH semantics (2026-07-23):

- Windows are ≤512 encoder tokens (the ColBERT limit IS the item grain), sliding at a
  configurable stride, spans cut by the encoder's own tokenizer.
- Every window sees the WHOLE aperture in one MaxSim query — that global visibility is
  the basis of contrastive selection. The GENUS view (scheme roots dropped, the armed
  filter's set) computes the admission margin; the UNFILTERED view feeds the root
  shadow-net.
- **Admitted items are NON-OVERLAPPING**: greedy suppression by genus margin — among
  potentially-overlapping admitted candidates, the final item is the highest-genus-
  margin match to a single aperture topic; overlapped rivals yield. Several high-margin
  topic matches per doc are expected; zero is normal.
- **Root preponderance** — a doc with ≥ ``REVIEW_MIN_STRONG`` STRONG ``skos:topConcept``
  matches (unfiltered top is a root AND its unfiltered rel_margin clears τ — i.e. the
  doc would have been root-ADMITTED under the pre-recomposition surface) while NO
  genus-level item admits — routes to the agent-mediated (ACP) review worklist: these
  docs are evidence of MISSING genus anchors, and the review accumulates ontology
  extension/refinement proposals (the escalation-organ pattern; never dropped).
  A weak nearest-neighbor drift toward a root is NOT a match and never reviews
  (measured: an unqualified trigger fire-hosed 332 out-of-domain docs in minutes).

Batch posture: all of a doc's windows encode in ONE forward pass (GPU-friendly; the
6-local-GPU sharding rides above this at the stream level).
"""
from __future__ import annotations

from pathlib import Path  # noqa: F401 — callers type against Path

REVIEW_MIN_STRONG = 1        # REGISTERED: ≥ this many STRONG root matches (τ-cleared,
                             # would-have-admitted) with zero genus items → ACP review


def token_windows(text: str, *, stride: int = 256, size: int = 512,
                  max_windows: int = 16) -> "list[tuple[int, int]]":
    """CHAR spans of ≤``size``-token windows at ``stride`` tokens, from the encoder's
    own tokenizer (one tokenization scheme, no drift)."""
    from aegir.ontology.colbert_encoder import get_encoder
    tk = get_encoder()._tokenizer(text, truncation=False, return_offsets_mapping=True,
                                  add_special_tokens=False)
    offs = [o for o in tk["offset_mapping"] if o[1] > o[0]]
    if not offs:
        return [(0, len(text))]
    out = []
    i = 0
    while i < len(offs) and len(out) < max_windows:
        j = min(i + size, len(offs))
        out.append((offs[i][0], offs[j - 1][1]))
        if j >= len(offs):
            break
        i += stride
    return out


def item_scan(text: str, *, url: "str | None" = None, collection: "str | None" = None,
              tau: float = 0.10, exclude_codes: "set | None" = None,
              stride: int = 256, max_windows: int = 16) -> dict:
    """Scan one doc into candidate windows and resolve the admitted item stream.

    Returns ``{candidates, items, n_admitted, review}``:
      candidates — every window scanned: span, unfiltered top, genus top + margin,
                   raw admission verdict, suppression state;
      items      — the RESOLVED non-overlapping stream in span order: admitted items
                   (topic + genus margin) with interstitial spans between them (each
                   carrying its best-overlapping candidate for reference);
      review     — the root-preponderance verdict (+ evidence) when no item admitted.
    """
    from aegir.ontology import domain_index as DI
    from aegir.ontology.colbert_encoder import get_encoder
    excl = set(exclude_codes or [])
    url = url or DI.DEFAULT_QDRANT_URL
    collection = collection or DI.DEFAULT_APERTURE
    enc = get_encoder()
    client = DI._client(url)

    spans = token_windows(text, stride=stride, max_windows=max_windows)
    vecs = enc.encode([text[c0:c1] for c0, c1 in spans])       # ONE forward pass

    def view(hits: "list[dict]") -> "tuple[dict, float]":
        scores = [h["score"] for h in hits[:5]]
        rel = ((scores[0] - scores[1]) / scores[0]) if len(scores) > 1 and scores[0] else 1.0
        return (hits[0] if hits else {}), rel

    candidates = []
    for ix, ((c0, c1), v) in enumerate(zip(spans, vecs)):
        res = client.query_points(collection_name=collection, query=v.tolist(),
                                  limit=10, with_payload=True).points
        hits = [{"score": float(p.score), **(p.payload or {})} for p in res]
        base_top, base_margin = view(hits)
        genus_top, genus_margin = view([h for h in hits
                                        if str(h.get("code", "")) not in excl])
        is_root_top = str(base_top.get("code", "")) in excl
        candidates.append({
            "ix": ix, "span": [c0, c1],
            "unfiltered_top": str(base_top.get("code", "")),
            "unfiltered_margin": round(base_margin, 4),
            "root_top": is_root_top,
            "root_strong": is_root_top and base_margin >= tau,
            "code": str(genus_top.get("code", "")),
            "label": str(genus_top.get("pref_label", "")),
            "genus_margin": round(genus_margin, 4),
            "admitted_raw": bool(genus_top) and genus_margin >= tau,
            "suppressed": False})

    # ── non-overlap resolution: greedy by genus margin (highest single-topic match
    #    among overlapping candidates wins; rivals yield) ──
    kept: "list[dict]" = []
    for c in sorted([c for c in candidates if c["admitted_raw"]],
                    key=lambda c: -c["genus_margin"]):
        c0, c1 = c["span"]
        if any(not (c1 <= k["span"][0] or c0 >= k["span"][1]) for k in kept):
            c["suppressed"] = True
            continue
        kept.append(c)
    kept.sort(key=lambda c: c["span"][0])

    # ── the resolved item stream: admitted items + interstitial spans ──
    items = []
    cursor = 0
    for k in kept:
        c0, c1 = k["span"]
        if c0 > cursor:
            gap = [cursor, c0]
            best = max((c for c in candidates
                        if c["span"][0] < gap[1] and c["span"][1] > gap[0]
                        and c is not k),
                       key=lambda c: c["genus_margin"], default=None)
            items.append({"span": gap, "admitted": False,
                          "code": (best or {}).get("code", ""),
                          "genus_margin": (best or {}).get("genus_margin", 0.0)})
        items.append({"span": [c0, c1], "admitted": True, "code": k["code"],
                      "label": k["label"], "genus_margin": k["genus_margin"]})
        cursor = c1
    if cursor < len(text):
        items.append({"span": [cursor, len(text)], "admitted": False, "code": "",
                      "genus_margin": 0.0})

    n_root_strong = sum(c["root_strong"] for c in candidates)
    review = None
    if not kept and n_root_strong >= REVIEW_MIN_STRONG:
        from collections import Counter
        review = {"reason": "STRONG topConcept match(es) without genus admission — the "
                            "old surface would have root-admitted this doc; ontology "
                            "extension/refinement candidate (ACP review)",
                  "n_windows": len(candidates), "n_root_strong": n_root_strong,
                  "n_root_top": sum(c["root_top"] for c in candidates),
                  "strong_share": round(n_root_strong / max(1, len(candidates)), 3),
                  "root_histogram": dict(Counter(c["unfiltered_top"]
                                                 for c in candidates
                                                 if c["root_strong"]).most_common()),
                  "best_root_margin": max((c["unfiltered_margin"] for c in candidates
                                           if c["root_strong"]), default=0.0),
                  "best_genus_margin": max((c["genus_margin"] for c in candidates),
                                           default=0.0)}
    return {"candidates": candidates, "items": items, "n_admitted": len(kept),
            "review": review}
