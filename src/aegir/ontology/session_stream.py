# Copyright 2026 zndx
# SPDX-License-Identifier: Apache-2.0
"""Window a streamed AgentRTC transcript and keep the genus spans.

The caller sends the text. This module cuts it, scores it against the
aperture, and returns character offsets. The segment text is not copied
into the result.
"""

from __future__ import annotations

TAU = 0.10
WINDOW_SIZE = 512
WINDOW_STRIDE = 256


class AdmitError(RuntimeError):
    """Aperture scoring failed. The message must not contain the transcript."""


def _margin(hits: list[dict]) -> tuple[dict, float]:
    if not hits:
        return {}, 0.0
    scores = [float(hit.get("score") or 0.0) for hit in hits[:5]]
    top = scores[0]
    rel = ((top - scores[1]) / top) if len(scores) > 1 and top else 1.0
    return hits[0], rel


def qualify(
    text: str,
    *,
    windows,
    query,
    tau: float = TAU,
    exclude_codes: set[str] | None = None,
) -> list[dict]:
    """Return kept windows as offsets into ``text``.

    ``windows(text)`` yields ``(start, end)`` character pairs.
    ``query(segment)`` yields MaxSim hits with ``score``, ``code``,
    ``iri``, and ``pref_label``. Hits whose code is in ``exclude_codes``
    are the scheme roots and do not count as a genus.
    """

    roots = set(exclude_codes or ())
    candidates = []
    for start, end in windows(text):
        if end <= start:
            continue
        hits = list(query(text[start:end]) or [])
        genus = [hit for hit in hits if str(hit.get("code") or "") not in roots]
        top, margin = _margin(genus)
        if not top or margin < tau:
            continue
        candidates.append(
            {
                "start": int(start),
                "end": int(end),
                "genus_iri": str(top.get("iri") or ""),
                "genus_code": str(top.get("code") or ""),
                "genus_label": str(top.get("pref_label") or ""),
                "rel_margin_h": round(float(margin), 4),
                "tau": float(tau),
            }
        )
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: -item["rel_margin_h"]):
        if any(not (candidate["end"] <= prior["start"] or candidate["start"] >= prior["end"]) for prior in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: item["start"])
    return kept


def stable_indexes(
    n_tokens: int,
    n_windows: int,
    *,
    flush: bool,
    size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> set[int]:
    """Windows whose text and overlapping rivals can no longer change.

    Window ``k`` starts at token ``k * stride``. A later append can still
    lengthen the partial tail and can still outrank a neighbor that overlaps
    that tail. A window is ready once the next overlapping start is a full
    window, which is ``size + stride`` tokens after its own start. ``flush``
    is the caller's last frame: the tail is then final.
    """

    if flush:
        return set(range(n_windows))
    ready: set[int] = set()
    for k in range(n_windows):
        if n_tokens >= k * stride + size + stride:
            ready.add(k)
    return ready


def select_admissions(
    text: str,
    spans: list[tuple[int, int]],
    *,
    n_tokens: int,
    flush: bool,
    query,
    exclude_codes: set[str] | None = None,
    tau: float = TAU,
    size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> list[dict]:
    """``qualify`` the spans, then drop any window that a later append could still move."""

    kept = qualify(text, windows=lambda _body: spans, query=query, tau=tau, exclude_codes=exclude_codes)
    ready = stable_indexes(n_tokens, len(spans), flush=flush, size=size, stride=stride)
    by_span = {spans[k]: k for k in ready}
    return [row for row in kept if (row["start"], row["end"]) in by_span]


def spans_for_session(
    text: str,
    *,
    size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[tuple[int, int]], int]:
    """Every 512/256 window of ``text``. The harvest cap of 16 does not apply."""

    try:
        from aegir.ontology.admission_items import token_windows
        from aegir.ontology.colbert_encoder import get_encoder

        tk = get_encoder()._tokenizer(
            text, truncation=False, return_offsets_mapping=True, add_special_tokens=False)
        n_tokens = len([pair for pair in tk["offset_mapping"] if pair[1] > pair[0]])
        cap = max(1, n_tokens)
        return token_windows(text, size=size, stride=stride, max_windows=cap), n_tokens
    except Exception as exc:
        raise AdmitError("aperture query failed") from exc


def fill_aperture_cache(segments: list[str], cache: dict[str, list]) -> None:
    """Encode each new window once and store its MaxSim hits under the segment text."""

    missing: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment in cache or segment in seen:
            continue
        seen.add(segment)
        missing.append(segment)
    if not missing:
        return
    try:
        from aegir.ontology import domain_index as DI
        from aegir.ontology.colbert_encoder import get_encoder

        vecs = get_encoder().encode(missing)
        client = DI._client(DI.DEFAULT_QDRANT_URL)
        for segment, vec in zip(missing, vecs):
            points = client.query_points(
                collection_name=DI.DEFAULT_APERTURE,
                query=vec.tolist(),
                limit=10,
                with_payload=True,
            ).points
            cache[segment] = [{"score": float(point.score), **(point.payload or {})} for point in points]
    except AdmitError:
        raise
    except Exception as exc:
        raise AdmitError("aperture query failed") from exc


def committed_spans(texts: list[str]) -> list[tuple[int, int]]:
    """One span per finished frame. The frame text does not grow after it is sent."""

    spans: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        start = cursor
        cursor += len(text or "")
        if text:
            spans.append((start, cursor))
    return spans


def admit_committed(text: str, spans: list[tuple[int, int]], *, cache: dict[str, list]) -> list[dict]:
    """Score each finished frame now.

    A 512-token window stays open until 768 tokens have arrived, which is
    too late for a live call. A frame is a whole utterance, so its genus
    can travel back before the next turn.
    """

    from aegir.ontology.domain_index import armed_admission_filter

    segments = [text[start:end] for start, end in spans if end > start]
    fill_aperture_cache(segments, cache)

    def query(segment: str) -> list:
        return cache.get(segment, [])

    exclude = set(armed_admission_filter().get("effective_exclude") or ())
    return qualify(text, windows=lambda _body: spans, query=query, exclude_codes=exclude, tau=TAU)


def default_admit(text: str, *, flush: bool, cache: dict[str, list]) -> list[dict]:
    """Window ``text`` on the aiming aperture and return the stable genus spans."""

    from aegir.ontology.domain_index import armed_admission_filter

    spans, n_tokens = spans_for_session(text)
    segments = [text[start:end] for start, end in spans if end > start]
    fill_aperture_cache(segments, cache)

    def query(segment: str) -> list:
        return cache.get(segment, [])

    exclude = set(armed_admission_filter().get("effective_exclude") or ())
    return select_admissions(
        text, spans, n_tokens=n_tokens, flush=flush, query=query, exclude_codes=exclude)


def concat(frames: list) -> tuple[str, int]:
    """Join ``SessionText`` frames in seq order. A repeated seq replaces the tail."""

    ordered = sorted(frames, key=lambda frame: int(getattr(frame, "seq", 0)))
    parts: list[str] = []
    seen: set[int] = set()
    last_seq = 0
    for frame in ordered:
        seq = int(getattr(frame, "seq", 0))
        if seq in seen:
            parts = parts[: seq - 1]
            seen = {item for item in seen if item < seq}
        while len(parts) < seq - 1:
            parts.append("")
        if len(parts) == seq - 1:
            parts.append(str(getattr(frame, "text", "") or ""))
        else:
            parts[seq - 1] = str(getattr(frame, "text", "") or "")
        seen.add(seq)
        last_seq = seq
    return "".join(parts), last_seq
