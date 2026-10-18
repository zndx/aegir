# Copyright 2026 zndx
# SPDX-License-Identifier: Apache-2.0
"""Window a streamed AgentRTC transcript and keep the genus spans.

The caller sends the text. This module cuts it, scores it against the
aperture, and returns character offsets. The segment text is not copied
into the result.
"""

from __future__ import annotations

TAU = 0.10


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
