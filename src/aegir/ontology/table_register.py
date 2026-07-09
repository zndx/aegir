"""table_register — the deterministic 'does this table read as a human document or a raw
schema?' metric (RH 2026-07-09).

Motivating measurement (`docs/scratch/2026-07-09/…_finepdfs_table_geometry.md`): across 40.8k
real FinePDFs tables vs our generated corpus, three tells separate synthetic tables from real
ones by 50–150×:

  1. LEADING KEY   — real tables lead with a natural/business key; ours lead with a surrogate
                     `id` (1.6% real vs 86% ours). Naming+position, NOT integer-monotonicity
                     (our ids are coded strings) — the test is the column *name*.
  2. HEADER REGISTER — real headers are prose with units ("Clone ID", "Chemotaxis IC50 (nM)");
                     ours are snake_case machine identifiers (0.6% machine real vs 88% ours).
  3. VALUE SHAPE   — real cell values are irregular (sentinels `ND`/`<0.1`, ranges, free text);
                     ours follow one rigid per-column format (mask-entropy 0.52 real vs 0.10).

This scores the three on parsed markdown tables (the surface a reader / the byte model sees).
`presentation.py` is the fix for tells 1–2 (a prose-render transform; schema identity is left
alone — presentation ≠ identifier). Tell 3 needs a materialization-level value lever and is
scored here as the standing gap. A DIAGNOSTIC by default (like table_fidelity), not folded into
the hard R composite until the strategy elects to gate on it.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# measured reference (essence_probe.py over 40,803 real FinePDFs tables) — the aiming point
REFERENCE = {
    "lead_id_rate": 0.016,
    "machine_header_rate": 0.006,
    "human_header_rate": 0.448,
    "unit_header_rate": 0.130,
    "value_mask_entropy": 0.516,
    "freetext_col_rate": 0.539,
}
# our pre-transform baseline (same probe, our corpus) — the starting point the transform moves
BASELINE_OURS = {
    "lead_id_rate": 0.860,
    "machine_header_rate": 0.876,
    "human_header_rate": 0.0,
    "unit_header_rate": 0.0,
    "value_mask_entropy": 0.096,
    "freetext_col_rate": 0.111,
}

_MACHINE = re.compile(r"^[a-z][a-z0-9_]*$")
_HAS_SPACE = re.compile(r"\s")
_HAS_UNIT = re.compile(r"[()%/µ]|\b(mg|ml|nm|nM|µM|kg|cm|mm|pH|IC50|USD|EUR|mol|°)\b")
_ID_NAME = re.compile(r"^(id|.*_id|no\.?|#|index|sr\.?\s*no\.?|s\.?no\.?)$", re.I)


def value_mask(v: str) -> str:
    """Class-mask of a value: letter-run→X, digit-run→9, punctuation/space literal, runs collapsed.
    '2605.0'→'9.9', 'CG11'→'X9', 'Vh1_DP-3'→'X9_X-9'. Rigid columns share one mask (low entropy)."""
    v = v.strip()
    out, prev = [], None
    for ch in v:
        c = "9" if ch.isdigit() else "X" if ch.isalpha() else ch
        if c != prev or c not in ("9", "X"):
            out.append(c)
        prev = c
    return "".join(out)


def col_mask_entropy(vals: "list[str]") -> "tuple[float | None, float | None]":
    """(normalized mask entropy, dominant-mask share) over a column's non-empty values."""
    vals = [v for v in vals if v and v.strip()]
    if len(vals) < 2:
        return None, None
    masks = Counter(value_mask(v) for v in vals)
    tot = sum(masks.values())
    ent = -sum((c / tot) * math.log2(c / tot) for c in masks.values())
    norm = ent / math.log2(len(masks)) if len(masks) > 1 else 0.0
    return norm, masks.most_common(1)[0][1] / tot


def leading_is_id(header0: str) -> bool:
    """The tell RH named: the first rendered column is a surrogate-key NAME (id / *_id / no / #)."""
    return bool(_ID_NAME.match((header0 or "").strip()))


def header_register(h: str) -> str:
    """'machine' (snake_case identifier), 'unit' (carries a unit/paren), or 'human' (has a space)."""
    hs = (h or "").strip()
    if _HAS_UNIT.search(hs):
        return "unit"
    if _MACHINE.match(hs):
        return "machine"
    if _HAS_SPACE.search(hs):
        return "human"
    return "other"


def register_metrics(tables: "list[dict]") -> dict:
    """Raw tell rates over parsed markdown tables ([{headers, cells}]). Cells = list of data rows."""
    n = len(tables)
    lead_id = 0
    reg = Counter()
    n_headers = 0
    ents, regular, freetext, n_cols = [], 0, 0, 0
    for t in tables:
        headers = t.get("headers") or []
        cells = t.get("cells") or []
        if not headers:
            continue
        if leading_is_id(headers[0]):
            lead_id += 1
        for h in headers:
            reg[header_register(h)] += 1
            n_headers += 1
        for c in range(len(headers)):
            col = [r[c] for r in cells if c < len(r)]
            norm, dom = col_mask_entropy(col)
            if norm is None:
                continue
            n_cols += 1
            ents.append(norm)
            if dom is not None and dom >= 0.8:
                regular += 1
            if norm >= 0.6:
                freetext += 1
    return {
        "n_tables": n,
        "lead_id_rate": round(lead_id / n, 3) if n else 0.0,
        "machine_header_rate": round(reg["machine"] / n_headers, 3) if n_headers else 0.0,
        "human_header_rate": round(reg["human"] / n_headers, 3) if n_headers else 0.0,
        "unit_header_rate": round(reg["unit"] / n_headers, 3) if n_headers else 0.0,
        "value_mask_entropy": round(sum(ents) / len(ents), 3) if ents else 0.0,
        "freetext_col_rate": round(freetext / n_cols, 3) if n_cols else 0.0,
        "format_regular_rate": round(regular / n_cols, 3) if n_cols else 0.0,
    }


def score_register(tables: "list[dict]") -> "tuple[float | None, dict]":
    """R_register ∈ [0,1] + the raw sub-metrics. None when a chapter has no tables.

    Composite (weighted arithmetic — forgiving, so the un-fixed value-register leg doesn't zero
    a chapter the transform already improved on headers/lead-key):
      • present_lead   = 1 − lead_id_rate            (want ~0 lead-with-id)         weight 0.40
      • present_header = 1 − machine_header_rate     (want prose/unit headers)      weight 0.40
      • value_register = 1 − |entropy − REF|/REF     (closeness to real irregularity) weight 0.20
    The first two are what presentation.py controls; the third is the standing materialization gap.
    """
    if not tables:
        return None, {"n_tables": 0}
    m = register_metrics(tables)
    ref_ent = REFERENCE["value_mask_entropy"]
    present_lead = 1.0 - m["lead_id_rate"]
    present_header = 1.0 - m["machine_header_rate"]
    value_register = max(0.0, 1.0 - abs(m["value_mask_entropy"] - ref_ent) / ref_ent)
    r = 0.40 * present_lead + 0.40 * present_header + 0.20 * value_register
    m["R_register"] = round(r, 3)
    m["legs"] = {"present_lead": round(present_lead, 3),
                 "present_header": round(present_header, 3),
                 "value_register": round(value_register, 3)}
    return round(r, 3), m
