"""presentation — the render register: how a table LOOKS in a document, distinct from its
schema IDENTITY (RH 2026-07-09, "presentation ≠ identifier").

Real PDF tables are human renderings *over* a schema: they lead with a natural/business key,
their headers are prose carrying units ("Clone ID", "Chemotaxis IC50 (nM)"), and a surrogate
`id` is rarely surfaced. Our pipeline instead printed the raw schema into the chapter prose —
snake_case identifiers, surrogate `id` first — the two 50–150× tells `table_register.py`
measures.

This transform fixes those two tells for the PROSE rendering ONLY. It is pure/deterministic and
touches neither the DDL nor the verifiable JSON footer: the physical schema keeps `id` + snake_case
(correct, load-bearing for CTA/CPA; a real warehouse DDL *does* lead with `id SERIAL PRIMARY KEY`)
while the rendered markdown table reads as a document. It reorders columns and rewrites header
TEXT; it never edits a cell value, so table_fidelity (a verbatim cell-value check) is preserved and
RI is untouched.

The third tell — rigid per-column value shape — is a materialization concern (values must change
in the footer too, RI-safely), out of scope here; `table_register` scores it as the standing gap.
"""
from __future__ import annotations

import re

# tokens that are units/measures — pulled into a trailing parenthetical on the display header
_UNIT_TOKENS = {
    "nm": "nM", "um": "µM", "mm": "mm", "cm": "cm", "km": "km", "kg": "kg", "mg": "mg",
    "ml": "mL", "ul": "µL", "l": "L", "pct": "%", "percent": "%", "usd": "$", "eur": "€",
    "gbp": "£", "ppm": "ppm", "ppb": "ppb", "mol": "mol", "mmol": "mmol", "nmol": "nmol",
    "hz": "Hz", "khz": "kHz", "mhz": "MHz", "ph": "pH", "psi": "psi", "rpm": "rpm",
    "days": "days", "hrs": "hrs", "sec": "s", "min": "min",
}
# multi-token unit stems, matched greedily on the header tail (order matters: longest first)
_UNIT_PHRASES = [
    (("mg", "ml"), "mg/mL"), (("ug", "ml"), "µg/mL"), (("ng", "ml"), "ng/mL"),
    (("g", "l"), "g/L"), (("mg", "kg"), "mg/kg"), (("deg", "c"), "°C"), (("deg", "f"), "°F"),
]
# tokens that display uppercased (acronyms) rather than title-cased
_ACRONYMS = {
    "id", "url", "uri", "uuid", "sku", "isbn", "ic50", "ec50", "ph", "dna", "rna", "pcr",
    "hplc", "usd", "eur", "gbp", "api", "sop", "cfr", "fda", "ema", "vh", "vl", "cdr",
    "mrn", "npi", "ssn", "vin", "gtin", "ndc",
}
# tokens whose canonical display is neither all-caps nor title-case (mixed case)
_ACRONYM_DISPLAY = {"ic50": "IC50", "ec50": "EC50", "ph": "pH", "vh": "VH", "vl": "VL"}

_SURROGATE_RE = re.compile(r"^(?:[ab]_)?id$", re.I)   # id / a_id / b_id (view join aliases)
_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


def _tok_display(tok: str) -> str:
    low = tok.lower()
    if low in _ACRONYM_DISPLAY:
        return _ACRONYM_DISPLAY[low]
    if low in _ACRONYMS:
        return low.upper()
    return tok[:1].upper() + tok[1:] if tok else tok


def display_header(name: str) -> str:
    """snake_case physical name → human document header.

    'clone_id'            → 'Clone ID'
    'chemotaxis_ic50_nm'  → 'Chemotaxis IC50 (nM)'
    'analyte_conc_mg_ml'  → 'Analyte Conc (mg/mL)'
    Non-identifier names (already prose) pass through unchanged.
    """
    if not name or not _IDENT.match(name):
        return name
    toks = [t for t in name.split("_") if t]
    if not toks:
        return name
    # drop a leading physical-table prefix 't' segment ('t_clone' → 'clone')
    if len(toks) > 1 and toks[0] == "t":
        toks = toks[1:]
    unit = ""
    # multi-token unit phrase on the tail
    for stems, disp in _UNIT_PHRASES:
        n = len(stems)
        if len(toks) > n and tuple(t.lower() for t in toks[-n:]) == stems:
            unit, toks = disp, toks[:-n]
            break
    if not unit and len(toks) > 1 and toks[-1].lower() in _UNIT_TOKENS:
        unit, toks = _UNIT_TOKENS[toks[-1].lower()], toks[:-1]
    label = " ".join(_tok_display(t) for t in toks) if toks else ""
    return f"{label} ({unit})" if unit and label else (label or name)


def _surrogate_index(headers: "list[str]", slot_refs: "list[str] | None") -> "int | None":
    """Index of the surrogate primary key, if the table has one — by slot_ref '__pk__' when
    available (authoritative), else by name (id / a_id / b_id)."""
    if slot_refs:
        for i, sr in enumerate(slot_refs):
            if sr == "__pk__":
                return i
    for i, h in enumerate(headers):
        if _SURROGATE_RE.match((h or "").strip()):
            return i
    return None


def present_table(headers: "list[str]", rows: "list[list[str]]", *,
                  slot_refs: "list[str] | None" = None, drop_surrogate: bool = False
                  ) -> "tuple[list[str], list[list[str]]]":
    """Presentation-register (display_headers, display_rows) for the PROSE markdown table.

    - a surrogate PK that LEADS is demoted to last (default) — killing the 'leads-with-id' tell
      while keeping every value present so table_fidelity is unaffected; ``drop_surrogate`` removes
      it from the display entirely (only when ≥2 other columns remain — more realistic, but the id
      VALUES then no longer appear in prose, lowering the fidelity diagnostic).
    - every header is rewritten to a human document header.
    Values are never touched. A no-op-safe transform: empty/degenerate tables pass through.
    """
    if not headers:
        return headers, rows
    order = list(range(len(headers)))
    sidx = _surrogate_index(headers, slot_refs)
    if sidx is not None and len(headers) > 1:
        if drop_surrogate and len(headers) >= 3:
            order = [i for i in order if i != sidx]
        elif sidx == 0:  # only reorder when it actually leads (the tell)
            order = order[1:] + [0]
    disp_headers = [display_header(headers[i]) for i in order]
    disp_rows = [[r[i] if i < len(r) else "" for i in order] for r in rows]
    return disp_headers, disp_rows
