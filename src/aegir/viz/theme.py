"""Color-mode for the live viz apps — the org design norm (cldr-design-template) carried into bokeh.

The UI's ``data-mode`` rides the embed request (``?mode=dark|light`` — PanelView appends it, the
gateway's ``server_document(arguments=…)`` forwards it, ``curdoc().session_context`` receives it) and
the doc gets a CUSTOM bokeh Theme built from the SAME Cloudera kumo tokens as ``theme-cloudera.css``,
so plots sit seamlessly on the panel surfaces in either mode. Toggling in the UI re-embeds the panel
(the mode is part of PanelView's session key), so a fresh session picks the matching theme.

Apps call ``apply_color_mode()`` once, immediately before ``curdoc().add_root``; it returns
``(mode, K)`` where ``K`` is the palette dict for any app-specific styling (guide lines, accents).
Default is dark (the org default) when no mode arg arrives — e.g. a bare /viz hit.
"""
from __future__ import annotations

from bokeh.io import curdoc
from bokeh.themes import Theme

# the kumo token values, verbatim from ui/src/styles/theme-cloudera.css (dark block / light block)
KUMO = {
    "dark": {"bg": "#12121a", "canvas": "#0a0a0f", "text": "#e5e7eb", "subtle": "#9ca3af",
             "inactive": "#6b7280", "line": "#2e2e3a", "accent": "#818cf8", "brand": "#6366f1"},
    "light": {"bg": "#ffffff", "canvas": "#f4f4f8", "text": "#1f2937", "subtle": "#6b7280",
              "inactive": "#9ca3af", "line": "#e5e7eb", "accent": "#4338ca", "brand": "#6366f1"},
}


def color_mode(default: str = "dark") -> str:
    """The requested mode from the session args; org default = dark."""
    sc = curdoc().session_context
    args = (sc.request.arguments if (sc and sc.request) else {}) or {}
    raw = args.get("mode", [b""])[0]
    mode = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw or "")
    return "light" if mode == "light" else ("dark" if mode == "dark" else default)


def _theme(k: dict) -> Theme:
    return Theme(json={"attrs": {
        "Plot": {"background_fill_color": k["bg"], "border_fill_color": k["bg"],
                 "outline_line_color": k["line"]},
        "Grid": {"grid_line_color": k["line"], "grid_line_alpha": 0.55},
        "Axis": {"major_label_text_color": k["subtle"], "axis_label_text_color": k["subtle"],
                 "major_tick_line_color": k["line"], "minor_tick_line_color": k["line"],
                 "axis_line_color": k["line"]},
        "Title": {"text_color": k["text"]},
        "Legend": {"background_fill_color": k["bg"], "background_fill_alpha": 0.85,
                   "label_text_color": k["text"], "border_line_color": k["line"]},
        "ColorBar": {"background_fill_color": k["bg"], "major_label_text_color": k["subtle"],
                     "title_text_color": k["subtle"]},
    }})


def apply_color_mode(default: str = "dark") -> "tuple[str, dict]":
    """Set the doc theme for the session's mode; → (mode, kumo palette) for app-specific styling."""
    mode = color_mode(default)
    k = KUMO[mode]
    curdoc().theme = _theme(k)
    return mode, k
