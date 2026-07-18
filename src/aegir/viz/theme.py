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


def themed(model, k: dict):
    """POST-render pass — the piece the Theme cannot do: hv writes EXPLICIT white fills on every
    figure (`background_fill_color='#ffffff'`, border, outline), and a bokeh Theme only fills unset
    properties. Walk the rendered tree and re-assert the palette on the explicit ones. Returns the
    model, so it wraps in place: ``curdoc().add_root(themed(hv.render(...), _K))``."""
    from bokeh.models import Axis, ColorBar, Legend, Plot, Title
    for p in model.select(dict(type=Plot)):
        p.background_fill_color = k["bg"]
        p.border_fill_color = k["bg"]
        p.outline_line_color = k["line"]
    for a in model.select(dict(type=Axis)):
        a.major_label_text_color = k["subtle"]
        a.axis_label_text_color = k["subtle"]
        a.major_tick_line_color = k["line"]
        a.minor_tick_line_color = k["line"]
        a.axis_line_color = k["line"]
    for lg in model.select(dict(type=Legend)):
        lg.background_fill_color = k["bg"]
        lg.background_fill_alpha = 0.85
        lg.label_text_color = k["text"]
        lg.border_line_color = k["line"]
    for t in model.select(dict(type=Title)):
        t.text_color = k["text"]
    for cb in model.select(dict(type=ColorBar)):
        cb.background_fill_color = k["bg"]
        cb.major_label_text_color = k["subtle"]
        cb.title_text_color = k["subtle"]
    return model


def apply_color_mode(default: str = "dark") -> "tuple[str, dict]":
    """Set the doc AND HoloViews-renderer theme for the session's mode; → (mode, kumo palette).

    Both are required: a bokeh doc Theme only fills UNSET properties, and HoloViews assigns many
    figure properties explicitly at render time — so hv output ignores the doc theme unless the
    RENDERER carries it too (hv applies renderer.theme during ``hv.render``). Call BEFORE render."""
    mode = color_mode(default)
    k = KUMO[mode]
    t = _theme(k)
    curdoc().theme = t
    try:
        import holoviews as hv
        hv.renderer("bokeh").theme = t
    except Exception:  # noqa: BLE001 — an app without hv still gets the doc theme
        pass
    return mode, k
