"""Live HoloViews parallel-coordinates ("sweeps") over training runs — served by the bokeh server
(topology B), embedded into the lineup Training ▸ Sweeps panel via ``<PanelView>``. ``?task=`` scopes
a sweep (default: all runs).

HoloViews-native by design: no canonical PCP *widget* is needed — each run is an ``hv.Path`` polyline
across normalized hyperparam→outcome axes, colored by the headline metric. Same situational awareness
as a W&B parallel-coordinates plot, on our stack, and the door is open to the *superior* version:
``datashade`` for thousands of runs and ``hv.link_selections`` for axis brushing (both native, fast-
follows). Renders through the same ``hv.render`` → bokeh-server → ``PanelView`` path as the chord and
the run curves (air-gapped). Reads runs live from ``outputs/runs`` (no copy into the KB).
"""
from __future__ import annotations

import json
from pathlib import Path

import holoviews as hv
import pandas as pd
from bokeh.io import curdoc

from aegir.config import load_config
from aegir.utils import runs as R

hv.extension("bokeh", logo=False)

_SIZE_ORD = {"tiny": 0, "small": 1, "base": 2}
# (axis label, accessor over metadata + metrics.final) — hyperparams first, then outcomes.
_DIMS = [
    ("model_size", lambda meta, f: _SIZE_ORD.get(meta.get("model_size"))),
    ("num_params", lambda meta, f: meta.get("num_params")),
    ("lr", lambda meta, f: meta.get("lr")),
    ("epochs", lambda meta, f: f.get("num_epochs_completed")),
    ("macro_f1", lambda meta, f: f.get("best_val_macro_f1")),
    ("micro_f1", lambda meta, f: f.get("best_val_micro_f1")),
    ("val_loss", lambda meta, f: f.get("last_val_loss")),
]
_COLOR_DIM = "macro_f1"


def _df(task: str) -> pd.DataFrame:
    rows = []
    for rd in R.iter_runs(Path(load_config().runs.dir)):
        meta = json.loads((rd / "metadata.json").read_text())
        if task and meta.get("task") != task:
            continue
        mp = rd / "metrics.json"
        final = (json.loads(mp.read_text()).get("final") or {}) if mp.exists() else {}
        row = {"run": meta.get("run_id", rd.name), "task": meta.get("task")}
        for name, acc in _DIMS:
            row[name] = acc(meta, final)
        rows.append(row)
    return pd.DataFrame(rows)


def _placeholder(msg: str):
    return hv.Div(f"<div style='padding:1em;color:#888;font-family:sans-serif'>{msg}</div>")


def _pcp(task: str):
    df = _df(task)
    dims = [n for n, _ in _DIMS if n in df.columns and pd.to_numeric(df[n], errors="coerce").notna().any()]
    if len(df) < 1 or len(dims) < 2:
        return _placeholder(f"not enough runs for a sweep yet (task={task or 'all'})")

    norm: dict[str, pd.Series] = {}
    rng: dict[str, tuple[float, float]] = {}
    for d in dims:
        col = pd.to_numeric(df[d], errors="coerce")
        col = col.fillna(col.median())
        lo, hi = float(col.min()), float(col.max())
        rng[d] = (lo, hi)
        norm[d] = (col - lo) / (hi - lo) if hi > lo else col * 0 + 0.5

    color = _COLOR_DIM if _COLOR_DIM in dims else dims[-1]
    cvals = pd.to_numeric(df[color], errors="coerce").fillna(0.0)
    paths = [{"x": list(range(len(dims))), "y": [float(norm[d].iloc[i]) for d in dims],
              "metric": float(cvals.iloc[i]), "run": str(df["run"].iloc[i])}
             for i in range(len(df))]
    pcp = hv.Path(paths, kdims=["x", "y"], vdims=["metric", "run"]).opts(
        hv.opts.Path(color="metric", cmap="viridis", colorbar=True, line_width=2, alpha=0.75,
                     tools=["hover"], width=720, height=420, yaxis=None, show_frame=False,
                     xticks=[(j, d) for j, d in enumerate(dims)], xrotation=30,
                     title=f"Sweep: {task or 'all runs'} · {len(df)} runs · color = best {color}"))
    guides = hv.Overlay([hv.VLine(j).opts(color="#d9d9d9", line_width=1) for j in range(len(dims))])
    labels = hv.Labels(
        [(j, 1.05, f"{rng[d][1]:.3g}") for j, d in enumerate(dims)]
        + [(j, -0.05, f"{rng[d][0]:.3g}") for j, d in enumerate(dims)],
        kdims=["x", "y"], vdims=["Label"]).opts(text_font_size="7pt", text_color="#999")
    return (guides * pcp * labels).opts(hv.opts.Overlay(show_frame=False))


def _task_arg() -> str:
    sc = curdoc().session_context
    a = sc.request.arguments if (sc and sc.request) else {}
    v = a.get("task", [b""])
    return v[0].decode() if v and isinstance(v[0], (bytes, bytearray)) else (v[0] if v else "")


curdoc().add_root(hv.render(_pcp(_task_arg()), backend="bokeh"))
