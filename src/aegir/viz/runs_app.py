"""Live HoloViews training-run plots — served by the **Bokeh server** (topology B), embedded into the
React Leaderboards via `<PanelView app="runs_app" params={run_id, plot}>`. The plot is chosen by the
`?run_id=&plot=` query args (plot = loss | f1 | boundary_stage{i}).

Reuses `runs.py`'s HoloViews curve builders (`_loss_curve`/`_f1_curve`/`_boundary_curve`) — the same
ones that used to be pre-rendered to static `.bokeh.json` — and renders them **live** off the run's
`metrics.json`. So the leaderboard plots load from the bokeh server's own BokehJS (air-gapped) instead
of the npm `@bokeh/bokehjs` build. Per-session render is cheap (reads one small JSON). Served by
`just viz-serve` alongside `lineup_app`.
"""
from __future__ import annotations

import json
from pathlib import Path

import holoviews as hv
from bokeh.io import curdoc

from aegir.config import load_config
from aegir.utils import runs as R

hv.extension("bokeh", logo=False)


def _epochs(run_id: str) -> list[dict]:
    if not run_id:
        return []
    mpath = Path(load_config().runs.dir) / run_id / "metrics.json"
    if not mpath.exists():
        return []
    return json.loads(mpath.read_text()).get("epochs", [])


def _placeholder(msg: str):
    return hv.Div(f"<div style='padding:1em;color:#888;font-family:sans-serif'>{msg}</div>")


def _plot(run_id: str, name: str):
    epochs = _epochs(run_id)
    if not epochs:
        return _placeholder(f"no metrics for run {run_id or '(none)'}")
    if name == "f1":
        return R._f1_curve(epochs)
    if name.startswith("boundary_stage"):
        try:
            stage = int(name.removeprefix("boundary_stage"))
        except ValueError:
            stage = 0
        return R._boundary_curve(epochs, stage) or _placeholder(f"no boundary data for stage {stage}")
    return R._loss_curve(epochs)   # default


def _args() -> tuple[str, str]:
    sc = curdoc().session_context
    a = sc.request.arguments if (sc and sc.request) else {}

    def g(key: str, default: str) -> str:
        v = a.get(key, [default.encode()])
        return v[0].decode() if v and isinstance(v[0], (bytes, bytearray)) else (v[0] if v else default)

    return g("run_id", ""), g("plot", "loss")


_rid, _name = _args()
from aegir.viz.theme import apply_color_mode, themed  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
curdoc().add_root(themed(hv.render(_plot(_rid, _name), backend="bokeh"), _K))
