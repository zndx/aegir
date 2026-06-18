"""Lightweight, repeatable verify gate for the live leaderboard viz.

Asserts the latest training run's plots build + render through `runs.py`'s HoloViews builders — the
exact path `aegir.viz.runs_app` uses to serve them live to the React leaderboard. No browser needed;
this is the committable gate. Run via `just verify-viz` (which first produces a real run with
`train.py --smoke-test`). For the full browser gate (air-gap + no JS errors): `just viz-serve`, then
open /leaderboards and click a run → the drawer plots render live.
"""
import json
from pathlib import Path

import holoviews as hv

from aegir.config import load_config
from aegir.utils import runs as R

hv.extension("bokeh", logo=False)


def main() -> None:
    runs_dir = Path(load_config().runs.dir)
    found = R.iter_runs(runs_dir)
    if not found:
        raise SystemExit(f"no runs under {runs_dir} — run `train.py --smoke-test` first")
    rd = found[0]
    plots = R.load_run_detail(rd)["plots"]
    assert "loss" in plots and "f1" in plots, f"unexpected plots: {plots}"
    epochs = json.loads((rd / "metrics.json").read_text())["epochs"]
    for name in plots:
        if name == "f1":
            fig = R._f1_curve(epochs)
        elif name.startswith("boundary_stage"):
            fig = R._boundary_curve(epochs, int(name.removeprefix("boundary_stage")))
        else:
            fig = R._loss_curve(epochs)
        hv.render(fig, backend="bokeh")   # raises if the builder/render path breaks
    print(f"verify-viz OK: run {rd.name} · plots {plots} · all build+render")
    print("browser gate: `just viz-serve`, then open /leaderboards (click a run → drawer plots render "
          "live, air-gapped)")


if __name__ == "__main__":
    main()
