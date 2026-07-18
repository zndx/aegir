"""Live reward-dynamics panel for a P5 GRPO/RLVR run — the GRPO **health monitor**: reward trend, the
**±std variance band (the collapse canary)**, the verifier **R_A pass-rate**, and the z-scored
advantage. Served by the bokeh server (topology B), embedded into the lineup Training ▸ Reward panel
via ``<PanelView app="reward_app">``.

Reads the GRPO ``metrics_jsonl`` (one ``GRPOMetrics`` row per iteration: rewards_mean/std/min/max,
advantage_mean/std, r_a_pass_rate, n_invalid_compositions) live from ``cfg.p5.output_dir`` — same
``hv.render`` → bokeh-server → ``PanelView`` path as the chord / curves / sweeps (air-gapped). Set
``GRPOConfig.metrics_jsonl_path`` on a run to populate it.
"""
from __future__ import annotations

import json
from pathlib import Path

import holoviews as hv
import pandas as pd
from bokeh.io import curdoc

from aegir.config import load_config

hv.extension("bokeh", logo=False)


def _metrics_path() -> Path:
    p5 = Path(load_config().p5.output_dir)
    direct = p5 / "grpo_metrics.jsonl"          # GRPOConfig.metrics_jsonl_path convention
    if direct.exists():
        return direct
    cands = sorted(p5.glob("**/grpo_metrics*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    return cands[0] if cands else direct


def _df() -> pd.DataFrame:
    mp = _metrics_path()
    if not mp.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(ln) for ln in mp.read_text().splitlines() if ln.strip()])


def _panel():
    df = _df()
    if df.empty or "rewards_mean" not in df.columns:
        return hv.Div("<div style='padding:1em;color:#888;font-family:sans-serif'>no GRPO metrics yet — "
                      "set <code>GRPOConfig.metrics_jsonl_path</code> on the run (cfg.p5.output_dir/"
                      "grpo_metrics.jsonl)</div>")
    x = "iteration" if "iteration" in df.columns else df.index.name or "index"
    if x not in df.columns:
        df = df.reset_index().rename(columns={"index": "iteration"})
        x = "iteration"

    # Reward (R ∈ [0,1]): mean ± std band (the variance canary) + faint min/max + the R_A pass-rate,
    # all on one [0,1] axis for at-a-glance health.
    reward = (
        hv.Spread((df[x], df["rewards_mean"], df["rewards_std"])).opts(alpha=0.22, color="#4f7cff")
        * hv.Curve((df[x], df["rewards_min"])).opts(color="#9fb0c7", line_dash="dotted", alpha=0.7)
        * hv.Curve((df[x], df["rewards_max"])).opts(color="#9fb0c7", line_dash="dotted", alpha=0.7)
        * hv.Curve((df[x], df["rewards_mean"]), label="reward R (mean±std)").opts(color=_K["accent"], line_width=2)
        * hv.Curve((df[x], df["r_a_pass_rate"]), label="R_A pass-rate").opts(color="#13a884", line_width=2)
    ).opts(width=720, height=250, ylabel="reward / pass-rate", xlabel="GRPO iteration",
           legend_position="bottom_right", tools=["hover"], title="")

    adv = (
        hv.Spread((df[x], df["advantage_mean"], df["advantage_std"])).opts(alpha=0.22, color="#d4880b")
        * hv.Curve((df[x], df["advantage_mean"]), label="advantage (mean±std)").opts(color="#a5620b", line_width=2)
    ).opts(width=720, height=160, ylabel="advantage (z)", xlabel="GRPO iteration",
           legend_position="bottom_right", tools=["hover"], title="")

    return (reward + adv).cols(1).opts(hv.opts.Layout(shared_axes=False))


from aegir.viz.theme import apply_color_mode, themed  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
curdoc().add_root(themed(hv.render(_panel(), backend="bokeh"), _K))
