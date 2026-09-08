#!/usr/bin/env python
"""Local-vs-remote scaling extrapolation from a chapter-generation run's stats.

Reads a ``generate_chapter`` run's ``chapters.parquet`` (the per-chapter stats the engine-backed run
captures: prompt/completion tokens, latency, finish_reason, reasoning/response chars) and produces the
data needed to choose between **local** Qwen3.8-27B serving and **remote** paid APIs as the corpus scales:

  * measured local characterization — generation throughput (tokens/s), end-to-end chapters/hour, tokens
    per chapter, the reasoning/answer split, and trace COMPLETENESS (finish_reason distribution: a "length"
    share > 0 means truncated — incomplete — thinking traces, which the run must avoid);
  * projection to target corpus sizes — local GPU-hours (+ $ at a configurable $/GPU-hr) vs remote $
    (tokens × GLM/Grok rates), and the break-even corpus size where local's GPU cost beats remote's.

The thinking traces themselves are retained in full elsewhere (``response_reasoning`` + ``raw.exchange``);
this tool only quantifies throughput/cost — it does not reduce the trace to a statistic.

    uv run --no-sync python scripts/analyze_generation_stats.py --run /raid/.../chapters_v0/<run_id>/
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq

# Remote $/M tokens (input, output) — mirror generate_chapter.RATES_USD_PER_M. ADJUST to live rates.
REMOTE_RATES = {
    "cerebras/zai-glm-4.7 (GLM)": (0.50, 1.50),
    "xai/grok-4.3 (Grok)": (3.00, 15.00),
}
_EST_CHARS_PER_TOKEN = 4.0  # only for the reasoning/answer split when a token count is unavailable


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def analyze(run: Path, *, n_gpus: int, gpu_hourly_usd: float, targets: list[int]) -> dict:
    rows = pq.read_table(run / "chapters.parquet").to_pylist()
    if not rows:
        raise SystemExit(f"no chapters in {run}/chapters.parquet")
    n = len(rows)

    pt = [r.get("prompt_tokens") or 0 for r in rows]
    ctok = [r.get("completion_tokens") or 0 for r in rows]
    lat_s = [(r.get("latency_ms") or 0) / 1000.0 for r in rows]
    resp_chars = [r.get("response_chars") or len(r.get("response_text") or "") for r in rows]
    reas_chars = [len(r.get("response_reasoning") or "") for r in rows]
    fins = [r.get("finish_reason") or "" for r in rows]

    # trace completeness — the load-bearing KPI for "retain COMPLETE traces"
    fin_dist: dict[str, int] = {}
    for f in fins:
        fin_dist[f or "(unset)"] = fin_dist.get(f or "(unset)", 0) + 1
    n_truncated = sum(v for k, v in fin_dist.items() if k == "length")

    sum_ct, sum_lat = sum(ctok), sum(lat_s)
    gen_tok_per_s = sum_ct / sum_lat if sum_lat else 0.0          # raw generation speed (GPU characterization)

    # end-to-end wall-clock from created_at span (includes sampling + table materialization + hx writes)
    times = sorted(r["created_at"] for r in rows if r.get("created_at"))
    span_s = (times[-1] - times[0]).total_seconds() if len(times) > 1 else sum_lat
    eff_chapters_per_hr = (n - 1) / (span_s / 3600.0) if span_s > 0 and n > 1 else (n / (sum_lat / 3600.0) if sum_lat else 0.0)

    mean_pt, mean_ct = st.mean(pt), st.mean(ctok)
    reas_share = (sum(reas_chars) / (sum(reas_chars) + sum(resp_chars))) if (sum(reas_chars) + sum(resp_chars)) else 0.0
    est_reas_tok = sum(reas_chars) / _EST_CHARS_PER_TOKEN

    measured = {
        "n_chapters": n,
        "trace_completeness": {"finish_reason": fin_dist, "truncated_share": round(n_truncated / n, 4),
                               "complete": n_truncated == 0},
        "tokens": {"mean_prompt": round(mean_pt, 1), "mean_completion": round(mean_ct, 1),
                   "total_prompt": sum(pt), "total_completion": sum_ct},
        "latency_s": {"mean": round(st.mean(lat_s), 1), "p50": round(_pct(lat_s, 50), 1),
                      "p90": round(_pct(lat_s, 90), 1)},
        "throughput": {"gen_tokens_per_s": round(gen_tok_per_s, 1),
                       "effective_chapters_per_hour": round(eff_chapters_per_hr, 2),
                       "gen_tokens_per_hour": round(gen_tok_per_s * 3600.0)},
        "reasoning": {"mean_reasoning_chars": round(st.mean(reas_chars), 0),
                      "mean_answer_chars": round(st.mean(resp_chars), 0),
                      "reasoning_char_share": round(reas_share, 3),
                      "est_reasoning_tokens_per_chapter": round(est_reas_tok / n, 0)},
        "gpu": {"n_gpus": n_gpus, "tokens_per_s_per_gpu": round(gen_tok_per_s / n_gpus, 1) if n_gpus else 0.0},
    }

    # ── projections ──
    def remote_cost(n_ch: int) -> dict:
        out = {}
        for name, (rin, rout) in REMOTE_RATES.items():
            out[name] = round(n_ch * (mean_pt * rin + mean_ct * rout) / 1e6, 2)
        return out

    def local_cost(n_ch: int) -> dict:
        hrs = n_ch / eff_chapters_per_hr if eff_chapters_per_hr else float("inf")
        gpu_hrs = hrs * n_gpus
        return {"wall_clock_hours": round(hrs, 1), "gpu_hours": round(gpu_hrs, 1),
                "usd_at_gpu_rate": round(gpu_hrs * gpu_hourly_usd, 2)}

    projections = []
    for n_ch in targets:
        loc, rem = local_cost(n_ch), remote_cost(n_ch)
        projections.append({"target_chapters": n_ch, "local": loc, "remote_usd": rem})

    # break-even corpus size: local_gpu_$ == remote_$ (per provider). local_$ per chapter is constant
    # (gpu_hrs/chapter × rate); remote_$ per chapter is constant — so they only cross if local's per-chapter
    # $ is below remote's, in which case local wins at EVERY scale (report that), else remote always wins.
    local_usd_per_ch = (n_gpus / eff_chapters_per_hr * gpu_hourly_usd) if eff_chapters_per_hr else float("inf")
    breakeven = {}
    for name, (rin, rout) in REMOTE_RATES.items():
        remote_usd_per_ch = (mean_pt * rin + mean_ct * rout) / 1e6
        breakeven[name] = {"local_usd_per_chapter": round(local_usd_per_ch, 5),
                           "remote_usd_per_chapter": round(remote_usd_per_ch, 5),
                           "local_cheaper": local_usd_per_ch < remote_usd_per_ch}

    return {"run": str(run), "params": {"n_gpus": n_gpus, "gpu_hourly_usd": gpu_hourly_usd},
            "measured": measured, "projections": projections, "break_even": breakeven}


def render_md(rep: dict) -> str:
    m, L = rep["measured"], []
    tc = m["trace_completeness"]
    L.append(f"# Generation scaling extrapolation — {Path(rep['run']).name}\n")
    tc_status = "✅ ALL complete" if tc["complete"] else f"⚠️ {tc['truncated_share']:.1%} truncated"
    L.append(f"**{m['n_chapters']} chapters** · thinking-trace completeness: {tc_status} "
             f"(finish_reason {tc['finish_reason']})\n")
    th = m["throughput"]
    L.append(f"## Measured (Qwen3.8-27B via the capability engine, {m['gpu']['n_gpus']} GPU)\n")
    L.append(f"- generation: **{th['gen_tokens_per_s']} tok/s** ({m['gpu']['tokens_per_s_per_gpu']} tok/s/GPU) · "
             f"**{th['effective_chapters_per_hour']} chapters/hour** end-to-end\n")
    L.append(f"- tokens/chapter: {m['tokens']['mean_prompt']} in / {m['tokens']['mean_completion']} out · "
             f"latency mean {m['latency_s']['mean']}s (p90 {m['latency_s']['p90']}s)\n")
    L.append(f"- reasoning: {m['reasoning']['mean_reasoning_chars']:.0f} chars/chapter "
             f"({m['reasoning']['reasoning_char_share']:.0%} of generated text is the thinking trace)\n")
    solar = rep["params"]["gpu_hourly_usd"] == 0.0
    chph = rep["measured"]["throughput"]["effective_chapters_per_hour"]
    L.append("\n## Projection — local (free solar compute, time-bound) vs remote (paid, parallel-fast)\n")
    L.append("| target chapters | local wall-clock | local GPU-hrs | local $ | remote $ (GLM) | remote $ (Grok) |")
    L.append("|---|---|---|---|---|---|")
    for p in rep["projections"]:
        loc, rem = p["local"], list(p["remote_usd"].values())
        loc_usd = "**$0 (solar)**" if solar else f"${loc['usd_at_gpu_rate']:,}"
        L.append(f"| {p['target_chapters']:,} | {loc['wall_clock_hours']:,}h | {loc['gpu_hours']:,} | "
                 f"{loc_usd} | ${rem[0]:,} | ${rem[1]:,} |")
    basis = "free renewable (solar)" if solar else f"${rep['params']['gpu_hourly_usd']}/GPU-hr × {rep['params']['n_gpus']} GPU"
    L.append(f"\n_Local compute = **{basis}**, wall-clock at the measured {chph} ch/hr on {rep['params']['n_gpus']} GPU "
             f"(scales ~linearly with added local GPUs). Remote APIs parallelize — their wall-clock is rate-limit-bound, "
             f"not throughput-bound. So the real trade is **time (free, local) vs money (fast, remote)**, not $ vs $._\n")
    L.append("## The decision — local is ~free; remote $ is the premium you'd pay for speed\n")
    mean_lat = rep["measured"]["latency_s"]["mean"]
    for name, b in rep["break_even"].items():
        L.append(f"- **{name}**: remote ≈ **${b['remote_usd_per_chapter']:.4f}/chapter** "
                 f"(${b['remote_usd_per_chapter']*1000:,.0f}/1k chapters) — the premium to skip ~{mean_lat:.0f}s of "
                 f"free local wall-clock per chapter. Local marginal cost: "
                 f"{'**$0** (solar/owned)' if solar else f'${b['local_usd_per_chapter']}/ch'}.")
    L.append(f"\n**Takeaway:** at {chph} ch/hr, local produces ~{chph*24:.0f} chapters/day on {rep['params']['n_gpus']} GPU "
             f"for $0 — scale GPUs (or daylight hours) for more. Reach for paid remote only when a deadline needs "
             f"parallel throughput the local fleet can't meet in time.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="chapters run dir (contains chapters.parquet)")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--gpu-hourly-usd", type=float, default=0.0,
                    help="$/GPU-hr for local compute. DEFAULT 0 — the local GPUs are SOLAR-powered "
                         "(free renewable marginal cost, owned hardware). Set e.g. 0.40 only to compare "
                         "against a cloud-rental counterfactual; local's real constraint is wall-clock, not $.")
    ap.add_argument("--targets", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    rep = analyze(run, n_gpus=args.n_gpus, gpu_hourly_usd=args.gpu_hourly_usd, targets=args.targets)
    out = Path(args.out) if args.out else run / "generation_scaling_report"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str))
    md = render_md(rep)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"→ {out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
