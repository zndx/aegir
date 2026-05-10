"""Step definitions for ``features/runs/run_artifacts.feature``.

Tier-0 only — no GPU, no network. Exercises the RunArtifacts writer on a
temp directory (cleaned up in after_scenario) so the tests don't pollute
``outputs/runs/`` with fixture rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from behave import given, then, when  # type: ignore[import]

from aegir.utils.runs import RunArtifacts


@given('a fresh RunArtifacts with task "{task}" and model_size "{model_size}"')
def step_given_fresh_run(context, task, model_size):
    tmp_root = Path("/tmp") / f"aegir_bdd_runs_{int(time.time() * 1000)}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        task=task, model_size=model_size, seed=4649, vocab_size=272,
        batch_size=4, lr=5e-4, epochs=3, max_length=128, max_context_cols=4,
        downsample_factor=2.0, lambda_lb=0.1, amp=False, smoke_test=True,
    )
    context.runs_root = tmp_root
    context.run = RunArtifacts.start(ns, runs_root=tmp_root)
    context._cleanups.append(lambda: _rmtree(tmp_root))


def _rmtree(path: Path) -> None:
    import shutil
    try:
        shutil.rmtree(path)
    except OSError:
        pass


@then("the run_id contains the UTC date as a prefix")
def step_then_run_id_date_prefix(context):
    import re
    assert re.match(r"^\d{8}T\d{6}Z_", context.run.run_id), (
        f"run_id {context.run.run_id!r} lacks UTC-compact prefix"
    )


@then('the run_id contains "{substr}"')
def step_then_run_id_contains(context, substr):
    assert substr in context.run.run_id, (
        f"{substr!r} not in run_id {context.run.run_id!r}"
    )


@when("I record {n:d} synthetic epochs and finalize")
def step_when_record_n_epochs(context, n):
    for e in range(1, n + 1):
        context.run.add_epoch_metrics({
            "epoch": e,
            "train_loss": 4.8 - 0.1 * e,
            "val_loss": 4.7 - 0.08 * e,
            "micro_f1": 0.01 + 0.02 * e,
            "macro_f1": 0.005 + 0.015 * e,
            "boundary": {
                "stage0_mean_F": 0.5 - 0.05 * e,
                "stage0_mean_G": 0.45 - 0.02 * e,
                "stage1_mean_F": 0.5 - 0.01 * e,
                "stage1_mean_G": 0.48 - 0.005 * e,
            },
        })
    context.run.finalize()


@then('the run directory contains "{name}"')
def step_then_contains(context, name):
    path = context.run.run_dir / name
    assert path.exists(), f"expected {path} to exist"


@then("the run directory has {n:d} plot files")
def step_then_plot_count(context, n):
    plots = list((context.run.run_dir / "plots").glob("*.bokeh.json"))
    assert len(plots) == n, f"expected {n} plot files, got {len(plots)}: {plots}"


@then('each plot file parses as a Bokeh JSON document with a "{key}" key')
def step_then_plots_parse(context, key):
    plots = list((context.run.run_dir / "plots").glob("*.bokeh.json"))
    assert plots, "no plot files found"
    for p in plots:
        doc = json.loads(p.read_text())
        assert key in doc, f"plot {p.name} missing {key!r}; keys={list(doc)}"


@when("I record an epoch with only loss and skip F1")
def step_when_record_partial(context):
    context.run.add_epoch_metrics({
        "epoch": 1,
        "train_loss": 4.5,
        "val_loss": 4.3,
        # micro_f1 + macro_f1 + boundary intentionally absent
    })
    context.run.finalize()


@then("the resulting metrics.json has one epoch with null F1 values")
def step_then_partial_metrics(context):
    metrics = json.loads((context.run.run_dir / "metrics.json").read_text())
    epochs = metrics["epochs"]
    assert len(epochs) == 1
    e = epochs[0]
    assert e["micro_f1"] is None, f"micro_f1 expected null, got {e['micro_f1']}"
    assert e["macro_f1"] is None, f"macro_f1 expected null, got {e['macro_f1']}"
