"""Step definitions for ``features/gateway/endpoints.feature``.

Uses FastAPI's in-process ``TestClient`` so every tier-0 scenario runs
without actually binding a port. Temp runs dirs are cleaned up in
``after_scenario`` via ``context._cleanups``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from behave import given, then, when  # type: ignore[import]

from aegir.config import Config, DataCfg, RunsCfg, UiCfg
from aegir.gateway.app import create_app
from aegir.utils.runs import RunArtifacts


def _synth_run(runs_root: Path, task: str, model_size: str, offset_ms: int = 0) -> RunArtifacts:
    ns = argparse.Namespace(
        task=task, model_size=model_size, seed=4649, vocab_size=272,
        batch_size=4, lr=5e-4, epochs=2, max_length=128, max_context_cols=4,
        downsample_factor=2.0, lambda_lb=0.1, amp=False, smoke_test=True,
    )
    time.sleep(offset_ms / 1000.0)  # ensure distinct run_id timestamps
    run = RunArtifacts.start(ns, runs_root=runs_root)
    for e in (1, 2):
        run.add_epoch_metrics({
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
    run.finalize()
    return run


def _rmtree(path: Path) -> None:
    import shutil
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def _build_client(runs_dir: Path, gittables_dir: Path | None = None):
    from fastapi.testclient import TestClient
    cfg = Config()
    cfg.runs = RunsCfg(dir=str(runs_dir))
    cfg.ui = UiCfg(dist_dir="/nonexistent-ui-dist-for-bdd")
    if gittables_dir is not None:
        cfg.data = DataCfg(gittables_signals_dir=str(gittables_dir))
    app = create_app(cfg)
    return TestClient(app)


@given("a gateway app bound to a temp runs dir with {n:d} runs")
def step_given_gateway_empty(context, n):
    # Placeholder for 0-run scenario; more specific variants below set up
    # synthetic data.
    runs_dir = Path("/tmp") / f"aegir_bdd_gateway_{int(time.time() * 1000)}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    context._cleanups.append(lambda: _rmtree(runs_dir))
    context.runs_dir = runs_dir
    context.client = _build_client(runs_dir)
    assert n == 0, (
        "the non-zero variant is handled by 'a temp runs dir with {n:d} synthetic runs'"
    )


@given("a gateway app bound to a temp runs dir with {n:d} synthetic runs")
def step_given_gateway_with_runs(context, n):
    runs_dir = Path("/tmp") / f"aegir_bdd_gateway_{int(time.time() * 1000)}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    context._cleanups.append(lambda: _rmtree(runs_dir))
    context.runs_dir = runs_dir
    # Stagger run_ids by a full second so lexical sort picks the last one.
    for i in range(n):
        _synth_run(runs_dir, task="gt-signals-dbpedia", model_size="tiny", offset_ms=1100 * i)
    context.client = _build_client(runs_dir)


@given("a gateway app configured to the cldr/signals gittables dir")
def step_given_gittables_configured(context):
    runs_dir = Path("/tmp") / f"aegir_bdd_gateway_{int(time.time() * 1000)}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    context._cleanups.append(lambda: _rmtree(runs_dir))
    context.client = _build_client(
        runs_dir,
        gittables_dir=Path("/home/rch/local/src/cldr/signals/build/datasets/gittables"),
    )


@when('I GET "{url}"')
def step_when_get(context, url):
    context.response = context.client.get(url)


@when("I GET the detail endpoint for the most recent run")
def step_when_get_detail(context):
    lb = context.client.get("/api/leaderboard").json()
    assert lb["count"] > 0, "need at least one run to fetch detail"
    run_id = lb["rows"][0]["run_id"]
    context.run_id = run_id
    context.response = context.client.get(f"/api/runs/{run_id}")


@when("I GET the loss plot for the most recent run")
def step_when_get_loss_plot(context):
    lb = context.client.get("/api/leaderboard").json()
    run_id = lb["rows"][0]["run_id"]
    context.run_id = run_id
    context.response = context.client.get(f"/api/runs/{run_id}/plot/loss")


@when('I GET a plot named "{name}"')
def step_when_get_bad_plot(context, name):
    lb = context.client.get("/api/leaderboard").json()
    run_id = lb["rows"][0]["run_id"] if lb["count"] else "fake_run_id"
    # Quote the bad name so the client sends it literally rather than
    # having urllib normalize the path traversal.
    from urllib.parse import quote
    context.response = context.client.get(
        f"/api/runs/{run_id}/plot/{quote(name, safe='')}"
    )


@then("the response status is {code:d}")
def step_then_status(context, code):
    assert context.response.status_code == code, (
        f"expected {code}, got {context.response.status_code}: {context.response.text}"
    )


@then('the response has field "{field}" equal to {value}')
def step_then_field_equal(context, field, value):
    data = context.response.json()
    # Parse the literal: true/false/null/number/string
    if value == "true":
        expected = True
    elif value == "false":
        expected = False
    elif value == "null":
        expected = None
    else:
        try:
            expected = int(value)
        except ValueError:
            try:
                expected = float(value)
            except ValueError:
                expected = value.strip('"')
    assert data.get(field) == expected, (
        f'field {field!r} expected {expected!r}, got {data.get(field)!r}'
    )


@then("each leaderboard row has a non-null run_id and task")
def step_then_rows_populated(context):
    data = context.response.json()
    for row in data["rows"]:
        assert row.get("run_id"), f"missing run_id in {row}"
        assert row.get("task"), f"missing task in {row}"


@then('the response has a "plots" list containing "{p1}" and "{p2}"')
def step_then_plots_list(context, p1, p2):
    data = context.response.json()
    assert p1 in data["plots"] and p2 in data["plots"], (
        f"expected {p1!r} and {p2!r} in plots; got {data['plots']}"
    )


@then("the response is a Bokeh JSON document")
def step_then_bokeh_doc(context):
    data = context.response.json()
    for key in ("doc", "root_id", "target_id"):
        assert key in data, f"expected key {key!r} in Bokeh JSON; got {list(data)}"


@then("the response has at least {n:d} types")
def step_then_types_count(context, n):
    data = context.response.json()
    assert len(data.get("types", [])) >= n, (
        f"expected >= {n} types, got {len(data.get('types', []))}"
    )


@then('the response text mentions "{substr}"')
def step_then_text_mentions(context, substr):
    assert substr in context.response.text, (
        f"expected {substr!r} in response text; got: {context.response.text[:200]}"
    )


def _nested_get(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


@then('the response has nested field "{path}" equal to {value}')
def step_then_nested_field(context, path, value):
    data = context.response.json()
    if value == "true":
        expected = True
    elif value == "false":
        expected = False
    elif value == "null":
        expected = None
    else:
        try:
            expected = int(value)
        except ValueError:
            try:
                expected = float(value)
            except ValueError:
                expected = value.strip('"')
    actual = _nested_get(data, path)
    assert actual == expected, f"{path!r} expected {expected!r}, got {actual!r}"


@then('the response has field "{path}" greater than {value:d}')
def step_then_field_greater(context, path, value):
    data = context.response.json()
    actual = _nested_get(data, path)
    assert isinstance(actual, (int, float)), f"{path!r} is not numeric: {actual!r}"
    assert actual > value, f"{path!r} expected > {value}, got {actual!r}"


@then('the response has nested field "{path}" greater than {value:d}')
def step_then_nested_greater(context, path, value):
    data = context.response.json()
    actual = _nested_get(data, path)
    assert isinstance(actual, (int, float)), f"{path!r} is not numeric: {actual!r}"
    assert actual > value, f"{path!r} expected > {value}, got {actual!r}"


@then('the response has an "{name}" root classification')
def step_then_root_classification(context, name):
    data = context.response.json()
    roots = data.get("classifications", [])
    assert any(r.get("name") == name for r in roots), (
        f"no root named {name!r}; got {[r.get('name') for r in roots]}"
    )


@then('every child has "{a}" and "{b}"')
def step_then_every_child_has(context, a, b):
    data = context.response.json()
    roots = data.get("classifications", [])
    for root in roots:
        for child in root.get("children", []):
            assert a in child, f"child {child.get('name')} missing {a!r}"
            assert b in child, f"child {child.get('name')} missing {b!r}"


@then('the response contains an "{iri}" node')
def step_then_contains_node(context, iri):
    data = context.response.json()
    nodes = data.get("nodes", [])
    assert any(n.get("id") == iri for n in nodes), (
        f"no node {iri!r}; got {[n.get('id') for n in nodes]}"
    )


@when('I POST to "{path}" with {n:d} terms')
def step_when_post_terms(context, path, n):
    terms = [{"name": f"Term{i}"} for i in range(n)]
    context.response = context.client.post(path, json={"terms": terms})


@then("the response has exactly {n:d} suggestions")
def step_then_suggestion_count(context, n):
    data = context.response.json()
    assert data.get("count") == n, f"expected {n} suggestions, got {data.get('count')}"


@then("the predictor is identified as a stub")
def step_then_predictor_stub(context):
    data = context.response.json()
    assert "stub" in data.get("predictor", ""), (
        f"expected stub predictor, got {data.get('predictor')!r}"
    )
