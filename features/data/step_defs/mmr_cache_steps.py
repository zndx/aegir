"""Step definitions for features/data/mmr_cache.feature."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import torch
from behave import given, then, when  # type: ignore[import]


TEMP_CACHE = Path("/tmp/aegir_mmr_cache_bdd")


@given("a fresh MMR cache dir under /tmp")
def step_given_fresh_cache(context):
    shutil.rmtree(TEMP_CACHE, ignore_errors=True)
    os.environ["AEGIR_MMR_CACHE_DIR"] = str(TEMP_CACHE)
    os.environ.pop("AEGIR_MMR_CACHE_DISABLE", None)
    # Force the module to pick up the new env var by resetting its handle.
    import aegir.data.context_select as cs
    cs._CACHE = None


@given("three toy columns")
def step_given_three_columns(context):
    context.cols = [
        ["apple", "banana", "cherry"],
        ["red", "blue", "green"],
        ["1", "2", "3"],
    ]


@given("four toy columns")
def step_given_four_columns(context):
    context.cols = [
        ["apple", "banana", "cherry"],
        ["red", "blue", "green"],
        ["1", "2", "3"],
        ["x", "y", "z"],
    ]


@given("the env var AEGIR_MMR_CACHE_DISABLE=1")
def step_given_disable(context):
    os.environ["AEGIR_MMR_CACHE_DISABLE"] = "1"
    import aegir.data.context_select as cs
    cs._CACHE = None


@when("I embed them twice in the same process")
def step_when_embed_twice(context):
    from aegir.data.context_select import embed_columns

    t0 = time.time()
    context.emb1 = embed_columns(context.cols)
    context.t_cold = time.time() - t0

    t0 = time.time()
    context.emb2 = embed_columns(context.cols)
    context.t_warm = time.time() - t0


@when("I embed four toy columns twice in the same process")
def step_when_embed_four_twice(context):
    from aegir.data.context_select import embed_columns

    t0 = time.time()
    context.emb1 = embed_columns(context.cols)
    context.t_cold = time.time() - t0

    t0 = time.time()
    context.emb2 = embed_columns(context.cols)
    context.t_warm = time.time() - t0


@then("the second call returns the same embeddings within {tol:g}")
def step_then_close(context, tol):
    diff = (context.emb1 - context.emb2).abs().max().item()
    assert diff < tol, f"max_diff={diff} >= {tol}"


@then("the second call is at least {factor:d}x faster than the first")
def step_then_faster(context, factor):
    speedup = context.t_cold / max(context.t_warm, 1e-6)
    assert speedup >= factor, f"speedup={speedup:.1f}x < {factor}x"


@then("both calls trigger the model encoder")
def step_then_both_encode(context):
    # When cache is disabled, both calls should take roughly the same time
    # (within 10×). If caching were active we'd see >1000× speedup.
    ratio = context.t_cold / max(context.t_warm, 1e-6)
    assert ratio < 100, (
        f"warm call was {ratio:.0f}× faster than cold — cache appears active "
        f"despite AEGIR_MMR_CACHE_DISABLE=1"
    )
