"""Step definitions for features/inference/rosa_step.feature."""

from __future__ import annotations

import torch
from behave import given, then, when  # type: ignore[import]


@given("a freshly constructed RWKV_ROSA block with d_model={d:d}")
def step_given_rosa_block(context, d):
    from aegir.models.config import RWKVConfig
    from aegir.modules.rwkv import RWKV_ROSA

    context.rosa = RWKV_ROSA(d_model=d, rwkv_cfg=RWKVConfig(), layer_idx=0)
    context.d_model = d


@when("I call step() on a fake (B={b:d}, L={l:d}, D={d:d}) input with allocated state")
def step_when_step_called(context, b, l, d):  # noqa: E741
    cache = context.rosa.allocate_inference_cache(batch_size=b, max_seqlen=l, dtype=torch.float32)
    # Wrap cache in the dict-style container step() expects
    class _Params:
        def __init__(self, cache):
            self.key_value_memory_dict = {0: cache}
    x = torch.zeros(b, l, d)
    context.caught_exc = None
    try:
        context.rosa.step(x, _Params(cache))
    except Exception as exc:
        context.caught_exc = exc


@then("the call raises NotImplementedError")
def step_then_raises(context):
    assert isinstance(context.caught_exc, NotImplementedError), (
        f"expected NotImplementedError, got {type(context.caught_exc).__name__}: {context.caught_exc}"
    )


@then('the error message mentions "{needle}"')
def step_then_message(context, needle):
    msg = str(context.caught_exc)
    assert needle in msg, f"{needle!r} not in error message: {msg!r}"
