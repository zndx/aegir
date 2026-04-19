"""Step definitions for features/chunking/ema_scan.feature.

Validates that the SSD-kernel EMA scan used by DeChunkLayer agrees with the
sequential reference within bf16 precision, for both forward values and
input gradients. Runs only on CUDA — @gpu tag skips on CPU-only boxes.
"""

from __future__ import annotations

import torch
from behave import given, then, when  # type: ignore[import]


@given("a random (B,L,D)=({b:d},{l:d},{d:d}) input with decay clamped to [1e-4, 1-1e-4]")
def step_given_input(context, b, l, d):  # noqa: E741  (reuse the feature's variable name)
    if not torch.cuda.is_available():
        context.scenario.skip("CUDA required for SSD EMA scan")
        return
    torch.manual_seed(0)
    context.x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    p = torch.rand(b, l, 1, device="cuda", dtype=torch.bfloat16).clamp(1e-3, 1 - 1e-3)
    context.p = p
    context.d = (1 - p).clamp(1e-4, 1 - 1e-4)


@given("the first-token decay forced to 1-1e-4 (matches RoutingModule pad)")
def step_given_pad_first(context):
    # RoutingModule pads boundary_prob[:, 0] = 1.0; DeChunkLayer clamps to 1-1e-4.
    # So first-token decay = 1 - (1-1e-4) = 1e-4.
    context.d[:, 0] = 1e-4


@when("I compute both sequential and SSD EMA scans")
def step_when_compute_scans(context):
    from aegir.modules.dc import _ema_scan_sequential, _ema_scan_ssd
    context.y_seq = _ema_scan_sequential(context.x, context.d)
    context.y_ssd = _ema_scan_ssd(context.x, context.d)


@then("their max relative error is below {tol:g}%")
def step_then_forward_close(context, tol):
    err = (context.y_seq.float() - context.y_ssd.float()).abs().max().item()
    ref = context.y_seq.float().abs().max().item() + 1e-9
    rel = err / ref
    assert rel * 100 < tol, f"fwd rel_err {rel * 100:.3f}% >= {tol}%"


@when("I compute both sequential and SSD EMA scan input gradients")
def step_when_compute_grads(context):
    from aegir.modules.dc import _ema_scan_sequential, _ema_scan_ssd

    g = torch.randn_like(context.x)
    x_seq = context.x.detach().clone().requires_grad_(True)
    _ema_scan_sequential(x_seq, context.d).backward(g)
    context.grad_seq = x_seq.grad

    x_ssd = context.x.detach().clone().requires_grad_(True)
    _ema_scan_ssd(x_ssd, context.d).backward(g)
    context.grad_ssd = x_ssd.grad


@then("their max relative gradient error is below {tol:g}%")
def step_then_grad_close(context, tol):
    err = (context.grad_seq.float() - context.grad_ssd.float()).abs().max().item()
    ref = context.grad_seq.float().abs().max().item() + 1e-9
    rel = err / ref
    assert rel * 100 < tol, f"bwd rel_err {rel * 100:.3f}% >= {tol}%"
