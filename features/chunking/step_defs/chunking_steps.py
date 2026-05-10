"""Step definitions for ``features/chunking/dynamic_boundaries.feature``.

These scenarios focus on the H-Net routing module's behavior: load-balancing
convergence, STE gradient flow, and per-stage diagnostics. Most rely on the
harness from ``features.annotation.step_defs.harness``; a few tier-0 steps
build synthetic ``RoutingModuleOutput`` instances to exercise the utility
function in isolation.
"""

from __future__ import annotations

import math

from behave import given, when, then  # type: ignore[import]


@when("I call boundary_diagnostics with an empty bpred_output list")
def step_when_bd_empty(context):
    from aegir.utils.train import boundary_diagnostics
    context.bd_result = boundary_diagnostics([], target_N=2.0)


@then("the result is an empty dict")
def step_then_bd_empty(context):
    assert context.bd_result == {}, f"expected empty dict, got {context.bd_result}"


@given("a synthetic bpred_output with 2 stages at selection rates {r0:g} and {r1:g}")
def step_given_synth_bpred(context, r0, r1):
    import torch
    from aegir.modules.dc import RoutingModuleOutput

    def _stage(rate: float) -> RoutingModuleOutput:
        B, L = 2, 64
        # boundary_prob is (B, L, 2): last slot is the positive-class probability.
        probs = torch.full((B, L, 2), 1.0 - rate)
        probs[..., -1] = rate
        mask = torch.bernoulli(torch.full((B, L), rate)).bool()
        # Deterministic mask at the requested rate (no randomness across runs):
        k = max(1, int(round(rate * L)))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, :k] = True
        return RoutingModuleOutput(
            boundary_prob=probs,
            boundary_mask=mask,
            selected_probs=torch.full((B, L), rate),
        )

    context.bpred = [_stage(r0), _stage(r1)]


@when("I call boundary_diagnostics with target_N {target_N:g}")
def step_when_bd_target(context, target_N):
    from aegir.utils.train import boundary_diagnostics
    context.bd_result = boundary_diagnostics(context.bpred, target_N=target_N)
    context.bd_target_N = target_N


@then("stage{i:d} mean_F is close to {expected:g}")
def step_then_stage_mean_f(context, i, expected):
    v = context.bd_result[f"stage{i}_mean_F"]
    assert math.isclose(v, expected, abs_tol=0.02), (
        f"stage{i}_mean_F={v:.3f} not close to {expected}"
    )


@then("stage{i:d} abs_ratio_err is close to {expected:g}")
def step_then_stage_err(context, i, expected):
    v = context.bd_result[f"stage{i}_abs_ratio_err"]
    assert math.isclose(v, expected, abs_tol=0.02), (
        f"stage{i}_abs_ratio_err={v:.3f} not close to {expected}"
    )


@then("each stage's achieved_N matches 1 over its mean_F")
def step_then_achieved_matches(context):
    for i in range(2):
        F = context.bd_result[f"stage{i}_mean_F"]
        N = context.bd_result[f"stage{i}_achieved_N"]
        if F < 1e-6:
            assert math.isinf(N)
        else:
            assert math.isclose(N, 1.0 / F, rel_tol=1e-3), (
                f"stage{i} achieved_N={N:.3f} != 1/{F:.3f}"
            )


# ── Tier-1: gradient + convergence on real batches ────────────


@then("every stage's RoutingModule has non-zero gradient magnitude")
def step_then_stage_grad_magnitude(context):
    stage_totals: dict[str, float] = {}
    for name, p in context.model.named_parameters():
        if "routing_module" not in name:
            continue
        if p.grad is None:
            continue
        # Group parameters by their routing-module path. The hierarchy produces
        # names like "backbone.routing_module..." for stage 0 and
        # "backbone.main_network.routing_module..." for stage 1.
        depth = name.count("main_network")
        key = f"stage{depth}"
        stage_totals[key] = stage_totals.get(key, 0.0) + p.grad.abs().sum().item()
    assert stage_totals, "no routing_module parameters with gradients — check module naming"
    dead = [k for k, v in stage_totals.items() if v == 0.0]
    assert not dead, f"stages with zero total routing gradient: {dead} (totals: {stage_totals})"


@then("the STE pass-through produces finite gradients on boundary probabilities")
def step_then_ste_finite(context):
    import torch
    bp_list = context.fwd_output.bpred_output
    assert bp_list, "no bpred_output from forward — confirm model has routing stages"
    for bp in bp_list:
        assert torch.isfinite(bp.boundary_prob).all(), "non-finite boundary_prob"
        # selected_probs is what STE multiplies against; it must stay finite.
        assert torch.isfinite(bp.selected_probs).all(), "non-finite selected_probs"


@then("at every routing stage the achieved N is finite")
def step_then_achieved_n_finite(context):
    b = context.train_summary["boundary"]
    n_keys = [k for k in b if k.endswith("_achieved_N")]
    assert n_keys, f"no achieved_N keys in {list(b)}"
    inf_stages = [k for k in n_keys if math.isinf(b[k])]
    assert not inf_stages, f"stages with infinite achieved_N: {inf_stages} (diag: {b})"


@then("the outermost stage's mean_F drifts from the untrained baseline")
def step_then_outer_meanf_drifts(context):
    # Why only the outermost: at init, routing Q/K projections are identity so
    # inner stages' boundary probabilities have zero gradient (see the note in
    # ``gittables_cta_steps::step_then_outer_routing_grads``). The outermost
    # stage sees raw embedding variation from step 0 and trains normally.
    # After ~100 steps, stage0's mean_F reliably drifts from 0.5 by 0.02+;
    # inner stages need more steps and are validated by the convergence
    # @slow scenario in the annotation feature.
    b = context.train_summary["boundary"]
    stage0_F = b.get("stage0_mean_F")
    assert stage0_F is not None, f"stage0_mean_F missing from diagnostics {b}"
    drift = abs(stage0_F - 0.5)
    assert drift >= 0.02, (
        f"stage0 mean_F={stage0_F:.4f} is within 0.02 of untrained baseline 0.5 "
        f"(drift={drift:.4f}); load balancing isn't pulling. diag={b}"
    )
