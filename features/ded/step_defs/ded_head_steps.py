"""Step definitions for features/ded/ded_head.feature.

Scaffolding-level contract checks for the DED head + contrastive loss.
Everything runs on CPU with the tiny config so these scenarios are tier-0.
"""

from __future__ import annotations

import torch
from behave import given, then, when  # type: ignore[import]


def _tiny_config(proj_dim: int = 64):
    from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig

    return AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192],
        d_intermediate=[0, 0, 0],
        vocab_size=260,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[8, 12, 12], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1,  # unused by DED head
        task_type="cta",
    )


@given("an AegirForDED head built from a tiny config with proj_dim={p:d}")
def step_given_ded_head(context, p):
    from aegir.models.heads import AegirForDED

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    context.ded = AegirForDED(
        config=_tiny_config(proj_dim=p), proj_dim=p, device=device, dtype=dtype
    )
    context.device = device
    context.dtype = dtype
    context.proj_dim = p


@given("a synthetic batch with {b:d} samples, seq_len={l:d}, {n_valid:d} valid columns + 1 padding each")
def step_given_batch(context, b, l, n_valid):  # noqa: E741
    torch.manual_seed(0)
    context.B = b
    context.L = l
    context.N = n_valid + 1
    context.n_valid = n_valid
    context.input_ids = torch.randint(0, 256, (b, l), device=context.device)
    context.role_ids = torch.zeros(b, l, dtype=torch.long, device=context.device)
    context.mask = torch.ones(b, l, dtype=torch.bool, device=context.device)
    # Valid CLS indexes evenly spaced, padding slot = -1
    step = l // (n_valid + 1)
    cls_valid = torch.tensor(
        [[step * (i + 1) for i in range(n_valid)] + [-1]], dtype=torch.long, device=context.device
    ).repeat(b, 1)
    context.cls_indexes = cls_valid


@when("I run the DED head forward")
def step_when_ded_forward(context):
    with torch.no_grad():
        context.out = context.ded(
            input_ids=context.input_ids,
            role_ids=context.role_ids,
            cls_indexes=context.cls_indexes,
            mask=context.mask,
        )


@then("every valid-column embedding has L2 norm close to 1")
def step_then_unit_norm(context):
    emb = context.out.embeddings.float()
    mask = context.out.col_mask
    for b in range(context.B):
        for n in range(context.N):
            if mask[b, n]:
                norm = emb[b, n].norm().item()
                assert abs(norm - 1.0) < 1e-2, f"[{b},{n}] norm={norm}"


@then("every padded-column embedding has L2 norm 0")
def step_then_pad_zero(context):
    emb = context.out.embeddings.float()
    mask = context.out.col_mask
    for b in range(context.B):
        for n in range(context.N):
            if not mask[b, n]:
                norm = emb[b, n].norm().item()
                assert norm < 1e-5, f"padded[{b},{n}] norm={norm}"


@then("col_mask flags exactly {n:d} valid columns per sample")
def step_then_n_valid(context, n):
    counts = context.out.col_mask.sum(dim=-1)
    assert (counts == n).all(), f"counts={counts.tolist()}"


@given("column embeddings for {b:d} samples with {n:d} valid columns each")
def step_given_embeddings(context, b, n):
    torch.manual_seed(0)
    D = 32
    # Embeddings clustered in 2 groups so supcon has positives to pull
    base = torch.randn(2, D) * 3.0
    raw = base[torch.randint(0, 2, (b, n))]
    raw = raw + 0.2 * torch.randn(b, n, D)
    raw = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    context.embeddings = raw
    context.col_mask = torch.ones(b, n, dtype=torch.bool)
    context.B = b
    context.N = n


@given("cluster_ids forming 2 clusters with at least 2 members each")
def step_given_2cluster_ids(context):
    # Alternate cluster assignment so every cluster has many members
    context.cluster_ids = (torch.arange(context.B * context.N) % 2).view(context.B, context.N)


@when("I compute supcon_loss")
def step_when_supcon(context):
    from aegir.utils.train import supcon_loss

    context.loss = supcon_loss(
        context.embeddings, context.cluster_ids, context.col_mask, temperature=0.1
    )


@then("the loss is finite")
def step_then_loss_finite(context):
    assert torch.isfinite(context.loss), f"loss={context.loss.item()}"


@then("the loss is positive")
def step_then_loss_positive(context):
    assert context.loss.item() > 0, f"loss={context.loss.item()}"


@given("every column assigned a unique cluster_id")
def step_given_unique_cluster_ids(context):
    context.cluster_ids = torch.arange(context.B * context.N).view(context.B, context.N)


@then("the loss is exactly 0")
def step_then_loss_zero(context):
    assert context.loss.item() == 0.0, f"loss={context.loss.item()}"


@given("ground-truth cluster_ids with {k:d} clusters of size 2")
def step_given_gt_clusters(context, k):
    B = 1
    N = k * 2
    # 0 0 1 1 2 2 ...
    context.cluster_ids = torch.arange(N).div(2, rounding_mode="floor").view(B, N)
    context.col_mask = torch.ones(B, N, dtype=torch.bool)


@when("I compute bcubed_f1 with predictions matching the ground truth")
def step_when_bcubed_perfect(context):
    from aegir.utils.train import bcubed_f1

    context.prf = bcubed_f1(context.cluster_ids.clone(), context.cluster_ids, context.col_mask)


@then("the F1 score is 1.0")
def step_then_f1_one(context):
    _, _, f1 = context.prf
    assert abs(f1 - 1.0) < 1e-6, f"f1={f1}"
