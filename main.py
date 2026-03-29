"""Aegir smoke test: instantiate model and verify forward pass shapes."""

import torch

from aegir.models.config import AegirConfig, SSMConfig, AttnConfig, RWKVConfig
from aegir.models.heads import AegirForCausalLM, AegirForColumnAnnotation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_causal_lm():
    """Smoke test for causal language modeling head."""
    config = AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[256, 384, 384],
        d_intermediate=[0, 0, 0],
        vocab_size=65536,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(
            num_heads=[4, 6, 6],
            rotary_emb_dim=[16, 24, 24],
            window_size=[],
        ),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
    )
    print(f"Config: {config}")
    print("Instantiating AegirForCausalLM...")
    model = AegirForCausalLM(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    B, L = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    print(f"Forward pass: input_ids {input_ids.shape}, mask {mask.shape}")
    output = model(input_ids, mask=mask)
    print(f"Output logits: {output.logits.shape}")
    print(f"Boundary predictions: {len(output.bpred_output)} stages")
    assert output.logits.shape == (B, L, config.vocab_size)
    print("CausalLM smoke test passed!")


def test_column_annotation():
    """Smoke test for column annotation head."""
    config = AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192],
        d_intermediate=[0, 0, 0],
        vocab_size=65536,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(
            num_heads=[2, 3, 3],
            rotary_emb_dim=[8, 12, 12],
            window_size=[],
        ),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=91,
        task_type="cta",
    )
    print(f"\nConfig: {config}")
    print("Instantiating AegirForColumnAnnotation...")
    model = AegirForColumnAnnotation(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    B, L = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
    role_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    role_ids[:, L // 2 :] = 1  # second half is context
    cls_indexes = torch.zeros(B, dtype=torch.long, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    print(f"Forward pass: input_ids {input_ids.shape}")
    output = model(input_ids, role_ids=role_ids, cls_indexes=cls_indexes, mask=mask)
    print(f"Output logits: {output.logits.shape}")
    print(f"Boundary predictions: {len(output.bpred_output)} stages")
    assert output.logits.shape == (B, config.num_labels)
    print("ColumnAnnotation smoke test passed!")


def main():
    print(f"=== Aegir Model Smoke Tests (device={device}) ===\n")
    test_causal_lm()
    test_column_annotation()
    print("\n=== All smoke tests passed! ===")


if __name__ == "__main__":
    main()
