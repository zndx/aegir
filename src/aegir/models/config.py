from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class AttnConfig:
    num_heads: List[int] = field(default_factory=list)
    rotary_emb_dim: List[int] = field(default_factory=list)
    window_size: List[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class RWKVConfig:
    head_size: int = 64
    dim_ffn_mult: float = 4.0
    rosa_mode: str = "1bit"  # "1bit", "4bit", or "qkv"
    # LoRA dimensions for RWKV-7 TimeMix (auto-computed from d_model if None)
    decay_low_rank_dim: Optional[int] = None
    gate_low_rank_dim: Optional[int] = None
    a_low_rank_dim: Optional[int] = None
    v_low_rank_dim: Optional[int] = None


@dataclass
class AegirConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 65536
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    rwkv_cfg: RWKVConfig = field(default_factory=RWKVConfig)
    tie_embeddings: bool = False
    num_labels: int = 0  # 0 = language modeling mode
    task_type: str = "cta"  # "cta" (single-label) or "cpa" (multi-label)
