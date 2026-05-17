---
license: apache-2.0
library_name: pytorch
tags:
- column-type-annotation
- cta
- table-understanding
- sotab
- structured-data
- rwkv
- h-net
- hierarchical
- aegir
language:
- en
datasets:
- wbsg-uni-mannheim/SOTAB
metrics:
- f1
pretty_name: "Aegir v0.1 — SOTAB-CTA Schema.org"
---

# Aegir v0.1 — SOTAB-CTA Schema.org

**Status:** v0.1, peer-review preview. Architecture and training methodology
may evolve before headline release.
**Curator:** [@zndx](https://huggingface.co/zndx)
**Project:** Aegir

An early **Aegir** checkpoint — hierarchical sequence model with all-RWKV-7
backbone and H-Net-style dynamic chunking — trained on the
**SOTAB v2 Schema.org Column Type Annotation (CTA)** benchmark.

This is the first peer-review-ready Aegir model artifact. The companion
[`zndx/sdg-bertopic-correspondence-v0.1`](https://huggingface.co/datasets/zndx/sdg-bertopic-correspondence-v0.1)
dataset documents the verifier methodology used elsewhere in the project; this
checkpoint demonstrates Aegir on a standard, externally-defined table-
understanding benchmark.

## Architecture

Recursive hierarchical structure (per `arch_layout`):

```
arch_layout = ["w4", ["w4", ["w8"], "w4"], "w4"]
d_model     = [256, 384, 384]
```

Each non-innermost stage flows: encoder (Isotropic RWKV-7 + SwiGLU) →
RoutingModule → ChunkLayer → main_network (Aegir recursive) →
DeChunkLayer → decoder (Isotropic RWKV-7 + SwiGLU), with an STE-gated
residual skip around the chunk/main/dechunk block.

| Component | Detail |
|---|---|
| Architecture | Aegir hierarchical (H-Net layout) |
| Time mixer | RWKV-7 TimeMix (fla `chunk_rwkv7` kernel) |
| FFN | SwiGLU (`W*` blocks) |
| Tokenization | Byte-level (vocab 65,536) |
| Dynamic chunking | RoutingModule + ChunkLayer + DeChunkLayer |
| Total params | _filled at eval time_ |
| Trainable params | All (no LoRA) |
| Task head | `AegirForColumnAnnotation` (CTA single-label, 82 classes) |
| Max sequence length | 128 (byte-level) |
| Context columns | 8 (MMR-selected via sentence-transformer) |

## Training

| Hyperparameter | Value |
|---|---|
| Task | SOTAB v2 Schema.org CTA |
| Train samples | 116,887 (table_name, column_index) pairs over 73,927 tables |
| Validation samples | _from sotab_v2_cta_validation_set.csv_ |
| Label vocabulary | 82 distinct Schema.org types |
| Effective batch size | 96 (16/GPU × 6 GPUs DDP) |
| Learning rate | 5e-4 |
| LR schedule | cosine with linear warmup |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Load-balancing loss weight (λ_lb) | 0.1 |
| Downsample factor | 2.0 |
| Mixed precision | bf16 AMP |
| Epochs | 12 |
| Optimizer | AdamW |
| Seed | 4649 |
| Hardware | 6× RTX 4090 (Tinybox) |

Trained with `torchrun --nproc_per_node=6 train.py --task sotab ...`.

The training set was used after MMR (sentence-transformers/all-mpnet-base-v2)
selected up to 8 context columns per table — the standard Aegir
column-annotation input shape.

## Evaluation

SOTAB v2 ships five test splits, evaluated head-to-head:

| Test split | Description | n samples | micro F1 | macro F1 | accuracy |
|---|---|---:|---:|---:|---:|
| `test` | main test (full union) | _eval_ | _eval_ | _eval_ | _eval_ |
| `random` | random subset | _eval_ | _eval_ | _eval_ | _eval_ |
| `corner_cases` | challenging examples | _eval_ | _eval_ | _eval_ | _eval_ |
| `missing_values` | missing-value rows | _eval_ | _eval_ | _eval_ | _eval_ |
| `format_heterogeneity` | varied formats | _eval_ | _eval_ | _eval_ | _eval_ |

_Eval metrics will be populated by `scripts/aegir_eval_sotab.py` after training._

## Use

```python
import torch
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForColumnAnnotation
from aegir.data.tokenizer import ByteTokenizer

config = AegirConfig(
    arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
    d_model=[256, 384, 384],
    d_intermediate=[0, 0, 0],
    vocab_size=65536,
    ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
    attn_cfg=AttnConfig(num_heads=[4, 6, 6],
                        rotary_emb_dim=[16, 24, 24], window_size=[]),
    rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
    num_labels=82,
    task_type="cta",
)
model = AegirForColumnAnnotation(config)
state_dict = torch.load("best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()
```

To run the full SOTAB-CTA test-split evaluation matrix:

```bash
LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/aegir_eval_sotab.py \
    --checkpoint best_model.pt \
    --model-size small \
    --num-classes 82 \
    --data-dir /path/to/sotab/v2
```

## Limitations

- **Single benchmark**: only the Schema.org CTA task; no DBpedia CTA or
  CPA results yet.
- **Pre-alpha architecture**: Aegir is v0.2.0; subsequent releases will
  almost certainly outperform this checkpoint on the same benchmark.
- **MMR context selection**: at inference time, the same MMR pipeline
  must select the 8 context columns the model expects.
- **Byte-level vocabulary**: while flexible, requires longer
  sequences for the same semantic coverage than subword tokenizers
  used by some baselines (TURL, DoDuo).

## Related artifacts

- [`zndx/sdg-bertopic-correspondence-v0.1`](https://huggingface.co/datasets/zndx/sdg-bertopic-correspondence-v0.1)
  — companion dataset with ontology-composition correspondence; same
  project, different artifact.
- [`zndx/sdg-sft-r1`](https://huggingface.co/zndx/sdg-sft-r1) —
  Qwen3.5-9B-Base LoRA for ontology-composition generation; a related
  but separate workstream within Aegir.

## Citation

```bibtex
@misc{aegir-sotab-cta-v01,
  title  = {Aegir v0.1 — SOTAB-CTA Schema.org},
  author = {Hill, Ryan and contributors},
  year   = {2026},
  url    = {https://huggingface.co/zndx/aegir-sotab-cta-v0.1}
}
```

## Changelog

- **v0.1** (2026-05-17) — initial release. Aegir-small (256/384 hidden,
  hierarchical recursive RWKV-7 + H-Net-style dynamic chunking) trained
  on SOTAB v2 Schema.org CTA for 12 epochs.
