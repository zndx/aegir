# Phase 1: Supervised Bootstrapping

Train the base Aegir model on Column Type Annotation (CTA) and Column Property Annotation (CPA) benchmarks with byte-level input. This phase establishes baseline performance and validates the hierarchical architecture on real tabular data.

## Objective

Produce a single Aegir checkpoint that achieves competitive F1 scores on standard CTA/CPA benchmarks, operating directly on raw byte sequences (no external tokenizer).

## Target Datasets

| Dataset | Task | Tables | Columns | Label Classes |
|---------|------|--------|---------|---------------|
| SOTAB-CTA | Column Type Annotation | ~50k | ~500k | 91 semantic types |
| GitTables | CTA (large-scale) | ~1.5M | ~15M | Schema.org types |
| WikiTables | CTA/CPA | ~1.7M | ~6M | DBpedia ontology |

## Baseline F1 Targets

These targets are based on published results from SOTAB and Retrieve-and-Verify:

| Benchmark | Metric | Target F1 |
|-----------|--------|-----------|
| SOTAB-CTA (easy) | Macro F1 | > 0.85 |
| SOTAB-CTA (hard) | Macro F1 | > 0.65 |
| SOTAB-CPA | Macro F1 | > 0.75 |

## Byte-Level Input

Aegir operates on raw byte sequences (`vocab_size=65536` to cover byte values plus special tokens). Tables are serialized into a linear byte stream with role markers distinguishing the target column from context columns.

**Dynamic chunking learns tokenization from raw bytes.** The `RoutingModule` in the hierarchical backbone predicts chunk boundaries based on cosine similarity between adjacent hidden states. This means the model discovers its own sub-word units during training, adapting segmentation to the statistics of tabular data rather than relying on a fixed tokenizer trained on natural language.

## Serialization Format

Tables are serialized using the format in `src/aegir/data/serialization.py`:

```
[CLS] col_name: val1 | val2 | val3 [SEP] ctx_col1: v1 | v2 [SEP] ctx_col2: ...
```

The target column comes first, followed by context columns selected via MMR (Maximal Marginal Relevance) to maximize diversity while staying within the byte budget.

## Training Configuration

### Single-GPU (Development)

```bash
uv run --no-sync python train.py \
    --model-size tiny \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-4
```

### Multi-GPU with DDP

```bash
uv run --no-sync torchrun --nproc_per_node=6 train.py \
    --model-size small \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4
```

Training uses:
- **DDP** (DistributedDataParallel) across GPUs
- **AMP** (Automatic Mixed Precision) with bf16
- **Cosine LR schedule** with linear warmup
- **Load balancing loss** adapted from H-Net to regularize dynamic chunking

### Model Sizes

| Size | `d_model` | Layers | Parameters | Use Case |
|------|-----------|--------|------------|----------|
| tiny | [128, 192, 192] | ~10 | ~2M | Smoke tests, CI |
| small | [256, 384, 384] | ~20 | ~15M | Development, ablations |
| base | [512, 768, 768] | ~40 | ~120M | Benchmark evaluation |

## Success Criteria

Phase 1 is complete when:

1. The base model meets or exceeds F1 targets on SOTAB-CTA/CPA
2. Dynamic chunking converges to stable boundary predictions (no degenerate all-boundary or no-boundary patterns)
3. The trained checkpoint can be frozen and used as a specialist in Phase 3
