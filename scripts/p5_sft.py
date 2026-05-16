#!/usr/bin/env python
"""P5 SFT trainer — supervised fine-tune of the policy on the
rejection-sampled high-reward composition corpus.

Pipeline position:
  1. ``scripts/p5_rejection_sample.py``  → rejection_samples.jsonl
  2. ``scripts/p5_sft.py``  (this script)  → /raid/checkpoints/p5-sft/
  3. ``scripts/p5_train.py --init-checkpoint /raid/checkpoints/p5-sft/``
                                            → GRPO warm-started

Why SFT before GRPO: the Base model emits zero-reward random JSON when
asked cold. GRPO needs reward variance to gradient-on; with 100% zero
reward and zero variance, no learning happens (the 24-hour 2026-05-11
run was the demonstration). A 1-epoch SFT pass on self-generated
high-reward samples teaches the schema + reward landscape shape, so
GRPO immediately has non-zero variance to work with.

Architectural choices matching p5_train:
  - LoRA on the same target modules (q/k/v/o/gate/up/down).
  - Whole-model bf16 cast (no peft fp32 promotion).
  - Forward pre-hook on every nn.Linear for dtype alignment.
  - FSDP FULL_SHARD across 2 GPUs by default (matches 9b-fsdp preset).
  - LoRA adapter weights saved standalone — p5_train.py loads them with
    ``--init-checkpoint`` and continues GRPO from that LoRA state.

Usage::

    bash -c 'source scripts/setup_jvm_env.sh && \\
        uv run --no-sync accelerate launch \\
            --use_fsdp --num_processes 2 \\
            --fsdp_sharding_strategy FULL_SHARD \\
            --fsdp_cpu_ram_efficient_loading true \\
            --fsdp_sync_module_states true \\
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \\
            --fsdp_use_orig_params true \\
            --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \\
            scripts/p5_sft.py --corpus /raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl'

The matching ``just`` recipe wraps this with the FSDP knobs already set;
see ``Justfile``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("p5-sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   help="Path to rejection_samples.jsonl from "
                        "scripts/p5_rejection_sample.py.")
    p.add_argument("--output-dir", default="/raid/checkpoints/p5-sft",
                   help="Where to save the SFT'd LoRA adapter.")
    p.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--epochs", type=int, default=1,
                   help="SFT is typically 1-2 epochs; longer overfits to the "
                        "narrow self-distilled distribution and harms GRPO.")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8,
                   help="Effective batch size = per_device × accum × n_gpus.")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    print("=" * 70)
    print("P5 SFT — warm-start policy on rejection-sampled corpus")
    print("=" * 70)

    # --- 1. Load + summarize the corpus ---------------------------------
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"corpus {corpus_path} not found — run "
            f"scripts/p5_rejection_sample.py first."
        )
    rows: list[dict] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(
            f"corpus {corpus_path} is empty. Lower --reward-threshold in "
            f"p5_rejection_sample.py and re-run."
        )

    rewards = [r["reward"] for r in rows]
    print(f"[1/5] corpus: {len(rows)} rows from {corpus_path}")
    print(f"      reward min/mean/max = "
          f"{min(rewards):.3f} / {sum(rewards)/len(rewards):.3f} / "
          f"{max(rewards):.3f}")

    # --- 2. Tokenizer + model -------------------------------------------
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)

    is_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
    )
    print(f"[2/5] loading {args.base_model_id} (bf16, "
          f"distributed={is_distributed})…")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict = {"torch_dtype": "bfloat16"}
    if not is_distributed:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, **load_kwargs
    )

    # --- 3. LoRA + dtype-alignment hook (same as p5_train) --------------
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    def _input_dtype_align_hook(module, args_):
        if not args_:
            return None
        x = args_[0]
        if not isinstance(x, torch.Tensor):
            return None
        wdtype = module.weight.dtype
        if x.dtype != wdtype:
            return (x.to(wdtype),) + tuple(args_[1:])
        return None

    n_linears = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(_input_dtype_align_hook)
            n_linears += 1
    print(f"[3/5] installed input-dtype-align hook on {n_linears} nn.Linear modules")

    # --- 4. Build SFT dataset -------------------------------------------
    # Format each row as a single concatenated text with the prompt
    # prefix masked (so loss is only on the completion). TRL's SFTTrainer
    # handles this when the dataset exposes ``prompt`` + ``completion``
    # fields.
    from datasets import Dataset
    ds = Dataset.from_list([
        {"prompt": r["prompt"], "completion": r["completion"]}
        for r in rows
    ])
    print(f"[4/5] SFT dataset: {len(ds)} rows (prompt+completion mode)")

    # --- 5. TRL SFTTrainer ---------------------------------------------
    from trl import SFTConfig, SFTTrainer
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        max_length=args.max_seq_length,
        report_to=[],
        seed=args.seed,
        save_strategy="steps",
        logging_strategy="steps",
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("[5/5] trainer.train()")
    trainer.train()
    trainer.save_model()

    # ---- 6. Convert FSDP-sharded checkpoint → PEFT adapter format ----
    # Under FSDP, ``trainer.save_model()`` emits sharded ``.distcp`` files
    # under ``checkpoint-*/pytorch_model_fsdp_0/`` rather than the
    # PEFT-format ``adapter_model.safetensors`` that
    # ``p5_train.py --init-checkpoint`` expects. Convert in-place on rank 0
    # so the SFT output is immediately consumable downstream.
    #
    # Mapping the FSDP keys to PEFT convention:
    #   FSDP:   model.base_model.model.<...>.lora_A.weight
    #   PEFT:   base_model.model.<...>.lora_A.default.weight
    # (strip leading ``model.``, insert ``.default`` between ``lora_X`` and
    # ``weight`` because PEFT stores LoRA layers in an nn.ModuleDict keyed
    # by adapter name; the default adapter is ``"default"``).
    # Sync ranks and tear down the process group BEFORE conversion. The
    # ``dcp.load`` call below uses ``torch.distributed`` for coordination
    # by default; with rank 1 already exited (since this is post-train),
    # rank 0 would hang in a barrier waiting for a peer that's gone.
    # Destroying the process group makes ``dcp.load`` fall back to
    # single-process mode, which is what we want for this single-rank
    # conversion step.
    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()  # wait for all ranks to finish save_model()
            is_rank0 = dist.get_rank() == 0
            dist.destroy_process_group()
        else:
            is_rank0 = True
    else:
        is_rank0 = True

    if is_rank0:
        out_root = Path(args.output_dir)
        ckpts = sorted(
            [d for d in out_root.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if ckpts:
            ckpt = ckpts[-1]
            distcp_dir = ckpt / "pytorch_model_fsdp_0"
            if distcp_dir.is_dir():
                print(f"[6/6] converting FSDP checkpoint → PEFT adapter "
                      f"({ckpt.name})…")
                import torch.distributed.checkpoint as dcp
                from safetensors.torch import save_file

                reader = dcp.FileSystemReader(str(distcp_dir))
                meta = reader.read_metadata()
                state_dict = {}
                for fqn, prop in meta.state_dict_metadata.items():
                    if hasattr(prop, "size") and hasattr(prop, "properties"):
                        state_dict[fqn] = torch.empty(
                            prop.size, dtype=prop.properties.dtype,
                        )
                dcp.load(state_dict, storage_reader=reader)

                peft_sd: dict[str, torch.Tensor] = {}
                for k, v in state_dict.items():
                    new_k = k[len("model."):] if k.startswith("model.") else k
                    new_k = new_k.replace(".lora_A.weight",
                                          ".lora_A.default.weight")
                    new_k = new_k.replace(".lora_B.weight",
                                          ".lora_B.default.weight")
                    peft_sd[new_k] = v

                save_file(peft_sd, str(out_root / "adapter_model.safetensors"))

                adapter_config = {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    "modules_to_save": [],
                    "base_model_name_or_path": args.base_model_id,
                    "bias": "none",
                    "fan_in_fan_out": False,
                    "inference_mode": False,
                    "init_lora_weights": True,
                    "use_rslora": False,
                }
                with (out_root / "adapter_config.json").open("w") as f:
                    json.dump(adapter_config, f, indent=2)
                print(f"       wrote adapter_model.safetensors "
                      f"({len(peft_sd)} LoRA tensors) + adapter_config.json")
            else:
                print(f"[6/6] no FSDP .distcp dir in {ckpt} — "
                      f"skipping conversion (single-GPU path?)")
        else:
            print("[6/6] no checkpoint-* directory — nothing to convert")

    print()
    print("=== SFT complete ===")
    print(f"  LoRA adapter saved to: {args.output_dir}")
    print(f"  next: scripts/p5_train.py --init-checkpoint {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
