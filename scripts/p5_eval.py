#!/usr/bin/env python
"""Held-out evaluation harness for a P5 policy checkpoint.

Loads ``Qwen3.5-9B-Base`` (optionally with a LoRA adapter from
``--init-checkpoint``), generates K compositions per held-out
scenario via constrained decoding (xgrammar + full 540-branch
schema), scores each with the locked verifier, and aggregates
metrics via ``aegir.rl.eval.evaluate``.

The held-out set lives at ``tests/p5_held_out_50/labels.json``
and was authored before P5 training began, so it's leakage-free.
Used to compare policies across training stages — base vs. SFT
vs. GRPO — with the same prompts and the same verifier.

Single-GPU inference. Output goes to ``--output`` (defaults to
``<checkpoint>/eval_held_out_50.json``).

Usage::

    bash -c 'source scripts/setup_jvm_env.sh && \\
        uv run --no-sync python scripts/p5_eval.py \\
            --init-checkpoint /raid/checkpoints/p5-sft-r1 \\
            --n-compositions 4'

To evaluate the base model (no adapter)::

    bash -c 'source scripts/setup_jvm_env.sh && \\
        uv run --no-sync python scripts/p5_eval.py --base-only'
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("p5-eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-checkpoint", default=None,
                   help="LoRA adapter dir (from p5_sft.py or p5_train.py "
                        "save). When omitted, evaluates the base model.")
    p.add_argument("--base-only", action="store_true",
                   help="Force base-model evaluation (no adapter), even "
                        "if --init-checkpoint is given. Mutually exclusive.")
    p.add_argument("--held-out", default="tests/p5_held_out_50/labels.json")
    p.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    p.add_argument("--t-i-cache", default="src/aegir/ontology/T_I.pkl")
    p.add_argument("--null-stats", default="src/aegir/ontology/null_stats.json")
    p.add_argument("--n-compositions", type=int, default=4,
                   help="Compositions to sample per scenario for the "
                        "per-scenario mean R.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Generation batch size on a single GPU.")
    p.add_argument("--max-new-tokens", type=int, default=640)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Lower than rejection-sample temperature; eval "
                        "rewards quality over exploration.")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Defaults to "
                        "<init-checkpoint>/eval_held_out_50.json or "
                        "/tmp/eval_held_out_50_base.json for base-only.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    from aegir.ontology.schema import load_catalog
    from aegir.ontology.verifier import CompositionEntry, verify
    from aegir.rl.decoding import (
        composition_json_schema,
        make_xgrammar_logits_processor_factory,
        parse_compositions,
    )
    from aegir.rl.prompt import SYSTEM_PROMPT

    print("=" * 70)
    print("P5 held-out evaluation")
    print("=" * 70)

    if args.base_only and args.init_checkpoint:
        raise SystemExit("--base-only and --init-checkpoint are mutually "
                         "exclusive")

    # --- 1. Load held-out scenarios ------------------------------------
    held_out_path = Path(args.held_out)
    if not held_out_path.exists():
        raise FileNotFoundError(f"held-out set not found: {held_out_path}")
    scenarios = json.loads(held_out_path.read_text())
    n_good = sum(1 for s in scenarios if s.get("label") == 1)
    n_bad = sum(1 for s in scenarios if s.get("label") == 0)
    print(f"[1/5] held-out scenarios: {len(scenarios)} "
          f"({n_good} good + {n_bad} bad)")

    catalog = load_catalog(args.catalog)
    schema = composition_json_schema(catalog)
    print(f"      catalog: {len(catalog.templates)} templates; "
          f"schema: {len(schema['items']['oneOf'])} branches")

    # --- 2. Load policy ------------------------------------------------
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)

    print(f"[2/5] loading {args.base_model_id} (bf16)…")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, torch_dtype="bfloat16",
    )
    model = model.to(args.device)
    model.eval()
    print(f"      base loaded in {time.time()-t0:.1f}s")

    if args.init_checkpoint and not args.base_only:
        adapter_dir = Path(args.init_checkpoint)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"adapter not found at {adapter_file}. "
                f"Run scripts/p5_sft.py first, which auto-emits PEFT "
                f"adapter_model.safetensors after training."
            )
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file

        cfg_json = json.loads(
            (adapter_dir / "adapter_config.json").read_text()
        )
        lora_cfg = LoraConfig(
            r=cfg_json["r"],
            lora_alpha=cfg_json["lora_alpha"],
            lora_dropout=cfg_json["lora_dropout"],
            target_modules=cfg_json["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
        model = model.to(torch.bfloat16)
        sd = load_file(str(adapter_file))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = sum(1 for k in sd if k in dict(model.named_parameters()))
        if unexpected:
            logger.warning("unexpected keys when loading adapter: %r",
                           unexpected[:3])
        print(f"      LoRA adapter loaded from {adapter_dir}: "
              f"{len(sd)} tensors")

    # --- 3. Build xgrammar constrained-decode factory -------------------
    print("[3/5] building xgrammar LogitsProcessor factory…")
    t0 = time.time()
    cfg_vocab = (
        getattr(model.config, "vocab_size", None)
        or getattr(getattr(model.config, "text_config", None),
                   "vocab_size", None)
        or tokenizer.vocab_size
    )
    xgrammar_factory = make_xgrammar_logits_processor_factory(
        tokenizer, schema, vocab_size=cfg_vocab,
    )
    print(f"      ready in {time.time()-t0:.1f}s (vocab={cfg_vocab})")

    # --- 4. Run evaluation ---------------------------------------------
    print(f"[4/5] generating {args.n_compositions} compositions per "
          f"scenario × {len(scenarios)} scenarios = "
          f"{args.n_compositions * len(scenarios)} total")
    print()

    results: list[dict] = []
    all_rs: list[float] = []
    all_r_a: list[int] = []  # 1 if R_A > 0 (parseable + in catalog)
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    t_start = time.time()

    with torch.no_grad():
        for i, scen in enumerate(scenarios, 1):
            scenario_text = scen.get("scenario_text", "")
            user_msg = (
                f"Compose an ontology that captures this scenario: "
                f"{scenario_text} Use catalog template_ids only."
            )
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_msg},
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").to(
                args.device
            )
            prompt_len = prompt_ids["input_ids"].shape[1]

            per_scen_rs: list[float] = []
            n_pass = 0

            for b_start in range(0, args.n_compositions, args.batch_size):
                b_size = min(
                    args.batch_size, args.n_compositions - b_start,
                )
                expanded = prompt_ids["input_ids"].expand(b_size, -1)
                mask = prompt_ids["attention_mask"].expand(b_size, -1)
                out = model.generate(
                    input_ids=expanded,
                    attention_mask=mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=[xgrammar_factory()],
                )
                completion_ids = out[:, prompt_len:]
                for j in range(b_size):
                    text = tokenizer.decode(
                        completion_ids[j], skip_special_tokens=True,
                    )
                    parsed = parse_compositions(text)
                    if not parsed:
                        per_scen_rs.append(0.0)
                        continue
                    entries = [
                        CompositionEntry(
                            template_id=e["template_id"],
                            slot_fillers=e["slot_fillers"],
                        )
                        for e in parsed
                    ]
                    try:
                        res = verify(
                            entries, catalog,
                            t_i_cache_path=args.t_i_cache,
                            null_stats_path=args.null_stats,
                        )
                        per_scen_rs.append(res.R)
                        if getattr(res, "R_A", 0) > 0:
                            n_pass += 1
                    except Exception as exc:
                        logger.warning("verify() failed: %s", exc)
                        per_scen_rs.append(0.0)

            mean_r = statistics.mean(per_scen_rs) if per_scen_rs else 0.0
            results.append({
                "ontology_id": scen["ontology_id"],
                "label": scen.get("label"),
                "n_compositions": len(per_scen_rs),
                "mean_r": mean_r,
                "min_r": min(per_scen_rs) if per_scen_rs else 0.0,
                "max_r": max(per_scen_rs) if per_scen_rs else 0.0,
                "r_a_pass_count": n_pass,
            })
            all_rs.extend(per_scen_rs)
            all_r_a.extend(
                1 if r > 0 else 0 for r in per_scen_rs
            )
            if scen.get("label") == 1:
                pos_scores.append(mean_r)
            elif scen.get("label") == 0:
                neg_scores.append(mean_r)

            elapsed = time.time() - t_start
            rate = i / elapsed
            print(
                f"  scenario {i:3d}/{len(scenarios)}  "
                f"id={scen['ontology_id'][:32]:32s}  "
                f"label={scen.get('label')}  "
                f"R_mean={mean_r:.3f}  rate={rate:.2f} scen/s"
            )

    # --- 5. Aggregate + write ------------------------------------------
    overall_mean = statistics.mean(all_rs) if all_rs else 0.0
    overall_median = statistics.median(all_rs) if all_rs else 0.0
    overall_std = (
        statistics.stdev(all_rs) if len(all_rs) > 1 else 0.0
    )
    pass_rate = sum(all_r_a) / max(len(all_r_a), 1)

    auc = None
    if pos_scores and neg_scores:
        # Wilcoxon / Mann-Whitney ranking AUC
        all_labels = ([1] * len(pos_scores) + [0] * len(neg_scores))
        all_means = pos_scores + neg_scores
        pairs = sorted(zip(all_means, all_labels))
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        rank_sum_pos = sum(
            rank for rank, (_, lbl) in enumerate(pairs, start=1)
            if lbl == 1
        )
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    pos_mean = statistics.mean(pos_scores) if pos_scores else 0.0
    neg_mean = statistics.mean(neg_scores) if neg_scores else 0.0

    print()
    print("=" * 70)
    print("=== held-out evaluation complete ===")
    elapsed = time.time() - t_start
    print(f"  elapsed: {elapsed/60:.1f} min")
    print(f"  total compositions scored: {len(all_rs)}")
    print(f"  overall R: mean={overall_mean:.3f} median={overall_median:.3f} "
          f"std={overall_std:.3f}")
    print(f"  R_A pass rate (in-catalog template_ids): {pass_rate:.3f}")
    print(f"  good scenarios R: mean={pos_mean:.3f}")
    print(f"  bad scenarios R:  mean={neg_mean:.3f}")
    print(f"  AUC (good vs bad mean_r): {auc:.3f}" if auc is not None
          else "  AUC: (insufficient labels)")

    # Output path
    if args.output is not None:
        out_path = Path(args.output)
    elif args.init_checkpoint and not args.base_only:
        out_path = Path(args.init_checkpoint) / "eval_held_out_50.json"
    else:
        out_path = Path("/tmp/eval_held_out_50_base.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(args.init_checkpoint) if not args.base_only else "(base)",
        "base_model_id": args.base_model_id,
        "n_scenarios": len(scenarios),
        "n_compositions_per_scenario": args.n_compositions,
        "overall_mean_r": overall_mean,
        "overall_median_r": overall_median,
        "overall_std_r": overall_std,
        "r_a_pass_rate": pass_rate,
        "good_mean_r": pos_mean,
        "bad_mean_r": neg_mean,
        "auc_good_vs_bad": auc,
        "elapsed_sec": elapsed,
        "per_scenario": results,
    }, indent=2))
    print(f"  results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
