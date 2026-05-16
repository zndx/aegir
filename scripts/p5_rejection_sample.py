#!/usr/bin/env python
"""P5 rejection-sampling SFT data generator.

Generates a high-reward composition corpus by sampling K compositions per
prompt variation under constrained decoding, scoring each with the locked
verifier R, and keeping every sample whose ``R >= reward_threshold``.

The corpus warm-starts the GRPO policy via SFT: ``Qwen3.5-9B-Base`` emits
zero-reward random compositions when asked cold (the 24-h run on
2026-05-11 made this very explicit). A 1-epoch SFT pass on self-generated
high-reward samples teaches the model the schema + the reward landscape
shape, so GRPO immediately has non-zero reward variance to gradient-on.

This script is intentionally inference-only — no LoRA, no SAE attach, no
FSDP. Single GPU, batched decoding. The 9B-bf16 envelope is ~18 GB so a
batch of 8 sequences with 384-token KV cache fits on one 24 GB 4090 with
margin.

Usage::

    bash -c 'source scripts/setup_jvm_env.sh \\
        && uv run --no-sync python scripts/p5_rejection_sample.py \\
        --n-iters 50 --k-per-iter 16 --batch-size 8'

Output: ``/raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl``
(one row per kept sample with ``prompt``, ``completion``, ``reward``).
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

logger = logging.getLogger("p5-reject")


# A small library of prompt variations so the SFT corpus isn't a single
# input mapped to N outputs. Each entry overrides PromptConfig fields.
PROMPT_VARIATIONS: list[dict] = [
    {"target_n_compositions": 3},
    {"target_n_compositions": 6},
    {"target_n_compositions": 9},
    {"target_n_compositions": 6,
     "user_template": "Compose a {target_n}-template ontology emphasizing "
                      "observation traces and event provenance. Use catalog "
                      "templates only; pick slot fillers in sdg: or cco: "
                      "namespaces."},
    {"target_n_compositions": 6,
     "user_template": "Compose a {target_n}-template ontology centered on "
                      "governance and policy attestation. Use catalog "
                      "templates only; pick slot fillers in sdg: or cco: "
                      "namespaces."},
    {"target_n_compositions": 6,
     "user_template": "Compose a {target_n}-template ontology with explicit "
                      "lineage and provenance relationships. Use catalog "
                      "templates only."},
    {"target_n_compositions": 6,
     "user_template": "Compose a {target_n}-template ontology that models "
                      "belief intervals and epistemic uncertainty. Use "
                      "catalog templates only."},
    {"target_n_compositions": 6,
     "user_template": "Compose a {target_n}-template ontology that combines "
                      "observation, governance, and lineage primitives. Use "
                      "catalog templates only."},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    p.add_argument("--t-i-cache", default="src/aegir/ontology/T_I.pkl")
    p.add_argument("--null-stats", default="src/aegir/ontology/null_stats.json")
    p.add_argument("--output",
                   default="/raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl")
    p.add_argument("--n-iters", type=int, default=50,
                   help="Number of outer iterations. Each iteration cycles "
                        "through the prompt variations and samples K "
                        "completions per variation. Total compositions "
                        "scored = n_iters × len(PROMPT_VARIATIONS) × k_per_iter.")
    p.add_argument("--k-per-iter", type=int, default=16,
                   help="Completions sampled per prompt variation per iter.")
    p.add_argument("--reward-threshold", type=float, default=0.3,
                   help="Keep only completions with R >= this threshold.")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Generation batch size. With 9B-bf16 + 384-token "
                        "KV cache + constrained-decode overhead, 8 is a "
                        "safe single-4090 default. Push higher only after "
                        "verifying you have GPU headroom.")
    p.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=4,
                   help="Print progress every N prompt-variation batches.")
    p.add_argument("--schema-mode", choices=("lite", "full", "none"),
                   default="full",
                   help="Constrained-decode schema strictness. ``full`` "
                        "(default) enumerates all 540 catalog template_ids "
                        "via discriminated oneOf — guarantees the policy "
                        "picks valid template_ids, no R_A hallucination "
                        "rejections. With xgrammar backend this is ~0.7s "
                        "to compile and O(1) per-token. ``lite`` enforces "
                        "only the array-of-objects shape (template_id "
                        "free string, R_A rejects hallucinations post-hoc). "
                        "``none`` skips constraint entirely.")
    p.add_argument("--decode-backend", choices=("xgrammar", "lmfe"),
                   default="xgrammar",
                   help="Constrained-decode backend. ``xgrammar`` (default) "
                        "pre-compiles a token-trie automaton at schema-build "
                        "time and runs O(1) per token — 5-15× faster than "
                        "lmfe at our 248K-vocab × 540-branch scale. ``lmfe`` "
                        "is the fallback (prefix_allowed_tokens_fn, O(V) "
                        "per token).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    from aegir.ontology.schema import load_catalog
    from aegir.ontology.verifier import CompositionEntry, verify
    from aegir.rl.decoding import (
        composition_json_schema,
        composition_json_schema_lite,
        make_lmfe_prefix_allowed_tokens_fn,
        make_xgrammar_logits_processor_factory,
        parse_compositions,
    )
    from aegir.rl.prompt import PromptConfig, build_messages

    print("=" * 70)
    print("P5 rejection-sampling SFT data generator")
    print("=" * 70)

    catalog = load_catalog(args.catalog)
    print(f"[1/5] catalog: {len(catalog.templates)} templates "
          f"(version={catalog.version})")
    if args.schema_mode == "full":
        schema = composition_json_schema(catalog)
        print(f"      schema (full): {len(schema['items']['oneOf'])} branches "
              f"— each token decision walks all branches (slow on 540×)")
    elif args.schema_mode == "lite":
        schema = composition_json_schema_lite()
        print(f"      schema (lite): array-of-objects shape only; "
              f"R_A rejects out-of-catalog template_ids post-hoc")
    else:  # "none"
        schema = None
        print(f"      schema (none): unconstrained generation; "
              f"reliant on prompt + model JSON-emitting ability")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)

    print(f"[2/5] loading {args.base_model_id} (bf16, no LoRA, no SAE)…")
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
    print(f"      loaded in {time.time()-t0:.1f}s")

    # Constrained-decode hook: backend choice determines which generation
    # kwarg the per-batch call uses.
    prefix_fn = None
    xgrammar_factory = None
    if schema is not None:
        if args.decode_backend == "xgrammar":
            # xgrammar needs the *model's* LM-head vocab size, not the
            # tokenizer's. For Qwen3.5-9B-Base: tokenizer.vocab_size=248044
            # but model.config.vocab_size=248320 (the LM head is padded for
            # alignment). Sampling from the LM head can produce token ids
            # >= tokenizer.vocab_size; if the grammar matcher was built
            # against 248044, those out-of-range tokens trigger
            # ``AssertionError: accept_token returned False``.
            cfg_vocab = (
                getattr(model.config, "vocab_size", None)
                or getattr(getattr(model.config, "text_config", None),
                           "vocab_size", None)
                or tokenizer.vocab_size
            )
            print("[3/5] building xgrammar LogitsProcessor factory…")
            t0 = time.time()
            xgrammar_factory = make_xgrammar_logits_processor_factory(
                tokenizer, schema, vocab_size=cfg_vocab,
            )
            print(f"      ready in {time.time()-t0:.1f}s "
                  f"(schema_mode={args.schema_mode}, backend=xgrammar, "
                  f"vocab_size={cfg_vocab})")
        else:  # lmfe
            print("[3/5] building lmfe prefix_allowed_tokens_fn…")
            t0 = time.time()
            prefix_fn = make_lmfe_prefix_allowed_tokens_fn(tokenizer, schema)
            print(f"      ready in {time.time()-t0:.1f}s "
                  f"(schema_mode={args.schema_mode}, backend=lmfe)")
    else:
        print("[3/5] schema=none: skipping constrained-decode build")

    # Pre-render every prompt variation once. Each variation gets its
    # own few-shot seed so the in-prompt examples rotate across the
    # 540-template catalog — without this, the Base model just copies
    # the same 3 template_ids over and over, and the SFT corpus has
    # near-zero template diversity.
    rendered_prompts: list[tuple[str, dict]] = []
    for i, variation in enumerate(PROMPT_VARIATIONS):
        cfg = PromptConfig(**variation) if variation else PromptConfig()
        chat = build_messages(catalog, cfg, few_shot_seed=args.seed * 1000 + i)
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        rendered_prompts.append((prompt_text, variation))
    print(f"[4/5] prompt variations: {len(rendered_prompts)} "
          f"(few-shot examples rotated per-variation)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] writing corpus to: {out_path}")
    print(f"      threshold: R >= {args.reward_threshold}")
    print(f"      budget: {args.n_iters} iters × {len(rendered_prompts)} "
          f"prompts × {args.k_per_iter} samples = "
          f"{args.n_iters * len(rendered_prompts) * args.k_per_iter} "
          f"compositions to score")
    print()

    n_kept = 0
    n_zero = 0
    n_parse_fail = 0
    n_verify_fail = 0
    rewards_seen: list[float] = []
    t_start = time.time()
    batch_idx = 0

    with out_path.open("w") as f, torch.no_grad():
        for it in range(args.n_iters):
            for p_idx, (prompt_text, variation) in enumerate(rendered_prompts):
                prompt_ids = tokenizer(prompt_text, return_tensors="pt").to(args.device)
                prompt_len = prompt_ids["input_ids"].shape[1]
                # Sample K completions in chunks of batch_size
                for b_start in range(0, args.k_per_iter, args.batch_size):
                    b_size = min(args.batch_size, args.k_per_iter - b_start)
                    expanded_ids = prompt_ids["input_ids"].expand(b_size, -1)
                    expanded_mask = prompt_ids["attention_mask"].expand(b_size, -1)
                    gen_kwargs = dict(
                        input_ids=expanded_ids,
                        attention_mask=expanded_mask,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    if prefix_fn is not None:
                        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
                    if xgrammar_factory is not None:
                        # xgrammar LogitsProcessor is single-use — fresh per call
                        gen_kwargs["logits_processor"] = [xgrammar_factory()]
                    out = model.generate(**gen_kwargs)
                    completion_ids = out[:, prompt_len:]
                    for i in range(b_size):
                        completion = tokenizer.decode(
                            completion_ids[i], skip_special_tokens=True
                        )
                        entries_raw = parse_compositions(completion)
                        if not entries_raw:
                            n_parse_fail += 1
                            continue
                        entries = [
                            CompositionEntry(
                                template_id=e["template_id"],
                                slot_fillers=e["slot_fillers"],
                            )
                            for e in entries_raw
                        ]
                        try:
                            res = verify(
                                entries, catalog,
                                t_i_cache_path=args.t_i_cache,
                                null_stats_path=args.null_stats,
                            )
                            r = res.R
                        except Exception as exc:
                            n_verify_fail += 1
                            logger.debug("verify() failed: %s", exc)
                            continue
                        rewards_seen.append(r)
                        if r == 0:
                            n_zero += 1
                        if r >= args.reward_threshold:
                            f.write(json.dumps({
                                "prompt": prompt_text,
                                "completion": completion,
                                "reward": r,
                                "variation_idx": p_idx,
                            }) + "\n")
                            f.flush()
                            n_kept += 1
                    batch_idx += 1

                if (batch_idx % args.log_every) == 0 and rewards_seen:
                    elapsed = time.time() - t_start
                    rate = batch_idx / elapsed
                    mean = statistics.mean(rewards_seen)
                    nonzero = [r for r in rewards_seen if r > 0]
                    nz_mean = statistics.mean(nonzero) if nonzero else 0.0
                    nz_frac = len(nonzero) / len(rewards_seen)
                    print(
                        f"  iter={it+1}/{args.n_iters} prompt={p_idx+1}/{len(rendered_prompts)}  "
                        f"kept={n_kept} scored={len(rewards_seen)} "
                        f"reward_mean={mean:.3f} nz_mean={nz_mean:.3f} "
                        f"nz_frac={nz_frac:.2f}  "
                        f"parse_fail={n_parse_fail} verify_fail={n_verify_fail}  "
                        f"rate={rate:.1f} batches/s"
                    )

    print()
    print("=" * 70)
    print("=== rejection sampling complete ===")
    elapsed = time.time() - t_start
    print(f"  elapsed: {elapsed/60:.1f} min")
    print(f"  kept: {n_kept} samples above R>={args.reward_threshold}")
    print(f"  parse failures: {n_parse_fail}")
    print(f"  verify failures: {n_verify_fail}")
    print(f"  zero-reward valid samples: {n_zero}")
    print(f"  total scored: {len(rewards_seen)}")
    if rewards_seen:
        nonzero = [r for r in rewards_seen if r > 0]
        print(f"  reward_mean: {statistics.mean(rewards_seen):.3f}")
        if len(rewards_seen) > 1:
            print(f"  reward_std:  {statistics.stdev(rewards_seen):.3f}")
        print(f"  reward_max:  {max(rewards_seen):.3f}")
        print(f"  non-zero fraction: {len(nonzero)/len(rewards_seen):.3f}")
        if nonzero:
            print(f"  non-zero mean: {statistics.mean(nonzero):.3f}")
    print(f"  corpus: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
