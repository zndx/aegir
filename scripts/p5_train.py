#!/usr/bin/env python
"""P5 launcher — full GRPO/RLVR training of SAE-Res-Qwen3.5-27B-W80K-L0_100
against the locked SDG verifier.

Two paths:

- **Fresh start** (default). Builds a new ``RunMetadata``,
  prints the run id, then either dry-runs (``--dry-run``) or
  invokes ``trl.GRPOTrainer.train()``.
- **Resume** (``--resume``). Discovers the latest checkpoint
  under ``--output-dir``, runs ``verify_resume_metadata(strict=True)``
  to confirm catalog version + locked verifier weights + null
  stats are unchanged, then continues training from that checkpoint.

The strict-resume default refuses silent reward-landscape changes;
override only with explicit intent (see
``aegir.rl.checkpointing.verify_resume_metadata`` docstring).

Usage::

    # pre-flight check, no GPU work, no trainer.train()
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_train.py --dry-run

    # actual training — 200 GPU-hours on 6× RTX 4090
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_train.py

    # resume after pause / crash / power event
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_train.py --resume

This script is intentionally thin — it stitches together the
pre-built `aegir.rl.*` machinery and TRL's `GRPOTrainer`. All
real logic lives in the imported modules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("p5-train")

# Default policy preset. ``just p5-train`` runs this on the Tinybox
# out of the box. Override only when you specifically want a
# different scale or L0 sparsity; see ``POLICY_PRESETS`` in
# ``aegir.rl.policy``.
DEFAULT_POLICY_PRESET = "9b-fsdp-l0-50"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint under --output-dir.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build configs + verify resume metadata + print "
                        "the ready-to-launch summary, but skip the actual "
                        "trainer.train() call.")
    p.add_argument("--init-only", action="store_true",
                   help="Load policy + dataset + instantiate GRPOTrainer, "
                        "but skip the actual trainer.train() call. Useful "
                        "for verifying the full multi-process wiring "
                        "without committing GPU-hours.")
    p.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    p.add_argument("--t-i-cache", default="src/aegir/ontology/T_I.pkl")
    p.add_argument("--null-stats", default="src/aegir/ontology/null_stats.json")
    p.add_argument("--output-dir", default="/raid/checkpoints/p5",
                   help="Default routes to /raid because the system drive is "
                        "space-constrained. Override only when /raid is unavailable.")
    p.add_argument("--num-generations", type=int, default=8,
                   help="GRPO group size (number of completions per prompt). "
                        "Defaults to 8 per the v0.5 concept brief.")
    p.add_argument("--per-device-batch-size", type=int, default=2,
                   help="``per_device_train_batch_size`` for TRL GRPOConfig. "
                        "Each rank generates ``batch_size × num_generations`` "
                        "sequences per step. The 24GB 4090 envelope on the "
                        "9b-fsdp preset (9GB sharded model + ~2GB LoRA/SAE/"
                        "optimizer + ~10GB KV cache + activations) means "
                        "batch_size=2 × num_generations=8 = 16 sequences is "
                        "the practical ceiling; TRL's default of 8 (= 64 "
                        "sequences per rank) OOMs at backward. Lower to 1 "
                        "if running parallel runs on the same box, or "
                        "raising num_generations.")
    p.add_argument("--n-prompts", type=int, default=2000,
                   help="Number of prompt rows in the training dataset. The "
                        "policy generates ``num_generations`` completions per "
                        "row, so total completions ≈ n_prompts × num_generations.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Hard cap on training steps. -1 (default) defers to "
                        "TRL's epoch-based budget computed from the prompt "
                        "dataset size × num_generations. Set to a small "
                        "positive int (e.g. 15) for short validation runs "
                        "before committing to the full 6000-step regime.")
    p.add_argument("--strict-resume", action="store_true", default=True,
                   help="Default policy: refuse to resume if catalog version, "
                        "locked verifier weights, or null stats drifted.")
    p.add_argument("--no-strict-resume", dest="strict_resume", action="store_false",
                   help="OVERRIDE: accept a drifted resume. Only with explicit "
                        "intent (e.g., deliberate catalog extension); the "
                        "policy's effective objective will have changed.")
    p.add_argument("--p5-brief-commit", default=None,
                   help="Optional git SHA of the v0.5 concept brief used to "
                        "drive this run. Recorded in RunMetadata for "
                        "post-hoc traceability.")
    p.add_argument("--init-checkpoint", default=None,
                   help="Warm-start LoRA adapter from a prior SFT run "
                        "(typically scripts/p5_sft.py's output directory). "
                        "The base model is still loaded fresh; only the "
                        "LoRA delta is replaced. Use this after rejection-"
                        "sampling SFT to give GRPO a non-zero-reward "
                        "starting policy. Mutually exclusive with --resume.")
    p.add_argument("--decode-backend", choices=("xgrammar", "lmfe"),
                   default="xgrammar",
                   help="Constrained-decode backend for GRPO generation. "
                        "``xgrammar`` (default) pre-compiles a token-trie "
                        "automaton and runs O(1) per token — 4-15× faster "
                        "than lmfe at our 248K-vocab × 540-branch scale. "
                        "``lmfe`` is the legacy backend "
                        "(prefix_allowed_tokens_fn, O(V) per token).")
    p.add_argument("--verify-workers", type=int, default=0,
                   help="ProcessPoolExecutor workers for parallel verify() "
                        "calls inside the reward function. Default 0 = "
                        "serial. Measurement on 2026-05-16 showed warm "
                        "verify is only ~16ms per completion — not the "
                        "actual GRPO step-time bottleneck. The pool helps "
                        "modestly (~2.7× on 8 workers) when compositions "
                        "are longer or verify cost is dominant in other "
                        "loops (e.g. eval, rejection sampling). The 18 "
                        "min/step GRPO floor on this box is elsewhere "
                        "(FSDP summon_full_params + peft + xgrammar "
                        "bitmask overhead per token).")
    p.add_argument("--policy-preset", default=DEFAULT_POLICY_PRESET,
                   choices=("9b-local-l0-50", "9b-local-l0-100",
                            "9b-fsdp-l0-50", "9b-fsdp-l0-100",
                            "27b-fsdp-l0-100"),
                   help="Advanced override; ``just p5-train`` already "
                        "selects the right preset for the Tinybox "
                        "(9b-fsdp-l0-50 — 2 GPUs, comfy headroom). "
                        "Override only when you specifically want a "
                        "different scale or L0 sparsity, e.g. for the "
                        "27B-FSDP LambdaLabs target.")
    p.add_argument("--sae-attach", choices=("auto", "off"), default="auto",
                   help="Attach SAE residual-stream observers. ``auto`` "
                        "downloads + attaches the default 8-layer subset; "
                        "``off`` disables SAE entirely (faster init, no "
                        "interpretability dividend).")
    p.add_argument("--sae-num-layers", type=int, default=2,
                   help="How many evenly-spaced layers to attach SAEs to. "
                        "Default 2 sized for the 9B-local case on a 24 GB "
                        "4090: 9B bf16 (18 GB) + 2 SAE layers (~2 GB at "
                        "hidden=4096, W=64K, bf16) + activations + KV "
                        "cache fits with margin. Push to 4-8 only on the "
                        "27B-FSDP path where each rank holds only 1/N of "
                        "the base shard, leaving more headroom for SAE "
                        "weights on rank 0.")
    p.add_argument("--sae-device", default="cuda", choices=("cuda", "cpu"),
                   help="Where to hold SAE weights. Default cuda; cpu "
                        "saves GPU memory but adds host-device transfer.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    from aegir.ontology.schema import load_catalog
    from aegir.ontology.verifier import CompositionEntry, verify
    from aegir.rl.checkpointing import (
        CheckpointConfig,
        build_run_metadata,
        latest_checkpoint,
        make_sidecar_callback,
        verify_resume_metadata,
    )
    from aegir.rl.decoding import (
        DecodingConfig,
        composition_json_schema,
        make_lmfe_prefix_allowed_tokens_fn,
        make_xgrammar_logits_processor_factory,
        parse_compositions,
    )
    from aegir.rl.policy import (
        attach_sae_adapters,
        default_layers_to_hook,
        download_sae_adapters,
        load_lora_adapter_into_policy,
        load_policy,
        policy_preset,
    )
    from aegir.rl.prompt import PromptConfig, build_messages
    from aegir.rl.sae_logging import SAELogConfig, SAELogger

    print("=" * 70)
    print("P5 launcher — GRPO/RLVR training")
    print("=" * 70)

    # Disk-routing pre-flight. The system drive is constrained;
    # all large artifacts must land on /raid. Surface the plan so
    # the user sees it in every invocation.
    import os
    import shutil
    hf_home = os.environ.get("HF_HOME", "(unset — relying on ~/.cache symlink)")
    print(f"\n[disk] HF_HOME             = {hf_home}")
    print(f"[disk] checkpoint output_dir = {args.output_dir}")
    for path in ("/raid", "/"):
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            print(f"[disk] {path:6s} free            = {free_gb:6.1f} GB / {total_gb:6.1f} GB")
        except Exception:
            pass

    # ---- 1. Configs --------------------------------------------------
    policy_cfg = policy_preset(args.policy_preset)
    print(f"[policy] preset            = {args.policy_preset}")
    print(f"[policy] base_model_id     = {policy_cfg.base_model_id}")
    print(f"[policy] sae_adapter       = {policy_cfg.sae_adapter_repo_id}")
    print(f"[policy] parallelism       = {policy_cfg.parallelism_strategy} "
          f"(n_gpus={policy_cfg.n_gpus})")
    decoding_cfg = DecodingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n_compositions_per_prompt=args.num_generations,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )
    prompt_cfg = PromptConfig()

    # ---- 2. Catalog + resume vs fresh start --------------------------
    catalog = load_catalog(args.catalog)
    print(f"\n[1/9] catalog: {len(catalog.templates)} templates "
          f"(version={catalog.version})")

    if args.resume:
        ck = latest_checkpoint(args.output_dir)
        if ck is None:
            raise RuntimeError(
                f"--resume requested but no checkpoint found under "
                f"{args.output_dir!r}. To start fresh, drop --resume."
            )
        metadata = verify_resume_metadata(
            ck,
            catalog_path=args.catalog,
            t_i_cache_path=args.t_i_cache,
            null_stats_path=args.null_stats,
            strict=args.strict_resume,
        )
        checkpoint_cfg.resume_from_checkpoint = str(ck)
        print(f"[2/9] resuming from {ck}  (run_id={metadata.run_id[:8]}…)")
        print(f"      last_checkpoint_step={metadata.last_checkpoint_step}")
    else:
        metadata = build_run_metadata(
            catalog_path=args.catalog,
            t_i_cache_path=args.t_i_cache,
            null_stats_path=args.null_stats,
            p5_brief_commit=args.p5_brief_commit,
        )
        print(f"[2/9] fresh start: run_id={metadata.run_id[:8]}…")
        print(f"      catalog_version={metadata.catalog_version}  "
              f"locked_weights_hash={metadata.locked_weights_hash}")

    # ---- 3. Constrained-decode schema --------------------------------
    schema = composition_json_schema(catalog)
    import json as _json
    schema_size = len(_json.dumps(schema))
    print(f"[3/9] decoding schema: {len(schema['items']['oneOf'])} branches, "
          f"{schema_size:,} chars  (backend={decoding_cfg.backend})")

    # ---- 4. Reward function (closure over catalog + verifier paths) -
    # TRL invokes reward_func via keyword args: ``reward_func(prompts=...,
    # completions=...)``. Parameter names must match exactly — the
    # underscore-prefix convention for "unused parameter" doesn't apply
    # to keyword-callable APIs (it just renames the parameter).
    #
    # Verify() can raise on degenerate completions (sklearn NaN
    # propagation through KMeans is the most common path). A failed
    # verify() is conceptually equivalent to a parse failure: the
    # completion is too pathological to score, so it gets R=0 and the
    # run continues. The parallel-verify worker handles this internally
    # too; both paths converge on the same fail-soft semantics.
    #
    # Verifier parallelism: when ``--verify-workers > 0``, the per-
    # completion verify() calls are dispatched to a ProcessPoolExecutor
    # rather than run serially. The verifier's R_D path
    # (sentence-transformer encode + KMeans cluster, all CPU) is the
    # dominant cost at 8-16 completions/step on this box — measured
    # at ~18 min/step on 2026-05-16. With workers=4 we expect ~4× the
    # throughput, putting step time near ~5 min.
    use_pool = args.verify_workers > 0
    if use_pool:
        from aegir.rl.parallel_verify import batch_verify_compositions
        print(f"[4/9] reward_fn bound + ProcessPool(workers={args.verify_workers}) "
              f"(verifier hash = {metadata.locked_weights_hash})")
    else:
        print(f"[4/9] reward_fn bound (serial; verifier hash = "
              f"{metadata.locked_weights_hash})")

    def reward_fn(prompts, completions, **kwargs):
        del prompts, kwargs  # unused; closure provides catalog + verifier paths
        parsed_list = [parse_compositions(text) for text in completions]

        if use_pool:
            return batch_verify_compositions(
                parsed_list,
                catalog_path=args.catalog,
                t_i_cache_path=args.t_i_cache,
                null_stats_path=args.null_stats,
                n_workers=args.verify_workers,
            )

        # Serial fallback (kept for the --verify-workers=0 escape hatch
        # and for direct comparison).
        rs: list[float] = []
        for entries_raw in parsed_list:
            entries = [
                CompositionEntry(
                    template_id=e["template_id"],
                    slot_fillers=e["slot_fillers"],
                )
                for e in entries_raw
            ]
            if not entries:
                rs.append(0.0)
                continue
            try:
                res = verify(
                    entries, catalog,
                    t_i_cache_path=args.t_i_cache,
                    null_stats_path=args.null_stats,
                )
                rs.append(res.R)
            except Exception as exc:
                logger.warning("verify() failed on a completion (%s); "
                               "scoring as R=0", type(exc).__name__)
                rs.append(0.0)
        return rs


    # ---- 5. Dry-run early exit ---------------------------------------
    # Pre-flight is complete. Print the launch summary and return without
    # doing any heavy lifting (no transformers / trl / model load).
    if args.dry_run:
        print("[5/9] (dry-run) prompt dataset would be:"
              f" {args.n_prompts} rows × {args.num_generations} generations"
              f" = {args.n_prompts * args.num_generations} completions")
        print(f"[6/9] (dry-run) skipping policy load "
              f"(strategy={policy_cfg.parallelism_strategy}, n_gpus={policy_cfg.n_gpus})")
        if args.init_checkpoint:
            print(f"      (dry-run) would warm-start LoRA from: "
                  f"{args.init_checkpoint}")
        _backend = args.decode_backend
        _hook = ("xgrammar LogitsProcessor factory"
                 if _backend == "xgrammar"
                 else "lm-format-enforcer prefix_allowed_tokens_fn")
        print(f"[6b/9] (dry-run) constrained decoding: would wrap "
              f"model.generate with {_hook} "
              f"(backend={_backend}, {len(schema['items']['oneOf'])} schema branches)")
        print("[7/9] (dry-run) skipping TRL GRPOConfig + SidecarCallback")
        print("[8/9] (dry-run) skipping GRPOTrainer instantiation")
        print("[9/9] dry-run complete — ready to launch")
        print()
        # Use the bare ``just p5-train`` form when the default preset
        # is selected (the common case); show the explicit
        # ``--policy-preset`` form only when the user overrode the
        # default. Keeps the "happy path" message a one-liner.
        flag = "" if args.policy_preset == DEFAULT_POLICY_PRESET else f" --policy-preset {args.policy_preset}"
        if policy_cfg.parallelism_strategy == "single":
            note = "plain uv run python (single GPU)"
        else:
            note = f"accelerate launch --use_fsdp --num_processes {policy_cfg.n_gpus}"
        print("Launch with:")
        print(f"  just p5-train{flag}   # {note}")
        return 0

    # ---- 6. Policy + tokenizer (HEAVY) -------------------------------
    print("[5/9] loading policy (this is the rate-limiting step)…")
    from datasets import Dataset
    from trl import GRPOConfig as TRL_GRPOConfig
    from trl import GRPOTrainer

    model, tokenizer = load_policy(policy_cfg)

    # Optional SFT warm-start. After rejection-sampling SFT
    # (scripts/p5_rejection_sample.py → scripts/p5_sft.py), this loads
    # the resulting LoRA delta into the freshly-loaded base, giving
    # GRPO a policy that already produces non-zero-reward compositions.
    if args.init_checkpoint:
        if args.resume:
            raise ValueError(
                "--init-checkpoint and --resume are mutually exclusive. "
                "--resume continues from a prior GRPO checkpoint; "
                "--init-checkpoint warm-starts a fresh GRPO run from SFT."
            )
        print(f"[5b/9] loading SFT init-checkpoint from {args.init_checkpoint}")
        load_lora_adapter_into_policy(model, args.init_checkpoint)

    chat_messages = build_messages(catalog, prompt_cfg)
    prompt_text = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    train_dataset = Dataset.from_list(
        [{"prompt": prompt_text} for _ in range(args.n_prompts)]
    )
    print(f"[6/9] prompt dataset built ({args.n_prompts} rows)")

    # ---- 6b. Wire constrained decoding into the generation path ------
    # TRL's GRPOTrainer calls ``model.generate(**generate_inputs,
    # generation_config=self.generation_config)`` internally; ``GenerationConfig``
    # has no field for ``prefix_allowed_tokens_fn`` or ``logits_processor``,
    # so the only reliable hook is to wrap ``model.generate`` so every call
    # carries the constraint.
    #
    # WITHOUT this wrap, the policy emits free-form text, ``parse_compositions``
    # returns ``[]``, reward is uniformly zero, and GRPO has no learning signal.
    # The 24-hour 0-reward / 0-variance run on 2026-05-11 was caused by exactly
    # this oversight — the schema was built but never connected to ``generate``.
    #
    # Backend choice: xgrammar (default) compiles a token-trie automaton and
    # runs O(1) per token; lmfe walks the full vocab per token (O(V)). On
    # Qwen3.5-9B-Base + 540-branch schema, xgrammar is 4-15× faster.
    import time as _time
    _original_generate = model.generate

    if args.decode_backend == "xgrammar":
        # xgrammar needs the *model's* LM-head vocab size, not the tokenizer's
        # (Qwen3.5-9B-Base: 248320 vs 248044). The LM head can sample tokens
        # in [tokenizer.vocab_size, config.vocab_size); a grammar built on
        # the smaller range raises AssertionError on those tokens.
        _cfg_vocab = (
            getattr(model.config, "vocab_size", None)
            or getattr(getattr(model.config, "text_config", None),
                       "vocab_size", None)
            or tokenizer.vocab_size
        )
        print("[6b/9] building xgrammar LogitsProcessor factory…")
        _t0 = _time.time()
        _xgrammar_factory = make_xgrammar_logits_processor_factory(
            tokenizer, schema, vocab_size=_cfg_vocab,
        )
        print(f"       ready in {_time.time()-_t0:.1f}s "
              f"(backend=xgrammar, vocab={_cfg_vocab})")

        def _generate_with_constraint(*g_args, **g_kwargs):
            # xgrammar processors maintain matcher state and are single-use;
            # instantiate fresh per generate() call.
            existing = g_kwargs.get("logits_processor", None) or []
            g_kwargs["logits_processor"] = list(existing) + [_xgrammar_factory()]
            return _original_generate(*g_args, **g_kwargs)
    else:  # lmfe
        print("[6b/9] building lmfe prefix_allowed_tokens_fn…")
        _t0 = _time.time()
        _prefix_fn = make_lmfe_prefix_allowed_tokens_fn(tokenizer, schema)
        print(f"       ready in {_time.time()-_t0:.1f}s "
              f"(backend=lmfe, 540-branch schema)")

        def _generate_with_constraint(*g_args, **g_kwargs):
            if "prefix_allowed_tokens_fn" not in g_kwargs:
                g_kwargs["prefix_allowed_tokens_fn"] = _prefix_fn
            return _original_generate(*g_args, **g_kwargs)

    model.generate = _generate_with_constraint
    print(f"       model.generate wrapped (backend={args.decode_backend})")

    # ---- 7. SAE feature logging --------------------------------------
    # Two attachment paths depending on parallelism strategy:
    #
    # - "single" (9B-local): attach inline now. The model is already
    #   on cuda:0 unsharded, no FSDP broadcast to wait for, no
    #   GPU-memory collision. Inline attach gives the fastest feedback
    #   loop for the iteration we're optimising for here.
    #
    # - "fsdp" (27B-LambdaLabs): defer to a TrainerCallback that fires
    #   on ``on_train_begin``. ``accelerator.prepare()`` (called inside
    #   ``trainer.train()``) FSDP-wraps the base model and broadcasts
    #   params layer-by-layer to each rank's GPU. Keeping ~7-14 GB of
    #   SAE weights on cuda:0 before that broadcast OOMs the staging
    #   path (observed at 27B + 8 SAE layers). Attaching AFTER FSDP
    #   shard completes lets the SAEs slot into the post-shard memory
    #   envelope.
    sae_logger = SAELogger(SAELogConfig())
    sae_attach_callback = None
    if args.sae_attach == "auto":
        # Read the actual transformer-layer count from the model
        # config so the layer-spread computation works for any base
        # (40 layers on Qwen3.5-9B-Base; 64 on 27B).
        model_cfg = getattr(model, "config", None)
        cfg_layers = getattr(model_cfg, "num_hidden_layers", None)
        if cfg_layers is None:
            text_cfg = getattr(model_cfg, "text_config", None)
            cfg_layers = getattr(text_cfg, "num_hidden_layers", None)
        if cfg_layers is None:
            cfg_layers = 64  # last-resort default
        chosen_layers = default_layers_to_hook(num_layers=cfg_layers,
                                                n=args.sae_num_layers)
        print(f"      SAE attach: downloading {len(chosen_layers)} layer adapters "
              f"({sorted(chosen_layers)}) from {policy_cfg.sae_adapter_repo_id}")
        sae_paths = download_sae_adapters(policy_cfg, num_layers=cfg_layers,
                                          layers=chosen_layers)

        if policy_cfg.parallelism_strategy == "single":
            # Inline attach: model already lives on cuda:0; no FSDP
            # to wait for. ``rank0_only=True`` is benign (rank=0 in
            # single-process) and keeps the same code path active
            # in case the launcher is later run under torchrun.
            attach_sae_adapters(
                model, sae_paths, sae_logger,
                layers_to_hook=chosen_layers, device=args.sae_device,
            )
            print(f"      SAE hooks active: {len(sae_logger._hook_handles)}")
        else:
            from transformers import TrainerCallback

            sae_device_setting = args.sae_device

            class SAEAttachCallback(TrainerCallback):
                """Attach SAE residual hooks once FSDP shard is in
                place. Keyed off ``on_train_begin`` because that
                fires after ``accelerator.prepare()`` completes —
                by then, rank 0's cuda:0 is at ~9 GB (1/N FSDP
                shard), leaving headroom for the SAE weights
                without colliding with the broadcast/staging
                path that triggers FSDP wrap."""
                _attached = False

                def on_train_begin(self, args, state, control, model=None, **_kw):
                    del args, state, control
                    if self._attached:
                        return
                    if model is None:
                        logger.warning("SAEAttachCallback: no model passed; skipping attach")
                        return
                    attach_sae_adapters(
                        model, sae_paths, sae_logger,
                        layers_to_hook=chosen_layers, device=sae_device_setting,
                    )
                    self._attached = True

            sae_attach_callback = SAEAttachCallback()
            print(f"      SAE attach deferred to on_train_begin "
                  f"(post-FSDP wrap; {len(sae_paths)} adapter files cached locally)")
    else:
        print("      SAE attach: off (--sae-attach=off)")

    # ---- 8. TRL GRPOConfig + SidecarCallback -------------------------
    print("[7/9] wiring TRL GRPOConfig + SidecarCallback")
    train_args = TRL_GRPOConfig(
        output_dir=checkpoint_cfg.output_dir,
        save_strategy=checkpoint_cfg.save_strategy,
        save_steps=checkpoint_cfg.save_steps,
        save_total_limit=checkpoint_cfg.save_total_limit,
        eval_strategy=checkpoint_cfg.eval_strategy,
        eval_steps=checkpoint_cfg.eval_steps,
        logging_strategy=checkpoint_cfg.logging_strategy,
        logging_steps=checkpoint_cfg.logging_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
    )
    sidecar_cb = make_sidecar_callback(
        metadata, sae_logger=sae_logger, cfg=checkpoint_cfg,
    )

    # ---- 9. Trainer + train -----------------------------------------
    # Pass ``processing_class=tokenizer`` explicitly so TRL skips
    # the ``AutoProcessor.from_pretrained`` lookup — the base
    # model is multimodal, and AutoProcessor would otherwise try
    # to load the image processor (Qwen2VLImageProcessor) which
    # transitively requires torchvision. Our use is text-only,
    # so the tokenizer alone is the correct ``processing_class``.
    print("[8/9] instantiating GRPOTrainer (text-only processing_class)")
    from transformers import TrainerCallback as _TrainerCallback
    callbacks_list: list[_TrainerCallback] = [sidecar_cb]
    if sae_attach_callback is not None:
        callbacks_list.append(sae_attach_callback)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=train_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks_list,
    )

    if args.init_only:
        print("[9/9] (init-only) skipping trainer.train()")
        print(f"   run_id: {metadata.run_id}")
        print(f"   trainer ready: {type(trainer).__name__}")
        if sae_attach_callback is not None:
            print(f"   SAE attach: deferred (would fire on_train_begin "
                  f"once FSDP wrap completes)")
        return 0

    print("[9/9] trainer.train() — this is the 200 GPU-hour run")
    trainer.train(resume_from_checkpoint=checkpoint_cfg.resume_from_checkpoint)
    trainer.save_model()
    print()
    print("=== P5 training complete ===")
    print(f"   run_id: {metadata.run_id}")
    print(f"   final checkpoint: {latest_checkpoint(args.output_dir)}")
    print(f"   next: held-out eval via aegir.rl.eval.evaluate(...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
