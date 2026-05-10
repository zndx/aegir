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
    p.add_argument("--sae-attach", choices=("auto", "off"), default="auto",
                   help="Attach SAE residual-stream observers. ``auto`` "
                        "downloads + attaches the default 8-layer subset; "
                        "``off`` disables SAE entirely (faster init, no "
                        "interpretability dividend).")
    p.add_argument("--sae-num-layers", type=int, default=4,
                   help="How many evenly-spaced layers to attach SAEs to. "
                        "Default 4 (out of 64 base layers) keeps GPU memory "
                        "footprint at ~7 GB bf16 on rank 0 — leaving room "
                        "for FSDP shards + activations + KV cache. Push "
                        "higher only if rank 0's cuda:0 has been verified "
                        "to have headroom at the chosen batch/length.")
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
        parse_compositions,
    )
    from aegir.rl.policy import (
        PolicyConfig,
        attach_sae_adapters,
        default_layers_to_hook,
        download_sae_adapters,
        load_policy,
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
    policy_cfg = PolicyConfig()
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
    def reward_fn(_prompts, completions, **_kwargs):
        rs: list[float] = []
        for completion_text in completions:
            entries_raw = parse_compositions(completion_text)
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
            res = verify(
                entries, catalog,
                t_i_cache_path=args.t_i_cache,
                null_stats_path=args.null_stats,
            )
            rs.append(res.R)
        return rs

    print(f"[4/9] reward_fn bound (verifier hash = {metadata.locked_weights_hash})")

    # ---- 5. Dry-run early exit ---------------------------------------
    # Pre-flight is complete. Print the launch summary and return without
    # doing any heavy lifting (no transformers / trl / model load).
    if args.dry_run:
        print("[5/9] (dry-run) prompt dataset would be:"
              f" {args.n_prompts} rows × {args.num_generations} generations"
              f" = {args.n_prompts * args.num_generations} completions")
        print(f"[6/9] (dry-run) skipping policy load "
              f"(strategy={policy_cfg.parallelism_strategy}, n_gpus={policy_cfg.n_gpus})")
        print("[7/9] (dry-run) skipping TRL GRPOConfig + SidecarCallback")
        print("[8/9] (dry-run) skipping GRPOTrainer instantiation")
        print("[9/9] dry-run complete — ready to launch")
        print()
        print("Launch with:")
        print("  just p5-train          # routes through accelerate launch --use_fsdp --num_processes 6")
        return 0

    # ---- 6. Policy + tokenizer (HEAVY) -------------------------------
    print("[5/9] loading policy (this is the rate-limiting step)…")
    from datasets import Dataset
    from trl import GRPOConfig as TRL_GRPOConfig
    from trl import GRPOTrainer

    model, tokenizer = load_policy(policy_cfg)

    chat_messages = build_messages(catalog, prompt_cfg)
    prompt_text = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    train_dataset = Dataset.from_list(
        [{"prompt": prompt_text} for _ in range(args.n_prompts)]
    )
    print(f"[6/9] prompt dataset built ({args.n_prompts} rows)")

    # ---- 7. SAE feature logging --------------------------------------
    # Download SAE adapter files upfront (file I/O only, no GPU
    # cost), but defer the actual ``attach_sae_adapters`` call to a
    # ``TrainerCallback.on_train_begin`` event. Rationale: GRPOTrainer's
    # ``accelerator.prepare()`` (called inside ``train()``) FSDP-wraps
    # the base model and broadcasts params layer-by-layer to each
    # rank's GPU. If 14 GB of SAE weights sit on rank 0's cuda:0
    # before that broadcast runs, the broadcast OOMs (observed:
    # rank-0 OOM at ``_sync_module_params_and_buffers`` allocating
    # 340 MiB into a card already at 21.3 / 23.5 GiB usage). Attaching
    # AFTER FSDP wrap completes lets the SAEs slot into the post-shard
    # memory envelope rather than competing with the broadcast staging.
    sae_logger = SAELogger(SAELogConfig())
    sae_attach_callback = None
    if args.sae_attach == "auto":
        chosen_layers = default_layers_to_hook(num_layers=64, n=args.sae_num_layers)
        print(f"      SAE attach: downloading {len(chosen_layers)} layer adapters "
              f"({sorted(chosen_layers)}) from {policy_cfg.sae_adapter_repo_id}")
        sae_paths = download_sae_adapters(policy_cfg, layers=chosen_layers)
        from transformers import TrainerCallback

        sae_device_setting = args.sae_device

        class SAEAttachCallback(TrainerCallback):
            """Attach SAE residual hooks once FSDP shard is in
            place. Keyed off ``on_train_begin`` because that fires
            after ``accelerator.prepare()`` completes — by then,
            rank 0's cuda:0 is at ~9 GB (1/6 FSDP shard), leaving
            headroom for the SAE weights without colliding with the
            broadcast/staging path that triggers FSDP wrap."""
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
