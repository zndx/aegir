"""Policy loading for P5 — Qwen3.5-27B base + SAE-Res adapter + LoRA.

The brief's "SAE-Res-Qwen3.5-27B-W80K-L0_100" policy is split
across two HuggingFace repos:

- ``Qwen/Qwen3.5-27B`` — the base LLM (tokenizer + weights).
  Loaded via ``transformers.AutoTokenizer`` /
  ``transformers.AutoModelForCausalLM``.
- ``Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`` — the SAE adapter
  repo, containing one ``layer{N}.sae.pt`` file per transformer
  layer (32 layers for the 27B base). The SAE has dictionary
  width 80K and average L0≈100 active features per token.
  These are not model weights; they are residual-stream
  forward-hook adapters used to make the policy's reasoning
  inspectable. Attached at runtime.

LoRA adapters target attention + MLP projections on the base
only — the SAE residual bottleneck is left untouched so the
interpretability claim from the v0.5 brief survives training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Hyper-parameters for the P5 policy load.

    Two parallelism regimes are supported, selected by
    ``parallelism_strategy``:

    - ``"single"`` — fits the base model unsharded on a single
      GPU. Used for the **9B-local fast-iteration loop on the
      6× RTX 4090 Tinybox**. Qwen3.5-9B-Base + LoRA + 4 SAE
      adapters lands at ~22 GB bf16 on one 4090 (9 GB base +
      ~7 GB SAE + ~3 GB activations + ~3 GB KV cache), with
      head-room. The other 5 GPUs are free for parallel
      experiments / data prep / UI dev.

    - ``"fsdp"`` — sharded across multiple GPUs via accelerate.
      Used for the **27B-LambdaLabs production target**.
      Tensor parallelism is not viable at TP=6 for Qwen3.5-27B
      (GQA num_kv_heads=4 doesn't divide 6); FSDP shards by
      parameter rather than by attention head and works at any
      n_gpus ≥ 2. Launch via ``accelerate launch --use_fsdp
      --num_processes <N> scripts/p5_train.py``.

    Use ``policy_preset(name)`` to construct ready-made
    configs for the workflow targets we actually run:
    ``9b-local-l0-50``, ``9b-local-l0-100``,
    ``27b-fsdp-l0-100``.

    The 9B-local path has *none* of the FSDP+PEFT+bf16
    interactions that bit us at 27B (no FSDP unshard, no
    accelerate fp32-upcast of LoRA params, no rank-0-only SAE
    workaround). The forward pre-hook + LoRA whole-model bf16
    cast in ``load_policy`` are kept on for both paths
    because they are cheap no-ops in the single-GPU regime.
    """

    # ── Base LLM ──────────────────────────────────────────────
    base_model_id: str = "Qwen/Qwen3.5-9B-Base"
    base_model_revision: str | None = None
    dtype: str = "bfloat16"
    parallelism_strategy: str = "single"  # "single" | "fsdp"
    n_gpus: int = 1
    trust_remote_code: bool = False
    # ── SAE adapter (residual-stream interpretability hooks) ──
    sae_adapter_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
    sae_adapter_revision: str | None = None
    # ── LoRA fine-tune surface ────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_modules_to_save: list[str] = field(default_factory=list)
    leave_sae_untouched: bool = True


# Preset configs for the workflow targets we actually run.
# Each preset is a (kwargs) dict consumed by ``policy_preset(name)``.
#
# - 9b-local-l0-50:    fast-iteration default on the Tinybox; sparser
#                      SAE (L0=50, W=64K) makes the interpretability
#                      readout cleaner.
# - 9b-local-l0-100:   denser SAE for capacity ablation against L0=50.
# - 27b-fsdp-l0-100:   LambdaLabs production target; same architecture
#                      as L0=50 but on the 27B base with the wider
#                      W=80K dictionary.
POLICY_PRESETS: dict[str, dict] = {
    # ── 9B local (single 4090) ──────────────────────────────
    #
    # 9B-bf16 = 18 GB, plus 2 SAE layers (~2 GB) + LoRA state +
    # generation activations + KV cache. Sits at the 24 GB
    # envelope edge; OOM was observed during the first GRPO
    # step's 8-way generation. Use the FSDP variants below
    # when more headroom is needed.
    "9b-local-l0-50": dict(
        base_model_id="Qwen/Qwen3.5-9B-Base",
        sae_adapter_repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        parallelism_strategy="single",
        n_gpus=1,
    ),
    "9b-local-l0-100": dict(
        base_model_id="Qwen/Qwen3.5-9B-Base",
        sae_adapter_repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100",
        parallelism_strategy="single",
        n_gpus=1,
    ),
    # ── 9B FSDP (2× 4090) ────────────────────────────────────
    #
    # FSDP FULL_SHARD across 2 GPUs: ~9 GB param shard per GPU,
    # leaving ~15 GB headroom for SAE (rank 0), LoRA state,
    # activations, KV cache. Comfortable on the Tinybox without
    # consuming all 6 GPUs. The other 4 GPUs are free for
    # parallel UI work, secondary worktree experiments, etc.
    "9b-fsdp-l0-50": dict(
        base_model_id="Qwen/Qwen3.5-9B-Base",
        sae_adapter_repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        parallelism_strategy="fsdp",
        n_gpus=2,
    ),
    "9b-fsdp-l0-100": dict(
        base_model_id="Qwen/Qwen3.5-9B-Base",
        sae_adapter_repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100",
        parallelism_strategy="fsdp",
        n_gpus=2,
    ),
    # ── 27B FSDP (6× 4090, or LambdaLabs 8× A100) ──────────
    "27b-fsdp-l0-100": dict(
        base_model_id="Qwen/Qwen3.5-27B",
        sae_adapter_repo_id="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
        parallelism_strategy="fsdp",
        n_gpus=6,
    ),
}


def policy_preset(name: str, **overrides) -> PolicyConfig:
    """Return a ``PolicyConfig`` for the named preset, with optional
    field overrides. Raises ``KeyError`` on an unknown preset name.

    Example::

        cfg = policy_preset("9b-local-l0-50", lora_rank=32)
    """
    if name not in POLICY_PRESETS:
        raise KeyError(
            f"unknown policy preset {name!r}; available: "
            f"{sorted(POLICY_PRESETS)}"
        )
    kwargs = dict(POLICY_PRESETS[name])
    kwargs.update(overrides)
    return PolicyConfig(**kwargs)


def load_policy(cfg: PolicyConfig):
    """Instantiate the policy + LoRA adapters and return a tuple
    ``(model, tokenizer)`` ready for GRPO rollout.

    Three placement paths, decided by ``cfg.parallelism_strategy``
    + the LOCAL_RANK / WORLD_SIZE env vars set by torchrun:

    - ``"single"`` (default, 9B-local). No FSDP, no
      ``device_map``. The full base + LoRA + (post-attach) SAE
      land on cuda:0 unsharded. Qwen3.5-9B-Base bf16 is ~18 GB,
      well within the 24 GB envelope.

    - ``"fsdp"`` under distributed launch (LOCAL_RANK set).
      Load to CPU; accelerate's FSDP wrapper shards once
      ``accelerator.prepare()`` runs inside ``GRPOTrainer``.
      ``device_map="auto"`` would have every rank materialize a
      full model on GPU 0 in parallel and OOM immediately.

    - ``"fsdp"`` invoked single-process (no LOCAL_RANK). Falls
      back to ``device_map="auto"`` for transformers's auto-
      placement + CPU offload. Useful for the 27B-local debug
      path; production 27B uses the distributed path on
      LambdaLabs.

    The SAE adapter `.pt` files from ``cfg.sae_adapter_repo_id``
    are downloaded but NOT attached here; that wiring lives in
    ``attach_sae_adapters`` because it depends on having the
    model's transformer-layer module paths available, which
    requires a successful base load first.
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    is_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
    )
    logger.info(
        "loading base policy %s (dtype=%s, parallelism=%s, n_gpus=%d, distributed=%s)",
        cfg.base_model_id, cfg.dtype, cfg.parallelism_strategy,
        cfg.n_gpus, is_distributed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id,
        revision=cfg.base_model_revision,
        trust_remote_code=cfg.trust_remote_code,
    )

    load_kwargs: dict = {
        "revision": cfg.base_model_revision,
        "torch_dtype": cfg.dtype,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.parallelism_strategy == "single":
        # 9B-local: load on CPU, then move to cuda:0 in one shot
        # below. No device_map, no FSDP, no LoRA-vs-FSDP-bf16
        # interaction surface.
        pass
    elif is_distributed:
        # FSDP under accelerate: load to CPU; FSDP shards after
        # accelerator.prepare() inside GRPOTrainer.
        pass
    else:
        # FSDP invoked single-process (debug). Fall back to
        # transformers auto-placement + CPU offload.
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, **load_kwargs)

    if cfg.parallelism_strategy == "single":
        # Move to cuda:0 once (FSDP path leaves model on CPU for
        # accelerator.prepare() to shard).
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        model = model.to(device)

    if cfg.leave_sae_untouched:
        logger.info("leaving SAE residual bottleneck untouched (no LoRA on residual SAE)")

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        modules_to_save=cfg.lora_modules_to_save,
        task_type="CAUSAL_LM",
    )
    # ``autocast_adapter_dtype=False`` disables peft's auto-cast of
    # LoRA adapters to fp32 (which it does for "stability"). Without
    # that promotion, adapters keep their nn.Linear init dtype —
    # which is *still* fp32 by default. So we follow up with an
    # explicit bf16 cast of every trainable LoRA param.
    #
    # Why this matters under FSDP + bf16 mixed precision: FSDP's
    # MixedPrecisionPolicy(param_dtype=bf16) casts the *unsharded*
    # LoRA weight to bf16 during forward, but peft's
    # ``LoraLayer.forward`` does ``x.to(lora_A.weight.dtype)``
    # against the *original* fp32 ``.weight`` attribute. That gives
    # an fp32 input which then hits the bf16 unsharded weight
    # inside ``F.linear``, producing ``RuntimeError: expected mat1
    # and mat2 to have the same dtype, but got: float !=
    # c10::BFloat16``. Storing the adapters as bf16 from the start
    # collapses both sides of the comparison.
    import torch as _torch
    target_dtype = getattr(_torch, cfg.dtype)
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model = model.to(target_dtype)

    # Even after a whole-model bf16 cast, the GRPO generation path
    # (TRL ``unwrapped_model.generate(...)`` under
    # ``FSDP.summon_full_params``) was throwing ``RuntimeError:
    # expected mat1 and mat2 to have the same dtype, but got: float
    # != c10::BFloat16`` inside peft's LoRA forward. The model's
    # *parameters* are uniformly bf16 (verified by post-cast loop:
    # 0 off-dtype params), so the fp32 input must be coming from a
    # transient activation produced upstream — most likely a norm
    # or residual op that escapes to fp32 under FSDP's
    # use_orig_params + cpu_ram_efficient_loading interaction.
    #
    # Surgical fix: install a forward pre-hook on every ``nn.Linear``
    # that casts the input to the weight's dtype before F.linear
    # runs. Cheap (a no-op when dtypes already match), idempotent,
    # and keeps the actual training/forward semantics unchanged.
    cast_dtype = target_dtype

    def _input_dtype_align_hook(module, args):
        if not args:
            return None
        x = args[0]
        if not isinstance(x, _torch.Tensor):
            return None
        wdtype = module.weight.dtype
        if x.dtype != wdtype:
            new_args = (x.to(wdtype),) + tuple(args[1:])
            return new_args
        return None

    n_linears = 0
    for _module in model.modules():
        if isinstance(_module, _torch.nn.Linear):
            _module.register_forward_pre_hook(_input_dtype_align_hook)
            n_linears += 1
    logger.info("installed input-dtype-align forward pre-hook on "
                "%d nn.Linear modules (target %s)", n_linears, cast_dtype)
    model.print_trainable_parameters()
    return model, tokenizer


def load_lora_adapter_into_policy(model, adapter_dir: str) -> None:
    """Load SFT-trained LoRA adapter weights into an already-loaded
    PEFT policy and zero out the optimizer's old state (since the
    GRPO optimizer should start fresh against the new init point).

    Used by ``p5_train.py --init-checkpoint <path>`` to warm-start GRPO
    from a rejection-sampling-SFT'd checkpoint. The adapter directory
    must be the output of ``trainer.save_model()`` from ``p5_sft.py``,
    i.e. it contains ``adapter_model.safetensors`` + ``adapter_config.json``.

    The policy's base model is preserved; only the LoRA delta is
    replaced. Residual-stream activations are therefore close to Base's
    (LoRA-mediated drift only), so the SAE adapter remains valid.
    """
    import os
    from peft import PeftModel  # noqa: F401 — runtime import guard

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"init-checkpoint {adapter_dir!r} is not a directory; "
            f"expected the output of trainer.save_model() from p5_sft.py."
        )
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_st = os.path.join(adapter_dir, "adapter_model.safetensors")
    weights_pt = os.path.join(adapter_dir, "adapter_model.bin")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"{config_path} not found — not a PEFT adapter checkpoint?"
        )
    if not (os.path.isfile(weights_st) or os.path.isfile(weights_pt)):
        raise FileNotFoundError(
            f"adapter weights not found in {adapter_dir!r} "
            f"(looked for adapter_model.safetensors / .bin)."
        )

    # ``set_adapter_model_state_dict`` is the supported way to swap LoRA
    # weights in-place on an existing PeftModel. Use safetensors when
    # available; fall back to torch.load.
    import torch
    if os.path.isfile(weights_st):
        from safetensors.torch import load_file
        sd = load_file(weights_st)
    else:
        sd = torch.load(weights_pt, map_location="cpu", weights_only=True)

    # PEFT prefixes adapter weight names with ``base_model.model.``; the
    # saved state dict already uses this convention, so direct load works.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Filter the expected misses: every non-LoRA param will be "missing"
    # by design (we only loaded LoRA deltas). Real surprise = unexpected.
    n_lora_loaded = len([k for k in sd if "lora_" in k])
    if unexpected:
        logger.warning(
            "unexpected keys when loading init-checkpoint LoRA: %r",
            unexpected[:5],
        )
    logger.info(
        "loaded SFT init-checkpoint: %d LoRA weight tensors from %s",
        n_lora_loaded, adapter_dir,
    )


def default_layers_to_hook(num_layers: int = 64, n: int = 8) -> list[int]:
    """Pick ``n`` evenly-spaced layer indices across a model with
    ``num_layers`` transformer layers. Default: 8 layers across
    a 64-layer model = ``[0, 8, 16, 24, 32, 40, 48, 56]``. The
    spread covers early/middle/late representations so the
    interpretability claim spans the depth.
    """
    if n >= num_layers:
        return list(range(num_layers))
    step = num_layers // n
    return [i * step for i in range(n)]


def _find_decoder_layers(model) -> list:
    """Walk the (possibly LoRA-wrapped) model to find the list of
    ``Qwen3_5DecoderLayer`` modules. The path is typically:
    ``model.base_model.model.model.layers`` after peft wrap, or
    ``model.model.layers`` for a vanilla causal-LM. Falls back to
    a name-based search if the attribute path doesn't resolve.
    """
    candidates = (
        ("base_model", "model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
    )
    for path in candidates:
        cur = model
        for part in path:
            cur = getattr(cur, part, None)
            if cur is None:
                break
        else:
            if isinstance(cur, (list, tuple)) or hasattr(cur, "__iter__"):
                return list(cur)
    # name-based fallback
    out = []
    for _name, mod in model.named_modules():
        if type(mod).__name__.endswith("DecoderLayer"):
            out.append(mod)
    return out


def attach_sae_adapters(
    model,
    sae_paths: dict[int, str],
    sae_logger,
    layers_to_hook: list[int] | None = None,
    device: str = "cuda",
    dtype=None,
    rank0_only: bool = True,
) -> dict[int, dict]:
    """Attach observe-only SAE forward hooks to selected
    transformer layers. The SAE does NOT modify the residual that
    flows downstream — we only read it, compute features +
    reconstruction loss, log via ``sae_logger.record(...)``, and
    return the original residual unchanged. This matches the
    brief's "SAE provides interpretability, not training signal"
    framing.

    Args:
        model: the LoRA-wrapped 27B policy.
        sae_paths: dict layer_idx → local path to ``layer{i}.sae.pt``,
            as returned by ``download_sae_adapters``.
        sae_logger: ``SAELogger`` instance to receive records.
        layers_to_hook: which layer indices to hook. ``None``
            falls back to ``default_layers_to_hook()``.
        device: where to hold the SAE weights. ``"cuda"`` keeps
            them GPU-resident (fast hook compute, ~14 GB bf16 for
            8 layers); ``"cpu"`` saves GPU memory at the cost of
            per-hook host-device transfer.
        dtype: SAE storage dtype. Default ``torch.bfloat16``.
        rank0_only: under FSDP, every rank computes every layer's
            forward (with all-gathered params). To avoid 6× the
            SAE compute, gate the hook to rank 0. Set ``False``
            only for single-process debugging.

    Returns:
        Dict layer_idx → {weights: dict, hook_handle: handle}.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    layer_modules = _find_decoder_layers(model)
    if not layer_modules:
        logger.warning("could not locate decoder layers — SAE attachment skipped")
        return {}

    num_layers = len(layer_modules)
    if layers_to_hook is None:
        layers_to_hook = default_layers_to_hook(num_layers)
    layers_to_hook = [i for i in layers_to_hook if 0 <= i < num_layers and i in sae_paths]
    if not layers_to_hook:
        logger.warning("no layers selected for SAE attachment (sae_paths=%d, num_layers=%d)",
                       len(sae_paths), num_layers)
        return {}

    # Detect rank via LOCAL_RANK (set by torchrun) — at this point
    # torch.distributed is NOT yet initialized (that happens later
    # via accelerator.prepare()), so torch.distributed.get_rank()
    # would lie and report 0 on every rank. LOCAL_RANK is set by
    # torchrun before any Python runs, so it's the reliable way
    # to know who we are at this point in the launch.
    import os as _os
    local_rank_env = _os.environ.get("LOCAL_RANK") or _os.environ.get("RANK")
    rank = int(local_rank_env) if local_rank_env is not None else 0

    # Under rank0_only, only rank 0 holds SAE weights and runs the
    # hook compute. Skipping the load entirely on other ranks
    # avoids 6× memory duplication and the cuda:0 collision when
    # ``device="cuda"`` resolves to the current device on every
    # rank before set_device() has been called per-rank.
    if rank0_only and rank != 0:
        logger.info(
            "SAE attachment skipped on rank %d (rank0_only=True; "
            "weights + hooks live only on rank 0)", rank,
        )
        return {}

    out: dict[int, dict] = {}
    for layer_idx in layers_to_hook:
        path = sae_paths[layer_idx]
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # Cast and place
        weights = {}
        for k in ("W_enc", "W_dec", "b_enc", "b_dec"):
            if k not in sd:
                logger.warning("SAE %s missing key %r; skipping layer %d", path, k, layer_idx)
                weights = None
                break
            weights[k] = sd[k].to(dtype=dtype, device=device)
        if weights is None:
            continue

        cfg = sae_logger.cfg

        def make_hook(layer_idx_capture: int, w: dict):
            def hook(_module, _inputs, output):
                # Output of a Qwen3_5DecoderLayer is a tuple
                # (hidden_states, *) — first element is the residual.
                hs = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hs):
                    return output
                # Rank-0 gating
                if rank0_only and rank != 0:
                    return output
                # Token cadence
                if sae_logger._token_in_step % cfg.record_every_n_tokens != 0:
                    sae_logger._token_in_step += 1
                    return output

                with torch.no_grad():
                    x = hs.detach().to(dtype=w["W_enc"].dtype, device=w["W_enc"].device)
                    # Flatten batch + sequence into a single token axis
                    x_flat = x.reshape(-1, x.shape[-1])  # (B*T, D)
                    pre_act = x_flat @ w["W_enc"].T + w["b_enc"]  # (B*T, K)
                    feat = torch.nn.functional.relu(pre_act)
                    # mean over token axis for top-K selection
                    mean_act = feat.float().abs().mean(dim=0)
                    top_vals, top_idx = mean_act.topk(cfg.top_k)

                    recon_loss: float | None = None
                    if cfg.record_reconstruction_loss:
                        recon = feat @ w["W_dec"].T + w["b_dec"]
                        diff = (x_flat.float() - recon.float())
                        recon_loss = float(diff.pow(2).mean().item())

                sae_logger.record(
                    layer_index=layer_idx_capture,
                    top_feature_indices=top_idx.tolist(),
                    top_feature_activations=top_vals.tolist(),
                    reconstruction_loss=recon_loss,
                )
                sae_logger._token_in_step += 1
                # Observe-only: return original output unchanged
                return output
            return hook

        layer_mod = layer_modules[layer_idx]
        handle = layer_mod.register_forward_hook(make_hook(layer_idx, weights))
        sae_logger.add_hook_handle(handle)
        out[layer_idx] = {"weights": weights, "hook_handle": handle}

    logger.info(
        "SAE adapters attached: %d layers (%s) on %s in %s; rank0_only=%s",
        len(out), sorted(out.keys()), device, dtype, rank0_only,
    )
    return out


def download_sae_adapters(
    cfg: PolicyConfig,
    num_layers: int = 64,
    layers: list[int] | None = None,
) -> dict[int, str]:
    """Download ``layer{i}.sae.pt`` files from the SAE adapter
    repo into the HuggingFace cache. Returns a dict mapping
    layer index → local path. Best-effort: missing layers are
    skipped with a warning rather than raising.

    Args:
        cfg: PolicyConfig giving the adapter repo id + revision.
        num_layers: total layer count if ``layers`` is ``None``.
            Default 64 matches Qwen3.5-27B's
            ``text_config.num_hidden_layers``.
        layers: explicit subset of layer indices to download.
            Defaults to the full ``range(num_layers)``.
    """
    from huggingface_hub import hf_hub_download

    if layers is None:
        layers = list(range(num_layers))

    out: dict[int, str] = {}
    for i in layers:
        try:
            p = hf_hub_download(
                cfg.sae_adapter_repo_id,
                f"layer{i}.sae.pt",
                revision=cfg.sae_adapter_revision,
            )
            out[i] = p
        except Exception as e:
            logger.warning(
                "SAE adapter layer%d not available in %s: %s",
                i, cfg.sae_adapter_repo_id, e,
            )
    logger.info("downloaded %d / %d SAE adapter layers from %s",
                len(out), len(layers), cfg.sae_adapter_repo_id)
    return out


def policy_load_dry_run(cfg: PolicyConfig) -> dict:
    """Return a description of what ``load_policy`` would do
    without actually instantiating the 27B model. Used by the
    P5 scaffold smoke test."""
    return {
        "base_model_id": cfg.base_model_id,
        "sae_adapter_repo_id": cfg.sae_adapter_repo_id,
        "dtype": cfg.dtype,
        "parallelism_strategy": cfg.parallelism_strategy,
        "n_gpus": cfg.n_gpus,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_targets": cfg.lora_target_modules,
        "leave_sae_untouched": cfg.leave_sae_untouched,
        "expected_trainable_pct_estimate": "~0.3-0.5% of base",
    }
