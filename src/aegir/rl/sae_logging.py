"""SAE feature-activation logging during GRPO rollout.

The SAE-Res-Qwen variant has an SAE bottleneck on the residual
stream of L0≈100 features within an 80K total feature
dictionary. We hook the residual SAE module's forward and
record the top-K active features for each generation step plus
the per-step **SAE reconstruction loss** ``||x - SAE(x)||²``.

The recorded data serves three purposes:

1. **Interpretability claim from v0.5 brief**. By logging which
   SAE features fire during high-R compositions vs low-R
   compositions, we can claim post-hoc that the policy learned
   to activate ontology-aligned features.
2. **Debugging tool**. If a particular feature consistently
   fires during R_A=0 generations, we can inspect that feature
   for a hint about the failure mode.
3. **Sparsity-budget robustness check**. By correlating
   per-step SAE reconstruction loss with the composition's
   final verifier reward R, we can detect if the L0≈100
   sparsity budget is dropping concepts that matter for
   cross-context compositions. A negative correlation between
   reconstruction loss and R on cross-context compositions
   would motivate a post-hoc L0=50 ablation on a saved
   checkpoint — without affecting the primary L0=100 training
   trajectory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SAELogConfig:
    top_k: int = 16
    record_every_n_tokens: int = 4
    record_reconstruction_loss: bool = True
    # ``sae_module_attr_path`` is retained for backwards-compat with
    # the original single-attach API used by the smoke test, but the
    # primary attachment path for P5 is ``attach_sae_adapters`` in
    # ``aegir.rl.policy`` which registers per-layer hooks externally
    # and feeds them to ``SAELogger.record(...)`` directly.
    sae_module_attr_path: str = "model.residual_sae"


@dataclass
class SAELogRecord:
    step: int
    token_index: int
    top_feature_indices: list[int]
    top_feature_activations: list[float]
    # Mean-squared error between the SAE's input residual and its
    # reconstructed output, averaged over the batch + sequence dims
    # for the recorded step. ``None`` if reconstruction-loss logging
    # is disabled or if the hook only sees the SAE's output (some
    # SAE wrappers expose the reconstruction directly without the
    # untouched input).
    reconstruction_loss: float | None = None
    # Index of the transformer layer whose residual produced this
    # record. ``-1`` when unknown (e.g., legacy single-attach hook).
    layer_index: int = -1


@dataclass
class SAELogger:
    cfg: SAELogConfig
    records: list[SAELogRecord] = field(default_factory=list)
    _hook_handles: list = field(default_factory=list)
    _step: int = 0
    _token_in_step: int = 0

    def record(self, layer_index: int, top_feature_indices: list[int],
               top_feature_activations: list[float],
               reconstruction_loss: float | None) -> None:
        """Append a record. Used by ``attach_sae_adapters`` hooks
        which compute features externally rather than wrapping a
        single SAE module."""
        self.records.append(SAELogRecord(
            step=self._step,
            token_index=self._token_in_step,
            top_feature_indices=top_feature_indices,
            top_feature_activations=top_feature_activations,
            reconstruction_loss=reconstruction_loss,
            layer_index=layer_index,
        ))

    def attach(self, model) -> None:
        """Legacy single-attach: attach a forward hook to a single
        SAE module discovered via ``cfg.sae_module_attr_path``.
        Used by the scaffold smoke test. The P5 launcher uses
        ``attach_sae_adapters`` in ``aegir.rl.policy`` instead,
        which registers per-layer hooks and calls ``record(...)``
        directly."""
        import torch

        sae_module = self._find_sae_module(model)
        if sae_module is None:
            logger.warning(
                "SAE module at %r not found; legacy single-attach SAE "
                "logging is disabled. The P5 launcher uses the per-layer "
                "attach_sae_adapters helper instead.",
                self.cfg.sae_module_attr_path,
            )
            return

        def hook(_module, inputs, output):
            if self._token_in_step % self.cfg.record_every_n_tokens != 0:
                self._token_in_step += 1
                return
            features = output if torch.is_tensor(output) else output[0]
            flat = features.detach().float().abs().reshape(-1, features.shape[-1])
            mean_act = flat.mean(dim=0)
            top_vals, top_idx = mean_act.topk(self.cfg.top_k)

            recon_loss: float | None = None
            if self.cfg.record_reconstruction_loss and inputs:
                x_in = inputs[0] if torch.is_tensor(inputs[0]) else inputs[0][0]
                if torch.is_tensor(features) and features.shape == x_in.shape:
                    recon = features.detach().float()
                    diff = (x_in.detach().float() - recon)
                    recon_loss = float(diff.pow(2).mean().item())

            self.record(
                layer_index=-1,  # legacy single-attach
                top_feature_indices=top_idx.tolist(),
                top_feature_activations=top_vals.tolist(),
                reconstruction_loss=recon_loss,
            )
            self._token_in_step += 1

        self._hook_handles.append(sae_module.register_forward_hook(hook))
        logger.info("SAE forward hook attached at %r", self.cfg.sae_module_attr_path)

    def add_hook_handle(self, handle: Any) -> None:
        """Register an externally-installed hook handle so the
        logger can detach it on cleanup."""
        self._hook_handles.append(handle)

    def detach(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def begin_step(self) -> None:
        self._step += 1
        self._token_in_step = 0

    def _find_sae_module(self, model):
        target = model
        for part in self.cfg.sae_module_attr_path.split("."):
            target = getattr(target, part, None)
            if target is None:
                return None
        return target

    def summary(self) -> dict:
        return {
            "n_records": len(self.records),
            "n_hooks": len(self._hook_handles),
            "config": {
                "top_k": self.cfg.top_k,
                "record_every_n_tokens": self.cfg.record_every_n_tokens,
            },
        }

    def spill_to_disk(self, path: str | Path, clear: bool = True) -> int:
        """Append in-memory records to ``path`` as JSONL, then
        clear the in-memory list (default). Append-only so a
        crash mid-spill loses at most the unflushed tail. Returns
        the number of records written.

        Use ``clear=False`` for snapshots without truncation
        (e.g., periodic safety dumps that don't hand off
        ownership).
        """
        if not self.records:
            return 0
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.records)
        with p.open("a") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")
        logger.info("SAE spill: %d records → %s", n, p)
        if clear:
            self.records = []
        return n

    def load_from_disk(self, path: str | Path) -> int:
        """Read previously-spilled JSONL into ``self.records``.
        Used at resume time to reconstruct the feature-log tail.
        Returns the number of records loaded."""
        p = Path(path)
        if not p.exists():
            return 0
        n = 0
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.records.append(SAELogRecord(**obj))
                n += 1
        logger.info("SAE load: %d records ← %s", n, p)
        return n
