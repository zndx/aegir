"""GRPO/RLVR scaffolding for ontology-policy training (P5).

The reward signal is the locked verifier R(O, I) at
``aegir.ontology.verifier.verify``; this package wires:

- ``policy`` — Qwen3.5-27B-W80K-L0_100 + LoRA load helper (SAE
  residual stream untouched).
- ``decoding`` — constrained-decode wrapper that emits valid
  ``(template_id, slot_fillers)`` JSON to keep R_A from clamping
  to zero on syntax errors during early training.
- ``sae_logging`` — forward-hook on the residual SAE to record
  top-k feature activations per generation step.
- ``grpo_loop`` — rollout → score → advantage → policy-gradient
  step. ``trl.GRPOTrainer``-compatible.
- ``eval`` — held-out evaluation against an extended C1 test set
  plus a fresh 50-ontology held-out set.

Heavy dependencies (``transformers``, ``peft``, ``trl``, plus the
constrained-decoding library) load lazily inside each module so
the smoke test in ``scripts/p5_scaffold_smoke.py`` runs without
the 27B weights actually present.
"""
