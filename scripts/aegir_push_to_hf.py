#!/usr/bin/env python
"""Push the Aegir SOTAB-CTA v0.1 checkpoint to HuggingFace.

Target: ``zndx/aegir-sotab-cta-v0.1`` (model repo).

Uploads:
  - best_model.pt           — the raw state_dict (from train.py)
  - config.json             — explicit AegirConfig dump for reproducibility
  - README.md               — model card with eval metrics filled in
  - sotab_test_metrics.json — output of scripts/aegir_eval_sotab.py
  - run_metadata.json       — the runs/<id>/metadata.json from training

Usage::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/aegir_push_to_hf.py \\
        --checkpoint /raid/checkpoints/aegir-sotab-cta-v0.1/runs/<run-id>/best_model.pt \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("aegir-push-hf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default="zndx")
    p.add_argument("--repo-name", default="aegir-sotab-cta-v0.1")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the trained best_model.pt.")
    p.add_argument("--card-dir", default="data/aegir-sotab-cta-v0.1",
                   help="Directory containing README.md (model card).")
    p.add_argument("--model-size", default="small")
    p.add_argument("--num-classes", type=int, default=82)
    p.add_argument("--metrics", default=None,
                   help="Path to sotab_test_metrics.json from "
                        "aegir_eval_sotab.py. Defaults to "
                        "<checkpoint-dir>/sotab_test_metrics.json.")
    p.add_argument("--run-metadata", default=None,
                   help="Path to the metadata.json from runs/<id>/. "
                        "Defaults to ../<run-id>/metadata.json relative "
                        "to the checkpoint.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    card_dir = REPO / args.card_dir
    readme_path = card_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"model card not found: {readme_path}")

    metrics_path = (
        Path(args.metrics) if args.metrics
        else ckpt_path.parent / "sotab_test_metrics.json"
    )
    run_metadata_path = (
        Path(args.run_metadata) if args.run_metadata
        else ckpt_path.parent / "metadata.json"
    )

    repo_id = f"{args.namespace}/{args.repo_name}"

    # Build an explicit config.json for the released checkpoint.
    from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
    if args.model_size == "small":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
            d_model=[256, 384, 384],
            d_intermediate=[0, 0, 0],
            vocab_size=65536,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[4, 6, 6],
                                rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=args.num_classes,
            task_type="cta",
        )
    else:
        raise ValueError(f"only --model-size small supported for v0.1, "
                         f"got {args.model_size}")

    def _to_dict(obj):
        if is_dataclass(obj):
            return asdict(obj)
        return obj

    config_dict = _to_dict(config)
    config_json_path = card_dir / "config.json"
    config_json_path.write_text(json.dumps(config_dict, indent=2,
                                            default=_to_dict))

    print(f"\n=== model → {repo_id} ===")
    files: list[tuple[Path, str]] = [
        (ckpt_path, "best_model.pt"),
        (config_json_path, "config.json"),
        (readme_path, "README.md"),
    ]
    if metrics_path.exists():
        files.append((metrics_path, "sotab_test_metrics.json"))
    else:
        print(f"  (note: metrics {metrics_path} not found — push without eval)")
    if run_metadata_path.exists():
        files.append((run_metadata_path, "run_metadata.json"))

    for src, remote in files:
        size_mb = src.stat().st_size / 1e6
        marker = "✓" if src.exists() else "✗"
        print(f"  {marker} {str(src):<70s} → {remote:<32s} ({size_mb:.2f} MB)")

    if args.dry_run:
        print("\n(dry-run: skipping create_repo + upload)")
        return 0

    from huggingface_hub import HfApi, whoami
    try:
        user = whoami()
        print(f"authenticated as: {user['name']}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Not authenticated to HuggingFace. Run ``hf auth login`` "
            "with a write-scope token and retry."
        ) from exc

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="model",
        private=args.private, exist_ok=True,
    )
    for src, remote in files:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=remote,
            repo_id=repo_id, repo_type="model",
            commit_message=f"upload {remote}",
        )
    print(f"\n  pushed {repo_id}")
    print(f"  https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
