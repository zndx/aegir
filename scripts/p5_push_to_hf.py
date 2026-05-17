#!/usr/bin/env python
"""Push the three v0.1 artifacts to HuggingFace under ``zndx/`` namespace.

Three repos:

  1. ``zndx/sdg-bertopic-correspondence-v0.1`` (dataset)
     uploads:
       - data/correspondence_v0.1/correspondence.parquet
       - data/correspondence_v0.1/T_I_centroids.npy
       - data/correspondence_v0.1/T_I_metadata.json
       - data/correspondence_v0.1/verifier_meta.json
       - data/correspondence_v0.1/catalog_meta.json
       - data/correspondence_v0.1/extraction_stats.json
       - data/correspondence_v0.1/README.md

  2. ``zndx/sdg-sft-r1`` (model)
     uploads:
       - /raid/checkpoints/p5-sft-r1/adapter_model.safetensors
       - /raid/checkpoints/p5-sft-r1/adapter_config.json
       - data/correspondence_v0.1/sft-r1-model-card.md  → README.md

  3. ``zndx/sdg-sft-r2`` (model)
     same layout as r1 but with sft-r2 files.

Prerequisite: user has run ``huggingface-cli login`` with a token that
has write access to the ``zndx`` namespace.

Usage::

    # Dry-run: print what would happen, no network calls
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_push_to_hf.py --dry-run

    # Actual push (after huggingface-cli login)
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_push_to_hf.py

    # Push one specific repo
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_push_to_hf.py --only dataset
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_push_to_hf.py --only sft-r1
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_push_to_hf.py --only sft-r2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("p5-push-hf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default="zndx",
                   help="HuggingFace user/org namespace.")
    p.add_argument("--dataset-name",
                   default="sdg-bertopic-correspondence-v0.1")
    p.add_argument("--sft-r1-name", default="sdg-sft-r1")
    p.add_argument("--sft-r2-name", default="sdg-sft-r2")
    p.add_argument("--dataset-dir",
                   default="data/correspondence_v0.1")
    p.add_argument("--sft-r1-dir",
                   default="/raid/checkpoints/p5-sft-r1")
    p.add_argument("--sft-r2-dir",
                   default="/raid/checkpoints/p5-sft-r2")
    p.add_argument("--only", choices=("dataset", "sft-r1", "sft-r2"),
                   default=None,
                   help="Push only one of the three artifacts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without touching the network.")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private (default: public).")
    return p.parse_args()


def _push_dataset(api, namespace: str, name: str, src_dir: Path,
                  dry_run: bool, private: bool) -> None:
    repo_id = f"{namespace}/{name}"
    print(f"\n=== dataset → {repo_id} ===")

    files = [
        ("correspondence.parquet", "correspondence.parquet"),
        ("T_I_centroids.npy", "T_I_centroids.npy"),
        ("T_I_metadata.json", "T_I_metadata.json"),
        ("verifier_meta.json", "verifier_meta.json"),
        ("catalog_meta.json", "catalog_meta.json"),
        ("extraction_stats.json", "extraction_stats.json"),
        ("README.md", "README.md"),
    ]
    for local, remote in files:
        path = src_dir / local
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        marker = "✓" if path.exists() else "✗ MISSING"
        print(f"  {marker} {local:<40s} → {remote:<40s} ({size_mb:.2f} MB)")

    if dry_run:
        print("  (dry-run: skipping create_repo + upload)")
        return

    api.create_repo(
        repo_id=repo_id, repo_type="dataset",
        private=private, exist_ok=True,
    )
    for local, remote in files:
        api.upload_file(
            path_or_fileobj=str(src_dir / local),
            path_in_repo=remote,
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"upload {remote}",
        )
    print(f"  pushed {repo_id}")


def _push_model(api, namespace: str, name: str, adapter_dir: Path,
                model_card_path: Path, dry_run: bool, private: bool) -> None:
    repo_id = f"{namespace}/{name}"
    print(f"\n=== model → {repo_id} ===")

    files = [
        (adapter_dir / "adapter_model.safetensors", "adapter_model.safetensors"),
        (adapter_dir / "adapter_config.json", "adapter_config.json"),
        (model_card_path, "README.md"),
    ]
    for path, remote in files:
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        marker = "✓" if path.exists() else "✗ MISSING"
        print(f"  {marker} {str(path):<60s} → {remote:<32s} ({size_mb:.2f} MB)")

    if dry_run:
        print("  (dry-run: skipping create_repo + upload)")
        return

    api.create_repo(
        repo_id=repo_id, repo_type="model",
        private=private, exist_ok=True,
    )
    for path, remote in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id, repo_type="model",
            commit_message=f"upload {remote}",
        )
    print(f"  pushed {repo_id}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    dataset_dir = REPO / args.dataset_dir
    sft_r1_dir = Path(args.sft_r1_dir)
    sft_r2_dir = Path(args.sft_r2_dir)
    sft_r1_card = dataset_dir / "sft-r1-model-card.md"
    sft_r2_card = dataset_dir / "sft-r2-model-card.md"

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi, whoami
        try:
            user = whoami()
            print(f"authenticated as: {user['name']}")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "Not authenticated to HuggingFace. Run "
                "``huggingface-cli login`` (or set HF_TOKEN) and retry."
            ) from exc
        api = HfApi()

    do_dataset = args.only is None or args.only == "dataset"
    do_r1 = args.only is None or args.only == "sft-r1"
    do_r2 = args.only is None or args.only == "sft-r2"

    if do_dataset:
        _push_dataset(api, args.namespace, args.dataset_name,
                      dataset_dir, args.dry_run, args.private)
    if do_r1:
        _push_model(api, args.namespace, args.sft_r1_name,
                    sft_r1_dir, sft_r1_card,
                    args.dry_run, args.private)
    if do_r2:
        _push_model(api, args.namespace, args.sft_r2_name,
                    sft_r2_dir, sft_r2_card,
                    args.dry_run, args.private)

    print()
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN — nothing was uploaded. Re-run without --dry-run "
              "to push.")
    else:
        print("done.")
        print()
        print("urls:")
        print(f"  https://huggingface.co/datasets/{args.namespace}/{args.dataset_name}")
        print(f"  https://huggingface.co/{args.namespace}/{args.sft_r1_name}")
        print(f"  https://huggingface.co/{args.namespace}/{args.sft_r2_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
