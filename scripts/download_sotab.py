#!/usr/bin/env python3
"""Idempotent, resumable download of SOTAB V2 benchmark into /raid/datasets/sotab/.

Canonical source: https://webdatacommons.org/structureddata/sotab/v2/

Four task bundles, each with train/val/test zips:
    Schema.org CTA ≈ 1.5 GB   DBpedia CTA ≈ 1.2 GB
    Schema.org CPA ≈ 1.1 GB   DBpedia CPA ≈ 0.9 GB

Layout after running::

    /raid/datasets/sotab/
    ├── CTA_training_schemaorg.zip          ← cached archives
    ├── CTA_training_schemaorg/             ← unpacked tree
    │   ├── *.json.gz                       ← per-table JSON files
    │   └── CTA_training_schemaorg_gt.csv   ← ground truth
    ├── ... (same pattern for every split)
    ├── CTA_CPA_label_set_schemaorg.xlsx
    └── CTA_CPA_label_set_dbpedia.xlsx

Re-running is a no-op. Partial downloads resume with ``curl -C -``. Each
zip is integrity-checked with ``unzip -t`` before it's considered done;
corrupted partials are deleted and re-fetched.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

BASE_URL = "https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/v2"
DEFAULT_DEST = Path("/raid/datasets/sotab")

# Task bundles — one (label-space, task) pair per bundle. The train/val/test
# suffix is the same for every bundle, so we derive filenames programmatically.
BUNDLES = [
    ("CTA", "schemaorg"),
    ("CPA", "schemaorg"),
    ("CTA", "dbpedia"),
    ("CPA", "dbpedia"),
]
SPLITS = ["training", "validation", "test"]
# Label spreadsheets (named with a space in the upstream URL; we quote).
# These are tiny (~200 KB) and ship alongside the zips.
LABEL_FILES = {
    "CTA_CPA_label_set_schemaorg.xlsx": "CTA_CPA label set schemaorg.xlsx",
    "CTA_CPA_label_set_dbpedia.xlsx": "CTA_CPA label set DBpedia.xlsx",
}


def fetch(url: str, dest: Path) -> None:
    """curl --continue-at - --fail --retry 3 --output dest url"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--continue-at", "-",
            "--retry", "3", "--retry-delay", "5",
            "--progress-bar",
            "--output", str(dest), url,
        ],
        check=True,
    )


def verify_zip(path: Path) -> bool:
    """Return True iff the zip is complete and every member extracts cleanly."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            # testzip() returns the name of the first bad file, or None.
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip_path under dest_dir. Idempotent: already-extracted files skipped."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f}TB"


def process_zip(dest_root: Path, task: str, label_space: str, split: str, *, force: bool = False) -> None:
    """Download + verify + extract one SOTAB zip.

    The zips extract into *shared* ``Train/`` / ``Validation/`` / ``Test/``
    directories at the package root — both CTA and CPA tables land in the
    same directory, each filename suffixed ``_CTA.json.gz`` or ``_CPA.json.gz``.
    The per-bundle CSV (``sotab_v2_{task.lower()}_{split}_set.csv`` for
    Schema.org, ``sotab_{task.lower()}_{split_short}_dbpedia.csv`` for DBpedia)
    is what distinguishes the label sources.  We use the presence of that
    CSV as the idempotence check rather than a per-bundle dir.
    """
    fname = f"{task}_{split}_{label_space}.zip"
    zip_path = dest_root / fname

    # Schema.org: sotab_v2_cta_training_set.csv
    # DBpedia:    sotab_cta_train_dbpedia.csv  (note: "train" not "training")
    split_short = {"training": "train", "validation": "validation", "test": "test"}[split]
    gt_name = (
        f"sotab_v2_{task.lower()}_{split}_set.csv"
        if label_space == "schemaorg"
        else f"sotab_{task.lower()}_{split_short}_{label_space}.csv"
    )
    gt_marker = dest_root / gt_name

    if gt_marker.exists() and not force:
        print(f"  [skip]  {fname} — already extracted ({gt_marker.name} present)")
        return

    if not verify_zip(zip_path) or force:
        if zip_path.exists():
            # partial or corrupted — curl --continue-at will resume
            print(f"  [resume] {fname} (partial/corrupt at {human(zip_path.stat().st_size)})")
        else:
            print(f"  [fetch]  {fname}")
        fetch(f"{BASE_URL}/{fname}", zip_path)

    if not verify_zip(zip_path):
        raise RuntimeError(f"verify failed for {zip_path}; delete + rerun to refetch")

    print(f"  [unpack] {fname} → {dest_root.name}/{{Train,Validation,Test}}/ + {gt_name}")
    extract(zip_path, dest_root)


def fetch_label_files(dest_root: Path, *, force: bool = False) -> None:
    for local, remote_spaced in LABEL_FILES.items():
        dest = dest_root / local
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"  [skip]  {local}")
            continue
        from urllib.parse import quote
        url = f"{BASE_URL}/{quote(remote_spaced)}"
        print(f"  [fetch]  {local}")
        fetch(url, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached files verify")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Subset of bundles to fetch, e.g. --only CTA_schemaorg CPA_schemaorg. Default: all.",
    )
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"SOTAB V2 → {args.dest}")
    print()

    for task, label_space in BUNDLES:
        bundle_id = f"{task}_{label_space}"
        if args.only and bundle_id not in args.only:
            continue
        print(f"== {bundle_id} ==")
        for split in SPLITS:
            process_zip(args.dest, task, label_space, split, force=args.force)
        print()

    print("== labels ==")
    fetch_label_files(args.dest, force=args.force)
    print()

    # Final sanity report — reflects the ACTUAL extracted layout
    # (shared Train/Validation/Test dirs, per-bundle GT CSVs at root).
    print("SOTAB summary:")
    for split_dir in ("Train", "Validation", "Test"):
        d = args.dest / split_dir
        if d.is_dir():
            n = sum(1 for _ in d.iterdir())
            print(f"  {split_dir}/  {n} table files")
        else:
            print(f"  MISSING  {split_dir}/")
    print("  GT CSVs:")
    for task, label_space in BUNDLES:
        bundle_id = f"{task}_{label_space}"
        if args.only and bundle_id not in args.only:
            continue
        for split in SPLITS:
            split_short = {"training": "train", "validation": "validation", "test": "test"}[split]
            gt_name = (
                f"sotab_v2_{task.lower()}_{split}_set.csv"
                if label_space == "schemaorg"
                else f"sotab_{task.lower()}_{split_short}_{label_space}.csv"
            )
            gt_path = args.dest / gt_name
            status = "OK" if gt_path.exists() else "MISSING"
            rows = sum(1 for _ in open(gt_path)) - 1 if gt_path.exists() else 0
            print(f"    [{status:7}] {gt_name}  ({rows} labeled columns)" if status == "OK" else f"    [{status:7}] {gt_name}")

    print("\nDone. Next: `just benchmarks` to run the registered SOTAB tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
