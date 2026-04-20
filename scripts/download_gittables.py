#!/usr/bin/env python3
"""Idempotent, resumable download of the full GitTables 1M benchmark into
``/raid/datasets/gittables/``.

Canonical source: https://zenodo.org/records/6517052 (Hulsebos et al., 2023)

~100 topic-organized zip files, ~12 GB total compressed. Each zip contains
per-table ``.parquet`` files with enriched DBpedia + Schema.org annotations.

Layout after running::

    /raid/datasets/gittables/
    ├── .manifest.json                      ← zenodo record snapshot
    ├── abstraction_tables_licensed.zip     ← cached archive
    ├── abstraction_tables_licensed/        ← unpacked (one dir per zip)
    │   ├── *.parquet
    ├── … (×100 topic directories)

Re-running is a no-op; partial downloads resume via ``curl --continue-at -``;
archives are md5-verified against the Zenodo manifest and re-fetched if corrupt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "6517052"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DEFAULT_DEST = Path("/raid/datasets/gittables")


def fetch_manifest(dest: Path) -> dict:
    """Get the Zenodo record JSON — cached for 24h to be polite to the API."""
    cache = dest / ".manifest.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        return json.loads(cache.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [fetch manifest] {ZENODO_API}")
    with urllib.request.urlopen(ZENODO_API, timeout=30) as r:
        body = r.read().decode("utf-8")
    cache.write_text(body)
    return json.loads(body)


def file_md5(path: Path, *, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def verify_zip(path: Path, expected_md5: str | None = None) -> bool:
    """True iff zip exists, every member extracts, and md5 matches (if given)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            if zf.testzip() is not None:
                return False
    except zipfile.BadZipFile:
        return False
    if expected_md5:
        return file_md5(path) == expected_md5.lower()
    return True


def fetch(url: str, dest: Path) -> None:
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


def extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f}TB"


def process_file(dest_root: Path, entry: dict, *, force: bool) -> None:
    """Download + verify + extract one Zenodo file entry.

    Zenodo entry shape::

        {"key": "foo_licensed.zip",
         "size": 12345,
         "checksum": "md5:abc123…",
         "links": {"self": "https://zenodo.org/api/records/…/files/foo/content"}}
    """
    name = entry["key"]
    size = entry["size"]
    url = entry["links"]["self"]
    md5_hex = entry["checksum"].split(":", 1)[1] if ":" in entry["checksum"] else entry["checksum"]

    zip_path = dest_root / name
    extract_stem = name.removesuffix(".zip")
    extract_marker = dest_root / extract_stem

    if extract_marker.is_dir() and not force:
        print(f"  [skip]  {name:60} already extracted")
        return

    need_fetch = True
    if zip_path.exists():
        if verify_zip(zip_path, expected_md5=md5_hex):
            need_fetch = False
        else:
            cur = zip_path.stat().st_size
            if cur < size:
                print(f"  [resume] {name:60} {human(cur)}/{human(size)}")
            else:
                print(f"  [refetch]{name:60} bad md5 — deleting and re-downloading")
                zip_path.unlink()

    if need_fetch:
        if not zip_path.exists():
            print(f"  [fetch]  {name:60} {human(size)}")
        fetch(url, zip_path)
        if not verify_zip(zip_path, expected_md5=md5_hex):
            raise RuntimeError(f"md5 mismatch after fetch: {name}")

    print(f"  [unpack] {name:60} → {extract_stem}/")
    extract(zip_path, dest_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached + verified")
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N zips (for smoke tests)")
    parser.add_argument("--only", nargs="*", default=None, help="Glob patterns to filter filenames")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"GitTables 1M → {args.dest}")
    print()

    manifest = fetch_manifest(args.dest)
    files = manifest.get("files", [])
    files.sort(key=lambda e: e["key"])

    if args.only:
        import fnmatch
        files = [e for e in files if any(fnmatch.fnmatch(e["key"], p) for p in args.only)]
    if args.limit:
        files = files[: args.limit]

    total_size = sum(e["size"] for e in files)
    print(f"  {len(files)} zips, total {human(total_size)}\n")

    for entry in files:
        process_file(args.dest, entry, force=args.force)

    # Summary
    parquets = sum(1 for _ in args.dest.rglob("*.parquet"))
    print(f"\nGitTables summary: {len(files)} archives, {parquets} parquet tables extracted")
    print("Next: `just benchmarks` to run the registered GitTables tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
