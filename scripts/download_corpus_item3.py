#!/usr/bin/env python3
"""Corpus item 3: Apache data-systems project documentation.

Pivot from "Stack v2 Python/Scala/Java filter" (gated) to "Apache
project docs directly from GitHub" — higher signal per byte, no gate.

Grabs README + docs/ directories from the major data-systems projects:
Spark, Flink, Hive, Iceberg, Trino, Kafka, Airflow, dbt-core, Atlas,
Ranger. These are the platforms whose vocabulary we actually need the
model to recognise.

Each project is cloned shallow (git clone --depth 1) to a temp dir,
relevant files (*.md, *.rst, *.txt under doc/docs/docs.*) are
concatenated into /raid/datasets/aegir-corpus-v1/code/docs/{project}.txt
with \\x03 doc-separators. The clones are then removed.

Each project: ~5-50 MB of clean documentation. Total ~200-500 MB.

Usage:
    uv run --no-sync python scripts/download_corpus_item3.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("corpus_item3")

DEFAULT_DEST = Path("/raid/datasets/aegir-corpus-v1/code/docs")
DOC_DELIM = b"\x03"

# (github_repo, doc_dirs) — shallow clone of master/main, then walk only
# these dirs + the top-level README/*.md files.
APACHE_PROJECTS = [
    ("apache/spark",        ["docs", "examples/src/main/python", "examples/src/main/scala"]),
    ("apache/flink",        ["docs"]),
    ("apache/hive",         ["docs"]),
    ("apache/iceberg",      ["docs"]),
    ("apache/trino",        ["docs"]),
    ("apache/kafka",        ["docs"]),
    ("apache/airflow",      ["docs"]),
    ("apache/atlas",        ["docs"]),
    ("apache/ranger",       ["docs"]),
    ("dbt-labs/dbt-core",   ["core/dbt/docs"]),
    ("apache/hadoop",       ["hadoop-common-project/hadoop-common/src/site/markdown"]),
    ("apache/beam",         ["website/www"]),
]

# Include these file extensions as text documentation
DOC_EXTS = {".md", ".rst", ".txt", ".adoc"}
# Hard cap per file to keep the corpus weighted toward breadth
MAX_FILE_BYTES = 200_000  # 200 KB per single doc file


def _git_clone_shallow(repo: str, dest: Path) -> bool:
    """Clone shallow. Returns True on success."""
    url = f"https://github.com/{repo}.git"
    log.info("  cloning %s ...", url)
    t0 = time.time()
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", url, str(dest)],
            check=True, capture_output=True, timeout=600,
        )
        log.info("    done in %.1fs", time.time() - t0)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("    git clone failed: %s", e.stderr.decode()[:400])
        return False
    except subprocess.TimeoutExpired:
        log.warning("    git clone timed out after 10 min")
        return False


def _collect_docs(repo_dir: Path, doc_dirs: list[str]) -> list[tuple[Path, bytes]]:
    """Walk the doc subdirectories + top-level README, collect file contents.

    Returns list of (relative_path, content_bytes) pairs.
    """
    results: list[tuple[Path, bytes]] = []

    # Top-level README and CHANGELOG
    for top_name in ("README.md", "README.rst", "CHANGELOG.md"):
        top = repo_dir / top_name
        if top.exists() and top.is_file():
            try:
                b = top.read_bytes()[:MAX_FILE_BYTES]
                if b:
                    results.append((Path(top_name), b))
            except Exception:
                pass

    # Doc directories
    for d in doc_dirs:
        root = repo_dir / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in DOC_EXTS:
                continue
            try:
                b = p.read_bytes()[:MAX_FILE_BYTES]
                if not b:
                    continue
                rel = p.relative_to(repo_dir)
                results.append((rel, b))
            except Exception:
                pass
    return results


def _process_project(repo: str, doc_dirs: list[str], dest: Path) -> int:
    """Clone, collect, concat, clean up. Returns bytes written."""
    out_name = repo.replace("/", "_") + ".txt"
    out_path = dest / out_name
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("  SKIP (exists): %s (%.2f MB)",
                 out_path.name, out_path.stat().st_size / 2**20)
        return out_path.stat().st_size

    with tempfile.TemporaryDirectory(prefix="aegir-doc-", dir="/raid/datasets/rwkv-7-world/.cache") as tmp:
        tmp_repo = Path(tmp) / "repo"
        if not _git_clone_shallow(repo, tmp_repo):
            return 0
        docs = _collect_docs(tmp_repo, doc_dirs)
        log.info("  %s: %d doc files", repo, len(docs))
        if not docs:
            return 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(out_path, "wb") as fh:
            for rel, b in docs:
                header = f"\n# {repo}: {rel}\n\n".encode("utf-8")
                fh.write(header)
                fh.write(b)
                fh.write(DOC_DELIM)
                total += len(header) + len(b) + 1
        log.info("  %s: %.2f MB written", repo, total / 2**20)
        return total


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--projects", nargs="*", default=None,
                    help="Only fetch these (github-owner/repo pairs).")
    args = ap.parse_args()

    projects = APACHE_PROJECTS
    if args.projects:
        projects = [(p, docs) for (p, docs) in APACHE_PROJECTS if p in args.projects]

    args.dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for repo, doc_dirs in projects:
        total += _process_project(repo, doc_dirs, args.dest)

    log.info("=" * 60)
    log.info("Done. Total docs corpus: %.2f MB", total / 2**20)
    for f in sorted(args.dest.glob("*.txt")):
        log.info("  %-50s  %.2f MB", f.name, f.stat().st_size / 2**20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
