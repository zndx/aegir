#!/usr/bin/env python
"""Populate ``verbal_templates`` (the diverse surface-frame SET) across the catalog — Comp 3 deterministic.

For each template, recompose DeepOnto's CfgNode parse tree (``deeponto_harness.extract_parts``) into a set
of slot-faithful, procedural frames (``verbalization.compose_frames``) and write them to the template's
``verbal_templates``. PURELY ADDITIVE: leaves ``verbal_template`` / ``mean_verbal_length`` / ``provenance``
untouched (no churn to the DeepOnto-derived baseline; ``frames()`` falls back to ``verbal_template`` where a
template yields no frames). Writes the family SoT files (``0[1-7]_*.json``) AND ``combined.json``.

Proven superior to DeepOnto's single string (scripts/compare_verbalizers.py): 5× distinct skeletons,
top-1 frame 32%→7%, 100% slot-faithful, and recovers multi-clause conjuncts DeepOnto drops.

    # compute (JVM) + apply to the catalog:
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_verbalization_frames.py --apply
    # fast path: apply a precomputed frames map without the JVM:
    uv run --no-sync python scripts/build_verbalization_frames.py --frames-in /tmp/verbal_frames.json --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import load_catalog, save_catalog  # noqa: E402

CATALOG_DIR = REPO / "src" / "aegir" / "ontology" / "catalog"
COMBINED = CATALOG_DIR / "combined.json"
K_DEFAULT = 5


def compute_map(k: int) -> dict[str, list[str]]:
    """JVM pass: {template_id: [frames]} for every template that yields ≥1 frame."""
    from aegir.ontology.deeponto_harness import ensure_jvm, extract_parts
    from aegir.ontology.verbalization import compose_frames
    ensure_jvm()
    cat = load_catalog(COMBINED)
    out: dict[str, list[str]] = {}
    for i, t in enumerate(cat.templates):
        parts = extract_parts(t)
        frames = compose_frames(parts, seed_key=t.template_id, k=k) if parts else []
        if frames:
            out[t.template_id] = frames
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(cat.templates)} ({len(out)} with frames)", flush=True)
    return out


def catalog_files() -> list[Path]:
    fams = [Path(f) for f in sorted(glob.glob(str(CATALOG_DIR / "0*.json")))
            if "candidate" not in f and "combined" not in f]
    return fams + [COMBINED]


def apply_map(frames_map: dict[str, list[str]], *, write: bool) -> dict:
    stats = {"files": [], "templates_set": 0}
    for path in catalog_files():
        cat = load_catalog(path)
        n_set = 0
        for t in cat.templates:
            fr = frames_map.get(t.template_id)
            if fr:
                t.verbal_templates = fr
                n_set += 1
        stats["files"].append({"file": path.name, "templates": len(cat.templates), "set": n_set})
        stats["templates_set"] += n_set
        if write:
            save_catalog(cat, path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="frames per template")
    ap.add_argument("--frames-in", default=None, help="precomputed {tid:[frames]} json (skips the JVM)")
    ap.add_argument("--frames-out", default=None, help="save the computed frames map here")
    ap.add_argument("--apply", action="store_true", help="write verbal_templates into the catalog files")
    args = ap.parse_args()

    if args.frames_in:
        frames_map = json.loads(Path(args.frames_in).read_text())
        print(f"loaded {len(frames_map)} frame sets from {args.frames_in}")
    else:
        print("computing frames via DeepOnto (JVM)...")
        frames_map = compute_map(args.k)
        print(f"computed {len(frames_map)} frame sets")
    if args.frames_out:
        Path(args.frames_out).write_text(json.dumps(frames_map, indent=1))
        print(f"wrote frames map → {args.frames_out}")

    total_frames = sum(len(v) for v in frames_map.values())
    print(f"frame sets: {len(frames_map)} | total frames: {total_frames} "
          f"(avg {total_frames / max(1, len(frames_map)):.1f})")

    stats = apply_map(frames_map, write=args.apply)
    for f in stats["files"]:
        print(f"  {f['file']:40s} {f['set']:>3}/{f['templates']} templates")
    print(f"{'WROTE' if args.apply else 'DRY RUN — would set'} verbal_templates on "
          f"{stats['templates_set']} template-rows across {len(stats['files'])} files")
    if not args.apply:
        print("(pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
