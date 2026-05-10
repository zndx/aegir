"""build_catalog — populate catalog metadata via the offline DeepOnto harness.

Reads a candidate catalog JSON (with placeholder ``is_complex``,
``verbal_template``, and ``mean_verbal_length`` fields), runs each
template through the DeepOnto harness, and writes an augmented
catalog JSON with those fields populated. Templates that fail to
load or verbalize are *retained* in the output with their failure
reason recorded under ``provenance.deeponto_error`` so authors can
diagnose; downstream filtering / dropping is the catalog-author's
call, not the harness's.

Usage:

    # Bootstrap surgical LD_LIBRARY_PATH for the JVM (handled by
    # the Justfile recipe `just build-catalog`):
    source scripts/setup_jvm_env.sh
    uv run --no-sync python scripts/build_catalog.py \\
        src/aegir/ontology/catalog/01_foundation.candidate.json \\
        src/aegir/ontology/catalog/01_foundation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aegir.ontology.schema import load_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("input", type=str, help="Candidate catalog JSON")
    parser.add_argument("output", type=str, help="Output catalog JSON")
    parser.add_argument("--memory", type=str, default="4g", help="JVM memory")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-template progress; emit only summary",
    )
    args = parser.parse_args()

    catalog = load_catalog(args.input)

    from aegir.ontology.deeponto_harness import ensure_jvm, probe_template
    ensure_jvm(memory=args.memory)

    n_total = len(catalog.templates)
    n_loaded = 0
    n_complex = 0
    n_verbalized = 0
    n_failed = 0
    failures: list[dict[str, str]] = []

    t0 = time.time()
    for i, template in enumerate(catalog.templates):
        result = probe_template(template)

        template.is_complex = result.is_complex
        template.verbal_template = result.verbal_template
        template.mean_verbal_length = result.mean_verbal_length

        if result.loaded:
            n_loaded += 1
        if result.is_complex:
            n_complex += 1
        if result.verbalized:
            n_verbalized += 1
        if not result.loaded or not result.verbalized:
            n_failed += 1
            failures.append({
                "template_id": template.template_id,
                "error_kind": result.error_kind,
                "error": result.error,
            })
            template.provenance = {
                **template.provenance,
                "deeponto_error_kind": result.error_kind,
                "deeponto_error": result.error,
            }

        if not args.quiet:
            status = "OK"
            if not result.loaded:
                status = f"LOAD-FAIL ({result.error_kind})"
            elif not result.verbalized:
                status = f"VERB-FAIL ({result.error_kind})"
            print(
                f"[{i+1:3d}/{n_total}] {template.template_id:40s} "
                f"{status} "
                f"complex={result.is_complex} "
                f"L={result.mean_verbal_length:.0f}"
            )

    dt = time.time() - t0

    out_payload = {
        "version": catalog.version,
        "templates": [asdict(t) for t in catalog.templates],
        "null_stats": catalog.null_stats,
    }
    Path(args.output).write_text(json.dumps(out_payload, indent=2) + "\n")

    print()
    print(f"=== build_catalog summary ({dt:.1f}s) ===")
    print(f"  total templates:     {n_total}")
    print(f"  loaded successfully: {n_loaded} ({n_loaded / max(n_total, 1):.0%})")
    print(f"  is_complex=True:     {n_complex} ({n_complex / max(n_total, 1):.0%})")
    print(f"  verbalized:          {n_verbalized} ({n_verbalized / max(n_total, 1):.0%})")
    print(f"  failed (load|verb):  {n_failed}")
    if failures:
        kinds: dict[str, int] = {}
        for f in failures:
            kinds[f["error_kind"]] = kinds.get(f["error_kind"], 0) + 1
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"    {kind}: {n}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
