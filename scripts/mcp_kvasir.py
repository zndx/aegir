#!/usr/bin/env python
"""kvasir as an MCP tool server (#141) — the reasoning/DDL toolbox an ACP agent iterates against.

RH direction (2026-07-04): the parallel prose harnesses call HermiT + kvasir as MCP server TOOLS and
load materialized artifacts (SHACL graph, DDL) as MCP RESOURCES. This is the kvasir half: a FastMCP
stdio server exposing the feature-complete kvasir DDL toolchain as tools an agent can call mid-
generation to CHECK its ontology and see the schema it yields — quick, JVM-free (ms).

Tools:
  - ddl_profile(ontology_omn)   → the DDL kvasir generates + its distance to real-world SchemaPile
                                  shape (tables/fks/junctions/lookups/width/EMD). "What schema does
                                  this ontology yield, and is it structurally realistic?"
  - shacl_shapes(ontology_omn)  → the SHACL Core shapes graph (Turtle) — the closed-world constraint
                                  view; the semantic harness's grounding resource.
  - check_consistency(ontology_omn) → kvasir's sound-refutation pre-pass (refuted / no-clash).

Membrane doctrine (client.py): the agent REQUESTS these; the CLIENT owns them — the agent can check
its schema against ground truth but cannot fake or bypass the check.

Run (stdio, for vibe-acp/local): uv run --no-sync python scripts/mcp_kvasir.py
Run (HTTP, for grok — its ACP advertises http/sse MCP only, no stdio):
    .devenv/state/venv/bin/python scripts/mcp_kvasir.py --http 8765   # streamable-http at /mcp
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO = Path(__file__).resolve().parents[1]
KVASIR = REPO / "components/kvasir/target/release/kvasir"
if str(REPO) not in sys.path:  # `scripts.` imports when exec'd as a file (sys.path[0] = scripts/)
    sys.path.insert(0, str(REPO))

_HTTP_PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else None
mcp = FastMCP("kvasir", host="127.0.0.1", port=_HTTP_PORT or 8765)


def _tmp(omn: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(omn)
        return f.name


def _dormant(omn: str) -> list:
    """kvasir --stats coverage: the DDL paths this ontology does NOT exercise — the
    agent's complexity brief (model junctions/enums/identity idioms where plausible)."""
    p = _tmp(omn)
    try:
        r = subprocess.run([str(KVASIR), "ddl", p, "--stats"],
                           capture_output=True, text=True, timeout=120)
        return (json.loads(r.stdout).get("coverage") or {}).get("paths_dormant", []) \
            if r.returncode == 0 else []
    except Exception:  # noqa: BLE001
        return []
    finally:
        Path(p).unlink(missing_ok=True)


@mcp.tool()
def ddl_profile(ontology_omn: str) -> str:
    """Generate the relational DDL for an OWL Manchester ontology and profile its structure
    against the empirical SchemaPile distribution. Returns JSON with table/FK/junction/lookup
    counts, the column-width distribution, and the shape EMD (0 = perfect real-world realism;
    lower is better). Use this to check whether an ontology yields a rich, realistic schema
    before writing prose about it."""
    from importlib import import_module
    score = import_module("scripts.score_ontology_ddl").score  # reuse the scorer
    path = _tmp(ontology_omn)
    try:
        prof = score(Path(path))
        return json.dumps({
            "n_tables": prof["n_elected"] + prof["n_reference"],
            "n_elected": prof["n_elected"], "total_fks": prof["total_fks"],
            "n_junctions": prof["n_junctions"], "n_lookups": prof["n_lookups"],
            "width_median": prof["width"]["median"], "width_p90": prof["width"]["p90"],
            "attr_zero_ratio": prof["attr_zero_ratio"], "sql_valid": prof["sql_valid"],
            "shape_emd_vs_schemapile": prof["shape_emd"],
            "hint": "shape_emd toward 0 = realistic; add DataProperties / cardinality-bounded "
                    "relations / enums to enrich",
            "dormant_ddl_paths": _dormant(ontology_omn),
        }, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})
    finally:
        Path(path).unlink(missing_ok=True)


@mcp.tool()
def shacl_shapes(ontology_omn: str) -> str:
    """Emit the SHACL Core shapes graph (Turtle) for an OWL Manchester ontology — the closed-world
    constraint view (sh:NodeShape per class with sh:datatype / sh:class / sh:minCount / sh:in).
    Use this to see the exact validation constraints the schema enforces."""
    path = _tmp(ontology_omn)
    try:
        p = subprocess.run([str(KVASIR), "shapes", path], capture_output=True, text=True, timeout=120)
        return p.stdout if p.returncode == 0 else json.dumps({"error": p.stderr[:300]})
    finally:
        Path(path).unlink(missing_ok=True)


@mcp.tool()
def check_consistency(ontology_omn: str) -> str:
    """Run kvasir's sound-for-refutation consistency pre-pass on an OWL Manchester ontology.
    Returns 'refuted' (a genuine clash, with the unsatisfiable classes) or 'no-clash' (passes the
    fast fragment check — full certification is HermiT's). Use before trusting an ontology fragment."""
    from aegir.ontology.kvasir_bridge import fast_refute
    v = fast_refute(ontology_omn, source="mcp")
    return json.dumps({"verdict": v.get("verdict"), "unsat_classes": v.get("unsat_classes", []),
                       "reason": v.get("reason", ""), "ms": v.get("ms")})


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if _HTTP_PORT else "stdio")
