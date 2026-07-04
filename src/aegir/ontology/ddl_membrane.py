"""The STRUCTURAL membrane — kvasir DDL-relevance at the derive boundary (#141).

The FinePDFs→ontology pipeline's leading agent proposes ontology extensions; the existing
membranes gate them for LOGICAL validity (parse / HermiT consistency / OntoClean /
property-reuse). This membrane adds the RELATIONAL axis: a proposed extension is scored by
what it lowers to as DDL — is it well-formed (parses, no clash, DDL-relevant) and does it
drive the corpus toward quantitatively-verified SchemaPile structure parity?

It runs the feature-complete kvasir DDL chain (``kvasir census``) on the extension's
rendered Manchester document and compares the resulting table-width distribution against
the empirical SchemaPile norms (``build/schemapile_shape_norms.json``, #139). The verdict +
reason is agent-consumable feedback (the [[agent_mediated_feedback_loop]] pattern on the
STRUCTURAL axis): an extension that adds only taxonomy (0 tables) or thin tables (no
attributes/relations) gets a reason telling the agent to propose DataProperties and
object-property restrictions with named-class fillers instead — the levers that lower into
width, FKs, junctions, and lookups.

Pure structural signal; it does NOT decide admission alone (HermiT/OntoClean own logical
validity). It is a forcing function toward relational richness, measured not asserted.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KVASIR_BIN = REPO / "components/kvasir/target/release/kvasir"
SCHEMAPILE_NORMS = REPO / "build/schemapile_shape_norms.json"

# SchemaPile width targets (non-identity, spine-comparable) — refreshed from the norms file
# when present; these are the mined 2026-07-04 values as a floor so the membrane works
# before the instrument has run.
_FALLBACK_MEDIAN = 4
_FALLBACK_P90 = 12


def _norms() -> dict:
    if SCHEMAPILE_NORMS.exists():
        try:
            return json.loads(SCHEMAPILE_NORMS.read_text())
        except Exception:
            pass
    return {}


# Standard prefix header so an instantiated primitive axiom (which uses sdg:/bfo:/cco:/xsd:
# prefixed forms) canonicalizes correctly under kvasir's own-prefix expansion.
_PREFIXES = (
    "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
    "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
    "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
    "Prefix: fhir: <http://hl7.org/fhir/>\n"
    "Prefix: obi: <http://purl.obolibrary.org/obo/OBI_>\n"
    "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "Prefix: skos: <http://www.w3.org/2004/02/skos/core#>\n"
)


import re as _re

# ``{Name:Type}`` slot markers → ``sdg:Name`` concrete IRIs. The slot's concept name IS the
# class name, so this measures the SAME DDL yield whether the axiom is still template-form
# (catalog storage) or already instantiated (the live derive loop, where no markers remain) —
# a no-op on instantiated axioms.
_SLOT = _re.compile(r"\{(\w+):\w+(?::\w+)?\}")


def signal_for_manchester(manchester: str) -> dict:
    """Score a single Manchester axiom (a primitive's ``manchester_template``, template-form
    or instantiated) for DDL yield — canonicalizes ``{Name:Type}`` slots to ``sdg:Name``,
    wraps it with prefix declarations (JVM-free, deterministic), and calls
    :func:`structural_signal`. The membrane-facing entry point for the derive loop."""
    concrete = _SLOT.sub(r"sdg:\1", manchester.strip())
    doc = _PREFIXES + "\n" + concrete + "\n"
    return structural_signal(doc)


def census_of(manchester_doc: str, *, timeout_s: int = 60) -> "dict | None":
    """Run ``kvasir census`` on a Manchester document; None if the binary is absent or errors.
    The census is diagnostic (never refuses), so it is safe on a derive-stage fragment."""
    if not KVASIR_BIN.exists():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(manchester_doc)
        path = f.name
    try:
        p = subprocess.run([str(KVASIR_BIN), "census", path, "--json"],
                           capture_output=True, text=True, timeout=timeout_s)
        if p.returncode != 0 or not p.stdout.strip():
            return None
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def structural_signal(manchester_doc: str) -> dict:
    """Score a proposed ontology extension by its DDL yield + SchemaPile-parity direction.

    Returns a signal dict the agent loop consumes:

    - ``verdict`` ∈ {``malformed``, ``inert``, ``thin``, ``rich``} — the coarse call.
    - ``reason`` — agent-facing re-prompt text (what to propose to improve).
    - ``well_formed`` / ``n_parse_issues`` / ``fragment_no_clash`` — the well-formedness leg.
    - ``ddl_yield`` — n_elected / n_junctions / n_lookups / attr_zero_ratio / width_*.
    - ``parity`` — schemapile_median, width_gap (signed: ours − SchemaPile), direction.

    The verdict ladder (structural only — HermiT/OntoClean own logical admission):
      malformed  parse issues OR a fragment clash — the extension is not well-formed
      inert      well-formed but 0 elected tables — pure taxonomy, no relational structure
      thin       elects tables but attr-poor / narrower than SchemaPile — some structure
      rich       elects tables with attributes/relations at/above SchemaPile width
    """
    c = census_of(manchester_doc)
    if c is None:
        return {"verdict": "unavailable", "reason": "kvasir census unavailable",
                "well_formed": None}

    n_issues = c.get("n_parse_issues", 0)
    no_clash = c.get("fragment_no_clash", True)
    well_formed = n_issues == 0 and no_clash
    width = c.get("width") or {}
    yield_ = {
        "n_elected": c.get("n_elected", 0),
        "n_junctions": c.get("n_junctions", 0),
        "n_lookups": c.get("n_lookups", 0),
        "attr_zero_ratio": c.get("attr_zero_ratio", 0.0),
        "width_median": width.get("median", 0),
        "width_p90": width.get("p90", 0),
        "width_max": width.get("max", 0),
    }

    norms = _norms()
    sp_median = (norms.get("width") or {}).get("median", _FALLBACK_MEDIAN)
    sp_p90 = (norms.get("width") or {}).get("p90", _FALLBACK_P90)
    our_median = yield_["width_median"]
    width_gap = our_median - sp_median  # negative = too narrow (the column-poverty direction)
    parity = {
        "schemapile_median": sp_median,
        "schemapile_p90": sp_p90,
        "width_gap": width_gap,
        # a small extension that ELECTS tables at/above the SchemaPile median moves toward
        # parity; one that elects nothing or stays thin does not
        "direction": ("toward" if yield_["n_elected"] > 0 and width_gap >= -1
                      else "away" if yield_["n_elected"] == 0 else "neutral"),
    }

    # ── verdict ladder + agent-facing reason ──────────────────────────────────
    if not well_formed:
        if n_issues > 0:
            reason = (f"malformed: {n_issues} parse issue(s) — the extension does not lower "
                      "to valid DDL; fix the axiom syntax/shape before re-proposing")
        else:
            reason = ("malformed: the extension introduces a fragment clash (inconsistency) — "
                      "no schema can be realized from it")
        verdict = "malformed"
    elif yield_["n_elected"] == 0:
        verdict = "inert"
        reason = ("inert: the extension adds only taxonomy (0 tables). To drive relational "
                  "structure, propose DataProperties (typed attributes) and object-property "
                  "restrictions with NAMED class fillers (some/exactly/min) — these lower into "
                  "columns, FKs, junctions and lookups")
    elif yield_["attr_zero_ratio"] >= 0.6 or width_gap < -1:
        verdict = "thin"
        reason = (f"thin: elects {yield_['n_elected']} table(s) but width median {our_median} "
                  f"vs SchemaPile {sp_median} (attr-zero {yield_['attr_zero_ratio']:.2f}). Add "
                  "typed DataProperties and cardinality-bounded relations to reach realistic "
                  "table width and FK fan-out")
    else:
        verdict = "rich"
        reason = (f"rich: elects {yield_['n_elected']} table(s) at width median {our_median} "
                  f"(SchemaPile {sp_median}), {yield_['n_junctions']} junction(s), "
                  f"{yield_['n_lookups']} lookup(s) — moves toward SchemaPile structure parity")

    return {
        "verdict": verdict,
        "reason": reason,
        "well_formed": well_formed,
        "n_parse_issues": n_issues,
        "fragment_no_clash": no_clash,
        "ddl_yield": yield_,
        "parity": parity,
    }
