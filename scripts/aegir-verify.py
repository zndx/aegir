"""aegir-verify — runtime verifier for the SDG catalog (R_A + R_B + R_C + R_D).

Implements the four-component verifier R(O, I) from the v0.5
concept brief. R_A (structural type-check, hard gate), R_B
(complex-template density), R_C (mean verbal length), and R_D
(topic alignment with the input corpus *I* via Hungarian
matching against the cached T_I and structural-shuffle null).

Inputs:
    --catalog PATH    Catalog JSON (default: examples.json).
    --composition PATH-or-stdin
                      Composition JSON: a list of
                      {"template_id": str, "slot_fillers": dict}
                      records, or a top-level object with
                      "templates" key. JSON is read from stdin
                      when --composition is "-".

Outputs deterministic R ∈ [0, 1] plus per-component
breakdown to stdout. Exits 0 on success, 1 on input errors.

The verifier is intentionally hash-stable across runs given
fixed inputs: no random seeds, no clock-dependent values, no
filesystem ordering dependencies (catalog templates iterate in
list order as authored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aegir.ontology.schema import Catalog, CatalogTemplate, load_catalog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "src" / "aegir" / "ontology" / "catalog" / "examples.json"
DEFAULT_T_I_CACHE = REPO_ROOT / "src" / "aegir" / "ontology" / "catalog" / "T_I_canonical.pkl"
DEFAULT_NULL_STATS = REPO_ROOT / "src" / "aegir" / "ontology" / "catalog" / "null_stats_canonical.json"

# Currently-tuned τ_B saturation point for R_B (≥ this many
# is_complex templates in a composition saturates R_B at 1).
# Replaced at P2 by null-distribution-derived value.
TAU_B_PLACEHOLDER = 5.0
# Target mean-verbal-length for R_C semantic-richness proxy. The
# expected centroid of well-authored verbal templates lies above
# this threshold; below saturates linearly to 0.
L_TARGET_PLACEHOLDER = 80.0
# Topic count for the runtime *T_V* fit. Smaller than T_I's k
# because *V(O)* is typically on the order of dozens of
# sentences; matches the value used in null-distribution sampling.
DEFAULT_TV_K = 12

# Aggregation weights {a, b, c} for R_B, R_C, R_D.
# **Locked at P2 (claim C1)** — tuned via AUC sweep against the
# labeled ontology test set in ``tests/ontology_test_set/`` per
# the v0.5 concept brief P2 exit gate. Selection criteria:
#  - AUC ≥ 0.85: PASS at AUC = 0.9956 (vs default 0.9867).
#  - separation ≥ 0.30: PASS at 0.3388 (vs default 0.3324).
#  - Preserve meaningful contribution from each component.
# R_C's small weight (0.05) reflects that semantic-richness-as-
# length is a weak discriminator on the current test set —
# bad ontologies with many shallow axioms produce high R_C
# despite being conceptually empty. R_B (complex-class density)
# carries most of the discrimination signal; R_D (topic
# alignment) preserves the four-component design intent.
# Full sweep results: tests/ontology_test_set/c1_results.json.
W_R_B, W_R_C, W_R_D = 0.50, 0.05, 0.45
assert abs(W_R_B + W_R_C + W_R_D - 1.0) < 1e-9

# Slot-syntax regex matching {name:Type} or {name:Type:Bound}.
SLOT_RE = re.compile(r"\{(?P<name>\w+):(?P<type>[\w:]+?)(?::(?P<bound>[\w:]+))?\}")


@dataclass
class CompositionEntry:
    """A single template instantiation in a composition."""
    template_id: str
    slot_fillers: dict[str, str]


@dataclass
class VerifierResult:
    """The output of running R(*O*, *I*) on a composition."""
    R: float
    R_A: float
    R_B: float
    R_C: float
    R_D: float
    diagnostics: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def load_composition(path_or_dash: str) -> list[CompositionEntry]:
    """Parse a composition file (or stdin if "-")."""
    if path_or_dash == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path_or_dash).read_text()
    data = json.loads(raw)
    if isinstance(data, dict) and "templates" in data:
        rows = data["templates"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(
            "composition JSON must be a list or an object with 'templates' key"
        )
    return [
        CompositionEntry(
            template_id=row["template_id"],
            slot_fillers=row.get("slot_fillers", {}),
        )
        for row in rows
    ]


def parse_template_slots(template: CatalogTemplate) -> dict[str, str]:
    """Extract {name: Type} pairs from the manchester_template
    string. Used for cross-checking against declared slot_types.
    """
    found: dict[str, str] = {}
    for m in SLOT_RE.finditer(template.manchester_template):
        name = m.group("name")
        type_ = m.group("type")
        if name in found and found[name] != type_:
            raise ValueError(
                f"template {template.template_id!r}: slot {name!r} appears with "
                f"conflicting types {found[name]!r} and {type_!r}"
            )
        found[name] = type_
    return found


def compute_r_a(
    composition: list[CompositionEntry],
    catalog: Catalog,
) -> tuple[float, list[str]]:
    """R_A — structural type-check (hard gate).

    Returns (R_A, errors). R_A = 1 iff every entry's template
    exists in the catalog and every slot filler is provided
    against a declared slot. R_A = 0 on any failure; aggregation
    short-circuits R(O) = 0 in that case.
    """
    errors: list[str] = []
    for i, entry in enumerate(composition):
        try:
            template = catalog.by_id(entry.template_id)
        except KeyError:
            errors.append(
                f"composition[{i}]: template_id {entry.template_id!r} not in catalog"
            )
            continue

        declared_slots = set(template.slot_types.keys())
        provided_slots = set(entry.slot_fillers.keys())

        missing = declared_slots - provided_slots
        if missing:
            errors.append(
                f"composition[{i}] (template={entry.template_id!r}): "
                f"missing slot fillers {sorted(missing)}"
            )
        extra = provided_slots - declared_slots
        if extra:
            errors.append(
                f"composition[{i}] (template={entry.template_id!r}): "
                f"unknown slot fillers {sorted(extra)}"
            )

        for slot_name, filler in entry.slot_fillers.items():
            if slot_name not in template.slot_types:
                continue
            if not isinstance(filler, str) or not filler.strip():
                errors.append(
                    f"composition[{i}] (template={entry.template_id!r}): "
                    f"slot {slot_name!r} filler must be a non-empty string"
                )

    return (0.0 if errors else 1.0, errors)


def compute_r_b(
    composition: list[CompositionEntry],
    catalog: Catalog,
    tau_b: float = TAU_B_PLACEHOLDER,
) -> float:
    """R_B — complex-template density.

    Counts templates in the composition flagged ``is_complex=True``
    in the catalog, divided by ``tau_b`` (saturating at 1).
    """
    if tau_b <= 0:
        return 1.0
    complex_count = 0
    for entry in composition:
        try:
            template = catalog.by_id(entry.template_id)
        except KeyError:
            continue
        if template.is_complex:
            complex_count += 1
    return min(complex_count / tau_b, 1.0)


def compute_r_c(
    composition: list[CompositionEntry],
    catalog: Catalog,
    l_target: float = L_TARGET_PLACEHOLDER,
) -> float:
    """R_C — mean verbal length / target (semantic-richness proxy).

    Reads each composition entry's pre-cached ``mean_verbal_length``
    from the catalog, averages, divides by ``l_target``, clips to
    [0, 1]. Returns 0.0 if the composition is empty or no entry has
    a non-zero length cached.
    """
    if l_target <= 0:
        return 1.0
    lengths: list[float] = []
    for entry in composition:
        try:
            template = catalog.by_id(entry.template_id)
        except KeyError:
            continue
        if template.mean_verbal_length > 0:
            lengths.append(template.mean_verbal_length)
    if not lengths:
        return 0.0
    mean_length = sum(lengths) / len(lengths)
    return max(0.0, min(mean_length / l_target, 1.0))


def render_composition_verbalizations(
    composition: list[CompositionEntry],
    catalog: Catalog,
) -> list[str]:
    """Render the composition's verbalization corpus *V(O)* by
    substituting slot fillers into pre-cached ``verbal_template``s.
    """
    out: list[str] = []
    for entry in composition:
        try:
            tmpl = catalog.by_id(entry.template_id)
        except KeyError:
            continue
        if not tmpl.verbal_template:
            continue
        verbal = tmpl.verbal_template
        for slot_name, filler in entry.slot_fillers.items():
            verbal = verbal.replace("{" + slot_name + "}", filler)
        out.append(verbal)
    return out


def compute_r_d(
    composition: list[CompositionEntry],
    catalog: Catalog,
    t_i_cache_path: Path | str,
    null_stats: dict | None,
    tv_k: int = DEFAULT_TV_K,
) -> tuple[float, dict]:
    """R_D — topic alignment with corpus *I*.

    Renders the verbalization corpus *V(O)* from the composition,
    fits a small topic model *T_V*, computes Hungarian-optimal
    cosine alignment against the cached *T_I*, then normalizes
    the raw score to [0, 1] using the catalog's null statistics
    (``null_mean → 0``, ``null_p95 → 1``).

    Returns ``(r_d, diagnostics)``. ``r_d = 0.0`` when the
    composition produces fewer than 2 verbalizable templates
    (insufficient for topic modeling).
    """
    from aegir.ontology.topic_alignment import (
        alignment_score,
        fit_topic_model,
        get_encoder,
        load_topic_model,
        normalize_alignment,
    )

    verbalizations = render_composition_verbalizations(composition, catalog)
    diagnostics: dict = {
        "n_verbalizations": len(verbalizations),
        "tv_k": tv_k,
    }
    if len(verbalizations) < 2:
        diagnostics["r_d_skipped"] = "fewer than 2 verbalizable templates"
        return 0.0, diagnostics

    if not Path(t_i_cache_path).exists():
        diagnostics["r_d_skipped"] = f"no cached T_I at {t_i_cache_path}"
        return 0.0, diagnostics

    t_i = load_topic_model(t_i_cache_path)
    encoder = get_encoder(t_i.encoder_model)
    t_v = fit_topic_model(verbalizations, k=tv_k, encoder=encoder)
    raw = alignment_score(t_v, t_i)

    if null_stats is None:
        diagnostics["r_d_unnormalized"] = True
        diagnostics["raw_alignment"] = raw
        return raw, diagnostics

    normalized = normalize_alignment(
        raw,
        null_mean=null_stats["null_mean"],
        null_p95=null_stats["null_p95"],
    )
    diagnostics["raw_alignment"] = raw
    diagnostics["null_mean"] = null_stats["null_mean"]
    diagnostics["null_p95"] = null_stats["null_p95"]
    return normalized, diagnostics


def aggregate(
    R_A: float,
    R_B: float,
    R_C: float,
    R_D: float,
) -> float:
    """R(O) = R_A · (a · R_B + b · R_C + c · R_D)."""
    return R_A * (W_R_B * R_B + W_R_C * R_C + W_R_D * R_D)


def verify(
    composition: list[CompositionEntry],
    catalog: Catalog,
    t_i_cache_path: Path | str = DEFAULT_T_I_CACHE,
    null_stats_path: Path | str = DEFAULT_NULL_STATS,
    skip_r_d: bool = False,
) -> VerifierResult:
    """Run the full verifier and return a :class:`VerifierResult`."""
    R_A, r_a_errors = compute_r_a(composition, catalog)
    R_B = compute_r_b(composition, catalog)
    R_C = compute_r_c(composition, catalog)

    null_stats: dict | None = None
    if Path(null_stats_path).exists():
        null_stats = json.loads(Path(null_stats_path).read_text())

    if skip_r_d or R_A == 0.0:
        # Skip R_D when R_A short-circuits (R(O) = 0 anyway) or
        # when caller explicitly requests R_A+R_B+R_C-only mode.
        R_D, r_d_diag = 0.0, {"r_d_skipped": "R_A=0 or skip-r-d flag"}
    else:
        R_D, r_d_diag = compute_r_d(
            composition, catalog,
            t_i_cache_path=t_i_cache_path,
            null_stats=null_stats,
        )

    R = aggregate(R_A, R_B, R_C, R_D)

    diagnostics: dict[str, object] = {
        "catalog_version": catalog.version,
        "n_templates_in_catalog": len(catalog.templates),
        "n_composition_entries": len(composition),
        "weights": {"a": W_R_B, "b": W_R_C, "c": W_R_D},
        "tau_b": TAU_B_PLACEHOLDER,
        "l_target": L_TARGET_PLACEHOLDER,
        "r_d": r_d_diag,
    }
    if r_a_errors:
        diagnostics["r_a_errors"] = r_a_errors

    return VerifierResult(
        R=R,
        R_A=R_A,
        R_B=R_B,
        R_C=R_C,
        R_D=R_D,
        diagnostics=diagnostics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(DEFAULT_CATALOG),
        help="Path to catalog JSON (default: examples.json).",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default="-",
        help="Path to composition JSON, or '-' for stdin (default: stdin).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only R as a single float; suppress JSON breakdown.",
    )
    parser.add_argument(
        "--ti-cache",
        type=str,
        default=str(DEFAULT_T_I_CACHE),
        help="Path to cached T_I topic model.",
    )
    parser.add_argument(
        "--null-stats",
        type=str,
        default=str(DEFAULT_NULL_STATS),
        help="Path to null-statistics JSON.",
    )
    parser.add_argument(
        "--skip-r-d",
        action="store_true",
        help="Skip R_D (topic alignment) — R_A + R_B + R_C only.",
    )
    args = parser.parse_args()

    try:
        catalog = load_catalog(args.catalog)
    except Exception as e:
        print(f"failed to load catalog at {args.catalog!r}: {e}", file=sys.stderr)
        return 1

    try:
        composition = load_composition(args.composition)
    except Exception as e:
        print(f"failed to load composition: {e}", file=sys.stderr)
        return 1

    result = verify(
        composition, catalog,
        t_i_cache_path=args.ti_cache,
        null_stats_path=args.null_stats,
        skip_r_d=args.skip_r_d,
    )
    if args.quiet:
        print(f"{result.R:.6f}")
    else:
        sys.stdout.write(result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
