"""Runtime verifier core — R(O, I) for catalog compositions.

Importable module form of the logic in ``scripts/aegir-verify.py``.
The CLI script remains the user-facing entry point; this module
exists so other Python code (P4 smoke test, future RL infra) can
import the verifier components cleanly.

Locked weights from the P2 / claim C1 sweep:
``W_R_B = 0.50``, ``W_R_C = 0.05``, ``W_R_D = 0.45``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from aegir.ontology.schema import Catalog

# Currently-tuned τ_B saturation point for R_B (≥ this many
# is_complex templates in a composition saturates R_B at 1).
TAU_B_PLACEHOLDER = 5.0
# Target mean-verbal-length for R_C semantic-richness proxy.
L_TARGET_PLACEHOLDER = 80.0
# Topic count for the runtime *T_V* fit.
DEFAULT_TV_K = 12

# Aggregation weights {a, b, c} for R_B, R_C, R_D, locked at P2.
W_R_B, W_R_C, W_R_D = 0.50, 0.05, 0.45
assert abs(W_R_B + W_R_C + W_R_D - 1.0) < 1e-9

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
    diagnostics: dict[str, object] = field(default_factory=dict)


def parse_template_slots(template) -> dict[str, str]:
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
    """R_A — structural type-check (hard gate)."""
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
    t_i_cache_path,
    null_stats: dict | None,
    tv_k: int = DEFAULT_TV_K,
) -> tuple[float, dict]:
    """R_D — topic alignment with corpus *I*."""
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
    w_b: float = W_R_B,
    w_c: float = W_R_C,
    w_d: float = W_R_D,
) -> float:
    return R_A * (w_b * R_B + w_c * R_C + w_d * R_D)


def verify(
    composition: list[CompositionEntry],
    catalog: Catalog,
    t_i_cache_path,
    null_stats_path,
    skip_r_d: bool = False,
) -> VerifierResult:
    """Run the full verifier and return a VerifierResult."""
    R_A, r_a_errors = compute_r_a(composition, catalog)
    R_B = compute_r_b(composition, catalog)
    R_C = compute_r_c(composition, catalog)

    null_stats: dict | None = None
    if Path(null_stats_path).exists():
        null_stats = json.loads(Path(null_stats_path).read_text())

    if skip_r_d or R_A == 0.0:
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
