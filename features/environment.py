"""Behave environment hooks — tier filtering, GPU gating, fixture setup.

Mirrors Atelier's pattern (``~/local/src/zndx/atelier/features/environment.py``)
with preconditions specific to Aegir: since Aegir has no running services to
probe, the "stack health" check instead asserts CUDA extensions are importable,
a GPU is present, the tiny fixture exists, and the sentence-transformers cache
is reachable. Cached on ``context._stack_verified`` so the cost is paid once
per session, not per scenario.

Env vars:
    AEGIR_BDD_TIER   — 0 or 1 (default 0). Max tier to run.
    AEGIR_BDD_SLOW   — 0 or 1 (default 0). Include @slow scenarios.
    AEGIR_BDD_FIXTURES — override fixture dir (default: ``features/fixtures``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("aegir.bdd")


# ── Tier logic ──────────────────────────────────────────────────

TIER_TAGS = {"tier-0": 0, "tier-1": 1}


def _tier_from_scenario(scenario) -> int:
    """Return the highest tier required by scenario + feature tags."""
    all_tags = set(scenario.tags) | set(scenario.feature.tags)
    max_tier = 0
    for tag in all_tags:
        if tag in TIER_TAGS:
            max_tier = max(max_tier, TIER_TAGS[tag])
    return max_tier


def _max_tier() -> int:
    """Max tier allowed by AEGIR_BDD_TIER env var (default: 0)."""
    raw = os.environ.get("AEGIR_BDD_TIER", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def _slow_enabled() -> bool:
    return os.environ.get("AEGIR_BDD_SLOW", "0") not in ("0", "", "false", "False")


# ── Stack health (cached, one-time) ────────────────────────────


def _check_cuda_extensions() -> None:
    """Verify the RWKV-7 Triton kernel is importable.

    fla's ``chunk_rwkv7`` is load-bearing — the w/W block types can't run
    without it. flash-attn and mamba-ssm are soft: ``aegir.modules.block``
    falls back to ``LayerNormPrenorm`` when flash-attn is unavailable, and
    only m/M blocks need mamba-ssm. We log warnings for the soft deps but
    only fail on fla.
    """
    from fla.ops.rwkv7 import chunk_rwkv7  # noqa: F401
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        log.warning(
            "flash_attn not importable (%s) — falling back to LayerNormPrenorm. "
            "See CLAUDE.md for the CXX11 ABI rebuild recipe to restore RMSNorm.",
            exc,
        )
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        log.info("mamba_ssm not installed (optional) — m/M blocks unavailable")


def _check_cuda_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for @tier-1 scenarios")
    return torch.device("cuda:0")


def _check_fixture_present(fixtures_dir: Path) -> Path:
    tiny = fixtures_dir / "gittables_tiny"
    parquet = tiny / "gittables_columns.parquet"
    gt = tiny / "gittables_gt.json"
    if not (parquet.exists() and gt.exists()):
        raise RuntimeError(
            f"gittables tiny fixture not found at {tiny}. "
            f"Run `just sotab-fixture` (alias for make_tiny_fixture.py) first."
        )
    return tiny


def _check_mpnet_cache() -> None:
    """Confirm sentence-transformers cache reachable. Don't trigger a download."""
    # Probe common cache roots. If neither exists, warn but don't fail —
    # scenarios that use MMR will fail loudly at first use, which is more
    # informative than a cryptic fixture-level error.
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    raid_cache = "/raid/cache/huggingface"
    candidates = [Path(hf_home), Path(raid_cache)]
    for root in candidates:
        target = root / "hub" / "models--sentence-transformers--all-mpnet-base-v2"
        if target.exists():
            return
    log.warning("all-mpnet-base-v2 not found in any HF cache; MMR may need to download")


_STACK_CACHE: dict = {}


def _ensure_stack_healthy(context) -> None:
    """Run expensive checks once per session; restore handles onto ``context`` each call.

    Behave rebuilds the Context object per scenario, so even though we cache the
    health check result, we still have to re-attach ``context.device`` and
    ``context.fixture_dir`` for each scenario that needs them.
    """
    if not _STACK_CACHE:
        _check_cuda_extensions()
        _STACK_CACHE["device"] = _check_cuda_device()
        _STACK_CACHE["fixture_dir"] = _check_fixture_present(context.fixtures_dir)
        _check_mpnet_cache()
        log.info("Aegir BDD stack verified (CUDA + extensions + fixtures)")
    context.device = _STACK_CACHE["device"]
    context.fixture_dir = _STACK_CACHE["fixture_dir"]
    context._stack_verified = True


# ── Hooks ───────────────────────────────────────────────────────


def before_all(context):
    context.project_root = Path(__file__).resolve().parent.parent
    default_fixtures = context.project_root / "features" / "fixtures"
    context.fixtures_dir = Path(
        os.environ.get("AEGIR_BDD_FIXTURES", str(default_fixtures))
    )
    context._cleanups = []
    logging.basicConfig(level=os.environ.get("AEGIR_BDD_LOG", "INFO"))


def before_scenario(context, scenario):
    tier = _tier_from_scenario(scenario)
    allowed = _max_tier()
    if tier > allowed:
        scenario.skip(f"Requires tier-{tier}, max allowed is {allowed}")
        return

    all_tags = set(scenario.tags) | set(scenario.feature.tags)

    if "slow" in all_tags and not _slow_enabled():
        scenario.skip("@slow scenario skipped: set AEGIR_BDD_SLOW=1 to run")
        return

    if "gpu" in all_tags:
        try:
            import torch
            if not torch.cuda.is_available():
                scenario.skip("@gpu scenario skipped: no CUDA device available")
                return
        except Exception as exc:
            scenario.skip(f"@gpu scenario skipped: torch import failed ({exc})")
            return

    if tier >= 1:
        try:
            _ensure_stack_healthy(context)
        except Exception as exc:
            scenario.skip(f"stack not healthy: {exc}")
            return

    # Fresh per-scenario state.
    context._cleanups = []


def after_scenario(context, scenario):
    for fn in getattr(context, "_cleanups", []):
        try:
            fn()
        except Exception as exc:
            log.warning("cleanup failed: %s", exc)
    context._cleanups = []
