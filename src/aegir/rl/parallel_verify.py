"""Process-pool wrapper around ``aegir.ontology.verifier.verify``.

The intrinsic per-GRPO-step bottleneck on the Tinybox is the verifier's
``compute_r_d`` path: sentence-transformer encoding + KMeans clustering
of every composition's verbalizations. With 16 completions per rank per
step the serial cost is ~18 min/step (measured on 2026-05-16); the
verify() calls themselves are independent, so wrapping them in a
``ProcessPoolExecutor`` with 4-8 workers should cut per-step time
proportionally.

Per-worker init (catalog load + null_stats parse + sentence-transformer
load) costs ~5-10s on first use; the encoder is then cached at module
level inside the worker. Subsequent calls in the same worker reuse it.

Threading model: the pool is created lazily on first
``batch_verify_compositions`` call and lives for the rest of the
process. Under FSDP each rank's reward_fn creates its own pool, so 2
ranks × N workers concurrent (16 cpu-bound jobs on a 32-thread Tinybox
is well within budget).

Pickling: catalog is loaded by path inside each worker (not pickled
across the boundary). The entries list passed to ``_worker_verify`` is
just JSON-shaped dicts → cheap.

Spawn vs fork: we use ``"spawn"`` to be safe under torch+CUDA. Fork
inherits the parent's CUDA context which can deadlock under FSDP's
NCCL communicator. Spawn is slower at init (~5s extra per worker) but
deterministic. This cost is amortized over hundreds of GRPO steps.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


# Module-level state — one pool per Python process. Each FSDP rank has
# its own process and thus its own pool. The pool itself spawns the
# workers, so a rank-0 pool with 4 workers means 5 Python processes
# total in rank 0's tree (1 main + 4 verifier workers).
_pool: ProcessPoolExecutor | None = None
_pool_workers: int = 0
_pool_paths: tuple[str, str, str] | None = None


# Per-worker state, set by ``_worker_init`` and read by ``_worker_verify``.
# These are global-in-the-worker; the parent never sees them.
_WORKER_CATALOG: Any = None
_WORKER_T_I_PATH: str | None = None
_WORKER_NULL_STATS_PATH: str | None = None


def _worker_init(catalog_path: str, t_i_cache_path: str,
                 null_stats_path: str,
                 torch_threads_per_worker: int = 2) -> None:
    """Per-process init: load the catalog once and pre-warm the
    sentence-transformer encoder. The first verify() in each worker
    would otherwise pay this cost on the hot path.

    Constrains each worker's torch intra-op thread count to prevent
    BLAS thread-thrashing when multiple workers run in parallel. The
    serial verifier already uses torch.set_num_threads(all-cores) by
    default — with N workers each spinning up that many threads, total
    is N × cores, far more than physical, and they fight for CPU
    instead of parallelizing.

    A worker-thread budget of 2 with 4 workers gives 8 effective torch
    threads (matching ~8 physical cores), leaving headroom for the
    other 8-16 cores on a 32-thread Tinybox.
    """
    global _WORKER_CATALOG, _WORKER_T_I_PATH, _WORKER_NULL_STATS_PATH

    # Set torch + BLAS thread caps BEFORE importing aegir.ontology
    # (which imports torch transitively). Once torch is imported with
    # default thread settings, set_num_threads has limited effect on
    # already-spawned BLAS thread pools.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(torch_threads_per_worker))

    import torch
    torch.set_num_threads(torch_threads_per_worker)

    from aegir.ontology.schema import load_catalog
    _WORKER_CATALOG = load_catalog(catalog_path)
    _WORKER_T_I_PATH = t_i_cache_path
    _WORKER_NULL_STATS_PATH = null_stats_path

    # Pre-load the encoder; this is the expensive part.
    try:
        from aegir.ontology.topic_alignment import get_encoder
        get_encoder()  # cached at module level for the rest of the worker
    except Exception as exc:  # noqa: BLE001
        # Don't fail init — just log; the first verify() call will retry.
        logger.warning("encoder pre-warm failed in worker: %s", exc)


def _worker_verify(entries_raw: list[dict] | None) -> float:
    """Score a single completion (already parsed). Returns R.

    Designed to be called inside a worker process. ``entries_raw`` is
    the JSON-shaped ``[{template_id: ..., slot_fillers: {...}}, ...]``
    output of ``parse_compositions``. An empty/None list scores 0;
    any verifier exception scores 0 (same convention as the inline
    reward_fn in p5_train.py).
    """
    if not entries_raw:
        return 0.0

    from aegir.ontology.verifier import CompositionEntry, verify

    entries = [
        CompositionEntry(
            template_id=e["template_id"],
            slot_fillers=e["slot_fillers"],
        )
        for e in entries_raw
    ]
    try:
        res = verify(
            entries, _WORKER_CATALOG,
            t_i_cache_path=_WORKER_T_I_PATH,
            null_stats_path=_WORKER_NULL_STATS_PATH,
        )
        return res.R
    except Exception as exc:  # noqa: BLE001
        # verify() can raise on degenerate completions (sklearn NaN
        # propagation through KMeans, for example). Score as R=0 and
        # let the outer training loop continue.
        logger.debug("worker verify() failed: %s", exc)
        return 0.0


def get_pool(catalog_path: str, t_i_cache_path: str,
             null_stats_path: str, n_workers: int = 4,
             torch_threads_per_worker: int = 2) -> ProcessPoolExecutor:
    """Return the lazily-initialized process pool. Rebuilds if the
    worker count or paths changed (rare; happens at most once per
    process)."""
    global _pool, _pool_workers, _pool_paths

    paths = (catalog_path, t_i_cache_path, null_stats_path)
    if _pool is None or _pool_workers != n_workers or _pool_paths != paths:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        ctx = mp.get_context("spawn")
        _pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(catalog_path, t_i_cache_path, null_stats_path,
                      torch_threads_per_worker),
        )
        _pool_workers = n_workers
        _pool_paths = paths
        logger.info(
            "spawned ProcessPoolExecutor: %d workers × %d torch threads "
            "(catalog=%s)",
            n_workers, torch_threads_per_worker, catalog_path,
        )
    return _pool


def batch_verify_compositions(
    entries_per_completion: list[list[dict] | None],
    catalog_path: str,
    t_i_cache_path: str,
    null_stats_path: str,
    n_workers: int = 4,
) -> list[float]:
    """Score a list of parsed completions in parallel.

    Each element of ``entries_per_completion`` is the output of
    ``parse_compositions`` on a single completion (a list of
    ``{template_id, slot_fillers}`` dicts). Returns a list of the same
    length with each ``R`` value.

    Empty/None entry lists score 0 without dispatching to a worker —
    cheap short-circuit for the common parse-failure case.

    First call pays the ~5-10s pool-spawn + worker-warmup cost.
    Subsequent calls reuse the pool.
    """
    pool = get_pool(
        catalog_path, t_i_cache_path, null_stats_path, n_workers,
    )

    # Short-circuit the empty/None entries; ``pool.map`` still has
    # per-task overhead even for trivial work.
    indices_to_dispatch: list[int] = []
    payloads: list[list[dict]] = []
    results: list[float] = [0.0] * len(entries_per_completion)
    for i, entries in enumerate(entries_per_completion):
        if entries:
            indices_to_dispatch.append(i)
            payloads.append(entries)

    if not payloads:
        return results

    for idx, r in zip(indices_to_dispatch,
                       pool.map(_worker_verify, payloads)):
        results[idx] = r
    return results


def shutdown_pool() -> None:
    """Shut down the pool — called from exit paths if needed."""
    global _pool, _pool_workers, _pool_paths
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        _pool_workers = 0
        _pool_paths = None
