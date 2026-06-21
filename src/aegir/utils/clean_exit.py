"""``os._exit``-based clean termination for scripts that load grpc / jpype / torch / HF-datasets.

These libraries spawn background (often C-level) threads — qdrant & engine gRPC pollers, the JVM,
HuggingFace ``datasets`` stream prefetchers, torch thread pools — whose teardown during *interpreter
finalization* can race the GIL and abort the process::

    Fatal Python error: PyGILState_Release: thread state ... must be current when releasing
    Python runtime state: finalizing

…with SIGABRT (exit 134) *after* ``main()`` has already done — and synchronously flushed — all of its
real work. The non-zero exit then poisons any caller that checks the return code (e.g. a Metaflow step's
``subprocess.run``). ``os._exit`` terminates immediately, skipping that finalization, so the genuine
exit code survives.

Contract: only call once ``main()`` has returned, i.e. all I/O is already committed (parquet/Iceberg
appends, manifest/cursor writes are synchronous in our pipeline scripts). ``os._exit`` skips ``atexit``
handlers and buffer flushing, so we flush std streams explicitly first.

    if __name__ == "__main__":
        from aegir.utils.clean_exit import clean_exit
        clean_exit(main())
"""
import os
import sys
from typing import NoReturn


def clean_exit(code: "int | None" = 0) -> NoReturn:
    """Flush std streams, then ``os._exit`` with ``code`` (None/falsy → 0, truthy non-int → 1)."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — never let a flush error mask the real exit code
        pass
    os._exit(code if isinstance(code, int) else (1 if code else 0))
