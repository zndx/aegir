"""Entry point: ``python -m aegir.gateway``.

Reads ``aegir.config.load_config()`` for host/port and launches uvicorn.
Environment overrides (``AEGIR_GATEWAY_PORT``, ``CDSW_APP_PORT``) take
effect automatically via the config loader. Set ``AEGIR_GATEWAY_RELOAD=1``
(dev only — the devenv gateway exec does) to enable uvicorn auto-reload so
code changes are picked up without a manual ``devenv processes restart``;
it watches only the ``aegir`` package source, so ``lineup build`` writes to
``build/dev`` never trigger a reload loop.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from aegir.config import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    reload = os.environ.get("AEGIR_GATEWAY_RELOAD") == "1"
    reload_dirs = None
    if reload:
        import aegir
        reload_dirs = [str(Path(aegir.__file__).resolve().parent)]
        logging.getLogger("aegir.gateway").info(
            "auto-reload ON (watching %s) — dev only", reload_dirs[0])
    uvicorn.run(
        "aegir.gateway.app:app",
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        log_level="info",
        access_log=True,
        reload=reload,
        reload_dirs=reload_dirs,
    )


if __name__ == "__main__":
    main()
