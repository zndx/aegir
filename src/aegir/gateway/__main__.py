"""Entry point: ``python -m aegir.gateway``.

Reads ``aegir.config.load_config()`` for host/port and launches uvicorn.
Environment overrides (``AEGIR_GATEWAY_PORT``, ``CDSW_APP_PORT``) take
effect automatically via the config loader.
"""

from __future__ import annotations

import logging

import uvicorn

from aegir.config import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    uvicorn.run(
        "aegir.gateway.app:app",
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
