"""HX storage layer for Aegir.

Reuses ``gaius.hx`` schemas and writers via the unified deployment env
(see ``pyproject.toml`` + commit ad14bf2). This module provides aegir-
specific helpers:

- ``get_catalog()``: load the shared PyIceberg postgres-backed catalog
- ``append_exchange()``: convenience wrapper for single-record writes

For deeper machinery (multi-record buffering, async writes, health
checks), call into ``gaius.hx.exchange.ExchangeCapture`` directly.
"""

from aegir.hx.catalog import append_exchange, get_catalog, get_exchange_table

__all__ = ["append_exchange", "get_catalog", "get_exchange_table"]
