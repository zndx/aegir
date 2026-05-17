"""PyIceberg catalog wiring for Aegir's HX storage.

Backed by:
- catalog metadata in postgres 16 (devenv-managed, port 5555,
  database ``iceberg_catalog``)
- warehouse data files at ``file:///raid/checkpoints/aegir-artifacts/iceberg-warehouse``

The catalog name is ``aegir`` (distinct from gaius's ``gaius`` catalog
to avoid namespace collisions while sharing the schema and writer
code via the unified venv).

Environment overrides:

- ``AEGIR_ICEBERG_URI``       postgres connection URI
- ``AEGIR_ICEBERG_WAREHOUSE`` warehouse path (file:// or s3://)
- ``AEGIR_ICEBERG_CATALOG``   catalog name (default ``aegir``)

When rustfs comes online, swap ``AEGIR_ICEBERG_WAREHOUSE`` to the
``s3://...`` endpoint and add s3 access env vars; no other code
changes needed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table


DEFAULT_URI = "postgresql+psycopg://rch@127.0.0.1:5555/iceberg_catalog"
DEFAULT_WAREHOUSE = "file:///raid/checkpoints/aegir-artifacts/iceberg-warehouse"
DEFAULT_CATALOG_NAME = "aegir"


def _config() -> dict[str, str]:
    return {
        "name": os.getenv("AEGIR_ICEBERG_CATALOG", DEFAULT_CATALOG_NAME),
        "uri": os.getenv("AEGIR_ICEBERG_URI", DEFAULT_URI),
        "warehouse": os.getenv("AEGIR_ICEBERG_WAREHOUSE", DEFAULT_WAREHOUSE),
    }


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Load the shared PyIceberg catalog. Cached per-process."""
    from pyiceberg.catalog.sql import SqlCatalog

    cfg = _config()
    catalog = SqlCatalog(cfg["name"], uri=cfg["uri"], warehouse=cfg["warehouse"])
    return catalog


def ensure_namespace(catalog: Catalog | None = None, name: str = "raw") -> None:
    """Idempotently create the given namespace."""
    cat = catalog or get_catalog()
    try:
        cat.create_namespace(name)
    except Exception:
        # Already exists — load to confirm
        try:
            cat.load_namespace_properties(name)
        except Exception:
            raise


def get_exchange_table(catalog: Catalog | None = None) -> Table:
    """Load (or idempotently create) ``raw.exchange`` table.

    Uses gaius's shared schema / partition / sort definitions so the
    bytes-on-disk are interoperable.
    """
    from gaius.hx.exchange_tables import (
        get_exchange_partition_spec,
        get_exchange_schema,
        get_exchange_sort_order,
    )

    cat = catalog or get_catalog()
    ensure_namespace(cat, "raw")
    table_id = ("raw", "exchange")
    try:
        return cat.create_table(
            table_id,
            schema=get_exchange_schema(),
            partition_spec=get_exchange_partition_spec(),
            sort_order=get_exchange_sort_order(),
        )
    except Exception:
        return cat.load_table(table_id)


def append_exchange(record: Any, catalog: Catalog | None = None) -> int:
    """Append a single ``gaius.hx.exchange.ExchangeRecord`` to the table.

    Returns the row count after append. Synchronous; raises on failure.

    For high-volume capture, buffer records in-memory in your script
    and call ``catalog.load_table(...).append(pa.Table.from_pylist(...))``
    directly with a batch.
    """
    import pyarrow as pa

    table = get_exchange_table(catalog)
    row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    rb = pa.Table.from_pylist([row], schema=table.schema().as_arrow())
    table.append(rb)
    return table.scan().to_arrow().num_rows
