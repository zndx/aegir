"""Aegir configuration loader.

Reads ``config/base.conf`` (HOCON) with environment-variable substitution and
returns a typed namespace. The HOCON syntax ``${?VAR}`` means "override with
env var if set; otherwise leave prior assignment". See
``config/base.conf`` for the full schema.

Three materialization outputs for M1:

- ``Config`` dataclass — typed Python handle used by the gateway and Python CLIs.
- ``build/config/aegir.env`` — shell-sourceable flat KEY=value pairs.
- ``build/config/aegir.json`` — structured JSON (no secrets in M1 because Aegir
  has no credentialed dependencies yet; the format is carried forward for M2+).

Pattern borrowed from ``/home/rch/local/src/zndx/atelier/src/atelier/config.py``
with the provider/classify/tooling sections stripped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE_CONF = Path(__file__).resolve().parent.parent.parent / "config" / "base.conf"


@dataclass
class GatewayCfg:
    host: str = "0.0.0.0"
    port: int = 8091


@dataclass
class DbCfg:
    url: str = "postgresql+psycopg://localhost:5555/aegir?sslmode=disable"


@dataclass
class QdrantCfg:
    host: str = "localhost"
    http_port: int = 6355
    grpc_port: int = 6356


@dataclass
class RunsCfg:
    dir: str = "outputs/runs"


@dataclass
class DataCfg:
    gittables_signals_dir: str = "/home/rch/local/src/cldr/signals/build/datasets/gittables"


@dataclass
class UiCfg:
    dist_dir: str = "ui/dist"


@dataclass
class Config:
    gateway: GatewayCfg = field(default_factory=GatewayCfg)
    db: DbCfg = field(default_factory=DbCfg)
    qdrant: QdrantCfg = field(default_factory=QdrantCfg)
    runs: RunsCfg = field(default_factory=RunsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    def to_flat_env(self) -> dict[str, str]:
        """Materialize for shell sourcing. Keys in ``AEGIR_*`` form."""
        return {
            "AEGIR_GATEWAY_HOST": self.gateway.host,
            "AEGIR_GATEWAY_PORT": str(self.gateway.port),
            "AEGIR_DB_URL": self.db.url,
            "AEGIR_QDRANT_HOST": self.qdrant.host,
            "AEGIR_QDRANT_HTTP_PORT": str(self.qdrant.http_port),
            "AEGIR_QDRANT_GRPC_PORT": str(self.qdrant.grpc_port),
            "AEGIR_RUNS_DIR": self.runs.dir,
            "AEGIR_GITTABLES_DIR": self.data.gittables_signals_dir,
            "AEGIR_UI_DIST": self.ui.dist_dir,
        }

    def to_json_dict(self) -> dict:
        return {
            "gateway": {"host": self.gateway.host, "port": self.gateway.port},
            "db": {"url": self.db.url},
            "qdrant": {
                "host": self.qdrant.host,
                "http_port": self.qdrant.http_port,
                "grpc_port": self.qdrant.grpc_port,
            },
            "runs": {"dir": self.runs.dir},
            "data": {"gittables_signals_dir": self.data.gittables_signals_dir},
            "ui": {"dist_dir": self.ui.dist_dir},
        }


def load_config(path: Path | str | None = None) -> Config:
    """Load ``config/base.conf`` with env substitution and return typed Config.

    HOCON env syntax (``${?VAR}``) is resolved by pyhocon at load time. On top
    of that, we apply an explicit final sweep so overrides like
    ``AEGIR_GATEWAY_PORT`` always win, even if pyhocon's handling differs
    across versions. This keeps behavior predictable in CAI/Zarf where the
    platform sets env vars just before process launch.
    """
    from pyhocon import ConfigFactory

    conf_path = Path(path) if path else DEFAULT_BASE_CONF
    if not conf_path.exists():
        # Fall back to pure-defaults if the file's missing; simplifies tests.
        return _apply_env_overrides(Config())

    parsed = ConfigFactory.parse_file(str(conf_path))
    root = parsed.get("aegir", {})

    def _g(section: str, key: str, default):
        """Look up a nested key with graceful fallback.

        pyhocon returns ``None`` for ``${?UNSET_VAR}`` substitutions, which
        overrides the preceding default assignment — so we treat ``None``
        as "use the Python default" instead of passing it through.
        """
        try:
            v = root.get(section).get(key)
        except Exception:
            return default
        return default if v is None else v

    cfg = Config(
        gateway=GatewayCfg(
            host=_g("gateway", "host", "0.0.0.0"),
            port=int(_g("gateway", "port", 8091)),
        ),
        db=DbCfg(url=_g("db", "url", DbCfg.url)),
        qdrant=QdrantCfg(
            host=_g("qdrant", "host", "localhost"),
            http_port=int(_g("qdrant", "http_port", 6355)),
            grpc_port=int(_g("qdrant", "grpc_port", 6356)),
        ),
        runs=RunsCfg(dir=_g("runs", "dir", "outputs/runs")),
        data=DataCfg(gittables_signals_dir=_g("data", "gittables_signals_dir", DataCfg.gittables_signals_dir)),
        ui=UiCfg(dist_dir=_g("ui", "dist_dir", "ui/dist")),
    )
    return _apply_env_overrides(cfg)


def _apply_env_overrides(cfg: Config) -> Config:
    """Final-pass overrides to guarantee env-var precedence."""
    g = os.environ.get
    if (v := g("AEGIR_GATEWAY_HOST")):
        cfg.gateway.host = v
    if (v := g("AEGIR_GATEWAY_PORT")) or (v := g("CDSW_APP_PORT")):
        try:
            cfg.gateway.port = int(v)
        except ValueError:
            pass
    if (v := g("AEGIR_DB_URL")):
        cfg.db.url = v
    if (v := g("AEGIR_QDRANT_HOST")) or (v := g("QDRANT_HOST")):
        cfg.qdrant.host = v
    if (v := g("AEGIR_QDRANT_HTTP_PORT")) or (v := g("QDRANT_HTTP_PORT")):
        try:
            cfg.qdrant.http_port = int(v)
        except ValueError:
            pass
    if (v := g("AEGIR_QDRANT_GRPC_PORT")) or (v := g("QDRANT_GRPC_PORT")):
        try:
            cfg.qdrant.grpc_port = int(v)
        except ValueError:
            pass
    if (v := g("AEGIR_RUNS_DIR")):
        cfg.runs.dir = v
    if (v := g("AEGIR_GITTABLES_DIR")):
        cfg.data.gittables_signals_dir = v
    if (v := g("AEGIR_UI_DIST")):
        cfg.ui.dist_dir = v
    return cfg


def materialize_config(cfg: Config, out_path: Path | str) -> None:
    """Write ``KEY=value`` lines suitable for ``source``-ing in bash."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={_shell_quote(v)}\n" for k, v in cfg.to_flat_env().items()]
    out.write_text("".join(lines))


def materialize_config_json(cfg: Config, out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.to_json_dict(), indent=2, sort_keys=True))


def _shell_quote(v: str) -> str:
    """Minimal quoting: wrap in double quotes if the value contains anything
    other than alphanum/path-safe characters. Not intended to guard against
    malicious input — configs come from the developer, not the network."""
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/-.:=,"
    if all(c in safe for c in v):
        return v
    return '"' + v.replace('"', r'\"') + '"'
