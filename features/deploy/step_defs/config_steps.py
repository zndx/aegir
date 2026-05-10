"""Step definitions for ``features/deploy/config_resolution.feature``.

Env-var isolation: each scenario stashes ``os.environ`` keys it sets into
``context._env_overrides``; ``after_scenario`` restores them via the
generic ``_cleanups`` list so tests don't leak mutations.
"""

from __future__ import annotations

import os

from behave import given, then, when  # type: ignore[import]

from aegir.config import load_config


def _set_env(context, key: str, value: str) -> None:
    prior = os.environ.get(key)
    os.environ[key] = value
    context._cleanups.append(
        lambda: (os.environ.__delitem__(key) if prior is None else os.environ.__setitem__(key, prior))
    )


@given('the env var {name} is set to "{value}"')
def step_given_env_set(context, name, value):
    _set_env(context, name, value)


@when("I load the default config")
def step_when_load_config(context):
    context.cfg = load_config()


@then("the gateway port is {port:d}")
def step_then_gateway_port(context, port):
    assert context.cfg.gateway.port == port, (
        f"expected {port}, got {context.cfg.gateway.port}"
    )


@then("the postgres URL targets port {port:d}")
def step_then_pg_port(context, port):
    assert f":{port}/" in context.cfg.db.url, (
        f"expected port {port} in {context.cfg.db.url}"
    )


@then("the qdrant HTTP port is {port:d}")
def step_then_qdrant_port(context, port):
    assert context.cfg.qdrant.http_port == port, (
        f"expected {port}, got {context.cfg.qdrant.http_port}"
    )


@then('the DB URL is "{expected}"')
def step_then_db_url(context, expected):
    assert context.cfg.db.url == expected, (
        f"expected {expected!r}, got {context.cfg.db.url!r}"
    )
