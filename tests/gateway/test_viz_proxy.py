"""Gateway /viz reverse-proxy: 502 when the Bokeh server is down.

Live static 200s are checked by `just stack-health` once devenv process `viz` is up.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from aegir.gateway.app import create_app


def test_viz_proxy_502_when_upstream_down(monkeypatch):
    monkeypatch.setenv("AEGIR_VIZ_UPSTREAM", "http://127.0.0.1:9")
    client = TestClient(create_app())
    r = client.get("/viz/static/js/bokeh-gl.min.js")
    assert r.status_code == 502
    assert b"just viz-serve" in r.content
