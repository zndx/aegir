"""Reverse-proxy ``/viz/*`` to the Bokeh server (topology B).

Vite does this in dev (``ui/vite.config.ts`` → ``:5006``). The gateway must do
the same so a built SPA on ``:8091`` (and Caddy ``:8080``) can load BokehJS
and the session websocket. Static ``/viz/static/js/*.min.js`` is what
``<PanelView>`` preloads; the websocket is the live chord session.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

_log = logging.getLogger("aegir.gateway.viz")

_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})


def _upstream() -> str:
    return os.environ.get("AEGIR_VIZ_UPSTREAM", "http://127.0.0.1:5006").rstrip("/")


def mount_viz_proxy(app: FastAPI) -> None:
    """Register HTTP + WebSocket proxy for ``/viz/{path}``. Call BEFORE the SPA mount."""

    @app.api_route("/viz/{path:path}", methods=["GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS"])
    async def viz_http(path: str, request: Request) -> Response:
        url = f"{_upstream()}/viz/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                r = await client.request(
                    request.method, url, headers=headers, content=await request.body())
        except httpx.ConnectError as e:
            _log.warning("viz upstream down (%s): %s", url, e)
            return Response(
                content="viz server not reachable on :5006 — start with `just viz-serve`",
                status_code=502, media_type="text/plain")
        out = {k: v for k, v in r.headers.items() if k.lower() not in _HOP}
        return Response(content=r.content, status_code=r.status_code, headers=out,
                        media_type=r.headers.get("content-type"))

    @app.websocket("/viz/{path:path}")
    async def viz_ws(websocket: WebSocket, path: str) -> None:
        import websockets

        q = websocket.scope.get("query_string", b"").decode()
        base = _upstream().replace("http://", "ws://").replace("https://", "wss://")
        url = f"{base}/viz/{path}" + (f"?{q}" if q else "")
        await websocket.accept()
        try:
            async with websockets.connect(url, max_size=None) as up:
                async def c2s() -> None:
                    try:
                        while True:
                            msg = await websocket.receive()
                            if msg["type"] == "websocket.disconnect":
                                break
                            if "text" in msg and msg["text"] is not None:
                                await up.send(msg["text"])
                            elif "bytes" in msg and msg["bytes"] is not None:
                                await up.send(msg["bytes"])
                    except (WebSocketDisconnect, Exception):
                        pass

                async def s2c() -> None:
                    try:
                        async for msg in up:
                            if isinstance(msg, bytes):
                                await websocket.send_bytes(msg)
                            else:
                                await websocket.send_text(msg)
                    except Exception:
                        pass

                await asyncio.gather(c2s(), s2c())
        except Exception as e:  # noqa: BLE001
            _log.warning("viz websocket failed (%s): %s", url, e)
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
