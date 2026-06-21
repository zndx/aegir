"""Monkeypatch for the Metaflow service metadata provider (lifted from Gaius, the Signals sibling).

Fixes a race condition where heartbeat/metadata requests return 404 because the run/task hasn't been
committed to the service DB yet. Upstream Metaflow only retries on 503; we also retry 404 for the
``/metadata`` and ``/heartbeat`` endpoints of freshly-created runs/tasks.

Usage:
    import aegir.metaflow_patch   # MUST import before any `from metaflow import …`
    from metaflow import FlowSpec, step
"""
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

_patched = False

# 404-retry config (race-condition fix): 15 attempts × 2s = 30s.
RETRY_404_MAX_ATTEMPTS = 15
RETRY_404_DELAY = 2.0


def patch_service_provider():
    """Patch ServiceMetadataProvider._request to retry on 404 for metadata/heartbeat endpoints."""
    global _patched
    if _patched:
        return
    try:
        from metaflow.exception import MetaflowInternalError
        from metaflow.metaflow_config import SERVICE_HEADERS, SERVICE_RETRY_COUNT  # type: ignore[import-untyped]
        from metaflow.plugins.metadata_providers.service import (  # type: ignore[import-untyped]
            ServiceException, ServiceMetadataProvider)
    except ImportError:
        logger.warning("could not import metaflow service provider; patch not applied")
        return

    @classmethod
    def patched_request(cls, monitor, path, method, data=None, retry_409_path=None, return_raw_resp=False):
        from metaflow.exception import MetaflowException
        if cls.INFO is None:
            raise MetaflowException("Missing Metaflow Service URL. Set METAFLOW_SERVICE_URL.")
        supported = ("GET", "PATCH", "POST")
        if method not in supported:
            raise MetaflowException(f"Only {supported} supported, got {method}")
        url = os.path.join(cls.INFO, path.lstrip("/"))
        is_retryable_path = any(x in path for x in ["/metadata", "/heartbeat"])

        def _do_request():
            hdr = SERVICE_HEADERS.copy()
            if method == "GET":
                return (cls._session.get(url, headers=hdr) if not monitor else
                        _measured(monitor, "get", lambda: cls._session.get(url, headers=hdr)))
            if method == "POST":
                return (cls._session.post(url, headers=hdr, json=data) if not monitor else
                        _measured(monitor, "post", lambda: cls._session.post(url, headers=hdr, json=data)))
            if method == "PATCH":
                return (cls._session.patch(url, headers=hdr, json=data) if not monitor else
                        _measured(monitor, "patch", lambda: cls._session.patch(url, headers=hdr, json=data)))
            raise MetaflowInternalError(f"Unexpected HTTP method {method}")

        max_attempts = RETRY_404_MAX_ATTEMPTS if is_retryable_path else SERVICE_RETRY_COUNT
        resp = None
        for i in range(max_attempts):
            try:
                resp = _do_request()
            except MetaflowInternalError:
                raise
            except Exception:
                if i == max_attempts - 1:
                    raise
                resp = None
                time.sleep(RETRY_404_DELAY if is_retryable_path else 2 ** i)
                continue
            if return_raw_resp:
                return resp, True
            if resp.status_code < 300:
                return resp.json(), True
            elif resp.status_code == 409 and data is not None:
                if retry_409_path:
                    v, _ = cls._request(monitor, retry_409_path, "GET")
                    return v, False
                return None, False
            elif resp.status_code == 404:
                if is_retryable_path and i < max_attempts - 1:
                    print(f"[AE-PATCH-001] retry 404 on {path} (attempt {i + 1}/{max_attempts}) — race",
                          file=sys.stderr, flush=True)
                    time.sleep(RETRY_404_DELAY)
                    continue
                raise ServiceException(f"Metadata request ({path}) failed (404): {resp.text}",
                                       resp.status_code, resp.text)
            elif resp.status_code == 503:
                if i < max_attempts - 1:
                    time.sleep(2 ** i)
                    continue
            else:
                raise ServiceException(f"Metadata request ({path}) failed ({resp.status_code}): {resp.text}",
                                       resp.status_code, resp.text)
        if resp:
            raise ServiceException(f"Metadata request ({path}) failed ({resp.status_code}): {resp.text}",
                                   resp.status_code, resp.text)
        raise ServiceException(f"Metadata request ({path}) failed")

    ServiceMetadataProvider._request = patched_request
    _patched = True
    logger.info("metaflow service provider patched (404 retry, max %d)", RETRY_404_MAX_ATTEMPTS)


def _measured(monitor, op, fn):
    with monitor.measure(f"metaflow.service_metadata.{op}"):
        return fn()


def patch_heartbeat():
    """Patch MetadataHeartBeat._heartbeat to treat 404 as transient (run/task not committed yet)."""
    try:
        from metaflow.metadata_provider.heartbeat import (  # type: ignore[import-untyped]
            HeartBeatException, MetadataHeartBeat)
    except ImportError:
        logger.warning("could not import metaflow heartbeat; patch not applied")
        return

    def patched_heartbeat(self):
        import json
        import requests
        if self.hb_url is None:
            return None
        try:
            response = requests.post(url=self.hb_url, data="{}", headers=self.headers.copy())
        except requests.exceptions.ConnectionError:
            raise HeartBeatException(f"HeartBeat ({self.hb_url}) failed (ConnectionError)")
        except requests.exceptions.Timeout:
            raise HeartBeatException(f"HeartBeat ({self.hb_url}) failed (Timeout)")
        except requests.exceptions.RequestException as e:
            raise HeartBeatException(f"HeartBeat ({self.hb_url}) failed (RequestException) {e}")
        if response.status_code == 200:
            return json.loads(response.json()).get("wait_time_in_seconds")
        elif response.status_code == 404:
            raise HeartBeatException(f"HeartBeat ({self.hb_url}) got 404 (transient, will retry)")
        raise HeartBeatException(
            f"HeartBeat ({self.hb_url}) failed ({response.status_code}): {response.text}")

    MetadataHeartBeat._heartbeat = patched_heartbeat
    logger.info("metaflow heartbeat patched (404 transient)")


# Auto-apply on import.
patch_service_provider()
patch_heartbeat()
