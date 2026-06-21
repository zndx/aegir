"""OTel instrumentation for Metaflow steps — the Step→OTel→NiFi spine.

``@traced_step`` wraps a Metaflow ``@step`` in an OpenTelemetry span (flow/step/run attributes + a per-run
correlation id); spans export over OTLP to the devenv collector (``OTEL_EXPORTER_OTLP_ENDPOINT``, default
``localhost:4317``), which forwards to NiFi ListenOTLP for flow visualization (the Gaius pattern). OTel is a
first-class, devenv-provided dependency — there is no untraced fallback. (The OTLP exporter batches/buffers
asynchronously, so a momentarily-down collector never blocks a flow; it just drops/retries spans.)

    import aegir.metaflow_patch            # before metaflow import
    from metaflow import FlowSpec, step
    from aegir.flows.trace import TracedFlow, traced_step

    class MyFlow(TracedFlow, FlowSpec):
        @traced_step
        @step
        def start(self): ...
"""
from __future__ import annotations

import functools
import logging
import os
import uuid

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_tracer = None


def get_tracer():
    """The aegir-flows tracer, built once with an OTLP→collector exporter."""
    global _tracer
    if _tracer is None:
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider = TracerProvider(resource=Resource.create({"service.name": "aegir-flows"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("aegir.flows")
        logger.info("OTel tracer → %s", endpoint)
    return _tracer


def traced_step(func):
    """Wrap a Metaflow step in an OTel span with flow/step/run/correlation attributes."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        flow, step = type(self).__name__, func.__name__
        with get_tracer().start_as_current_span(f"aegir/{flow}/{step}") as span:
            span.set_attribute("metaflow.flow_name", flow)
            span.set_attribute("metaflow.step_name", step)
            span.set_attribute("aegir.correlation_id", self.correlation_id)
            rid = getattr(getattr(self, "_graph", None), "run_id", None) or os.environ.get("METAFLOW_RUN_ID", "")
            if rid:
                span.set_attribute("metaflow.run_id", str(rid))
            return func(self, *args, **kwargs)
    return wrapper


class TracedFlow:
    """Mixin for FlowSpec: a lazily-minted per-run correlation id + an event emitter onto the active span."""

    @property
    def correlation_id(self) -> str:
        cid = getattr(self, "_aegir_cid", None)
        if cid is None:
            cid = uuid.uuid4().hex[:16]
            self._aegir_cid = cid
        return cid

    def emit_event(self, name: str, attributes: "dict | None" = None) -> None:
        trace.get_current_span().add_event(name, attributes or {})
