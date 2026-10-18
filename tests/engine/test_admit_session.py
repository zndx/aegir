"""AdmitSession returns genus offsets and does not echo the transcript."""
from __future__ import annotations

from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg
from aegir.engine.server import ZndxEngineServicer
from aegir.ontology.session_stream import select_admissions


def _mgr():
    return SimpleNamespace(status=lambda: [])


def _serve(admit=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    zpbg.add_EngineServicer_to_server(ZndxEngineServicer(_mgr(), admit=admit), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    return server, channel, zpbg.EngineStub(channel)


def _hello_admit(text, *, flush, cache):
    if "HELLO" not in text:
        return []
    start = text.index("HELLO")
    return [{
        "start": start,
        "end": start + 5,
        "genus_iri": "https://signals.zndx.org/sdg#LIMS",
        "genus_code": "9",
        "genus_label": "LIMS",
        "rel_margin_h": 0.5556,
        "tau": 0.10,
    }]


def test_two_frames_return_offsets_for_the_qualified_span() -> None:
    server, channel, stub = _serve(_hello_admit)
    try:
        frames = [
            zpb.SessionText(session_id="agent-rtc-1", seq=1, text="xxxx"),
            zpb.SessionText(session_id="agent-rtc-1", seq=2, text="HELLO", end=True),
        ]
        got = list(stub.AdmitSession(iter(frames), timeout=3))
    finally:
        channel.close()
        server.stop(grace=0)
    assert len(got) == 1
    row = got[0]
    assert row.session_id == "agent-rtc-1"
    assert row.seq == 2
    assert (row.start, row.end) == (4, 9)
    assert row.genus_iri == "https://signals.zndx.org/sdg#LIMS"
    assert row.genus_code == "9"
    assert "text" not in zpb.SegmentAdmission.DESCRIPTOR.fields_by_name
    assert b"HELLO" not in row.SerializeToString()


def test_a_root_only_hit_yields_nothing() -> None:
    server, channel, stub = _serve(lambda text, *, flush, cache: [])
    try:
        frames = [zpb.SessionText(session_id="s", seq=1, text="xxxx", end=True)]
        got = list(stub.AdmitSession(iter(frames), timeout=3))
    finally:
        channel.close()
        server.stop(grace=0)
    assert got == []


def test_flush_sends_the_higher_margin_overlap_only() -> None:
    text = "AAAAABBBBB"

    def query(segment: str):
        if segment == text[0:8]:
            return [
                {"score": 0.5, "code": "A", "iri": "iri:A", "pref_label": "A"},
                {"score": 0.4, "code": "B", "iri": "iri:B", "pref_label": "B"},
            ]
        return [
            {"score": 0.9, "code": "B", "iri": "iri:B", "pref_label": "B"},
            {"score": 0.2, "code": "A", "iri": "iri:A", "pref_label": "A"},
        ]

    def admit(body, *, flush, cache):
        if body != text:
            return []
        return select_admissions(
            body, [(0, 8), (4, 10)], n_tokens=10, flush=flush, query=query, exclude_codes=set(),
        )

    server, channel, stub = _serve(admit)
    try:
        frames = [zpb.SessionText(session_id="s", seq=1, text=text, end=True)]
        got = list(stub.AdmitSession(iter(frames), timeout=3))
    finally:
        channel.close()
        server.stop(grace=0)
    assert [(row.start, row.end, row.genus_iri) for row in got] == [(4, 10, "iri:B")]
    assert b"AAAAA" not in got[0].SerializeToString()


def test_a_second_session_id_is_invalid() -> None:
    server, channel, stub = _serve(lambda text, *, flush, cache: [])
    try:
        frames = [
            zpb.SessionText(session_id="one", seq=1, text="aa"),
            zpb.SessionText(session_id="two", seq=2, text="bb", end=True),
        ]
        with pytest.raises(grpc.RpcError) as caught:
            list(stub.AdmitSession(iter(frames), timeout=3))
    finally:
        channel.close()
        server.stop(grace=0)
    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_decide_is_unimplemented() -> None:
    server, channel, stub = _serve()
    try:
        with pytest.raises(grpc.RpcError) as caught:
            stub.Decide(zpb.DecideRequest(), timeout=3)
    finally:
        channel.close()
        server.stop(grace=0)
    assert caught.value.code() == grpc.StatusCode.UNIMPLEMENTED
