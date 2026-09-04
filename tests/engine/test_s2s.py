"""Status.surfaces and ServerQuery remotes / peers / surfaces."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.s2s import (
    advertise_host,
    advertised_head,
    announced_peers,
    collect_peer_surfaces,
    configured_peers,
    is_loopback_host,
    list_named_remotes,
    local_primary_ui,
    local_response,
    local_surfaces,
    remember_announce,
    reset_announced,
    rewrite_public_url,
)
from aegir.engine.server import ZndxEngineServicer, build_status_response


def _mgr(endpoints=()):
    from types import SimpleNamespace
    return SimpleNamespace(status=lambda: list(endpoints))


@pytest.fixture(autouse=True)
def _clear_announce_roster():
    reset_announced()
    yield
    reset_announced()


def test_list_named_remotes_from_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:zndx/aegir.git"],
        cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "upstream", "git@github.com:example/aegir.git"],
        cwd=tmp_path, check=True, capture_output=True)
    remotes = dict(list_named_remotes(tmp_path))
    assert remotes["origin"] == "git@github.com:zndx/aegir.git"
    assert remotes["upstream"] == "git@github.com:example/aegir.git"


def test_list_named_remotes_empty_when_not_a_repo(tmp_path: Path) -> None:
    assert list_named_remotes(tmp_path) == []
    assert advertised_head(tmp_path) == ""


def test_advertise_host_never_loopback(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    assert advertise_host() == "tinybox.dev.vista.zndx.org"
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "127.0.0.1")
    monkeypatch.setenv("SIGNALS_KRB_HOST", "tinybox.dev.vista.zndx.org")
    assert advertise_host() == "tinybox.dev.vista.zndx.org"
    assert is_loopback_host("localhost")
    assert not is_loopback_host("tinybox.dev.vista.zndx.org")


def test_short_hostname_promotes_to_zt_fqdn(monkeypatch) -> None:
    monkeypatch.delenv("AEGIR_ADVERTISE_HOST", raising=False)
    monkeypatch.delenv("AEGIR_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_ADVERTISE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_KRB_HOST", raising=False)
    monkeypatch.delenv("GAIUS_ADVERTISE_HOST", raising=False)
    import aegir.engine.s2s as s2s
    monkeypatch.setattr(s2s.socket, "getfqdn", lambda: "tinybox")
    monkeypatch.setattr(s2s.socket, "gethostname", lambda: "tinybox")
    monkeypatch.setattr(s2s.socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("10.0.0.1", 0))])
    assert advertise_host() == "tinybox.dev.vista.zndx.org"


def test_primary_ui_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.setenv("AEGIR_PRIMARY_UI", "http://127.0.0.1:5173")
    assert local_primary_ui() == "http://tinybox.dev.vista.zndx.org:5173"
    monkeypatch.delenv("AEGIR_PRIMARY_UI")
    monkeypatch.setenv("AEGIR_UI_BIND", "0.0.0.0:15173")
    assert local_primary_ui() == "http://tinybox.dev.vista.zndx.org:15173"
    assert "127.0.0.1" not in local_primary_ui()
    assert "localhost" not in local_primary_ui()


def test_rewrite_public_url_leaves_lan_alone(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    assert rewrite_public_url("http://gaius.lan:9890/") == "http://gaius.lan:9890/"


def test_status_advertises_primary_surface(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("AEGIR_PRIMARY_UI", raising=False)
    monkeypatch.delenv("AEGIR_UI_URL", raising=False)
    monkeypatch.delenv("AEGIR_UI_BIND", raising=False)
    resp = build_status_response(_mgr())
    assert resp.project == "aegir"
    assert [(s.kind, s.url) for s in resp.surfaces] == [
        ("primary", "http://tinybox.dev.vista.zndx.org:5173"),
    ]


def test_server_query_remotes_and_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@example.com:aegir.git"],
        cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "t"],
        cwd=tmp_path, check=True, capture_output=True)
    resp = local_response(zpb.SERVER_QUERY_KIND_REMOTES, root=tmp_path)
    assert resp.project == "aegir"
    assert [(r.name, r.url) for r in resp.remotes] == [
        ("origin", "git@example.com:aegir.git"),
    ]
    assert resp.head == advertised_head(tmp_path)
    assert len(resp.head) == 40


def test_server_query_surfaces_matches_status(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    q = local_response(zpb.SERVER_QUERY_KIND_SURFACES)
    assert [(s.kind, s.url) for s in q.surfaces] == [
        (s.kind, s.url) for s in local_surfaces()
    ]
    assert all("127.0.0.1" not in s.url for s in q.surfaces)


def test_server_query_peers_from_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("AEGIR_LATTICE_HOST", raising=False)
    monkeypatch.delenv("SIGNALS_LATTICE_HOST", raising=False)
    contract = tmp_path / "peer-contract.json"
    contract.write_text(
        json.dumps({
            "engine_grpc_lattice": {
                "gaius": 50051,
                "aegir": 50151,
                "signals": 50551,
                "service": "zndx.engine.v1.Engine",
            }
        }),
        encoding="utf-8",
    )
    peers = configured_peers(contract)
    assert ("gaius", "tinybox.dev.vista.zndx.org:50051") in peers
    assert ("signals", "tinybox.dev.vista.zndx.org:50551") in peers
    assert all(p[0] != "aegir" for p in peers)
    q = local_response(zpb.SERVER_QUERY_KIND_PEERS, contract=contract)
    assert {p.project for p in q.peers} == {"gaius", "signals"}


def test_servicer_server_query_and_yield() -> None:
    svc = ZndxEngineServicer(_mgr())
    resp = svc.ServerQuery(
        zpb.ServerQueryRequest(kind=zpb.SERVER_QUERY_KIND_REMOTES), None)
    assert resp.project == "aegir"
    names = {r.name for r in resp.remotes}
    assert "origin" in names
    assert all(r.url for r in resp.remotes)
    y = svc.Yield(zpb.YieldRequest(workload_id="none"), None)
    assert y.ok is True
    assert y.process_ended is False


def test_server_query_workloads_advertises_model_and_tp_pp() -> None:
    from aegir.engine.s2s import declared_workloads

    q = local_response(zpb.SERVER_QUERY_KIND_WORKLOADS)
    assert q.project == "aegir"
    offers = list(declared_workloads())
    assert [w.model for w in q.workloads] == [h.model for h in offers]
    instruct = next(w for w in q.workloads if "instruct" in list(w.capabilities))
    assert instruct.model
    tp = instruct.requirements.parallelism.tensor_parallel
    pp = instruct.requirements.parallelism.pipeline_parallel
    assert tp >= 1
    assert pp >= 1
    assert instruct.requirements.footprint.gpu == tp * pp
    via_svc = ZndxEngineServicer(_mgr()).ServerQuery(
        zpb.ServerQueryRequest(kind=zpb.SERVER_QUERY_KIND_WORKLOADS), None)
    assert [w.model for w in via_svc.workloads] == [w.model for w in q.workloads]


def test_announce_joins_peers_until_ttl() -> None:
    ok, ttl, err = remember_announce(
        project="hermes",
        engine_target="otherbox.lan:50651",
        ttl_seconds=30,
    )
    assert ok is True
    assert err == ""
    assert ttl == 30
    assert announced_peers() == [("hermes", "otherbox.lan:50651")]
    q = local_response(zpb.SERVER_QUERY_KIND_PEERS)
    assert any(p.project == "hermes" and p.target == "otherbox.lan:50651" for p in q.peers)
    ack = ZndxEngineServicer(_mgr()).Announce(
        zpb.PeerAnnounce(project="hermes", engine_target="otherbox.lan:50651", ttl_seconds=45),
        None,
    )
    assert ack.accepted is True
    assert ack.ttl_seconds == 45
    refused = remember_announce(project="aegir", engine_target="tinybox:50151")
    assert refused[0] is False
    empty = remember_announce(project="hermes", engine_target="not-a-target")
    assert empty[0] is False


def test_announce_expires(monkeypatch) -> None:
    import aegir.engine.s2s as s2s

    clock = {"t": 100.0}
    monkeypatch.setattr(s2s.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(s2s, "MIN_ANNOUNCE_TTL_S", 1)
    ok, ttl, _ = remember_announce(
        project="hermes", engine_target="otherbox.lan:50651", ttl_seconds=1,
    )
    assert ok and ttl == 1
    assert announced_peers() == [("hermes", "otherbox.lan:50651")]
    clock["t"] = 102.0
    assert announced_peers() == []


def test_collect_skips_peers_without_primary_ui(monkeypatch) -> None:
    monkeypatch.setattr("aegir.engine.s2s.configured_peers", lambda contract=None: [])
    monkeypatch.setattr("aegir.engine.s2s.directory_seeds", lambda: [("", "127.0.0.1:50551")])
    monkeypatch.setattr(
        "aegir.engine.s2s.status_peer",
        lambda target, timeout=4.0: zpb.StatusResponse(project="signals"),
    )
    monkeypatch.setattr(
        "aegir.engine.s2s.query_peer",
        lambda *a, **k: zpb.ServerQueryResponse(project="signals"),
    )
    monkeypatch.delenv("AEGIR_PRIMARY_UI", raising=False)
    monkeypatch.delenv("AEGIR_ADVERTISE_HOST", raising=False)
    rows = collect_peer_surfaces()
    assert all(r["project"] != "signals" for r in rows)


def test_collect_walks_peers_to_foreign_hermes(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("AEGIR_PRIMARY_UI", raising=False)
    monkeypatch.setenv("AEGIR_UI_BIND", "0.0.0.0:5173")
    monkeypatch.setattr("aegir.engine.s2s.configured_peers", lambda contract=None: [])
    monkeypatch.setattr("aegir.engine.s2s.directory_seeds", lambda: [("", "127.0.0.1:50551")])

    def _status(target, timeout=4.0):
        if "50651" in target:
            return zpb.StatusResponse(
                project="hermes",
                surfaces=[zpb.Surface(
                    kind="primary", url="http://otherbox.lan:9119", healthy=True,
                )],
            )
        if "50551" in target:
            return zpb.StatusResponse(
                project="signals",
                surfaces=[zpb.Surface(
                    kind="primary",
                    url="http://tinybox.dev.vista.zndx.org:9889",
                    healthy=True,
                )],
            )
        return None

    def _query(target, **_k):
        if "50551" in target:
            return zpb.ServerQueryResponse(
                project="signals",
                peers=[zpb.PeerHint(project="hermes", target="otherbox.lan:50651")],
            )
        return zpb.ServerQueryResponse(project="hermes")

    monkeypatch.setattr("aegir.engine.s2s.status_peer", _status)
    monkeypatch.setattr("aegir.engine.s2s.query_peer", _query)
    rows = {r["project"]: r for r in collect_peer_surfaces()}
    assert rows["hermes"]["title"] == "Hermes"
    assert rows["hermes"]["primary_ui"] == "http://otherbox.lan:9119"
    assert rows["hermes"]["engine_target"] == "otherbox.lan:50651"
    assert rows["signals"]["primary_ui"] == "http://tinybox.dev.vista.zndx.org:9889"


def test_collect_walks_own_engine_peers_from_gateway(monkeypatch) -> None:
    """Gateway has no in-process roster; it must ServerQuery this engine."""
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("AEGIR_PRIMARY_UI", raising=False)
    monkeypatch.setenv("AEGIR_UI_BIND", "0.0.0.0:5173")
    monkeypatch.setattr("aegir.engine.s2s.configured_peers", lambda contract=None: [])
    monkeypatch.setattr("aegir.engine.s2s.directory_seeds", lambda: [])
    monkeypatch.setattr("aegir.engine.s2s.announced_peers", lambda: [])

    def _status(target, timeout=4.0):
        if "50151" in target:
            return zpb.StatusResponse(
                project="aegir",
                surfaces=[zpb.Surface(
                    kind="primary",
                    url="http://tinybox.dev.vista.zndx.org:5173",
                    healthy=True,
                )],
            )
        if "50651" in target:
            return zpb.StatusResponse(
                project="hermes",
                surfaces=[zpb.Surface(
                    kind="primary", url="http://otherbox.lan:9119", healthy=True,
                )],
            )
        return None

    def _query(target, **_k):
        if "50151" in target:
            return zpb.ServerQueryResponse(
                project="aegir",
                peers=[zpb.PeerHint(project="hermes", target="otherbox.lan:50651")],
            )
        return zpb.ServerQueryResponse(project="hermes")

    monkeypatch.setattr("aegir.engine.s2s.status_peer", _status)
    monkeypatch.setattr("aegir.engine.s2s.query_peer", _query)
    rows = {r["project"]: r for r in collect_peer_surfaces()}
    assert rows["hermes"]["primary_ui"] == "http://otherbox.lan:9119"
    assert rows["hermes"]["engine_target"] == "otherbox.lan:50651"


def test_collect_lists_announced_hermes_same_box(monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_ADVERTISE_HOST", "tinybox.dev.vista.zndx.org")
    monkeypatch.delenv("AEGIR_PRIMARY_UI", raising=False)
    monkeypatch.setenv("AEGIR_UI_BIND", "0.0.0.0:5173")
    monkeypatch.setattr("aegir.engine.s2s.configured_peers", lambda contract=None: [])
    monkeypatch.setattr("aegir.engine.s2s.directory_seeds", lambda: [])
    remember_announce(project="hermes", engine_target="127.0.0.1:50651", ttl_seconds=60)

    def _status(target, timeout=4.0):
        return zpb.StatusResponse(
            project="hermes",
            surfaces=[zpb.Surface(
                kind="primary", url="http://127.0.0.1:9119", healthy=True,
            )],
        )

    monkeypatch.setattr("aegir.engine.s2s.status_peer", _status)
    monkeypatch.setattr(
        "aegir.engine.s2s.query_peer",
        lambda *a, **k: zpb.ServerQueryResponse(project="hermes"),
    )
    rows = {r["project"]: r for r in collect_peer_surfaces()}
    assert rows["hermes"]["primary_ui"] == "http://tinybox.dev.vista.zndx.org:9119"
    assert rows["hermes"]["engine_target"] == "tinybox.dev.vista.zndx.org:50651"
