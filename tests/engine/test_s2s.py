"""Status.surfaces and ServerQuery remotes / peers / surfaces."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.s2s import (
    advertise_host,
    advertised_head,
    configured_peers,
    is_loopback_host,
    list_named_remotes,
    local_primary_ui,
    local_response,
    local_surfaces,
    rewrite_public_url,
)
from aegir.engine.server import ZndxEngineServicer, build_status_response


def _mgr(endpoints=()):
    from types import SimpleNamespace
    return SimpleNamespace(status=lambda: list(endpoints))


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
