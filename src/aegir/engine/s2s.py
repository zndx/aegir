"""Server-to-server query helpers (signals-protocol Engine/ServerQuery).

Pairwise snapshot — not gossip. Do not invent remotes, peers, or peer UI URLs.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb

logger = logging.getLogger(__name__)

PROJECT = "aegir"
LATTICE_SKIP = frozenset(
    {"service", "proto", "server_reflection", "first_party_clients"}
)
_LOOPBACK = frozenset(
    {"", "localhost", "127.0.0.1", "0.0.0.0", "::1", "::", "[::1]", "[::]"}
)
_LAB_CONTRACT = Path.home() / "local/src/wxs/signals/config/platform/peer-contract.json"

DEFAULT_ANNOUNCE_TTL_S = 90
MIN_ANNOUNCE_TTL_S = 15
MAX_ANNOUNCE_TTL_S = 600


@dataclass(frozen=True)
class AnnouncedPeer:
    project: str
    target: str
    expires_at: float


_ANNOUNCE_LOCK = threading.Lock()
_ANNOUNCED: dict[str, AnnouncedPeer] = {}


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().strip("[]")
    if not h:
        return True
    if h.lower() in _LOOPBACK or h.startswith("127."):
        return True
    return False


def advertise_host() -> str:
    """LAN hostname or IP. Never loopback — empty is honest."""
    for key in (
        "AEGIR_ADVERTISE_HOST",
        "AEGIR_LATTICE_HOST",
        "SIGNALS_ADVERTISE_HOST",
        "SIGNALS_LATTICE_HOST",
        "SIGNALS_KRB_HOST",
        "GAIUS_ADVERTISE_HOST",
    ):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        host = raw.split("/")[-1].split(":")[0].strip("[]")
        if host and not is_loopback_host(host):
            return host
    try:
        fqdn = (socket.getfqdn() or "").strip()
        if fqdn and not is_loopback_host(fqdn) and "." in fqdn:
            return fqdn
        hn = (socket.gethostname() or "").strip()
        if hn and not is_loopback_host(hn) and "." in hn:
            return hn
        # Short hostname (tinybox) → Zero Trust / lab FQDN when it resolves.
        if hn and not is_loopback_host(hn):
            zt = f"{hn}.dev.vista.zndx.org"
            try:
                socket.getaddrinfo(zt, None)
                return zt
            except OSError:
                return hn
        for info in socket.getaddrinfo(hn or "localhost", None, socket.AF_INET):
            ip = info[4][0]
            if ip and not is_loopback_host(ip):
                return ip
    except OSError:
        pass
    return ""


def rewrite_public_url(url: str) -> str:
    """Replace a loopback URL host with advertise_host(). Non-loopback unchanged."""
    host = advertise_host()
    if not url or not host:
        return url
    parsed = urlparse(url)
    if not parsed.hostname or not is_loopback_host(parsed.hostname):
        return url
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunparse(
        (parsed.scheme or "http", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def repo_root() -> Path:
    raw = (os.environ.get("AEGIR_REPO_ROOT") or os.environ.get("DEVENV_ROOT") or "").strip()
    if raw:
        return Path(raw)
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".git").exists() and (cand / "src" / "aegir").exists():
            return cand
    return Path.cwd()


def list_named_remotes(root: Path | None = None) -> list[tuple[str, str]]:
    """Unique (name, fetch_url) from `git remote -v`. Do not invent remotes."""
    checkout = root or repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "remote", "-v"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    seen: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        if "(push)" in line and name in seen:
            continue
        if name not in seen:
            seen[name] = url
    return [(name, seen[name]) for name in seen]


def advertised_head(root: Path | None = None) -> str:
    checkout = root or repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def local_primary_ui() -> str:
    """This engine's product UI as a hostname URL. Never loopback.

    Default port is the devenv Vite surface (:5173). Override with
    AEGIR_PRIMARY_UI or AEGIR_UI_BIND. Browser waffle rebases this-host
    URLs from the request Host when the client arrived on a LAN IP.
    """
    raw = (os.environ.get("AEGIR_PRIMARY_UI") or os.environ.get("AEGIR_UI_URL") or "").strip()
    if raw:
        rewritten = rewrite_public_url(raw)
        parsed = urlparse(rewritten)
        if parsed.hostname and not is_loopback_host(parsed.hostname):
            return rewritten
    host = advertise_host()
    if not host:
        return ""
    port = "5173"
    bind = (os.environ.get("AEGIR_UI_BIND") or "").strip()
    if bind:
        maybe = bind.rsplit(":", 1)[-1]
        if maybe.isdigit():
            port = maybe
    return f"http://{host}:{port}"


def local_surfaces() -> list[zpb.Surface]:
    url = local_primary_ui()
    if not url:
        return []
    return [zpb.Surface(kind="primary", url=url, healthy=True)]


def surface_title(project: str) -> str:
    raw = (project or "").strip()
    known = {
        "gaius": "Gaius",
        "signals": "Signals",
        "aegir": "Ægir",
        "atelier": "Atelier",
        "metabase": "Metabase",
        "synth": "Synth",
        "hermes": "Hermes",
    }
    if raw.lower() in known:
        return known[raw.lower()]
    if not raw:
        return "Peer"
    return raw.replace("-", " ").replace("_", " ").title()


def peer_contract_path() -> Path | None:
    for key in ("AEGIR_PEER_CONTRACT", "SIGNALS_PEER_CONTRACT"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_file() else None
    if _LAB_CONTRACT.is_file():
        return _LAB_CONTRACT
    return None


def configured_peers(contract: Path | None = None) -> list[tuple[str, str]]:
    """Lattice Engine targets from peer-contract. Skip self. Empty is honest."""
    path = contract if contract is not None else peer_contract_path()
    if path is None or not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lattice = doc.get("engine_grpc_lattice") or {}
    host = (os.environ.get("AEGIR_LATTICE_HOST") or os.environ.get("SIGNALS_LATTICE_HOST")
            or advertise_host()).strip()
    if ":" in host and not host.startswith("["):
        host = host.rsplit(":", 1)[0]
    if not host or is_loopback_host(host):
        return []
    out: list[tuple[str, str]] = []
    for name, port in lattice.items():
        if name in LATTICE_SKIP or name == PROJECT:
            continue
        if not isinstance(port, int):
            continue
        out.append((str(name), f"{host}:{port}"))
    return out


def directory_seeds() -> list[tuple[str, str]]:
    """Engine targets to probe. Not a UI roster."""
    hub = (os.environ.get("SIGNALS_ENGINE_TARGET") or "").strip()
    if not hub:
        return []
    return [("", hub.replace("grpc://", ""))]


def clamp_announce_ttl(raw: int) -> int:
    if raw <= 0:
        return DEFAULT_ANNOUNCE_TTL_S
    return max(MIN_ANNOUNCE_TTL_S, min(MAX_ANNOUNCE_TTL_S, int(raw)))


def parse_engine_target(raw: str) -> str:
    """Lattice Engine host:port, or empty if the string is not a target."""
    addr = (raw or "").replace("grpc://", "").strip()
    if not addr:
        return ""
    host, sep, port = addr.rpartition(":")
    if not sep or not host or not port.isdigit():
        return ""
    return addr


def remember_announce(
    *,
    project: str,
    engine_target: str,
    ttl_seconds: int = 0,
) -> tuple[bool, int, str]:
    """Record a PeerAnnounce. Process-local, TTL'd — not a lineage store."""
    pid = (project or "").strip()
    if not pid:
        return False, 0, "empty project"
    if pid.lower() == PROJECT:
        return False, 0, "self-announce"
    target = parse_engine_target(engine_target)
    if not target:
        return False, 0, "engine_target must be host:port"
    ttl = clamp_announce_ttl(ttl_seconds)
    with _ANNOUNCE_LOCK:
        _ANNOUNCED[pid.lower()] = AnnouncedPeer(
            project=pid,
            target=target,
            expires_at=time.monotonic() + ttl,
        )
    return True, ttl, ""


def announced_peers() -> list[tuple[str, str]]:
    """Live Announce roster. Expired rows are dropped here (honest miss)."""
    now = time.monotonic()
    with _ANNOUNCE_LOCK:
        for key in [k for k, row in _ANNOUNCED.items() if row.expires_at <= now]:
            del _ANNOUNCED[key]
        return [(row.project, row.target) for row in _ANNOUNCED.values()]


def reset_announced() -> None:
    with _ANNOUNCE_LOCK:
        _ANNOUNCED.clear()


# ── Source posture (kind=SOURCE_POSTURE): what code this peer is running ─────
# specification/protocol/source_posture.md. Every field is best-effort and
# honest: what we cannot read stays empty/zero ("not reported"), never invented.
# `head` is read live; `running_sha` is stamped once at engine start so a
# long-lived process reveals when its checkout moved under it.

_RUNNING_SHA: str | None = None


def stamp_running_sha(root: Path | None = None) -> str:
    """Record the commit the live process started from (call once in serve())."""
    global _RUNNING_SHA
    _RUNNING_SHA = advertised_head(root)
    return _RUNNING_SHA


def running_sha() -> str:
    return _RUNNING_SHA or ""


def _git(root: Path, *args: str) -> str:
    """Best-effort ``git -C root ...`` → stripped stdout, or "" on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def working_tree_dirty(root: Path | None = None) -> bool:
    """TRACKED modifications only — untracked scratch/notes do not change what runs."""
    return bool(_git(Path(root or repo_root()), "status", "--porcelain", "--untracked-files=no"))


def current_branch(root: Path | None = None) -> str:
    b = _git(Path(root or repo_root()), "rev-parse", "--abbrev-ref", "HEAD")
    return "" if b == "HEAD" else b  # "" == detached


def upstream_ahead_behind(root: Path | None = None) -> tuple[str, int, int]:
    """(tracking ref, ahead, behind); ("", 0, 0) when there is no upstream."""
    checkout = Path(root or repo_root())
    up = _git(checkout, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not up:
        return "", 0, 0
    parts = _git(checkout, "rev-list", "--left-right", "--count", f"{up}...HEAD").split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return up, int(parts[1]), int(parts[0])
    return up, 0, 0


def parse_submodule_status_line(line: str) -> tuple[str, str, str] | None:
    """One ``git submodule status`` line → (prefix, checked_out_sha, path).

    prefix: " " in sync · "+" moved off the pin · "-" uninitialized · "U" conflict.
    The in-sync leading space may be stripped by callers, so a line that starts
    with a hex digit is in sync.
    """
    raw = line.rstrip("\n")
    if not raw.strip():
        return None
    if raw[0] in "+-U":
        prefix, body = raw[0], raw[1:].strip()
    else:
        prefix, body = " ", raw.strip()
    parts = body.split()
    if len(parts) < 2 or not all(c in "0123456789abcdef" for c in parts[0]):
        return None
    return prefix, parts[0], parts[1]


def submodule_postures(root: Path | None = None) -> list[dict[str, object]]:
    """Per-submodule superproject gitlink vs on-disk sha, plus tracked dirtiness."""
    checkout = Path(root or repo_root())
    out: list[dict[str, object]] = []
    for raw in _git(checkout, "submodule", "status").splitlines():
        parsed = parse_submodule_status_line(raw)
        if parsed is None:
            continue
        prefix, checked_out, path = parsed
        pinned = ""
        ls = _git(checkout, "ls-tree", "HEAD", path).split()
        if len(ls) >= 3 and ls[1] == "commit":
            pinned = ls[2]
        dirty = False
        if prefix != "-":  # initialized → we can look inside
            dirty = bool(_git(checkout / path, "status", "--porcelain", "--untracked-files=no"))
        out.append({
            "path": path, "name": path,
            "pinned_sha": pinned, "checked_out_sha": checked_out, "dirty": dirty,
        })
    return out


def migration_postures(root: Path | None = None) -> list[dict[str, object]]:
    """dbmate-compatible posture for ``migrations/*.sql`` vs ``schema_migrations``.

    Versions are file stems (what ``aegir.db.bootstrap`` records). If the applied
    set cannot be read within a few seconds the list is empty — not reported, not
    "clean". A partial tracking table is itself posture and is reported as such.
    """
    checkout = Path(root or repo_root())
    mig_dir = checkout / "migrations"
    if not mig_dir.is_dir():
        return []
    present = [f.stem for f in sorted(mig_dir.glob("*.sql"))]
    if not present:
        return []
    try:
        from sqlalchemy import create_engine, text

        from aegir.config import load_config

        engine = create_engine(load_config().db.url, connect_args={"connect_timeout": 5})
        try:
            with engine.connect() as conn:
                applied = {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}
        finally:
            engine.dispose()
    except Exception:
        return []
    unapplied = [v for v in present if v not in applied]
    current = max(applied) if applied else ""
    return [{"source": "dbmate", "current": current, "unapplied": unapplied}]


def build_source_posture(root: Path | None = None, *, project: str = PROJECT) -> zpb.SourcePosture:
    checkout = Path(root or repo_root())
    up, ahead, behind = upstream_ahead_behind(checkout)
    posture = zpb.SourcePosture(
        project=project,
        checkout=str(checkout),
        branch=current_branch(checkout),
        head=advertised_head(checkout),
        running_sha=running_sha(),
        dirty=working_tree_dirty(checkout),
        upstream=up,
        ahead=ahead,
        behind=behind,
    )
    for s in submodule_postures(checkout):
        posture.submodules.add(
            path=str(s["path"]), name=str(s["name"]),
            pinned_sha=str(s["pinned_sha"]), checked_out_sha=str(s["checked_out_sha"]),
            dirty=bool(s["dirty"]),
        )
    for m in migration_postures(checkout):
        posture.migrations.add(
            source=str(m["source"]), current=str(m["current"]),
            unapplied=[str(x) for x in m["unapplied"]],  # type: ignore[union-attr]
        )
    return posture


def local_response(
    kind: int,
    *,
    root: Path | None = None,
    contract: Path | None = None,
) -> zpb.ServerQueryResponse:
    """Answer ServerQuery. Unknown kind → project only (honest empty payload)."""
    resp = zpb.ServerQueryResponse(project=PROJECT)
    if kind in (
        zpb.SERVER_QUERY_KIND_UNSPECIFIED,
        zpb.SERVER_QUERY_KIND_REMOTES,
    ):
        resp.remotes.extend(
            zpb.GitRemote(name=n, url=u) for n, u in list_named_remotes(root)
        )
        resp.head = advertised_head(root)
    if kind == zpb.SERVER_QUERY_KIND_PEERS:
        seen: set[str] = set()
        for pid, tgt in (*configured_peers(contract), *announced_peers()):
            key = (pid or "").strip().lower() or tgt
            if key in seen or key == PROJECT:
                continue
            seen.add(key)
            resp.peers.append(zpb.PeerHint(project=pid, target=tgt))
    if kind == zpb.SERVER_QUERY_KIND_SURFACES:
        resp.surfaces.extend(local_surfaces())
    if kind == zpb.SERVER_QUERY_KIND_QUEUES:
        resp.queues.extend(declared_queues())
    if kind == zpb.SERVER_QUERY_KIND_WORKLOADS:
        resp.workloads.extend(declared_workloads())
    if kind == zpb.SERVER_QUERY_KIND_SOURCE_POSTURE:
        resp.posture.CopyFrom(build_source_posture(root))
    return resp


def declared_queues() -> list:
    """Declared leaf shape (QueueHint). Occupancy-over-time is RequestQueueShare."""
    from aegir.engine.queue_share import HEAVY, LIGHT, MEDIUM

    return [
        zpb.QueueHint(
            path=LIGHT.queue,
            resource_class=LIGHT.name,
            gpu_guarantee=1,
            gpu_max=2,
            max_applications=LIGHT.max_applications,
            preemption_delay="5s",
            role="light",
            examples="",
        ),
        zpb.QueueHint(
            path=MEDIUM.queue,
            resource_class=MEDIUM.name,
            gpu_guarantee=2,
            gpu_max=2,
            max_applications=MEDIUM.max_applications,
            preemption_delay="5s",
            role="medium",
            examples="",
        ),
        zpb.QueueHint(
            path=HEAVY.queue,
            resource_class=HEAVY.name,
            gpu_guarantee=HEAVY.gpu_tokens,
            gpu_max=HEAVY.gpu_tokens,
            max_applications=HEAVY.max_applications,
            preemption_policy="fence",
            role="heavy",
            examples="",
        ),
    ]


def _resource_class_enum(gpu_tokens: int) -> int:
    """Physical gpu_tokens → ResourceClass need (former 'extract' folds into LIGHT)."""
    if gpu_tokens <= 0:
        return zpb.RESOURCE_CLASS_COMPUTE
    if gpu_tokens == 1:
        return zpb.RESOURCE_CLASS_LIGHT
    if gpu_tokens == 2:
        return zpb.RESOURCE_CLASS_MEDIUM
    return zpb.RESOURCE_CLASS_HEAVY


def declared_workloads(peer: str = "aegir") -> list:
    """This peer's WorkloadOffers: NONE — Ægir hosts no model (2026-09-08).

    `instruct` / `thinking` are served by the peer that hosts Qwen3.8-27B (Gaius) and forwarded by
    this engine (`forwarder.py`); an engine that does not host a model MUST NOT advertise the
    capability (capabilities.md §Operating profiles). Ægir's own GPU workloads (training / eval
    windows) will arrive as workload-catalogue entries, not as model offers. Empty is honest.
    """
    return []


def primary_ui_of(status: zpb.StatusResponse) -> str:
    for surf in status.surfaces:
        if (surf.kind or "primary") == "primary" and (surf.url or "").strip():
            return surf.url.strip()
    return ""


def status_peer(target: str, timeout: float = 4.0) -> zpb.StatusResponse | None:
    import grpc
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = grpc.insecure_channel(addr)
    try:
        stub = zpb_grpc.EngineStub(channel)
        return stub.Status(zpb.StatusRequest(), timeout=timeout)
    except grpc.RpcError as e:
        logger.info("Status failed at %s: %s", addr, e.code())
        return None
    finally:
        channel.close()


def query_peer(
    target: str,
    *,
    kind: int = zpb.SERVER_QUERY_KIND_REMOTES,
    timeout: float = 10.0,
) -> zpb.ServerQueryResponse | None:
    """Ask a lattice peer ServerQuery. UNIMPLEMENTED → None."""
    import grpc
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpb_grpc

    addr = target.replace("grpc://", "").strip()
    channel = grpc.insecure_channel(addr)
    try:
        stub = zpb_grpc.EngineStub(channel)
        return stub.ServerQuery(
            zpb.ServerQueryRequest(kind=kind, origin_project=PROJECT),
            timeout=timeout,
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            logger.info("ServerQuery UNIMPLEMENTED at %s — peer has not adopted S2S yet", addr)
            return None
        logger.warning("ServerQuery failed at %s: %s %s", addr, e.code(), e.details())
        return None
    finally:
        channel.close()


def _canonicalize_same_box(url: str, reached_via: str) -> str:
    """Rewrite a same-box peer's advertised URL host to our canonical host.

    When we reached the peer via loopback or our own advertise host, it
    lives on this machine — whatever host it advertises (WAN reverse-DNS
    like customer.*.isp.starlink.com included) refers to this box and is
    canonicalized so links resolve on any network path (LAN, WARP, road).
    A peer reached at a genuinely different host keeps its own identity.
    """
    host = advertise_host()
    if not url or not host:
        return url
    via_host = (reached_via or "").rsplit(":", 1)[0]
    if via_host and not is_loopback_host(via_host) and via_host != host:
        return url
    parsed = urlparse(url)
    if not parsed.hostname or parsed.hostname == host:
        return url
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunparse(
        (parsed.scheme or "http", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _canonical_target(addr: str) -> str:
    """Loopback peer targets are same-box; display the canonical host."""
    host = advertise_host()
    if not host or not addr:
        return addr
    tgt_host, sep, port = addr.rpartition(":")
    if sep and is_loopback_host(tgt_host):
        return f"{host}:{port}"
    return addr


def _url_is_loopback(url: str) -> bool:
    return is_loopback_host(urlparse(url).hostname or "")


def local_engine_targets() -> list[tuple[str, str]]:
    """This engine's lattice listen addresses. Gateway collectors have no
    in-process Announce roster — they must ServerQuery PEERS on :50151.
    """
    port = "50151"
    out: list[tuple[str, str]] = [(PROJECT, f"127.0.0.1:{port}")]
    host = advertise_host()
    if host and not is_loopback_host(host):
        out.append((PROJECT, f"{host}:{port}"))
    return out


def collect_peer_surfaces(*, skip_project: str = PROJECT) -> list[dict[str, str]]:
    """S2S waffle roster: self + PEERS + Announce + Status.surfaces.

    Only advertised primary UIs. Foreign hosts keep their own identity;
    same-box loopback is canonicalized onto advertise_host().
    """
    queue: list[tuple[str, str]] = list(configured_peers())
    queue.extend(announced_peers())
    queue.extend(directory_seeds())
    queue.extend(local_engine_targets())
    seen_addr: set[str] = set()
    by_project: dict[str, dict[str, str]] = {}
    self_ui = local_primary_ui()
    if self_ui:
        host = advertise_host() or "localhost"
        by_project[PROJECT] = {
            "project": PROJECT,
            "title": surface_title(PROJECT),
            "engine_target": f"{host}:50151",
            "primary_ui": self_ui,
        }
    while queue:
        hint_project, target = queue.pop(0)
        addr = target.replace("grpc://", "").strip()
        if not addr or addr in seen_addr:
            continue
        seen_addr.add(addr)
        status = status_peer(addr)
        if status is None:
            continue
        project = (status.project or hint_project or "").strip()
        ui = primary_ui_of(status)
        key = (project or addr).lower()
        # Skip adding our own Status row (already seeded) but still walk
        # PEERS — Announce lives on the engine process, not the gateway.
        if ui and project != skip_project:
            prev = by_project.get(key)
            if prev is None or _url_is_loopback(prev.get("primary_ui") or ""):
                by_project[key] = {
                    "project": project or addr,
                    "title": surface_title(project or ""),
                    "engine_target": _canonical_target(addr),
                    "primary_ui": _canonicalize_same_box(ui, addr),
                }
        peers = query_peer(addr, kind=zpb.SERVER_QUERY_KIND_PEERS)
        if peers is None:
            continue
        for peer in peers.peers:
            tgt = (peer.target or "").strip()
            if tgt:
                queue.append((peer.project or "", tgt))
    return sorted(by_project.values(), key=lambda row: row["project"])
