#!/usr/bin/env python3
"""Reclaim listening ports for a devenv-managed process — the zombie/orphan resilience step.

    python3 bin/reclaim-port.py <process-name> <port> [<port>...]

A devenv process DECLARES its ports; anything else found LISTENing on one is a squatter
(a manually-launched orphan that outlived its session, or a zombie from a previous run)
and gets SIGTERM → grace → SIGKILL, loudly. Without this, process-compose crash-loops
the managed process forever on `[Errno 98] Address already in use` — and a
build-then-serve exec chain (the gateway) re-runs its expensive build on every retry
(measured 2026-08-07: 1,130 restarts over ~1.5 days, each wiping+rebuilding the lineup
KB under the stale orphan actually serving it).

stdlib + /proc only (no ss/lsof dependency — process env is nix-curated). Never kills
self or an ancestor. Exits 0 when every port ends free; 2 if any squatter survived
(the caller then fails on bind exactly as before, but with the cause already named).
"""
from __future__ import annotations

import os
import signal
import sys
import time


def listen_inodes(port: int) -> set[str]:
    """Socket inodes in LISTEN state (st == 0A) on `port`, any local address, v4+v6."""
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                lines = f.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            if int(parts[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(parts[9])
    return inodes


def owners(inodes: set[str]) -> set[int]:
    """Pids whose fd table holds any of these socket inodes (own-user procs only)."""
    targets = {f"socket:[{i}]" for i in inodes}
    pids = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") in targets:
                        pids.add(int(pid))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def ancestors() -> set[int]:
    chain, pid = {os.getpid()}, os.getpid()
    while pid > 1:
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().split(") ")[-1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        chain.add(pid)
    return chain


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:160]
    except OSError:
        return "?"


def main() -> int:
    name, ports = sys.argv[1], [int(p) for p in sys.argv[2:]]
    protected = ancestors()
    failed = False
    for port in ports:
        pids = owners(listen_inodes(port)) - protected
        for pid in sorted(pids):
            print(f"[reclaim:{name}] :{port} held by pid {pid} ({cmdline(pid)}) — terminating", flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                print(f"[reclaim:{name}] cannot signal pid {pid} (other user) — leaving it", flush=True)
                failed = True
                continue
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and owners(listen_inodes(port)) - protected:
            time.sleep(0.2)
        for pid in sorted(owners(listen_inodes(port)) - protected):
            print(f"[reclaim:{name}] pid {pid} ignored SIGTERM — SIGKILL", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.3)
        if leftover := owners(listen_inodes(port)) - protected:
            print(f"[reclaim:{name}] :{port} STILL held by {sorted(leftover)} — bind will fail", flush=True)
            failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
