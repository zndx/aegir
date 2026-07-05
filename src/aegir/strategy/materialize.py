"""Materialize a strategy's qdrant collections FROM its declarative sources (#148 inc-3).

Truth flows repo → runtime: a strategy ref carries the SKOS sources (``lens/*.skos.ttl``) and
the binding (collection names). Materialization writes the sources to a scratch dir, builds the
collections with the EXISTING index builder, then verifies point counts against the ref's
snapshots. Shadow branches thus become runnable with one command:

    uv run --no-sync python -m aegir.strategy.materialize <ref>

Idempotent-ish: collections that already exist with matching point counts are left alone;
``--force`` rebuilds.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from aegir.strategy.manifest import lens_binding, read_component


def materialize(ref: "str | None" = None, *, force: bool = False) -> dict:
    from aegir.ontology import domain_index as DI
    binding = lens_binding(ref)
    url = binding.get("qdrant_url") or DI.DEFAULT_QDRANT_URL
    cl = DI._client(url)
    out = {}
    jobs = (("vocab", binding.get("vocab_collection"), "lens/vocab.skos.ttl",
             "lens/vocab.snapshot.json"),
            ("aiming", binding.get("aiming_collection"), "lens/aiming.skos.ttl",
             "lens/aiming.snapshot.json"))
    for kind, coll, skos_path, snap_path in jobs:
        if not coll:
            out[kind] = "unbound"
            continue
        want_n = None
        try:
            want_n = json.loads(read_component(snap_path, ref))["n"]
        except Exception:  # noqa: BLE001 — no snapshot (fresh shadow): build without count check
            pass
        try:
            have_n = cl.count(coll).count
        except Exception:  # noqa: BLE001
            have_n = None
        if have_n is not None and not force and (want_n is None or have_n == want_n):
            out[kind] = f"{coll}: exists ({have_n} points)"
            continue
        try:
            skos = read_component(skos_path, ref)
        except Exception as e:  # noqa: BLE001
            out[kind] = f"{coll}: NO SOURCE ({skos_path}: {str(e)[:80]})"
            continue
        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
            f.write(skos)
            tmp = f.name
        rep = DI.build_index(vocab=tmp, url=url, collection=coll)
        got = cl.count(coll).count
        status = "OK" if (want_n is None or got == want_n) else f"COUNT MISMATCH want {want_n}"
        out[kind] = f"{coll}: built {got} points ({status})"
        Path(tmp).unlink(missing_ok=True)
        del rep
    return out


if __name__ == "__main__":
    ref = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    res = materialize(ref, force="--force" in sys.argv)
    for k, v in res.items():
        print(f"  {k}: {v}")
    sys.exit(0 if all("NO SOURCE" not in v and "MISMATCH" not in v for v in res.values()) else 1)
