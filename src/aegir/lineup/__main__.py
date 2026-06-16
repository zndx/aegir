"""Executable entry: ``python -m aegir.lineup {build|sync}``."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aegir.lineup",
        description="The lineup's data backend: build/sync the ontology, relational, "
                    "and content Data Products.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="project the Data Products → build/dev KB projection")
    b.add_argument("--kb-dir", default=None, help="override the KB root (else AEGIR_KB_DIR or build/dev)")
    m = sub.add_parser("maintain", help="run a KB upkeep op (the scheduled-task handlers)")
    m.add_argument("op", choices=["upkeep", "snapshot", "reproject"])
    sub.add_parser("install-cron", help="register the upkeep pg_cron jobs (in postgres, targeting aegir)")
    sub.add_parser("sync", help="(stub) federate Data Products across disclosure tiers")
    args = ap.parse_args(argv)

    import os
    if getattr(args, "kb_dir", None):
        os.environ["AEGIR_KB_DIR"] = args.kb_dir

    if args.cmd == "build":
        from aegir.lineup import build
        return build.run(args)
    if args.cmd == "maintain":
        from aegir.lineup import maintain
        print(maintain.run(args.op))
        return 0
    if args.cmd == "install-cron":
        from aegir.config import load_config
        from aegir.lineup import cron
        print(cron.install(load_config().db.url))
        return 0
    if args.cmd == "sync":
        from aegir.lineup import sync
        return sync.run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
