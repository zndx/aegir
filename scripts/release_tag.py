#!/usr/bin/env python3
"""Compute (and optionally create) the sdg-corpora release tag the CERTIFICATE earns.

RH 2026-07-26: a HermiT backstop is not a failed release — it certifies RELEASE-CANDIDATE
status. So the tag is not a judgement call: the certification tier stamped by
`realize_sdg._certification` decides it.

    level=certified    → vX.Y            (un-suffixed; the reasoner covered everything shipped)
    level=rc           → vX.Y-rcNN       (zero-padded, monotonic within the target version)
    level=refused      → no tag          (a sick TBox ships at no level)
    level=uncertified  → no tag          (no verdict at all — budget trip or --skip-hermit)

BOUNDARY (L11, pre-1.0 identity-without-versioning): `-rcNN` is a tag on the corpora
ARTIFACT repo — the thing we SHARE. It is deliberately NOT `owl:versionInfo` on the
ontology and not a version on the Current/lineup surface, which stay unversioned until
the 1.0 publication. Nothing here writes into the ontology.

    uv run --no-sync python scripts/release_tag.py v0.7                     # what would it be
    uv run --no-sync python scripts/release_tag.py v0.7 --tag               # create it
    uv run --no-sync python scripts/release_tag.py v0.7 --certificate <p>   # explicit cert
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPORA = REPO / "corpora"
VERSION_RE = re.compile(r"^v\d+\.\d+$")


def default_certificate() -> "Path | None":
    """Where a SHIPPED graded certificate actually lives in corpora.

    kb-sync copies the run's `certificate.json` into `corpora/ddl-comprehensive/<run>/`
    (content-addressed) — NOT into `ontology/`, which carries the markdown
    HERMIT_CERTIFICATE.md instead. Newest wins when several run dirs are present. The tag
    must be earned by a certificate that actually shipped with the artifact, so a run-local
    certificate under /raid has to be passed explicitly.
    """
    direct = CORPORA / "ontology" / "certificate.json"
    if direct.exists():
        return direct
    cands = sorted(CORPORA.glob("ddl-comprehensive/*/certificate.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

LEVEL_TAGGABLE = {"certified", "rc"}


def _git(*args: str, repo: Path = CORPORA) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(p.stderr or '').strip()}")
    return p.stdout.strip()


def existing_tags(repo: Path = CORPORA) -> "list[str]":
    out = _git("tag", repo=repo)
    return [t for t in out.splitlines() if t.strip()]


def next_rc(target: str, tags: "list[str]") -> str:
    """Next zero-padded rc number within `target` — monotonic, never reused."""
    pat = re.compile(rf"^{re.escape(target)}-rc(\d+)$")
    used = [int(m.group(1)) for t in tags if (m := pat.match(t))]
    return f"{target}-rc{max(used, default=0) + 1:02d}"


def plan_tag(cert: dict, target: str, tags: "list[str]") -> "tuple[str | None, str]":
    """(tag, rationale). tag is None when the certificate earns no release."""
    level = cert.get("level")
    if level is None:
        return None, ("certificate carries no `level` — it predates the certification tiers "
                      "(RH 2026-07-26). Re-run realize so the grade is stamped; an ungraded "
                      "certificate is not evidence for a release.")
    if level not in LEVEL_TAGGABLE:
        residue = cert.get("uncertified_residue") or []
        detail = f" Residue: {'; '.join(residue)}" if residue else ""
        why = {"refused": "HermiT found unsatisfiable classes — a sick TBox ships at no level",
               "uncertified": "no reasoner verdict at all (budget trip or --skip-hermit)"}
        return None, f"level={level}: {why.get(level, 'not releasable')}.{detail}"
    if level == "certified":
        if target in tags:
            return None, (f"level=certified earns {target}, but that tag already exists. "
                          "Pick the next target version.")
        return target, (f"level=certified — the reasoner covered every shipped class and enum "
                        f"member ({cert.get('method')}), so the release needs no rc suffix.")
    tag = next_rc(target, tags)
    residue = cert.get("uncertified_residue") or []
    return tag, (f"level=rc — the reasoner completed but did not establish satisfiability for "
                 f"everything shipped, so this is a candidate, not a release. Uncertified: "
                 f"{'; '.join(residue) if residue else '(residue not itemised)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_version", help="target version, e.g. v0.7 (rc suffixes hang off it)")
    ap.add_argument("--certificate", type=Path, default=None,
                    help="graded certificate.json to read the tier from (default: the one "
                         "shipped in corpora — ontology/, else the newest "
                         "ddl-comprehensive/<run>/)")
    ap.add_argument("--tag", action="store_true",
                    help="actually create the annotated tag (default: print what it would be)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="tag even with uncommitted corpora changes (default: refuse — RH's "
                         "precondition is that changes are properly arranged there first)")
    a = ap.parse_args()

    if not VERSION_RE.match(a.target_version):
        print(f"target_version must look like v0.7, got {a.target_version!r}", file=sys.stderr)
        return 2
    certificate = a.certificate or default_certificate()
    if certificate is None:
        print("no shipped certificate found in corpora (looked for ontology/certificate.json "
              "and ddl-comprehensive/*/certificate.json) — run `just kb-sync` so the graded "
              "certificate ships with the artifact, or pass --certificate explicitly.",
              file=sys.stderr)
        return 2
    if not certificate.exists():
        print(f"no certificate at {certificate} — nothing to grade", file=sys.stderr)
        return 2

    cert = json.loads(certificate.read_text())
    tags = existing_tags()
    tag, why = plan_tag(cert, a.target_version, tags)

    print(f"certificate: {certificate}")
    print(f"  level={cert.get('level')} method={cert.get('method')} "
          f"generated={cert.get('generated_at')}")
    print(f"  scope certified: {cert.get('scope_certified')}")
    print(f"  scope shipped:   {cert.get('scope_shipped')}")
    print(f"rationale: {why}")
    if tag is None:
        print("→ NO TAG", file=sys.stderr)
        return 1
    print(f"→ tag: {tag}")

    if not a.tag:
        print("(dry run — pass --tag to create it)")
        return 0
    dirty = _git("status", "--porcelain")
    if dirty and not a.allow_dirty:
        print("refusing to tag: corpora has uncommitted changes — arrange them first "
              f"(or --allow-dirty).\n{dirty}", file=sys.stderr)
        return 1
    msg = (f"sdg-corpora {tag}\n\ncertification level: {cert.get('level')}\n"
           f"method: {cert.get('method')}\nscope certified: {cert.get('scope_certified')}\n"
           f"scope shipped: {cert.get('scope_shipped')}\n")
    residue = cert.get("uncertified_residue") or []
    if residue:
        msg += "uncertified residue:\n" + "".join(f"  - {r}\n" for r in residue)
    _git("tag", "-a", tag, "-m", msg)
    print(f"created annotated tag {tag} in corpora (push: git -C corpora push origin {tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
