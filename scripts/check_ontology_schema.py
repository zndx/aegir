"""Tier-0 mechanical check for the ontology catalog schema.

Loads ``src/aegir/ontology/catalog/examples.json`` (and any other
``catalog/*.json`` files) and validates each row against
``aegir.ontology.schema.CatalogTemplate``. Surfaces:

- JSON parse failures
- Missing or extra top-level keys (``version``, ``templates``)
- Per-row dataclass-construction failures (missing required fields,
  wrong field shapes)
- Slot-type cross-checks: every ``{name:Type}`` placeholder in
  ``manchester_template`` has a matching entry in ``slot_types``
  with consistent ``Type``

Exits 0 on success, 1 on any failure. Intended to run as part of
``just bdd-0``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aegir.ontology.schema import Catalog, CatalogTemplate, load_catalog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "src" / "aegir" / "ontology" / "catalog"

# Match ``{name:Type}`` and ``{name:Type:Bound}`` placeholders.
SLOT_RE = re.compile(r"\{(?P<name>\w+):(?P<type>[\w:]+?)(?::(?P<bound>[\w:]+))?\}")


def check_template(t: CatalogTemplate) -> list[str]:
    """Return a list of error strings for the given template; empty
    list means OK."""
    errors: list[str] = []

    declared_slots = set(t.slot_types.keys())
    found_slots: dict[str, str] = {}

    for m in SLOT_RE.finditer(t.manchester_template):
        name = m.group("name")
        type_ = m.group("type")
        if name in found_slots and found_slots[name] != type_:
            errors.append(
                f"  slot {name!r} appears with conflicting types "
                f"{found_slots[name]!r} and {type_!r}"
            )
        found_slots[name] = type_

    found_set = set(found_slots.keys())

    missing_in_decl = found_set - declared_slots
    if missing_in_decl:
        errors.append(
            f"  slots in template but not in slot_types: "
            f"{sorted(missing_in_decl)}"
        )

    unused_in_template = declared_slots - found_set
    if unused_in_template:
        errors.append(
            f"  slot_types declared but never used in template: "
            f"{sorted(unused_in_template)}"
        )

    for name, declared_type in t.slot_types.items():
        if name in found_slots and found_slots[name] != declared_type:
            errors.append(
                f"  slot {name!r}: declared type {declared_type!r} "
                f"!= type in template {found_slots[name]!r}"
            )

    return errors


def check_catalog_file(path: Path) -> tuple[int, int]:
    """Validate a single catalog JSON. Returns (n_templates, n_errors)."""
    print(f"checking {path.relative_to(REPO_ROOT)}")
    try:
        catalog: Catalog = load_catalog(path)
    except Exception as e:
        print(f"  FAIL: cannot load catalog: {e}")
        return 0, 1

    n_errors = 0
    for t in catalog.templates:
        template_errors = check_template(t)
        if template_errors:
            print(f"  template {t.template_id!r}:")
            for err in template_errors:
                print(err)
            n_errors += len(template_errors)

    return len(catalog.templates), n_errors


def main() -> int:
    # null_stats*.json are R_D normalization artifacts, not catalogs (no
    # `templates` key) — skip them so the glob only validates catalog files.
    catalog_files = [f for f in sorted(CATALOG_DIR.glob("*.json"))
                     if not f.name.startswith("null_stats") and ".candidate." not in f.name]
    if not catalog_files:
        print(f"no catalog files found under {CATALOG_DIR.relative_to(REPO_ROOT)}")
        return 0

    total_templates = 0
    total_errors = 0
    for f in catalog_files:
        n_t, n_e = check_catalog_file(f)
        total_templates += n_t
        total_errors += n_e

    print()
    if total_errors:
        print(f"{total_templates} templates checked, {total_errors} errors")
        return 1
    print(f"{total_templates} templates validated, 0 errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
