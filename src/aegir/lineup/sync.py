"""``aegir.lineup sync`` — federate Data Products across disclosure tiers (stub).

Planned second verb of the lineup capability. ``build`` materializes the products
locally (what we KNOW); ``sync`` distributes them across the disclosure tiers:

  • KNOW  → Atelier, via the shared Apache Iceberg / ``hx`` plane — the FULL working
            products, EXCEPT the held-out reference/labels (the evidence boundary that
            keeps Atelier a valid independent gate; narrower than the know/share line).
  • SHARE → the ``corpora`` (sdg-corpora) submodule — the reasoner-gated publish, i.e.
            the know→share promotion (charter review), attribution-clean, public.

Not implemented yet; gated on the lineup navigation proving out first.
"""
from __future__ import annotations


def run(args=None) -> int:
    print(
        "aegir.lineup sync — not implemented yet.\n"
        "Planned: federate the Data Products across disclosure tiers —\n"
        "  • KNOW  → Atelier via shared Iceberg/hx (held-out reference withheld)\n"
        "  • SHARE → sdg-corpora submodule (reasoner-gated publish; the know→share gate)\n"
        "Gated on the lineup navigation proving out first."
    )
    return 0
