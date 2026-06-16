"""aegir.lineup — the lineup's data backend (build + sync of the Data Products).

A core capability, not a script: it MATERIALIZES the ontology / relational / content
Data Products into the gitignored ``build/dev/{current,scratch,archive}`` KB projection
(``build``) and will FEDERATE them across disclosure tiers (``sync``). The React app +
gateway route merely *present* what this package produces and the reasoner structures.

Run: ``python -m aegir.lineup build``  (wired to ``just kb-build``).

Credit: the *lineup* navigation primitive is Ward Cunningham's (federated wiki). We
adopt the primitive only — navigation over a reasoner-structured, machine-curated
substrate — not a wiki (no human authoring, editing, or forking).
"""
from __future__ import annotations

# Submodules (build, sync, notes, sources) are imported on demand to keep package
# import cheap (build pulls in ddl/pyarrow only when invoked).
