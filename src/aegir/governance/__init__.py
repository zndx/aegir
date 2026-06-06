"""CDP Governance SDK — Atlas and Ranger REST clients for CDP environments.

Provides AtlasClient and RangerClient for direct REST access to Apache
Atlas v2 and Apache Ranger v2 APIs.  GovernanceClient is a convenience
facade exposing both through a single configuration.

Designed as a standalone SDK with only ``requests`` as a dependency,
integrated into Atelier via HOCON config.
"""

from aegir.governance.client import GovernanceClient, ClientConfig
from aegir.governance.atlas import AtlasClient, ClassificationTag, QualifiedName
from aegir.governance.ranger import RangerClient, RangerRole

__all__ = [
    "GovernanceClient", "ClientConfig",
    "AtlasClient", "ClassificationTag", "QualifiedName",
    "RangerClient", "RangerRole",
]
