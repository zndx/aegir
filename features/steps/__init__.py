"""Central step re-exports for behave discovery.

Behave discovers steps from ``features/steps/`` only. Domain step definitions
live in ``<domain>/step_defs/`` directories (not ``steps/``, to avoid behave
auto-discovery executing them before the project is on ``sys.path``).

Pattern borrowed from ``~/local/src/zndx/atelier/features/steps/__init__.py``.
"""

from features.annotation.step_defs.gittables_cta_steps import *  # noqa: F401,F403
from features.annotation.step_defs.context_selection_steps import *  # noqa: F401,F403
from features.annotation.step_defs.sotab_steps import *  # noqa: F401,F403
from features.chunking.step_defs.chunking_steps import *  # noqa: F401,F403
from features.chunking.step_defs.ema_scan_steps import *  # noqa: F401,F403
from features.inference.step_defs.rosa_step_steps import *  # noqa: F401,F403
from features.ded.step_defs.ded_head_steps import *  # noqa: F401,F403
from features.data.step_defs.mmr_cache_steps import *  # noqa: F401,F403
from features.runs.step_defs.run_artifacts_steps import *  # noqa: F401,F403
from features.gateway.step_defs.endpoints_steps import *  # noqa: F401,F403
from features.deploy.step_defs.config_steps import *  # noqa: F401,F403
