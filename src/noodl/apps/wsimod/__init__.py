"""WSIMOD 0.8.1's daily step, transcribed: a WSIMOD ``config.yml`` as a noodl ``Model``.

``read_wsimod_config`` reads WSIMOD's own model description; ``build_model`` turns it into a
``Model`` whose one closure, ``WsimodDay``, runs WSIMOD's day (orchestration, river order,
``end_timestep``) with WSIMOD's arithmetic, batched and differentiable. ``initial_state`` and
``initial_drivers`` take values by name, as every noodl application does. See
``docs/applications/wsimod.md``.
"""

from noodl.apps.wsimod.build import build_model, initial_drivers, initial_state
from noodl.apps.wsimod.config import WsimodNetwork, read_wsimod_config
from noodl.apps.wsimod.day import DayRecord, WsimodDay
from noodl.apps.wsimod.run import EVENT_NAMES, WsimodMassBalance, WsimodResults, wsimod_run

__all__ = [
    "EVENT_NAMES",
    "DayRecord",
    "WsimodDay",
    "WsimodMassBalance",
    "WsimodNetwork",
    "WsimodResults",
    "build_model",
    "initial_drivers",
    "initial_state",
    "read_wsimod_config",
    "wsimod_run",
]
