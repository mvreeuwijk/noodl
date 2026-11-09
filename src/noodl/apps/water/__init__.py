"""A differentiable, pressurised water distribution network with EPANET 2.2 as the reference
implementation."""

from noodl.apps.water.demand import PressureDrivenDemand
from noodl.apps.water.elements import (
    DW_SI,
    EPANET_QCF,
    HW_SI,
    BreakerValve,
    CompositeDarcyWeisbach,
    EpanetDarcyWeisbach,  # noqa: F401  (alias)
    FlowControlledPipe,
    FlowControlledPump,
    G,
    GeneralPurposeValve,
    HazenWilliams,
    MinorLoss,
    PumpCurve,
    composite_friction_factor,
    epanet_friction_factor,  # noqa: F401  (alias)
    three_point_curve,
)
from noodl.apps.water.inp import FLOW_UNITS, read_epanet_inp
from noodl.apps.water.network import (
    Junction,
    Pump,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterOptions,
    WaterPipe,
    build_model,
    initial_drivers,
    initial_state,
    tank_inflow,
    tank_levels,
    twoloop,
    water_event_step,
    water_steady,
    water_step,
)
from noodl.apps.water.report import link_table, pressure_head, to_kilopascal
from noodl.apps.water.tanks import Control, TankLevels

# Reference-model aliases (the imports marked `# noqa: F401  (alias)`) are importable from
# here but kept out of `__all__`: an alias is named in its target's docstring, not catalogued.
__all__ = [
    "DW_SI", "EPANET_QCF", "FLOW_UNITS", "G", "HW_SI", "BreakerValve", "Control",
    "CompositeDarcyWeisbach", "FlowControlledPipe", "FlowControlledPump",
    "GeneralPurposeValve", "HazenWilliams",
    "Junction",
    "MinorLoss", "PressureDrivenDemand", "Pump", "PumpCurve", "Reservoir", "Tank",
    "TankLevels", "Valve", "WaterNetwork", "WaterOptions", "WaterPipe",
    "build_model", "initial_drivers", "initial_state", "link_table", "pressure_head",
    "composite_friction_factor", "read_epanet_inp", "tank_inflow", "tank_levels",
    "three_point_curve",
    "to_kilopascal", "twoloop", "water_event_step", "water_step", "water_steady",
]
