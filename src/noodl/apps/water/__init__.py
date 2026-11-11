"""A differentiable, pressurised water distribution network with EPANET 2.2 as the reference
implementation."""

from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN, Action, Control, Premise, Rule
from noodl.apps.water.demand import EmitterOutflow, PressureDrivenDemand
from noodl.apps.water.elements import (
    DW_SI,
    HW_SI,
    BreakerValve,
    ChezyManning,
    CompositeDarcyWeisbach,
    ConstantPowerPump,
    EpanetDarcyWeisbach,  # noqa: F401  (alias)
    FlowControlledPipe,
    FlowControlledPump,
    FlowControlValve,
    G,
    GeneralPurposeValve,
    HazenWilliams,
    MinorLoss,
    PiecewisePumpCurve,
    PressureReducingValve,
    PressureSustainingValve,
    PumpCurve,
    ShutLinks,
    ThrottleValve,
    composite_friction_factor,
    epanet_friction_factor,  # noqa: F401  (alias)
    three_point_curve,
)
from noodl.apps.water.energy import PumpEnergy
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    Emitter,
    EnergyOptions,
    Junction,
    Pump,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterOptions,
    WaterPipe,
    apply_rules,
    build_model,
    initial_drivers,
    initial_state,
    link_states,
    power_curve,
    tank_inflow,
    tank_levels,
    twoloop,
    water_steady,
    water_step,
)
from noodl.apps.water.quality import QualityRouter, SegmentPool, quality_step, sort_nodes
from noodl.apps.water.quality_data import QualitySource, WaterQuality
from noodl.apps.water.quality_react import (
    bulk_rate,
    pipe_coefficients,
    tank_coefficients,
    wall_coefficients,
    wall_rate,
)
from noodl.apps.water.report import link_table, pressure_head, to_kilopascal
from noodl.apps.water.rules import evaluate_rules, take_actions
from noodl.apps.water.simulate import WaterResults, pattern_factor, water_run
from noodl.apps.water.tanks import LinkData, TankLevels, TankVolumeCurve
from noodl.apps.water.units import EPANET_QCF, FLOW_UNITS

# Reference-model aliases (the imports marked `# noqa: F401  (alias)`) are importable from
# here but kept out of `__all__`: an alias is named in its target's docstring, not catalogued.
__all__ = [
    "ACTIVE", "CLOSED", "DW_SI", "EPANET_QCF", "FLOW_UNITS", "G", "HW_SI", "OPEN", "Action",
    "BreakerValve", "ChezyManning", "Control", "CompositeDarcyWeisbach", "ConstantPowerPump",
    "Emitter", "EmitterOutflow",
    "EnergyOptions",
    "FlowControlledPipe", "FlowControlledPump", "FlowControlValve",
    "GeneralPurposeValve", "HazenWilliams", "PressureReducingValve",
    "PressureSustainingValve",
    "Junction", "LinkData",
    "MinorLoss", "PiecewisePumpCurve", "Premise", "PressureDrivenDemand", "Pump", "PumpCurve",
    "PumpEnergy", "QualityRouter", "QualitySource",
    "Reservoir", "Rule", "SegmentPool", "ShutLinks",
    "Tank",
    "TankLevels", "TankVolumeCurve", "ThrottleValve", "Valve", "WaterNetwork", "WaterOptions",
    "WaterPipe", "WaterQuality", "WaterResults",
    "apply_rules", "build_model", "bulk_rate", "initial_drivers", "initial_state", "link_states",
    "link_table", "pressure_head",
    "composite_friction_factor", "evaluate_rules", "pattern_factor", "pipe_coefficients",
    "power_curve", "quality_step", "read_epanet_inp", "sort_nodes", "tank_coefficients",
    "tank_inflow", "tank_levels",
    "take_actions", "three_point_curve",
    "to_kilopascal", "twoloop", "wall_coefficients", "wall_rate", "water_run", "water_step",
    "water_steady",
]
