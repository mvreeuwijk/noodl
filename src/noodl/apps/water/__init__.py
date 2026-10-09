"""A differentiable, pressurised water distribution network with EPANET 2.2 as the reference
implementation."""

from noodl.apps.water.demand import PressureDrivenDemand
from noodl.apps.water.elements import (
    DW_SI,
    EPANET_QCF,
    HW_SI,
    CompositeDarcyWeisbach,
    EpanetDarcyWeisbach,
    G,
    HazenWilliams,
    MinorLoss,
    PumpCurve,
    composite_friction_factor,
    epanet_friction_factor,
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
    twoloop,
    water_steady,
)
from noodl.apps.water.report import link_table, pressure_head, to_kilopascal
from noodl.apps.water.tanks import Control, TankLevels

__all__ = [
    "DW_SI", "EPANET_QCF", "FLOW_UNITS", "G", "HW_SI", "Control", "CompositeDarcyWeisbach",
    "HazenWilliams", "Junction",
    "MinorLoss", "PressureDrivenDemand", "Pump", "PumpCurve", "Reservoir", "Tank",
    "TankLevels", "Valve", "WaterNetwork", "WaterOptions", "WaterPipe",
    "build_model", "initial_drivers", "initial_state", "link_table", "pressure_head",
    "composite_friction_factor", "read_epanet_inp", "tank_inflow", "three_point_curve",
    "to_kilopascal", "twoloop", "water_steady",
    # aliases, the pre-rename names
    "EpanetDarcyWeisbach", "epanet_friction_factor",
]
