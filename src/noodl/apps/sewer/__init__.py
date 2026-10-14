"""A differentiable gravity sewer: water, headspace air and sulfide, coupled."""

from noodl.apps.sewer.air import F_AIR_DEFAULT, F_I_DEFAULT, Drag, Headspace
from noodl.apps.sewer.dynwave import (
    DynamicWaveConduit,
    DynamicWaveHydraulics,
    DynamicWaveOptions,
    JunctionStorage,
    OverflowLoss,
    StructureLink,
    build_dynamic_wave_model,
    simulate_dynamic_wave,
    tidal_stage,
)
from noodl.apps.sewer.geometry import (
    H_MAX_RATIO,
    air_geometry,
    capacity_flow,
    manning_flow,
    normal_depth,
)
from noodl.apps.sewer.hydraulics import SewerHydraulics
from noodl.apps.sewer.inp import read_swmm_inp, unit_factors
from noodl.apps.sewer.network import (
    ROUTING_ALIASES,
    Manhole,
    Orifice,
    Outfall,
    Outlet,
    Pipe,
    Pump,
    SewerNetwork,
    Weir,
    build_model,
    initial_drivers,
    initial_state,
    sewer_steady,
    storage_unit,
    tree_steady,
)
from noodl.apps.sewer.quality import (
    H2STransfer,
    SulfideGeneration,
    henry_h2s,
    kla_h2s,
)
from noodl.apps.sewer.report import pipe_table, to_mg_per_litre, to_ppm
from noodl.apps.sewer.runoff import (
    Adjustments,
    CurveNumber,
    Evaporation,
    GreenAmpt,
    Horton,
    Hydrology,
    RainGage,
    RunoffInflows,
    RunoffModel,
    Subareas,
    Subcatchment,
)
from noodl.apps.sewer.storage import STORAGE_SHAPES, StorageCurves
from noodl.apps.sewer.xsect_spec import Transect, Xsect

__all__ = [
    "ROUTING_ALIASES", "DynamicWaveConduit", "DynamicWaveHydraulics", "DynamicWaveOptions",
    "JunctionStorage", "OverflowLoss", "build_dynamic_wave_model", "simulate_dynamic_wave",
    "tidal_stage", "STORAGE_SHAPES",
    "StorageCurves",
    "storage_unit", "StructureLink", "Orifice", "Weir", "Outlet", "Pump",
    "Drag", "F_AIR_DEFAULT", "F_I_DEFAULT", "H2STransfer", "H_MAX_RATIO", "Headspace",
    "Manhole", "Outfall", "Pipe", "SewerHydraulics", "SewerNetwork", "SulfideGeneration",
    "air_geometry", "build_model", "capacity_flow", "henry_h2s", "initial_drivers", "initial_state",
    "kla_h2s", "manning_flow", "normal_depth", "pipe_table", "read_swmm_inp", "sewer_steady",
    "to_mg_per_litre", "to_ppm", "tree_steady", "unit_factors", "Transect", "Xsect",
    "Adjustments", "CurveNumber", "Evaporation", "GreenAmpt", "Horton", "Hydrology", "RainGage",
    "RunoffInflows", "RunoffModel", "Subareas", "Subcatchment",
]
