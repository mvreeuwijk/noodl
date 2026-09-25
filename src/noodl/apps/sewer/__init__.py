"""A differentiable gravity sewer: water, headspace air and sulfide, coupled."""

from noodl.apps.sewer.air import F_AIR_DEFAULT, F_I_DEFAULT, Drag, Headspace
from noodl.apps.sewer.geometry import (
    H_MAX_RATIO,
    air_geometry,
    capacity_flow,
    manning_flow,
    normal_depth,
)
from noodl.apps.sewer.hydraulics import SewerHydraulics
from noodl.apps.sewer.inp import read_swmm_inp
from noodl.apps.sewer.network import (
    Manhole,
    Outfall,
    Pipe,
    SewerNetwork,
    build_model,
    initial_state,
    sewer_steady,
    tree_steady,
)
from noodl.apps.sewer.quality import (
    H2STransfer,
    SulfideGeneration,
    henry_h2s,
    kla_h2s,
)
from noodl.apps.sewer.report import pipe_table, to_mg_per_litre, to_ppm

__all__ = [
    "Drag", "F_AIR_DEFAULT", "F_I_DEFAULT", "H2STransfer", "H_MAX_RATIO", "Headspace",
    "Manhole", "Outfall", "Pipe", "SewerHydraulics", "SewerNetwork", "SulfideGeneration",
    "air_geometry", "build_model", "capacity_flow", "henry_h2s", "initial_state",
    "kla_h2s", "manning_flow", "normal_depth", "pipe_table", "read_swmm_inp", "sewer_steady",
    "to_mg_per_litre", "to_ppm", "tree_steady",
]
