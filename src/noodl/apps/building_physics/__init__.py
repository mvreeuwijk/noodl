"""The building application: zones, walls, heat and species layers, and `build_model`.

A model built ON TOP of the core package. The core supplies every solve; this package
decides which elements, drives, closures and layers a building network has, and in which
units (milestone 2 spec 6.4). Nothing under `src/noodl/` outside `apps/` imports from
here.
"""

# `contamx` drives NIST's ContamX through the OPTIONAL `contamxpy` package. Importing it
# here is safe: the module imports `contamxpy` lazily, inside the call, so a noodl
# installed without the `contam` extra still imports this package cleanly and only a call
# to `run_steady`/`run_transient` raises the ImportError that names the missing package.
from noodl.apps.building_physics.contamx import (
    run_steady,
    run_transient,
)
from noodl.apps.building_physics.elements import (
    add_large_opening,
    mass_orifice,
    orifice_elements_from_edges,
)
from noodl.apps.building_physics.epw import read_epw, write_wth
from noodl.apps.building_physics.modelica import read_modelica
from noodl.apps.building_physics.prj import (
    Project,
    project_to_model,
    read_prj,
)
from noodl.apps.building_physics.sources import (
    BurstSource,
    ConstantSource,
    CutoffSource,
    DecayingSource,
    assemble_sources,
    sources_from_project,
)
from noodl.apps.building_physics.thermal import (
    IdealGasDensity,
    LinearDensity,
    WallMass,
    Zone,
    add_zone,
    build_model,
    initial_state,
    species_layer,
    thermal_layer,
)
from noodl.apps.building_physics.wth import Weather, read_wth

__all__ = [
    "BurstSource",
    "ConstantSource",
    "CutoffSource",
    "DecayingSource",
    "IdealGasDensity",
    "LinearDensity",
    "Project",
    "WallMass",
    "Weather",
    "Zone",
    "add_large_opening",
    "add_zone",
    "assemble_sources",
    "build_model",
    "initial_state",
    "mass_orifice",
    "orifice_elements_from_edges",
    "project_to_model",
    "read_epw",
    "read_modelica",
    "read_prj",
    "read_wth",
    "run_steady",
    "run_transient",
    "species_layer",
    "sources_from_project",
    "thermal_layer",
    "write_wth",
]
