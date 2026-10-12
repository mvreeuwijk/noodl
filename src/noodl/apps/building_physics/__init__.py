"""The building application: zones, walls, heat and species layers, and `build_model`.

A model built ON TOP of the core package. The core supplies every solve; this package
decides which elements, drives, closures and layers a building network has, and in which
units. Nothing under `src/noodl/` outside `apps/` imports from
here.
"""

# `contamx` drives NIST's ContamX through the OPTIONAL `contamxpy` package. Importing it
# here is safe: the module imports `contamxpy` lazily, inside the call, so a noodl
# installed without the `contam` extra still imports this package cleanly and only a call
# to `run_steady`/`run_transient`/`run_project` raises the ImportError that names the
# missing package.
from noodl.apps.building_physics.attributes import EDGE_ATTRIBUTES, NODE_ATTRIBUTES
from noodl.apps.building_physics.components import crack, door, room, shaft, window
from noodl.apps.building_physics.contamx import (
    run_project,
    run_steady,
    run_transient,
)
from noodl.apps.building_physics.ctm import AmbientConcentrations, read_ctm
from noodl.apps.building_physics.elements import (
    add_large_opening,
    mass_orifice,
    orifice_elements_from_edges,
)
from noodl.apps.building_physics.epw import read_epw, write_wth
from noodl.apps.building_physics.filters import (
    ConstantEfficiencyFilter,
    FilterSeries,
    ParticleFilter,
    filter_transmission,
)
from noodl.apps.building_physics.inputs import Input, apply_inputs, bind_inputs, scatter
from noodl.apps.building_physics.modelica import read_modelica
from noodl.apps.building_physics.prj import (
    Project,
    RunControl,
    project_to_model,
    read_prj,
)
from noodl.apps.building_physics.schedules import (
    Calendar,
    DaySchedule,
    Signal,
    TimeSeries,
    WeekSchedule,
)
from noodl.apps.building_physics.simulation import simulate, simulate_project
from noodl.apps.building_physics.sources import (
    BurstSource,
    ConstantSource,
    CutoffSource,
    DecayingSource,
    DepositionRateSink,
    DepositionVelocitySink,
    PeakEmissionSource,
    PowerLawDecaySource,
    PressureDrivenSource,
    SpeciesSources,
    assemble_sources,
    sources_from_project,
)
from noodl.apps.building_physics.species import KineticReaction, Species, reaction_kinetics
from noodl.apps.building_physics.thermal import (
    IdealGasDensity,
    LinearDensity,
    WallMass,
    Zone,
    add_zone,
    build_model,
    initial_drivers,
    initial_state,
    species_layer,
    thermal_layer,
)
from noodl.apps.building_physics.wth import Weather, read_wth

__all__ = [
    "AmbientConcentrations",
    "EDGE_ATTRIBUTES",
    "NODE_ATTRIBUTES",
    "BurstSource",
    "Calendar",
    "ConstantEfficiencyFilter",
    "ConstantSource",
    "CutoffSource",
    "DaySchedule",
    "DecayingSource",
    "DepositionRateSink",
    "DepositionVelocitySink",
    "FilterSeries",
    "IdealGasDensity",
    "Input",
    "KineticReaction",
    "LinearDensity",
    "ParticleFilter",
    "PeakEmissionSource",
    "PowerLawDecaySource",
    "PressureDrivenSource",
    "Project",
    "RunControl",
    "Signal",
    "Species",
    "SpeciesSources",
    "TimeSeries",
    "WallMass",
    "Weather",
    "WeekSchedule",
    "Zone",
    "add_large_opening",
    "add_zone",
    "apply_inputs",
    "assemble_sources",
    "bind_inputs",
    "build_model",
    "crack",
    "door",
    "filter_transmission",
    "initial_drivers",
    "initial_state",
    "mass_orifice",
    "orifice_elements_from_edges",
    "project_to_model",
    "read_ctm",
    "read_epw",
    "read_modelica",
    "read_prj",
    "reaction_kinetics",
    "read_wth",
    "room",
    "run_project",
    "run_steady",
    "run_transient",
    "scatter",
    "shaft",
    "simulate",
    "simulate_project",
    "sources_from_project",
    "species_layer",
    "thermal_layer",
    "window",
    "write_wth",
]
