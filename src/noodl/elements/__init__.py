"""Typed branch constitutive laws: the ConstitutiveLaw base class and the built-in laws.

Laws are named for the physics they express. Where a law is a transcription of a specific
tool's equations, the class docstring says so (for example ``RegularizedPowerLaw``,
``SplineFlowTable`` and the door elements follow the Modelica Buildings Library, and
``PowerLaw``/``Quadratic``/``Duct``/``FanCurve`` follow CONTAM's element library).
"""

from noodl.elements.air_duct import AirDuct, DuctTerminal, air_viscosity
from noodl.elements.base import ConstitutiveLaw, ImplicitLaw, LawContext
from noodl.elements.conductance import Conductance
from noodl.elements.curve import CubicSplineFlow, SplineCurve
from noodl.elements.damper import Damper
from noodl.elements.door import (
    OpenDoor,
    OperableDoor,
    open_door_pair,
    operable_door_pair,
)
from noodl.elements.door_discretized import (
    DoorCompartment,
    DoorCompartmentHead,
    DoorPortStream,
    OperableDoorCompartment,
    discretized_door,
    discretized_operable_door,
)
from noodl.elements.duct import Duct
from noodl.elements.fan import AffinityFan, FanCurve
from noodl.elements.fixed import FixedFlow, FixedVolumeFlow, PrescribedFlow
from noodl.elements.media import AirMedium, medium
from noodl.elements.powerlaw import Orifice, PowerLaw
from noodl.elements.powerlaw_regularized import (
    RegularizedPowerLaw,
    effective_leakage_area,
    power_law_coefficient,
    power_law_from_point,
    power_law_from_points,
    regularized_orifice,
)
from noodl.elements.quadratic import Quadratic
from noodl.elements.scaled import Scaled
from noodl.elements.table import SplineFlowTable
from noodl.elements.upstream import UpstreamDensityPowerLaw

__all__ = [
    "AffinityFan",
    "AirDuct",
    "AirMedium",
    "Conductance",
    "CubicSplineFlow",
    "Damper",
    "DoorCompartment",
    "DoorCompartmentHead",
    "DoorPortStream",
    "Duct",
    "ConstitutiveLaw",
    "DuctTerminal",
    "FanCurve",
    "FixedFlow",
    "FixedVolumeFlow",
    "ImplicitLaw",
    "LawContext",
    "OpenDoor",
    "OperableDoor",
    "OperableDoorCompartment",
    "Orifice",
    "PowerLaw",
    "PrescribedFlow",
    "Quadratic",
    "RegularizedPowerLaw",
    "Scaled",
    "SplineCurve",
    "SplineFlowTable",
    "UpstreamDensityPowerLaw",
    "air_viscosity",
    "discretized_door",
    "discretized_operable_door",
    "effective_leakage_area",
    "medium",
    "open_door_pair",
    "operable_door_pair",
    "power_law_coefficient",
    "power_law_from_point",
    "power_law_from_points",
    "regularized_orifice",
]
