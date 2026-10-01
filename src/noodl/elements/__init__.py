"""Typed branch constitutive laws: the Element base class and the built-in laws.

Elements are named for the physics they express. Where a law is a transcription of a specific
tool's equations, the class docstring says so (for example ``RegularizedPowerLaw``,
``SplineFlowTable`` and the door elements follow the Modelica Buildings Library, and
``PowerLaw``/``Quadratic``/``Duct``/``FanCurve`` follow CONTAM's element library). The
pre-rename ``MBL*``/``mbl_*`` names remain importable as aliases.
"""

from noodl.elements.base import Element
from noodl.elements.conductance import Conductance
from noodl.elements.damper import Damper
from noodl.elements.door import (
    MBLDoorOpen,
    MBLDoorOperable,
    OpenDoor,
    OperableDoor,
    mbl_door_pair,
    mbl_operable_door_pair,
    open_door_pair,
    operable_door_pair,
)
from noodl.elements.door_discretized import (
    DoorCompartment,
    DoorCompartmentHead,
    DoorPortStream,
    MBLDoorCompartment,
    MBLDoorCompartmentOperable,
    MBLDoorPortStream,
    OperableDoorCompartment,
    discretized_door,
    discretized_operable_door,
    mbl_discretized_door,
    mbl_discretized_operable_door,
)
from noodl.elements.duct import Duct
from noodl.elements.fan import FanCurve
from noodl.elements.fixed import FixedFlow
from noodl.elements.media import AirMedium, MBLMedium, medium
from noodl.elements.powerlaw import Orifice, PowerLaw
from noodl.elements.powerlaw_regularized import (
    MBLPowerLaw,
    RegularizedPowerLaw,
    effective_leakage_area,
    mbl_coefficient,
    mbl_ela,
    mbl_orifice,
    mbl_point,
    mbl_points,
    power_law_coefficient,
    power_law_from_point,
    power_law_from_points,
    regularized_orifice,
)
from noodl.elements.quadratic import Quadratic
from noodl.elements.table import MBLTable, SplineFlowTable
from noodl.elements.upstream import UpstreamDensityPowerLaw

__all__ = [
    "AirMedium",
    "Conductance",
    "Damper",
    "DoorCompartment",
    "DoorCompartmentHead",
    "DoorPortStream",
    "Duct",
    "Element",
    "FanCurve",
    "FixedFlow",
    "OpenDoor",
    "OperableDoor",
    "OperableDoorCompartment",
    "Orifice",
    "PowerLaw",
    "Quadratic",
    "RegularizedPowerLaw",
    "SplineFlowTable",
    "UpstreamDensityPowerLaw",
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
    # aliases, the pre-rename names
    "MBLDoorCompartment",
    "MBLDoorCompartmentOperable",
    "MBLDoorOpen",
    "MBLDoorOperable",
    "MBLDoorPortStream",
    "MBLMedium",
    "MBLPowerLaw",
    "MBLTable",
    "mbl_coefficient",
    "mbl_discretized_door",
    "mbl_discretized_operable_door",
    "mbl_door_pair",
    "mbl_ela",
    "mbl_operable_door_pair",
    "mbl_orifice",
    "mbl_point",
    "mbl_points",
]
