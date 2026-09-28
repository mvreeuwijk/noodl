"""Typed branch constitutive laws: the Element base class and the built-in laws.

Elements are grouped by the physics they express. Where a law is a transcription of a
specific tool's equations, the class docstring says so (for example the ``MBL*`` classes follow
the Modelica Buildings Library, and ``PowerLaw``/``Quadratic``/``Duct``/``FanCurve`` follow
CONTAM's element library).
"""

from noodl.elements.base import Element
from noodl.elements.conductance import Conductance
from noodl.elements.damper import Damper
from noodl.elements.door import (
    MBLDoorOpen,
    MBLDoorOperable,
    mbl_door_pair,
    mbl_operable_door_pair,
)
from noodl.elements.door_discretized import (
    DoorCompartmentHead,
    MBLDoorCompartment,
    MBLDoorCompartmentOperable,
    MBLDoorPortStream,
    mbl_discretized_door,
    mbl_discretized_operable_door,
)
from noodl.elements.duct import Duct
from noodl.elements.fan import FanCurve
from noodl.elements.fixed import FixedFlow
from noodl.elements.media import MBLMedium, medium
from noodl.elements.powerlaw import Orifice, PowerLaw
from noodl.elements.powerlaw_mbl import (
    MBLPowerLaw,
    mbl_coefficient,
    mbl_ela,
    mbl_orifice,
    mbl_point,
    mbl_points,
)
from noodl.elements.quadratic import Quadratic
from noodl.elements.table import MBLTable
from noodl.elements.upstream import UpstreamDensityPowerLaw

__all__ = [
    "Conductance",
    "Damper",
    "DoorCompartmentHead",
    "Duct",
    "Element",
    "FanCurve",
    "FixedFlow",
    "MBLDoorCompartment",
    "MBLDoorCompartmentOperable",
    "MBLDoorPortStream",
    "MBLDoorOpen",
    "MBLDoorOperable",
    "MBLMedium",
    "MBLPowerLaw",
    "MBLTable",
    "Orifice",
    "PowerLaw",
    "Quadratic",
    "UpstreamDensityPowerLaw",
    "mbl_coefficient",
    "mbl_discretized_door",
    "mbl_discretized_operable_door",
    "mbl_door_pair",
    "mbl_ela",
    "mbl_operable_door_pair",
    "mbl_orifice",
    "mbl_point",
    "mbl_points",
    "medium",
]
