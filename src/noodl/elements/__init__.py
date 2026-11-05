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
    MBLDoorOpen,  # noqa: F401  (alias)
    MBLDoorOperable,  # noqa: F401  (alias)
    OpenDoor,
    OperableDoor,
    mbl_door_pair,  # noqa: F401  (alias)
    mbl_operable_door_pair,  # noqa: F401  (alias)
    open_door_pair,
    operable_door_pair,
)
from noodl.elements.door_discretized import (
    DoorCompartment,
    DoorCompartmentHead,
    DoorPortStream,
    MBLDoorCompartment,  # noqa: F401  (alias)
    MBLDoorCompartmentOperable,  # noqa: F401  (alias)
    MBLDoorPortStream,  # noqa: F401  (alias)
    OperableDoorCompartment,
    discretized_door,
    discretized_operable_door,
    mbl_discretized_door,  # noqa: F401  (alias)
    mbl_discretized_operable_door,  # noqa: F401  (alias)
)
from noodl.elements.duct import Duct
from noodl.elements.fan import AffinityFan, FanCurve
from noodl.elements.fixed import FixedFlow, FixedVolumeFlow, PrescribedFlow
from noodl.elements.media import (
    AirMedium,
    MBLMedium,  # noqa: F401  (alias)
    medium,
)
from noodl.elements.powerlaw import Orifice, PowerLaw
from noodl.elements.powerlaw_regularized import (
    MBLPowerLaw,  # noqa: F401  (alias)
    RegularizedPowerLaw,
    effective_leakage_area,
    mbl_coefficient,  # noqa: F401  (alias)
    mbl_ela,  # noqa: F401  (alias)
    mbl_orifice,  # noqa: F401  (alias)
    mbl_point,  # noqa: F401  (alias)
    mbl_points,  # noqa: F401  (alias)
    power_law_coefficient,
    power_law_from_point,
    power_law_from_points,
    regularized_orifice,
)
from noodl.elements.quadratic import Quadratic
from noodl.elements.scaled import Scaled
from noodl.elements.table import (
    MBLTable,  # noqa: F401  (alias)
    SplineFlowTable,
)
from noodl.elements.upstream import UpstreamDensityPowerLaw

# Reference-model aliases (the imports marked `# noqa: F401  (alias)`) are importable from
# here but kept out of `__all__`: an alias is named in its target's docstring, not catalogued.
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
