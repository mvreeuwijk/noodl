"""Every element, whichever tool its equations come from, is importable from noodl.elements."""

import noodl.elements as elements

NAMES = [
    "Conductance", "Damper", "Duct", "ConstitutiveLaw", "FanCurve", "FixedFlow", "Orifice",
    "PowerLaw",
    "Quadratic", "UpstreamDensityPowerLaw",
    "DoorCompartmentHead", "MBLDoorCompartment", "MBLDoorCompartmentOperable", "MBLDoorOpen",
    "MBLDoorPortStream",
    "MBLDoorOperable", "MBLMedium", "MBLPowerLaw", "MBLTable", "medium", "mbl_coefficient",
    "mbl_discretized_door", "mbl_discretized_operable_door", "mbl_door_pair", "mbl_ela",
    "mbl_operable_door_pair", "mbl_orifice", "mbl_point", "mbl_points",
]


def test_every_element_is_exported():
    for name in NAMES:
        assert name in elements.__all__, name
        assert getattr(elements, name) is not None, name
