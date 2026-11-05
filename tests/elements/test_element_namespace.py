"""Every element, whichever tool its equations come from, is importable from noodl.elements."""

import noodl.elements as elements

NAMES = [
    "Conductance", "Damper", "Duct", "ConstitutiveLaw", "FanCurve", "FixedFlow", "Orifice",
    "PowerLaw", "Quadratic", "UpstreamDensityPowerLaw", "DoorCompartmentHead",
    "DoorCompartment", "OperableDoorCompartment", "OpenDoor", "DoorPortStream", "OperableDoor",
    "AirMedium", "RegularizedPowerLaw", "SplineFlowTable", "medium", "power_law_coefficient",
    "discretized_door", "discretized_operable_door", "open_door_pair", "effective_leakage_area",
    "operable_door_pair", "regularized_orifice", "power_law_from_point",
    "power_law_from_points",
]


def test_every_element_is_exported():
    for name in NAMES:
        assert name in elements.__all__, name
        assert getattr(elements, name) is not None, name
