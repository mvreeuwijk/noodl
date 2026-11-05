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

MBL_ALIASES = [
    "MBLDoorCompartment", "MBLDoorCompartmentOperable", "MBLDoorOpen", "MBLDoorPortStream",
    "MBLDoorOperable", "MBLMedium", "MBLPowerLaw", "MBLTable", "mbl_coefficient",
    "mbl_discretized_door", "mbl_discretized_operable_door", "mbl_door_pair", "mbl_ela",
    "mbl_operable_door_pair", "mbl_orifice", "mbl_point", "mbl_points",
]


def test_every_element_is_exported():
    for name in NAMES:
        assert name in elements.__all__, name
        assert getattr(elements, name) is not None, name


def test_the_mbl_names_are_importable_aliases_outside_all():
    for name in MBL_ALIASES:
        assert name not in elements.__all__, name
        assert getattr(elements, name) is not None, name
