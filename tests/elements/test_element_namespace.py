"""Every element, whichever tool its equations come from, is importable from noodl.elements."""

import importlib

import pytest

import noodl.elements as elements

NAMES = [
    "Conductance", "Damper", "Duct", "Element", "FanCurve", "FixedFlow", "Orifice", "PowerLaw",
    "Quadratic", "UpstreamDensityPowerLaw",
    "DoorCompartmentHead", "MBLDoorCompartment", "MBLDoorCompartmentOperable", "MBLDoorOpen",
    "MBLDoorPortStream",
    "MBLDoorOperable", "MBLMedium", "MBLPowerLaw", "MBLTable", "medium", "mbl_coefficient",
    "mbl_discretized_door", "mbl_discretized_operable_door", "mbl_door_pair", "mbl_ela",
    "mbl_operable_door_pair", "mbl_orifice", "mbl_point", "mbl_points",
]


@pytest.mark.parametrize("name", NAMES)
def test_element_is_exported(name):
    assert name in elements.__all__
    assert getattr(elements, name) is not None


def test_no_tool_named_subpackage():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("noodl.elements.mbl")
