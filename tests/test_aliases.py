"""The pre-rename names of noodl physics identifiers are aliases of the physical names.

Identifiers say what something does physically; the earlier, model-named identifiers stay
importable. Each alias must be the SAME object as its new name, and each old option value must
give the same result as the new one.
"""

import dataclasses
import importlib
from pathlib import Path

import pytest
import torch

import noodl.couple as couple
import noodl.elements as elements
from noodl.apps import water
from noodl.apps.water import elements as water_elements

DATA = Path(__file__).parent / "data"

OBJECT_ALIASES = [
    # (module, old, new)
    (elements, "MBLPowerLaw", "RegularizedPowerLaw"),
    (elements, "MBLTable", "SplineFlowTable"),
    (elements, "MBLDoorOpen", "OpenDoor"),
    (elements, "MBLDoorOperable", "OperableDoor"),
    (elements, "MBLDoorCompartment", "DoorCompartment"),
    (elements, "MBLDoorCompartmentOperable", "OperableDoorCompartment"),
    (elements, "MBLDoorPortStream", "DoorPortStream"),
    (elements, "MBLMedium", "AirMedium"),
    (elements, "mbl_orifice", "regularized_orifice"),
    (elements, "mbl_ela", "effective_leakage_area"),
    (elements, "mbl_point", "power_law_from_point"),
    (elements, "mbl_points", "power_law_from_points"),
    (elements, "mbl_coefficient", "power_law_coefficient"),
    (elements, "mbl_door_pair", "open_door_pair"),
    (elements, "mbl_operable_door_pair", "operable_door_pair"),
    (elements, "mbl_discretized_door", "discretized_door"),
    (elements, "mbl_discretized_operable_door", "discretized_operable_door"),
    (water, "EpanetDarcyWeisbach", "CompositeDarcyWeisbach"),
    (water, "epanet_friction_factor", "composite_friction_factor"),
    (water_elements, "EPANET_NU", "NU_WATER_REF"),
    (water_elements, "EPANET_A1", "W_RE_4000"),
    (water_elements, "EPANET_A2", "W_RE_2000"),
    (water_elements, "EPANET_A8", "SWAMEE_JAIN_W_COEFF"),
    (water_elements, "EPANET_A9", "SWAMEE_JAIN_LOG_COEFF"),
    (water_elements, "EPANET_AB", "DUNLOP_AB"),
    (water_elements, "EPANET_AC", "DUNLOP_AC"),
    (water_elements, "EPANET_FOOT", "M_PER_FT"),
    (water_elements, "EPANET_G_FT", "G_FT_S2"),
    (water_elements, "EPANET_KM", "MINOR_LOSS_K_FT"),
    (water_elements, "EPANET_VISCOS_FT2", "NU_WATER_FT2"),
    (couple, "STREET_RAD_TO_CONTAM_DEG", "MATH_RAD_TO_COMPASS_DEG"),
    (couple, "CONTAM_DEG_TO_STREET_RAD", "COMPASS_DEG_TO_MATH_RAD"),
]


@pytest.mark.parametrize(("module", "old", "new"), OBJECT_ALIASES,
                         ids=[old for _, old, _ in OBJECT_ALIASES])
def test_alias_is_the_same_object(module, old, new):
    assert getattr(module, old) is getattr(module, new)


@pytest.mark.parametrize("old", [o for m, o, _ in OBJECT_ALIASES if m in (elements, water)])
def test_alias_is_still_exported(old):
    module = elements if hasattr(elements, old) else water
    assert old in module.__all__


@pytest.mark.parametrize(("old", "new", "names"), [
    ("noodl.elements.powerlaw_mbl", "noodl.elements.powerlaw_regularized",
     ["MBLPowerLaw", "RegularizedPowerLaw", "mbl_orifice", "regularized_orifice", "_f64"]),
    ("noodl.apps.sewer.swmm_xsect", "noodl.apps.sewer.xsect_tables",
     ["kinwave_steady", "full_flow", "a_circular", "s_circular", "LCF", "A_CIRC"]),
])
def test_shim_module_reexports(old, new, names):
    shim, module = importlib.import_module(old), importlib.import_module(new)
    for name in names:
        assert getattr(shim, name) is getattr(module, name)


def test_old_conversion_names_give_the_new_conversions():
    theta = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    wd = torch.tensor([0.0, 90.0, 270.0], dtype=torch.float64)
    assert torch.equal(couple.apply_conversion("street_rad_to_contam_deg", theta, {}),
                       couple.apply_conversion("math_rad_to_compass_deg", theta, {}))
    assert torch.equal(couple.apply_conversion("contam_deg_to_street_rad", wd, {}),
                       couple.apply_conversion("compass_deg_to_math_rad", wd, {}))


@pytest.mark.parametrize(("physical", "path"), [
    ("moist_air", "Buildings.Media.Air"),
    ("moist_air_ideal_gas", "Buildings.Media.Specialized.Air.PerfectGas"),
    ("dry_air_ideal_gas", "Modelica.Media.Air.SimpleAir"),
])
def test_medium_physical_key_and_modelica_path_give_the_same_medium(physical, path):
    assert elements.medium(physical) == elements.medium(path)
    assert elements.medium(physical).name == path


def test_friction_epanet_is_friction_composite():
    net = dataclasses.replace(water.twoloop(), headloss="D-W")
    out = []
    for friction in ("epanet", "composite"):
        model, state, drivers = water.build_model(net, friction=friction)
        (pipe,) = [e for e in model.potential["water"]._elements
                   if isinstance(e, water.CompositeDarcyWeisbach)]
        out.append((pipe.nu, pipe.cfs_per_m3s, pipe.scale, model.notes["headloss"]))
    assert out[0] == out[1]


def test_geometry_swmm_is_geometry_tabulated():
    from noodl.apps.sewer import build_model, read_swmm_inp

    results = []
    for geometry in ("swmm", "tabulated"):
        net, _, _ = read_swmm_inp(DATA / "sewer" / "tree_kinwave.inp", geometry=geometry)
        assert net.geometry == "tabulated"
        model, state, drivers = build_model(net, air=False, quality=False)
        results.append(model._apply_closures(state, drivers))
    for key in ("sewer.h", "sewer.v", "sewer.V_wet"):
        assert torch.equal(results[0][key], results[1][key])
