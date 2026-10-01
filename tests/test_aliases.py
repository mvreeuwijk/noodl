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
from noodl.apps import street_aq, water
from noodl.apps.street_aq import canyon as street_canyon
from noodl.apps.street_aq import chemistry as street_chemistry
from noodl.apps.street_aq import closures as street_closures
from noodl.apps.street_aq import network as street_network
from noodl.apps.street_aq import plume as street_plume
from noodl.apps.street_aq import routing as street_routing
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
    (street_canyon, "KAPPA_MUNICH", "KAPPA_041"),
    (street_canyon, "GAMMA_E", "EULER_GAMMA_TRUNCATED"),
    (street_canyon, "SIRANE_EXCHANGE", "EXCHANGE_SIGMA_W_RATIO"),
    (street_canyon, "SCHULTE_BETA", "ASPECT_RATIO_EXCHANGE_BETA"),
    (street_canyon, "soulhac_residual", "bessel_shape_residual"),
    (street_canyon, "soulhac_shape", "bessel_shape_parameter"),
    (street_canyon, "macdonald_profile", "canopy_displacement_roughness"),
    (street_routing, "MAX_SIGMA_THETA_SIRANE", "MAX_SIGMA_THETA_EXACT"),
    (street_routing, "SIRANE_WINDOW", "EXACT_GAUSSIAN_WINDOW"),
    (street_routing, "sigma_theta_munich", "sigma_theta_turbulence_intensity"),
    (street_routing, "n_theta_munich", "n_theta_rectangle_rule"),
    (street_routing, "sirane_direction_samples", "exact_gaussian_direction_samples"),
    (street_plume, "SEUIL_GAUSS", "GAUSS_CUTOFF_SIGMAS"),
    (street_network, "munich_idealised", "twelve_street_grid"),
    (street_chemistry, "SIRANE_K3_PREFACTOR", "K_NO_O3_SOULHAC2011_PREFACTOR"),
    (street_chemistry, "SIRANE_K3_ACTIVATION", "K_NO_O3_SOULHAC2011_ACTIVATION"),
    (street_chemistry, "SIRANE_K_FLOOR_PPB", "PHOTOSTATIONARY_FLOOR_PPB"),
    (street_chemistry, "j_no2_sirane", "j_no2_elevation_cloud"),
    (street_chemistry, "k_no_o3_sirane", "k_no_o3_soulhac2011"),
]

STREET_PACKAGE_ALIASES = [
    ("KAPPA_MUNICH", "KAPPA_041"),
    ("GAMMA_E", "EULER_GAMMA_TRUNCATED"),
    ("SIRANE_EXCHANGE", "EXCHANGE_SIGMA_W_RATIO"),
    ("SCHULTE_BETA", "ASPECT_RATIO_EXCHANGE_BETA"),
    ("SEUIL_GAUSS", "GAUSS_CUTOFF_SIGMAS"),
    ("soulhac_shape", "bessel_shape_parameter"),
    ("macdonald_profile", "canopy_displacement_roughness"),
    ("sigma_theta_munich", "sigma_theta_turbulence_intensity"),
    ("n_theta_munich", "n_theta_rectangle_rule"),
    ("sirane_direction_samples", "exact_gaussian_direction_samples"),
    ("munich_idealised", "twelve_street_grid"),
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


@pytest.mark.parametrize(("old", "new"), STREET_PACKAGE_ALIASES,
                         ids=[old for old, _ in STREET_PACKAGE_ALIASES])
def test_street_alias_is_exported_from_the_package(old, new):
    assert getattr(street_aq, old) is getattr(street_aq, new)
    assert old in street_aq.__all__
    assert new in street_aq.__all__


def test_street_von_karman_constants_are_named_by_value():
    assert street_closures.KAPPA_040 == 0.40
    assert street_closures.KAPPA_041 == 0.41
    assert street_canyon.KAPPA == street_closures.KAPPA_040
    assert street_closures.PRESETS["sirane"]["kappa"] == street_closures.KAPPA_040
    assert street_closures.PRESETS["munich"]["kappa"] == street_closures.KAPPA_041
    assert street_closures.infer_kappa({"canyon_wind": "exponential_profile"}) == 0.41
    assert street_closures.infer_kappa({"canyon_wind": "bessel_profile"}) == 0.40


def test_street_euler_gamma_is_the_truncated_value():
    assert street_canyon.EULER_GAMMA_TRUNCATED == 0.577


def _t(*values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize(("function", "new", "value", "args", "kwargs"), [
    ("canyon_velocity", "canyon_wind", "bessel_profile",
     (_t(20.0), _t(20.0), _t(0.4)), {"u_star": _t(0.5)}),
    ("roof_wind", "roof_wind", "canopy_log_law",
     (_t(0.5), _t(20.0), _t(20.0)), {"h_mean": _t(15.0), "w_mean": _t(20.0)}),
    ("exchange_velocity", "roof_exchange", "aspect_ratio_scaled",
     (_t(0.4), _t(6.9), _t(7.5)), {}),
])
def test_street_form_keyword_is_the_option_keyword(function, new, value, args, kwargs):
    fn = getattr(street_canyon, function)
    expected = fn(*args, **{new: value}, **kwargs)
    with pytest.warns(DeprecationWarning, match=f"'form' is deprecated; use '{new}'"):
        got = fn(*args, form=value, **kwargs)
    assert torch.equal(got, expected)
    with pytest.raises(TypeError, match="give only one"):
        fn(*args, form=value, **{new: value}, **kwargs)


def test_street_model_keyword_is_junction_routing():
    flux_in, flux_out = _t(10.0, 4.0), _t(6.0, 8.0)
    expected = street_routing.routing_matrix(flux_in, flux_out,
                                             junction_routing="non_crossing_streamlines")
    with pytest.warns(DeprecationWarning, match="'model' is deprecated"):
        got = street_routing.routing_matrix(flux_in, flux_out,
                                            model="non_crossing_streamlines")
    assert torch.equal(got, expected)
    with pytest.raises(TypeError, match="junction_routing is required"):
        street_routing.routing_matrix(flux_in, flux_out)


def test_street_scheme_keyword_is_direction_averaging():
    sigma = _t(0.1, 0.2)
    expected = street_routing.direction_offsets("rectangle_rule", sigma)
    keyword = street_routing.direction_offsets(direction_averaging="rectangle_rule",
                                               sigma_theta=sigma)
    with pytest.warns(DeprecationWarning, match="'scheme' is deprecated"):
        old = street_routing.direction_offsets(scheme="rectangle_rule", sigma_theta=sigma)
    for got in (keyword, old):
        assert all(torch.equal(a, b) for a, b in zip(got, expected, strict=True))
    with pytest.raises(TypeError, match="both required"):
        street_routing.direction_offsets("rectangle_rule")
