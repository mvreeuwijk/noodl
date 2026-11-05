"""The reference-model names of noodl physics identifiers are aliases of the physical names.

Identifiers say what something does physically; the names of the reference model a relation
comes from (MBL, EPANET, SWMM, SIRANE, MUNICH, ...) stay importable, so its source stays
visible. Each alias must be the SAME object as its physical name, and each model-named option
value must give the same result as the physical one.
"""

import dataclasses
from pathlib import Path

import torch

import noodl.elements as elements
from noodl.apps import street_aq, water
from noodl.apps.street_aq import canyon as street_canyon
from noodl.apps.street_aq import chemistry as street_chemistry
from noodl.apps.street_aq import closures as street_closures
from noodl.apps.street_aq import network as street_network
from noodl.apps.street_aq import plume as street_plume
from noodl.apps.street_aq import reactions as street_reactions
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
    (street_closures, "KAPPA_MUNICH", "KAPPA_041"),
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
    (street_chemistry, "k_no_o3_sirane", "k_no_o3_soulhac2011"),
    (street_chemistry, "j_no2_sirane", "j_no2_elevation_cloud"),
    (street_reactions, "k_no_o3_munich", "k_no_o3_jpl2003"),
]

STREET_PACKAGE_ALIASES = [
    "KAPPA_MUNICH", "GAMMA_E", "SIRANE_EXCHANGE", "SCHULTE_BETA", "SEUIL_GAUSS",
    "soulhac_shape", "macdonald_profile", "sigma_theta_munich", "n_theta_munich",
    "sirane_direction_samples", "munich_idealised", "j_no2_sirane", "k_no_o3_sirane",
    "k_no_o3_munich",
]


def test_reference_model_names_are_the_same_objects_as_the_physical_names():
    for module, old, new in OBJECT_ALIASES:
        assert getattr(module, old) is getattr(module, new), old


def test_reference_model_names_are_importable_but_not_catalogued():
    """An alias is importable from its package but kept out of `__all__` (so the catalogue
    lists only the physical name); the physical name's docstring names the alias."""
    for module, old, new in OBJECT_ALIASES:
        assert old not in getattr(module, "__all__", ()), old
        obj = getattr(module, new)
        if callable(obj):
            assert f"Alias: `{old}`" in obj.__doc__, old


def test_street_aliases_are_importable_from_the_package():
    new_names = {old: new for module, old, new in OBJECT_ALIASES
                 if module not in (elements, water)}
    for old in STREET_PACKAGE_ALIASES:
        new = new_names[old]
        assert getattr(street_aq, old) is getattr(street_aq, new)
        assert old not in street_aq.__all__
        assert new in street_aq.__all__


def test_friction_epanet_is_friction_composite():
    net = dataclasses.replace(water.twoloop(), headloss="D-W")
    out = []
    for friction in ("epanet", "composite"):
        model, state, drivers = water.build_model(net, friction=friction)
        (pipe,) = [getattr(e, "pipe", e) for e in model.kirchhoff["water"]._laws
                   if isinstance(getattr(e, "pipe", e), water.CompositeDarcyWeisbach)]
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

