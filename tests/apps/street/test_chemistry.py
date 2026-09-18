"""Photostationary chemistry on a street model -- spec section 4.6."""

from __future__ import annotations

import pytest
import torch

from tellegen.apps.street.chemistry import photostationary_for_streets, street_steady
from tellegen.apps.street.network import Street, StreetNetwork, build_street_model
from tellegen.layers.reaction import K_NO_O3, MOLAR_MASS

DT = torch.float64


def _network():
    return StreetNetwork(
        streets=[Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 Street("s2", "b", "c", 100.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0},
        y={"a": 0.0, "b": 0.0, "c": 0.0},
    )


def _drivers(model, *, j=5.0e-3):
    net = model.net
    sources = torch.zeros(net.n, 3, dtype=DT)
    sources[net.node_index("s1"), 0] = 2.0e-3     # NO
    sources[net.node_index("s1"), 1] = 4.0e-4     # NO2
    background = torch.zeros(1, 3, dtype=DT)
    background[0, 2] = 8.0e-8                      # O3 aloft
    return {
        "street.x_boundary": background,
        "street.sources": sources,
        "U_ref": torch.tensor(3.0, dtype=DT),
        "theta_w": torch.tensor(0.0, dtype=DT),
        "h_abl": torch.tensor(800.0, dtype=DT),
        "J_NO2": torch.tensor(j, dtype=DT),
    }


def test_photostationary_for_streets_finds_the_three_columns():
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    assert reaction.columns == (0, 1, 2)
    shuffled = photostationary_for_streets(("o3", "nox", "no", "no2"))
    assert shuffled.columns == (2, 3, 0)
    # 'no' is the first of the three that is missing, so it is the one named.
    with pytest.raises(ValueError, match=r"photostationary_for_streets.*'no'.*'nox'"):
        photostationary_for_streets(("nox",))


def test_model_steady_does_not_apply_the_reaction():
    """Spec section 5 claims `Model.steady` iterates transport AND reaction; it does not
    (`Model._pass` applies reactions only when `dt` is not None, and `Model.steady`'s own
    docstring says so). `street_steady` is what the street application uses instead."""
    sn = _network()
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    model, state, _ = build_street_model(sn, species=("no", "no2", "o3"),
                                         chemistry=reaction, pblh_floor=False)
    drivers = _drivers(model)
    transport_only = model.steady(state, drivers)
    relaxed = reaction.apply(transport_only["street.x"], None, drivers)
    assert float((relaxed - transport_only["street.x"]).abs().max()) > 0.0


def test_street_steady_reaches_the_photostationary_state_on_every_street():
    sn = _network()
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    model, state, _ = build_street_model(sn, species=("no", "no2", "o3"),
                                         chemistry=reaction, pblh_floor=False)
    drivers = _drivers(model)
    out = street_steady(model, state, drivers, reaction=reaction, tol=1e-18, max_iter=200)
    x = out["street.x"]
    masses = torch.tensor([MOLAR_MASS["no"], MOLAR_MASS["no2"], MOLAR_MASS["o3"]],
                          dtype=DT)
    c = x / masses
    k_mol = K_NO_O3 * MOLAR_MASS["o3"]
    residual = 5.0e-3 * c[:, 1] - k_mol * c[:, 0] * c[:, 2]
    assert float((residual / (5.0e-3 * c[:, 1])).abs().max()) < 1e-10
    assert bool((x >= 0).all())


def test_street_steady_without_a_reaction_is_exactly_model_steady():
    sn = _network()
    model, state, _ = build_street_model(sn, species=("nox",), pblh_floor=False)
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    sources[net.node_index("s1")] = 1.0e-3
    drivers = {
        "street.x_boundary": torch.zeros(1, dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(3.0, dtype=DT),
        "theta_w": torch.tensor(0.0, dtype=DT),
        "h_abl": torch.tensor(800.0, dtype=DT),
    }
    torch.testing.assert_close(
        street_steady(model, state, drivers)["street.x"],
        model.steady(state, drivers)["street.x"], rtol=0, atol=0,
    )


def test_street_steady_reports_a_fixed_point_it_cannot_reach():
    sn = _network()
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    model, state, _ = build_street_model(sn, species=("no", "no2", "o3"),
                                         chemistry=reaction, pblh_floor=False)
    # One pass cannot converge: the change from the all-zero initial state to the first
    # solve is the whole solution. (Two passes DO converge on this case -- the second
    # pass's change is exactly zero -- so the budget has to be one.)
    with pytest.raises(RuntimeError, match=r"street_steady.*did not converge within 1"):
        street_steady(model, state, _drivers(model), reaction=reaction, tol=1e-30,
                      max_iter=1)


def test_j_no2_reproduces_munich_s_tabulation_and_interpolates_between_it():
    from tellegen.apps.street.chemistry import (
        J_NO2_CLEAR_SKY,
        J_NO2_ZENITH_DEG,
        j_no2,
    )

    torch.testing.assert_close(
        j_no2(torch.tensor(J_NO2_ZENITH_DEG, dtype=DT)),
        torch.tensor(J_NO2_CLEAR_SKY, dtype=DT), rtol=1e-15, atol=0,
    )
    # Halfway between the 0 and 10 degree entries, and held flat outside the table.
    assert abs(float(j_no2(5.0)) - 0.5 * (9.31026e-3 + 9.21901e-3)) < 1e-15
    # Held flat outside the table. The interpolation is `a + w (b - a)` at w = 0 and 1,
    # which is not bit-identical to the endpoint, so this is at double precision.
    assert abs(float(j_no2(-5.0)) / J_NO2_CLEAR_SKY[0] - 1.0) < 1e-15
    assert abs(float(j_no2(120.0)) / J_NO2_CLEAR_SKY[-1] - 1.0) < 1e-15
    # `Attenuation` is the only modulation MUNICH applies (chem.f:232-234).
    assert abs(float(j_no2(0.0, 0.25)) - 0.25 * 9.31026e-3) < 1e-18
