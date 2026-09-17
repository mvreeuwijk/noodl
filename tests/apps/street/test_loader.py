"""`read_aqdt` on hand-written AQ_DT products — spec section 6.3.

Nothing here reads the real AQ_DT tree: `tests/data/street/aqdt_fixture.py` writes the same
five products, with the same property and variable names, into `tmp_path`.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tellegen.apps.street.loader import RHO_AIR, read_aqdt
from tellegen.apps.street.network import build_street_model
from tests.data.street.aqdt_fixture import (
    BACKGROUND_CONCENTRATION,
    EMISSION_NORMALIZED,
    EMISSION_TIME_MODULATION,
    WIND_SPEED_MPS,
    build,
)

DT = torch.float64
SELECTED = [0, 1, 3, 4]


def _read(tmp_path, **options):
    stage1, stage2 = build(tmp_path, **{k: v for k, v in options.items()
                                        if k in ("aligned", "kg_per_year")})
    rest = {k: v for k, v in options.items() if k not in ("aligned", "kg_per_year")}
    rest.setdefault("wind_height_m", 30.0)
    rest.setdefault("trust_file_height", True)
    return read_aqdt(stage1, stage2, year=2024, **rest)


def test_only_the_network_transport_features_become_streets(tmp_path):
    data = _read(tmp_path)
    assert [s.name for s in data.net.streets] == [str(i) for i in SELECTED]
    assert data.feature_index == SELECTED
    assert data.osmid == [900001, 900002, 900004, 900005]
    assert "162" not in data.notes["selection"]
    assert data.notes["selection"] == "aq_solver_type == 'network_transport': 4 of 5"


def test_lengths_and_azimuths_come_from_the_node_coordinates(tmp_path):
    data = _read(tmp_path)
    # Feature 0 runs due east from 1001 to 1002, one thousandth of a degree of longitude.
    first = data.net.streets[0]
    expected = 6371000.0 * math.radians(0.001) * math.cos(math.radians(52.1604))
    assert abs(first.length / expected - 1.0) < 1e-3
    azimuth = dict(zip([s.name for s in data.net.streets], data.net.azimuth, strict=True))
    assert abs(azimuth["0"]) < 1e-12                       # due east
    assert abs(azimuth["1"] - 0.5 * math.pi) < 1e-12       # due north
    assert abs(abs(azimuth["4"]) - 0.5 * math.pi) < 1e-12  # due south


def test_widths_heights_and_the_roughness_default(tmp_path):
    data = _read(tmp_path)
    assert [s.width for s in data.net.streets] == [20.0, 15.0, 18.0, 22.0]
    assert [s.height for s in data.net.streets] == [12.0, 10.0, 9.0, 11.0]
    # Feature 3 carries `roughness_m`; the others fall back to the loader's default.
    assert [s.z0_b for s in data.net.streets] == [0.15, 0.15, 0.2, 0.15]


def test_the_background_is_converted_from_a_mass_mixing_ratio(tmp_path):
    data = _read(tmp_path)
    torch.testing.assert_close(
        data.forcing.background,
        torch.tensor(BACKGROUND_CONCENTRATION, dtype=DT) * RHO_AIR,
        rtol=1e-14, atol=0,
    )
    assert "kg/kg" in data.notes["background"] and str(RHO_AIR) in data.notes["background"]


def test_the_wind_height_must_be_stated_and_the_file_s_label_is_not_believed(tmp_path):
    stage1, stage2 = build(tmp_path)
    with pytest.raises(ValueError, match=r"read_aqdt.*30\.0 m.*wind_height_m is 10\.0 m"):
        read_aqdt(stage1, stage2, year=2024)
    data = read_aqdt(stage1, stage2, year=2024, wind_height_m=30.0,
                     trust_file_height=True)
    assert data.forcing.wind_height_m == 30.0
    assert data.forcing.reference_height_m == 30.0
    ten = read_aqdt(stage1, stage2, year=2024, wind_height_m=10.0,
                    trust_file_height=True)
    assert ten.forcing.wind_height_m == 10.0
    assert "30.0 m" in ten.notes["wind_height"]
    assert "10.0 m" in ten.notes["wind_height"]


def test_the_forcing_series_are_read_verbatim(tmp_path):
    data = _read(tmp_path)
    torch.testing.assert_close(data.forcing.u_ref,
                               torch.tensor(WIND_SPEED_MPS, dtype=DT), rtol=0, atol=0)
    assert data.forcing.time_hours.shape == (4,)
    assert data.forcing.h_abl.dtype is torch.float64


def test_emissions_are_matched_by_the_emission_key_when_the_orders_differ(tmp_path):
    data = _read(tmp_path, aligned=False)
    expected = torch.tensor(
        [[m * EMISSION_NORMALIZED[i] for i in SELECTED]
         for m in EMISSION_TIME_MODULATION], dtype=DT,
    )
    torch.testing.assert_close(data.emission, expected, rtol=1e-14, atol=0)
    assert data.notes["alignment"] == "emission rows matched to features by 'emission_key'"


def test_edge_index_alignment_works_when_the_products_agree(tmp_path):
    data = _read(tmp_path, aligned=True, align="edge_index")
    expected = torch.tensor(
        [[m * EMISSION_NORMALIZED[i] for i in SELECTED]
         for m in EMISSION_TIME_MODULATION], dtype=DT,
    )
    torch.testing.assert_close(data.emission, expected, rtol=1e-14, atol=0)


def test_edge_index_alignment_raises_when_the_products_are_out_of_step(tmp_path):
    with pytest.raises(ValueError, match=r"align='edge_index' puts \d+ emission rows"):
        _read(tmp_path, aligned=False, align="edge_index")


def test_kg_per_year_is_refused_when_it_is_not_finite(tmp_path):
    with pytest.raises(ValueError,
                       match=r"edge_emission_rate_nox_kg_per_year is not finite at 5"):
        _read(tmp_path, kg_per_year=False, emissions="kg_per_year")
    data = _read(tmp_path, kg_per_year=True, emissions="kg_per_year")
    assert data.emission.shape == (4, 4)
    assert bool((data.emission > 0).all())
    assert "kg/s" in data.notes["emissions"]


def test_times_selects_a_subset_of_the_forcing_and_the_emissions(tmp_path):
    data = _read(tmp_path, times=slice(1, 3))
    assert data.forcing.u_ref.tolist() == WIND_SPEED_MPS[1:3]
    assert data.emission.shape == (2, 4)
    picked = _read(tmp_path, times=[0, 3])
    assert picked.forcing.u_ref.tolist() == [WIND_SPEED_MPS[0], WIND_SPEED_MPS[3]]


def test_a_bad_option_is_named(tmp_path):
    stage1, stage2 = build(tmp_path)
    with pytest.raises(ValueError, match=r"emissions must be 'normalized'"):
        read_aqdt(stage1, stage2, year=2024, emissions="tonnes",
                  wind_height_m=30.0, trust_file_height=True)
    with pytest.raises(ValueError, match=r"align must be 'emission_key'"):
        read_aqdt(stage1, stage2, year=2024, align="position",
                  wind_height_m=30.0, trust_file_height=True)
    with pytest.raises(ValueError, match=r"no feature.*aq_solver_type == 'hybrid'"):
        read_aqdt(stage1, stage2, year=2024, select="hybrid",
                  wind_height_m=30.0, trust_file_height=True)


def test_the_wind_angle_is_the_direction_the_wind_blows_towards(tmp_path):
    """ERA5's `arctan2(v, u)` is the direction the wind blows TOWARDS, radians CCW from
    east -- the same convention as the street azimuths. Step 0 of the fixture has
    `wind_angle_rad = 0`, so the due-east street 0 must carry a POSITIVE canyon velocity
    (its `u` end upwind) and the due-north street 1 must carry essentially none."""
    data = _read(tmp_path)
    model, state, _ = build_street_model(data.net, pblh_floor=True, z_ref=30.0)
    resolved = model._apply_closures(state, {
        "U_ref": data.forcing.u_ref[0],
        "theta_w": data.forcing.theta_w[0],
        "h_abl": data.forcing.h_abl[0],
    })
    u_canyon = resolved["street.u_canyon"]
    assert float(u_canyon[0]) > 0.0
    assert abs(float(u_canyon[1])) < 1e-12
    assert abs(float(u_canyon[3])) < 1e-12


def test_the_loaded_network_builds_a_model_that_solves_and_conserves(tmp_path):
    data = _read(tmp_path)
    model, _state, _ = build_street_model(data.net, routing="sirane", pblh_floor=True,
                                          z_ref=30.0)
    net = model.net
    n_time = data.emission.shape[0]
    sources = torch.zeros(n_time, net.n, dtype=DT)
    for column, street in enumerate(data.net.streets):
        sources[:, net.node_index(street.name)] = data.emission[:, column]
    out = model.steady({}, {
        "street.x_boundary": data.forcing.background.reshape(-1, 1),
        "street.sources": sources,
        "U_ref": data.forcing.u_ref,
        "theta_w": data.forcing.theta_w,
        "h_abl": data.forcing.h_abl,
    })
    assert out["street.x"].shape == (n_time, len(data.net.streets))
    assert bool(torch.isfinite(out["street.x"]).all())
    assert bool((out["street.x"] > 0).all())


def test_a_colliding_emission_key_is_never_resolved_silently(tmp_path):
    """Feature 0 (selected, `network_transport`) and feature 2 (unselected,
    `gaussian_fallback`) are made to share one `(osmid, u, v)` key. Last-write-wins would
    make feature 0 silently receive feature 2's emission row; this must raise instead,
    naming both feature indices."""
    stage1, stage2 = build(tmp_path)
    key_path = stage2 / "edge_emissions_normalized.geojson"
    collection = json.loads(key_path.read_text(encoding="utf-8"))
    donor = collection["features"][0]["properties"]
    victim = collection["features"][2]["properties"]
    victim["osmid"] = donor["osmid"]
    victim["u"] = donor["u"]
    victim["v"] = donor["v"]
    key_path.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match=r"emission key .* occurs at features 0 and 2"):
        read_aqdt(stage1, stage2, year=2024, wind_height_m=30.0, trust_file_height=True)


def test_the_selection_note_names_is_canyon_when_unlabelled(tmp_path):
    """When no feature carries `aq_solver_type` at all, selection falls back to `is_canyon
    is True`, and the note must say so rather than repeating the `aq_solver_type` wording
    that was never actually applied."""
    stage1, stage2 = build(tmp_path)
    edges_path = stage1 / "repaired_edges_canyon.geojson"
    collection = json.loads(edges_path.read_text(encoding="utf-8"))
    for feature in collection["features"]:
        feature["properties"].pop("aq_solver_type", None)
    edges_path.write_text(json.dumps(collection), encoding="utf-8")
    data = read_aqdt(stage1, stage2, year=2024, wind_height_m=30.0, trust_file_height=True)
    assert "is_canyon" in data.notes["selection"]
    assert "aq_solver_type ==" not in data.notes["selection"]


def test_a_present_zero_roughness_is_not_treated_as_absent(tmp_path):
    """`roughness_m: 0.0` is a present, invalid value, not a missing one; it must reach
    `StreetNetwork`'s positivity check (naming the street) rather than being silently
    replaced by the loader's default."""
    stage1, stage2 = build(tmp_path)
    edges_path = stage1 / "repaired_edges_canyon.geojson"
    collection = json.loads(edges_path.read_text(encoding="utf-8"))
    collection["features"][0]["properties"]["roughness_m"] = 0.0
    edges_path.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match=r"street '0' has z0_b 0\.0"):
        read_aqdt(stage1, stage2, year=2024, wind_height_m=30.0, trust_file_height=True)
