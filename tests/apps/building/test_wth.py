"""CONTAM .wth reader (TN 1887r1 section 3.15 format)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tellegen.apps.building.elements import orifice_elements_from_edges
from tellegen.apps.building.thermal import (
    R_AIR,
    Zone,
    add_zone,
    build_model,
    initial_state,
)
from tellegen.apps.building.wth import Weather, read_wth
from tellegen.drives import Stack
from tellegen.topology import Network

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"

SYNTHETIC = """WeatherFile ContamW 2.0
synthetic
1/1\t1/2
!Date\tDofW\tDtype\tDST\tTgrnd
1/1\t1\t1\t0\t283.15
1/2\t2\t1\t0\t283.15
!Date\tTime\tTa\tPb\tWs\tWd\tHr\tIth\tIdn\tTs\tRn\tSn
1/1\t00:00:00\t280.0\t101000\t2.0\t90\t0\t0\t0\t0\t0\t0
1/1\t12:00:00\t290.0\t101200\t4.0\t180\t0\t0\t0\t0\t0\t0
1/1\t24:00:00\t282.0\t101100\t3.0\t270\t0\t0\t0\t0\t0\t0
1/2\t00:00:00\t282.0\t101100\t3.0\t270\t0\t0\t0\t0\t0\t0
1/2\t24:00:00\t286.0\t101300\t1.0\t0\t0\t0\t0\t0\t0\t0
"""


def test_reads_the_nist_sample():
    w = read_wth(DATA / "valThreeZonesWthCtm.wth")
    assert isinstance(w, Weather)
    assert w.t.tolist() == [0.0, 86400.0]
    assert w.Ta.tolist() == pytest.approx([293.15, 293.15])
    assert w.Ws.tolist() == pytest.approx([5.23, 5.23])
    assert w.Wd.tolist() == pytest.approx([270.0, 270.0])


def test_interpolates_linearly_between_rows_and_across_days(tmp_path):
    f = tmp_path / "s.wth"
    f.write_text(SYNTHETIC)
    w = read_wth(f)
    assert w.t.tolist() == [0.0, 43200.0, 86400.0, 86400.0, 172800.0]
    at = w.at(6 * 3600.0)
    assert at["Ta"] == pytest.approx(285.0)
    assert at["Ws"] == pytest.approx(3.0)
    assert w.at(36 * 3600.0)["Ta"] == pytest.approx(284.0)
    d = w.drivers_at(6 * 3600.0)
    assert set(d) == {"thermal.x_boundary", "P_ref", "V_met", "theta_w"}
    assert d["V_met"].dtype == torch.float64 and d["V_met"].dim() == 0


def test_drivers_at_emits_the_keys_a_build_model_model_actually_consumes(tmp_path):
    """The seam test: not that `drivers_at` returns four keys, but that they are the keys a
    `build_model`-built model reads. Updating such a model's drivers with them must move the
    ambient temperature the density closure sees -- which `"T_amb"`/`"P_amb"` never did."""
    f = tmp_path / "s.wth"
    f.write_text(SYNTHETIC)
    w = read_wth(f)

    net = Network(dtype=torch.float64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("z", volume=50.0, T0=293.15))
    net.add_edge("ambient", "z", kind="airpath", z_path=0.0, Cd=0.6, area=0.01)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")])
    state = initial_state(model)
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
        "thermal.x_boundary": torch.tensor([300.0], dtype=torch.float64),
    }
    (closure,) = model.closures
    amb = net.node_index("ambient")
    before = closure(state, drivers)["rho"][amb].item()

    d = w.drivers_at(6 * 3600.0)
    assert set(d) == {"thermal.x_boundary", "P_ref", "V_met", "theta_w"}
    assert d["thermal.x_boundary"].shape == (model.layers["thermal"].n_b,)
    assert d["thermal.x_boundary"].dtype == torch.float64
    assert d["P_ref"].dtype == torch.float64 and d["P_ref"].dim() == 0
    drivers.update(d)
    after = closure(state, drivers)["rho"][amb].item()

    assert d["thermal.x_boundary"].item() == pytest.approx(285.0)      # interpolated Ta
    assert d["P_ref"].item() == pytest.approx(101100.0)                # interpolated Pb
    assert after == pytest.approx(101100.0 / (R_AIR * 285.0))
    assert after != before


def test_rejects_a_file_that_is_not_a_contam_weather_file(tmp_path):
    f = tmp_path / "bad.wth"
    f.write_text("EPW,SomeCity\n")
    with pytest.raises(ValueError, match="WeatherFile ContamW"):
        read_wth(f)


def test_wind_direction_interpolates_on_the_shorter_arc_across_the_0_360_wrap():
    # A step from 350 deg to 10 deg sweeps forward through 0 (a 20 deg arc), not
    # backwards through 180 (a 340 deg arc): the midpoint must be 0, not 180.
    w = Weather(
        t=torch.tensor([0.0, 100.0], dtype=torch.float64),
        Ta=torch.tensor([280.0, 280.0], dtype=torch.float64),
        Pb=torch.tensor([101000.0, 101000.0], dtype=torch.float64),
        Ws=torch.tensor([1.0, 1.0], dtype=torch.float64),
        Wd=torch.tensor([350.0, 10.0], dtype=torch.float64),
    )
    assert w.at(50.0)["Wd"] == pytest.approx(0.0, abs=1e-9)
