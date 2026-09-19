"""The headline demo — a real CONTAM building on a real street canyon.

Design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`, sections 3,
4 and 10; amendments A2 (units), A3 (wind conventions), A5 (layouts).

The street's `StreetFlows` closure REQUIRES the drivers `U_ref` (m/s at `z_ref`), `theta_w`
(radians CCW from east, wind TOWARD) and `h_abl` (m), which `build_street_model`'s returned
driver template does NOT include (`apps/street/routing.py:414-416`,
`tests/verification/test_street_parity.py:53-59`); the fixtures below supply them. The
building is built at the .prj's own ambient (`Ws=5.23`, `Wd=270`, west wind) and then
receives the street's wind through the `DriverAlias`es.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from tellegen.apps.building.prj import project_to_model, read_prj
from tellegen.apps.street.network import Street, StreetNetwork, build_street_model, street_index
from tellegen.couple import (
    CONCENTRATION_TO_MASS_FRACTION,
    STREET_RAD_TO_CONTAM_DEG,
    DriverAlias,
    ValueLink,
    union,
)

F64 = torch.float64
CONTAM_DIR = Path(__file__).resolve().parent.parent / "data" / "contam"
BUILDING_PRJ = CONTAM_DIR / "valThreeZonesWthCtm-UseApi.prj"
U_REF, THETA_W, H_ABL = 2.0, 0.0, 1200.0        # 2 m/s WEST wind (toward east), 1200 m ABL
BACKGROUND = 2.0e-8                              # kg/m3 NOx, ~20 ug/m3
EMISSION = 1.0e-7                                # kg/s per street
SHARED = "r2"


def _small_street_network() -> StreetNetwork:
    """`from_test_network()`'s topology at 1/10 scale with 2 m x 3 m canyons, so one CONTAM
    building's ~0.05 m3/s of exhaust is a few percent of a segment's own ventilation and the
    back-coupling code path produces a MEASURABLE change (design spec section 10)."""
    x = {"n0": 0.0, "n1": 30.0, "n2": 60.0, "n3": 30.0}
    y = {"n0": 40.0, "n1": 30.0, "n2": 40.0, "n3": 0.0}
    spec = [("r1", "n0", "n1"), ("r2", "n1", "n2"), ("r3", "n3", "n1")]
    streets = [
        Street(name, a, b, math.hypot(x[b] - x[a], y[b] - y[a]), 2.0, 3.0)
        for name, a, b in spec
    ]
    return StreetNetwork(streets=streets, x=x, y=y)


def _street(net: StreetNetwork, *, u_ref=U_REF, theta_w=THETA_W, h_abl=H_ABL,
            background=BACKGROUND, emission=EMISSION):
    """`(model, state, drivers)` with EVERY driver `StreetFlows` needs. `z_ref=10.0`: the
    street's reference wind is the 10 m wind the building's `V_met` also names (spec A3)."""
    model, state, drivers = build_street_model(net, species=("nox",), z_ref=10.0)
    graph = model.net
    sources = torch.zeros(graph.n, dtype=F64)
    for street in net.streets:
        sources[graph.node_index(street.name)] = emission
    drivers = dict(drivers)
    drivers.update({
        "street.x_boundary": torch.tensor([background], dtype=F64),
        "street.sources": sources,
        "U_ref": torch.tensor(u_ref, dtype=F64),
        "theta_w": torch.tensor(theta_w, dtype=F64),
        "h_abl": torch.tensor(h_abl, dtype=F64),
    })
    # Start every segment at the street's OWN steady state for this forcing, so a coupled
    # step measures the building's effect and not the segment filling up from zero.
    state = model.steady(state, drivers)
    return model, state, drivers


def _building():
    project = read_prj(BUILDING_PRJ)
    return project, project_to_model(project)


def _link(street_model, segment: str) -> ValueLink:
    # `convert_back=None`: the feedback is a mass flux, kg/s on both sides (spec A2).
    return ValueLink(
        from_model="street", from_key="street.x",
        from_index=street_index(street_model)[segment],
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert=CONCENTRATION_TO_MASS_FRACTION, two_way=True,
    )


def _aliases() -> list[DriverAlias]:
    return [
        DriverAlias(source=("street", "U_ref"), targets=(("building", "V_met", None),)),
        DriverAlias(source=("street", "theta_w"),
                    targets=(("building", "theta_w", STREET_RAD_TO_CONTAM_DEG),)),
    ]


def _coupled(street_triple, building_triple, street_model, *, two_way=True, **kwargs):
    link = _link(street_model, SHARED)
    if not two_way:
        link = ValueLink(**{**link.__dict__, "two_way": False})
    return union({"street": street_triple, "building": building_triple},
                 shared=[link, *_aliases()], **kwargs)


def test_small_street_shows_measurable_back_coupling_and_reports_it(record_property):
    net = _small_street_network()
    street_model, street_state, street_drivers = _street(net)
    _project, (building_model, building_state, building_drivers) = _building()
    seg = street_index(street_model)[SHARED]
    street_triple = (street_model, street_state, street_drivers)
    building_triple = (building_model, building_state, building_drivers)

    city, state, drivers = _coupled(
        street_triple, building_triple, street_model,
        substeps={"building": 60}, iterate_rtol=1e-10, iterate_max=100,
    )
    diag: dict = {}
    coupled = city.step(state, drivers, dt=3600.0, diagnostics=diag)

    # The same hour with the building NOT feeding back (one-way): the street is unaffected.
    # Identical forcing -- `union` and `step` both copy, so the triples are reusable as-is.
    loose, s1, d1 = _coupled(
        street_triple, building_triple, street_model,
        two_way=False, substeps={"building": 60},
    )
    one_way = loose.step(s1, d1, dt=3600.0)

    c_two = coupled["street"]["street.x"][seg].item()
    c_one = one_way["street"]["street.x"][seg].item()
    rel = abs(c_two - c_one) / abs(c_one)
    record_property("segment", SHARED)
    record_property("street_conc_one_way_kg_m3", c_one)
    record_property("street_conc_two_way_kg_m3", c_two)
    record_property("back_coupling_relative_change", rel)
    record_property("passes", diag["passes"])
    assert diag["converged"]
    assert rel > 1e-4, f"back-coupling {rel:.3e} on a 2x3 m canyon should be visible"
    # Infiltration draws segment air into the building and returns cleaner air: a SINK.
    assert c_two < c_one


def test_the_wind_reaches_the_building_through_the_aliases():
    """The building's V_met/theta_w must be the street's wind in CONTAM's convention after a
    step -- not the .prj's own 5.23 m/s / 270 deg."""
    net = _small_street_network()
    # theta_w = pi/2: the wind blows TOWARD the north.
    street_triple = _street(net, u_ref=3.0, theta_w=0.5 * math.pi)
    street_model = street_triple[0]
    _project, (building_model, building_state, building_drivers) = _building()
    city, state, drivers = _coupled(
        street_triple, (building_model, building_state, building_drivers),
        street_model, iterate_max=100,
    )
    # A DriverAlias is applied to the (copied) drivers at the start of every step; observe it
    # by spying on the building model's step.
    seen: dict = {}
    real_step = building_model.step

    def spy(s, d, dt, **kw):
        seen["V_met"], seen["theta_w"] = float(d["V_met"]), float(d["theta_w"])
        return real_step(s, d, dt, **kw)

    building_model.step = spy
    try:
        city.step(state, drivers, dt=60.0)
    finally:
        building_model.step = real_step
    assert seen["V_met"] == pytest.approx(3.0)
    # Wind TOWARD the north == CONTAM's "from the south" == Wd 180 deg (spec A3).
    assert seen["theta_w"] == pytest.approx(180.0)


# ----------------------------------------------------------------- the real data

AQDT_DATA = Path(os.environ.get(
    "TELLEGEN_AQDT_DATA", r"<workspace>\tmp\2026_AQ_DT\data"
))
DOMAIN, YEAR, STEP = "leiden_small", 2024, 1000
needs_aqdt = pytest.mark.skipif(
    not (AQDT_DATA / "stage1_geometry" / DOMAIN / "repaired_edges_canyon.geojson").exists(),
    reason=f"the AQ_DT products are not at {AQDT_DATA}; set TELLEGEN_AQDT_DATA",
)


@needs_aqdt
def test_real_leiden_small_building_back_coupling_magnitude(record_property):
    """The headline pairing on real data: one leiden_small hour (forcing step 1000, the
    middle of test_street_parity.py's own sampled steps), the busiest street as the shared
    segment, the three-zone CONTAM building on it. The magnitude is RECORDED (design spec
    section 4: negligible is an acceptable, reportable result); the assertion is only that
    the coupled step converged and the sink sign is right."""
    from tellegen.apps.street.loader import read_aqdt

    data = read_aqdt(
        AQDT_DATA / "stage1_geometry" / DOMAIN, AQDT_DATA / "stage2_inputs" / DOMAIN,
        year=YEAR, wind_height_m=30.0, trust_file_height=True, times=[STEP],
    )
    model, state, drivers = build_street_model(data.net, species=("nox",), z_ref=30.0)
    graph = model.net
    sources = torch.zeros(graph.n, dtype=F64)
    for column, street in enumerate(data.net.streets):
        sources[graph.node_index(street.name)] = data.emission[0, column]
    drivers = dict(drivers)
    drivers.update({
        "street.x_boundary": data.forcing.background[0].reshape(1),
        "street.sources": sources,
        "U_ref": data.forcing.u_ref[0], "theta_w": data.forcing.theta_w[0],
        "h_abl": data.forcing.h_abl[0],
    })
    state = model.steady(state, drivers)
    busiest = data.net.streets[int(torch.argmax(data.emission[0]))].name
    _project, (building_model, building_state, building_drivers) = _building()
    link = ValueLink(**{**_link(model, busiest).__dict__})
    city, s, d = union(
        {
            "street": (model, state, drivers),
            "building": (building_model, building_state, building_drivers),
        },
        shared=[link, *_aliases()], substeps={"building": 60},
        iterate_rtol=1e-10, iterate_max=100,
    )
    diag: dict = {}
    coupled = city.step(s, d, dt=3600.0, diagnostics=diag)
    seg = street_index(model)[busiest]
    c_two, c_one = coupled["street"]["street.x"][seg].item(), state["street.x"][seg].item()
    record_property("segment", busiest)
    record_property("street_conc_steady_kg_m3", c_one)
    record_property("street_conc_coupled_kg_m3", c_two)
    record_property("back_coupling_relative_change", abs(c_two - c_one) / abs(c_one))
    record_property("passes", diag["passes"])
    assert diag["converged"]
    assert c_two <= c_one


def test_sequential_file_exchange_disagrees_with_the_coupled_result(record_property):
    """Today's practice: run the street for the hour, hand the resulting concentration to
    the building, no feedback (framework spec section 7 item 5). Against the two-way coupled
    step on the same fixture and forcing, report the indoor discrepancy."""
    net = _small_street_network()
    street_model, street_state, street_drivers = _street(net)
    _project, (building_model, building_state, building_drivers) = _building()
    seg = street_index(street_model)[SHARED]

    street_triple = (street_model, street_state, street_drivers)
    building_triple = (building_model, building_state, building_drivers)
    city, state, drivers = _coupled(
        street_triple, building_triple,
        street_model, substeps={"building": 60}, iterate_rtol=1e-10, iterate_max=100,
    )
    coupled = city.step(state, drivers, dt=3600.0)

    # Sequential exchange: street first, frozen, then the building on that value, sixty
    # 60 s steps with the SAME wind the aliases would have supplied.
    street_after = street_model.step(dict(street_state), dict(street_drivers), dt=3600.0)
    frozen = street_after["street.x"][seg] / building_drivers["rho_amb"]
    loose_drivers = dict(building_drivers)
    loose_drivers["species.x_boundary"] = frozen.reshape(1, 1)
    loose_drivers["V_met"] = street_drivers["U_ref"]
    wind_from_deg = 270.0 - torch.rad2deg(street_drivers["theta_w"])
    loose_drivers["theta_w"] = torch.remainder(wind_from_deg, 360.0)
    indoor = dict(building_state)
    for _ in range(60):
        indoor = building_model.step(indoor, loose_drivers, dt=60.0)

    a, b = coupled["building"]["species.x"].flatten(), indoor["species.x"].flatten()
    discrepancy = ((a - b).abs() / b.abs().clamp_min(1e-300)).max().item()
    record_property("indoor_mass_fraction_coupled", a.tolist())
    record_property("indoor_mass_fraction_sequential", b.tolist())
    record_property("loose_coupling_relative_discrepancy", discrepancy)
    assert discrepancy > 1e-4
