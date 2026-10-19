"""Evaporation and seepage losses of the dynamic wave (`noodl.apps.sewer.losses`): the pieces,
without SWMM (tests/verification/test_sewer_losses_parity.py compares the whole against it)."""

import dataclasses
import datetime
import math

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.losses import (
    ClimateClock,
    ClimateValues,
    StorageLosses,
    exfiltration_geometry,
)
from noodl.apps.sewer.network import Exfiltration, Manhole, Pipe, SewerNetwork, storage_unit
from noodl.apps.sewer.runoff import Adjustments, Evaporation, Hydrology, evaporation_can_be_positive
from noodl.apps.sewer.xsect_tables import LCF, QCF

F64 = torch.float64

NET = """[TITLE]
losses

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
START_DATE           01/01/2026
START_TIME           00:00:00
END_DATE             01/01/2026
END_TIME             00:10:00
WET_STEP             00:01:00
DRY_STEP             00:01:00
ROUTING_STEP         5
VARIABLE_STEP        0
LENGTHENING_STEP     0
MIN_SURFAREA         0
MAX_TRIALS           8
HEAD_TOLERANCE       1e-9

[EVAPORATION]
CONSTANT         {evap}
DRY_ONLY         NO

[JUNCTIONS]
J1               10.0       3          0          0          0
J2               9.4        3          0          0          0

[OUTFALLS]
Outfall          8.0        FIXED      8.2              NO

[CONDUITS]
C1               J1               J2               300        0.020      0          0
C2               J2               Outfall          200        0.013      0          0

[XSECTIONS]
C1               {shape1}
C2               {shape2}

[LOSSES]
C1 0 0 0 NO {seep1}
C2 0 0 0 NO {seep2}

[INFLOWS]
J1               FLOW             ""               FLOW     1.0      1.0      0.04
J2               FLOW             ""               FLOW     1.0      1.0      0.02
"""

OPEN = "TRAPEZOIDAL  0.8          0.5        1.0        1.0        1"
CIRCLE = "CIRCULAR     0.60         0          0          0          1"
RECT = "RECT_CLOSED   0.6          0.8        0          0          1"


def build(tmp_path, *, evap=0.0, shape1=OPEN, shape2=CIRCLE, seep1=0.0, seep2=0.0, **kwargs):
    path = tmp_path / "net.inp"
    path.write_text(NET.format(evap=evap, shape1=shape1, shape2=shape2, seep1=seep1,
                               seep2=seep2))
    net, _, _ = read_swmm_inp(path)
    return net, build_model(net, **kwargs)


# ------------------------------------------------------------------------------ geometry
def test_a_tabular_curve_gives_the_banks_of_exfil_init_state():
    """exfil.c:78-114: the bottom area is the curve at depth 0 (0 if the first depth is above
    0), the banks start where the area stops being flat and end where it stops growing."""
    curve = ("tabular", (0.0, 0.3, 0.6, 1.2, 2.0), (8.0, 8.0, 20.0, 35.0, 35.0))
    bottom, low, high, top = exfiltration_geometry(curve, us_units=False)
    assert (bottom, low, high, top) == pytest.approx(
        (8 / LCF**2, 0.3 / LCF, 1.2 / LCF, 35 / LCF**2))
    shifted = ("tabular", (0.1, 0.6, 1.2), (4.0, 20.0, 35.0))
    assert exfiltration_geometry(shifted, us_units=True)[0] == 0.0          # table_lookupEx
    shrinking = ("tabular", (0.0, 1.0, 2.0), (8.0, 10.0, 6.0))             # a < alast: stop
    assert exfiltration_geometry(shrinking, us_units=True)[3] == pytest.approx(10 / LCF**2)


def test_the_other_shapes_have_a0_for_a_bottom_and_banks_from_the_floor():
    """exfil.c:116-135: FUNCTIONAL a0 (plus a1 for an exponent 0), CYLINDRICAL, CONICAL and
    PYRAMIDAL the quadratic's a0; banks from depth 0 up without limit. SWMM reads that area in
    the file's length unit as ft2: for an SI file the number of m2 is taken as ft2."""
    area = math.pi * 2.0 * 1.5
    cylinder = ("cylindrical", 4.0, 3.0)
    assert exfiltration_geometry(cylinder, us_units=True)[0] == pytest.approx(area / LCF**2)
    bottom, low, high, top = exfiltration_geometry(cylinder, us_units=False)
    assert bottom == pytest.approx(area) and low == 0.0 and high == top == 1.0e10
    assert exfiltration_geometry(("functional", 20.0, 1.5, 5.0), us_units=False)[0] == 5.0
    assert exfiltration_geometry(("functional", 20.0, 0.0, 5.0), us_units=False)[0] == 25.0
    assert exfiltration_geometry(("conical", 4.0, 2.0, 0.5), us_units=False)[0] == pytest.approx(
        math.pi * 2.0 * 1.0)
    with pytest.raises(ValueError, match="paraboloid"):
        exfiltration_geometry(("paraboloid", 2.0, 1.0, 3.0), us_units=False)


def test_the_seepage_depth_limit_is_each_sections_widest_point(tmp_path):
    """link.c:1370-1374: a circle's width stops growing at half its height, a closed
    rectangle's is wMax at any depth, an open trapezoid's first widest point is its top."""
    _, (model, _, _) = build(tmp_path, shape1=OPEN, shape2=RECT, seep1=10.0, seep2=10.0)
    geo = model.dynamic_wave.geo
    assert geo.seep_depth.tolist() == pytest.approx([0.8 / LCF, 0.6 / LCF])
    assert geo.seep_rect.tolist() == [False, True]
    _, (model, _, _) = build(tmp_path, shape1=CIRCLE, shape2=CIRCLE, seep1=10.0)
    geo = model.dynamic_wave.geo
    assert geo.seep_depth.tolist() == pytest.approx([0.3 / LCF] * 2) and geo.seep_rect is None


# ----------------------------------------------------------------------------- the climate
def hydrology(kind="constant", monthly=(0.0,) * 12, start=(2026, 1, 31, 23, 59), adjust=None):
    return Hydrology(
        gages=(), subcatchments=(), start=datetime.datetime(*start), total_duration=3600.0,
        us_units=False, evaporation=Evaporation(kind, monthly),
        adjustments=Adjustments(evaporation=adjust or (0.0,) * 12))


def test_the_clock_reads_the_month_at_the_start_of_the_step():
    """Without subcatchments `climate_setState` runs at the start of each routing step, read
    one millisecond late (swmm5.c:556): mm/day to ft/s, the month's rate plus its
    adjustment."""
    jan_feb = (1.0, 2.0) + (3.0,) * 10
    clock = ClimateClock(hydrology("monthly", jan_feb, adjust=(0.0, 0.5) + (0.0,) * 10))
    ucf = 26334720.0
    assert clock.at(0.0, 5000.0) == ClimateValues(1.0 / ucf, 1.0, 1.0, 1.0)
    assert clock.at(59_000.0, 64_000.0).evap == pytest.approx(1.0 / ucf)   # 23:59:59.001
    assert clock.at(59_999.0, 64_999.0).evap == pytest.approx(2.5 / ucf)   # 00:00:00.000
    assert evaporation_can_be_positive(hydrology("monthly", jan_feb))
    assert not evaporation_can_be_positive(hydrology())
    assert evaporation_can_be_positive(hydrology(adjust=(0.0, 0.2) + (0.0,) * 10))


def test_a_clock_without_a_climate_is_dry_with_unit_factors():
    assert ClimateClock(None).at(0.0, 5000.0) == ClimateValues(0.0, 1.0, 1.0, 1.0)


def test_with_subcatchments_the_clock_follows_the_runoff_model():
    """The climate is the last runoff step's (a runoff step is cut at the next evaporation
    change), shared with the inflow driver when it has one."""
    net, _, _ = read_swmm_inp("tests/data/sewer/runoff_evap_series_dynwave.inp")
    clock = ClimateClock(net.hydrology)
    first = clock.at(0.0, 10_000.0)
    assert clock.model is not None and clock.model.new_ms >= 10_000.0
    assert first.evap == pytest.approx(clock.climate.rate)
    assert first.infil == clock.model.env.hydcon
    driver = net.inflows.driver(list(net.manholes and [m.name for m in net.manholes]))
    shared = ClimateClock(net.hydrology, driver.model)
    assert shared.model is driver.model


# ---------------------------------------------------------------------------- the storage
def unit(curve=("cylindrical", 4.0, 3.0), fevap=0.0, exfiltration=None):
    return storage_unit("S", 0.0, 3.0, curve, evaporation_factor=fevap,
                        exfiltration=exfiltration)


def clim(evap=2.0e-7, hydcon=1.0, infil=1.0, recovery=1.0):
    return ClimateValues(evap / LCF, hydcon, infil, recovery)


def losses_of(manhole):
    from noodl.apps.sewer.storage import StorageCurves
    curves = StorageCurves([manhole.storage], torch.tensor([3.0], dtype=F64))
    return StorageLosses([manhole], curves)


def test_a_storage_unit_evaporates_fevap_times_the_rate_times_its_area():
    """node.c:1072-1085, in cfs: Fevap Evap.rate A(y); just Fevap Evap.rate when the volume is
    at most FUDGE (SWMM's slip); nothing for a unit without Fevap."""
    area = math.pi * 2.0 * 1.5
    s = losses_of(unit(fevap=1.5))
    depth = torch.tensor([0.4 / LCF], dtype=F64)
    rate = s.rates(depth, 5.0, clim())
    assert rate.item() * LCF**3 == pytest.approx(1.5 * 2.0e-7 * area)
    dry = s.rates(torch.tensor([2.0e-7], dtype=F64), 5.0, clim())     # V = 4e-5 ft3 < FUDGE
    assert dry.item() == pytest.approx(1.5 * 2.0e-7 / LCF)                # not times the area
    assert losses_of(unit()).rates(depth, 5.0, clim()).item() == 0.0


def test_the_loss_over_a_step_cannot_exceed_the_stored_volume():
    """node.c:1096-1101: evaporation and exfiltration are scaled together."""
    s = losses_of(unit(fevap=1.0, exfiltration=Exfiltration(conductivity=1.0e-3)))
    depth = torch.tensor([0.01 / LCF], dtype=F64)
    volume = math.pi * 2.0 * 1.5 * 0.01 / QCF          # SI: UCF(VOLUME) = 0.02832
    rate = s.rates(depth, 5.0, clim(evap=1.0e-3))
    assert rate.item() * 5.0 == pytest.approx(volume)


def test_a_constant_exfiltration_is_ksat_over_the_bottom_and_the_banks():
    """exfil.c:158-182: Ksat hydconFactor times the bottom area, and over the surface area
    beyond it (a CONICAL unit's quadratic area has a bottom of a0 and banks above)."""
    ks = 2.0e-5                                                         # m/s
    s = losses_of(unit(("conical", 4.0, 2.0, 0.5), exfiltration=Exfiltration(conductivity=ks)))
    y = 0.8
    area = math.pi * 2 * 1 + y * (2 * math.pi * 1 * 0.5) + y * y * (math.pi * 0.5 * 0.25)
    for hydcon in (1.0, 0.5):
        rate = s.rates(torch.tensor([y / LCF], dtype=F64), 5.0, clim(evap=0.0, hydcon=hydcon))
        assert rate.item() * LCF**3 == pytest.approx(ks * hydcon * area, rel=1e-12)
    assert s.has_exfil.tolist() == [True] and not s.any_evap


def test_green_ampt_exfiltration_keeps_its_state_between_steps():
    """A positive IMDmax runs SWMM's modified Green-Ampt model: wetter soil takes less water,
    and `reset` returns the unit to the start of a run."""
    x = Exfiltration(conductivity=2.0e-5, suction=0.1, initial_deficit=0.4)
    s = losses_of(unit(("cylindrical", 4.0, 3.0), exfiltration=x))
    depth = torch.tensor([1.0 / LCF], dtype=F64)
    first = s.rates(depth, 5.0, clim(evap=0.0))
    later = [s.rates(depth, 5.0, clim(evap=0.0)) for _ in range(40)][-1]
    assert 0.0 < later.item() < first.item()
    s.reset()
    assert s.rates(depth, 5.0, clim(evap=0.0)).item() == first.item()
    with pytest.raises(ValueError, match="cannot be batched"):
        s.rates(depth.expand(2, 1), 5.0, clim(evap=0.0))


# ------------------------------------------------------------------------------- the network
def test_the_network_refuses_what_it_cannot_lose(tmp_path):
    net, _ = build(tmp_path)
    plain = Manhole("J", 0.0, ground=1.0)
    for bad, match in (
            (Pipe("P", "J1", "Outfall", 1.0, 0.3, 0.013, 0.01, seepage=-1.0), "seepage"),
            (Pipe("P", "J1", "Outfall", 1.0, 0.3, 0.013, 0.01, seepage=math.inf), "seepage")):
        with pytest.raises(ValueError, match=match):
            SewerNetwork(manholes=net.manholes, pipes=(bad,), outfalls=net.outfalls,
                         routing="dynamic_wave").validate()
    for manhole, match in (
            (unit(exfiltration=Exfiltration(conductivity=0.0)), "exfiltration needs"),
            (unit(exfiltration=Exfiltration(conductivity=1e-5, initial_deficit=1.5)),
             "exfiltration needs"),
            (unit(("paraboloid", 2.0, 1.0, 3.0),
                  exfiltration=Exfiltration(conductivity=1e-5)), "PARABOLOID"),
            (unit(fevap=math.nan), "evaporation_factor"),
            (Manhole("J", 0.0, ground=1.0, evaporation_factor=1.0), "not a storage unit"),
            (Manhole("J", 0.0, ground=1.0, exfiltration=Exfiltration(conductivity=1e-5)),
             "not a storage unit")):
        with pytest.raises(ValueError, match=match):
            dataclass_net(net, manhole).validate()
    assert plain.evaporation_factor == 0.0
    kinematic = dataclass_net(net, unit(fevap=1.0), routing="kinematic_wave")
    with pytest.raises(ValueError, match="storage unit"):
        kinematic.validate()
    seeping = SewerNetwork(
        manholes=net.manholes, pipes=(Pipe("P", "J1", "Outfall", 100.0, 0.3, 0.013, 0.01,
                                           seepage=1e-6),), outfalls=net.outfalls,
        routing="kinematic_wave")
    with pytest.raises(ValueError, match="seepage"):
        seeping.validate()


def dataclass_net(net, manhole, routing="dynamic_wave"):
    """`net` with its first manhole replaced by `manhole`."""
    manhole = dataclasses.replace(manhole, name="J1")
    return SewerNetwork(manholes=(manhole, *net.manholes[1:]), pipes=net.pipes,
                        outfalls=net.outfalls, routing=routing, geometry=net.geometry,
                        dynamic_wave=net.dynamic_wave)


def test_a_lossless_network_builds_none_of_it(tmp_path):
    net, (model, state, drivers) = build(tmp_path, shape1=CIRCLE)
    hyd = model.dynamic_wave
    assert not hyd.has_wet_losses and hyd.storage_losses is None
    assert "sewer.pipe_depth" not in state and "evap_rate" not in drivers


def test_a_network_with_losses_carries_the_link_depth_and_the_climate_drivers(tmp_path):
    net, (model, state, drivers) = build(tmp_path, evap=1000.0, seep2=10.0)
    hyd = model.dynamic_wave
    assert hyd.has_wet_losses and "sewer.pipe_depth" in hyd.state_keys
    assert state["sewer.pipe_depth"].shape == (2,)
    assert drivers["evap_rate"].item() == pytest.approx(1000.0 / 86.4e6)   # mm/day to m/s
    assert drivers["hydcon"].item() == 1.0
    with pytest.raises(KeyError, match="evap_rate"):
        hyd(state, {k: v for k, v in drivers.items() if k != "evap_rate"})
    out = hyd(state, drivers)
    assert "sewer.pipe_depth" in out


def test_a_closure_without_the_storage_driver_names_it(tmp_path):
    text = NET.format(evap=1000.0, shape1=OPEN, shape2=CIRCLE, seep1=0.0, seep2=0.0)
    text = text.replace("[OUTFALLS]", "[STORAGE]\nS1 9.0 3 0 CYLINDRICAL 4 3 0 0 1.5\n\n"
                                      "[OUTFALLS]").replace(
        "J2               9.4        3          0          0          0\n", "")
    text = text.replace("J2               FLOW", "S1               FLOW").replace(
        "C1               J1               J2", "C1               J1               S1").replace(
        "C2               J2               Outfall", "C2               S1               Outfall")
    path = tmp_path / "s.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_model(net)
    assert drivers["storage_loss"].shape == (len(net.manholes),)
    with pytest.raises(KeyError, match="storage_loss"):
        model.dynamic_wave(state, {k: v for k, v in drivers.items() if k != "storage_loss"})


def test_water_quality_with_storage_losses_is_refused_by_name(tmp_path):
    text = NET.format(evap=1000.0, shape1=CIRCLE, shape2=CIRCLE, seep1=0.0, seep2=0.0)
    text = text.replace("[OUTFALLS]", "[STORAGE]\nS1 9.0 3 0 CYLINDRICAL 4 3 0 0 1.5\n\n"
                                      "[OUTFALLS]").replace(
        "J2               9.4        3          0          0          0\n", "")
    text = text.replace("J2               FLOW", "S1               FLOW").replace(
        "C1               J1               J2", "C1               J1               S1").replace(
        "C2               J2               Outfall", "C2               S1               Outfall")
    path = tmp_path / "s.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    with pytest.raises(ValueError, match="storage units that evaporate or exfiltrate"):
        build_model(net, quality=True)


def _tracer_steps(tmp_path, seep, steps=24, c=0.1, contents=50.0):
    """A tracer at the uniform concentration `c` (kg/m3) everywhere (inflows and contents) in
    the circular two-conduit network with C1 seeping at `seep` mm/hr: per step the
    junctions' tracer concentration and amount (kg) and quality volume (m3)."""
    net, (model, state, drivers) = build(tmp_path, shape1=CIRCLE, seep1=seep, quality=True,
                                         species=("tracer",), iterate_tol=1e-11, iterate_max=500)
    load = torch.zeros(model.net.n, dtype=F64)
    load[: len(model.dynamic_wave.junction_names)] = c
    drivers = dict(drivers, tracer_in=load)
    # `contents` None: the dry start's own few litres, which one step's seepage exceeds
    state = dict(state, **{"water_quality.x": torch.full_like(state["water_quality.x"], c)})
    if contents is not None:
        state.update({key: torch.full_like(state["sewer.quality_volume"], contents)
                      for key in ("sewer.quality_volume", "water_quality.capacity")})
    rows = []
    for _ in range(steps):
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
        x = state["water_quality.x"]
        volume = state["water_quality.capacity"]
        rows.append((x.clone(), (x * volume).sum().item(), volume.sum().item()))
    return rows


def test_seepage_removes_the_mass_it_carries_so_the_concentration_stays_put(tmp_path):
    """SWMM's qualrout.c takes a conduit's seepage out at the conduit's concentration, so a
    well-mixed conduit seeping keeps its concentration (probed against SWMM 5.2.4: a 100 mg/L
    tracer through a seeping conduit leaves at 100 mg/L). The seeping network holds less
    water and, in the same proportion, less mass than the one without seepage."""
    c = 0.1
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dry = _tracer_steps(tmp_path / "a", 0.0, c=c)
    seeping = _tracer_steps(tmp_path / "b", 100.0, c=c)
    x, mass, volume = seeping[-1]
    torch.testing.assert_close(x, torch.full_like(x, c), rtol=1e-8, atol=0.0)
    assert volume < dry[-1][2] * (1.0 - 1e-3)               # the seepage took water
    assert mass == pytest.approx(c * volume, rel=1e-8)      # and mass in proportion
    assert mass < dry[-1][1] * (1.0 - 1e-3)


def test_evaporation_takes_water_and_leaves_the_mass(tmp_path):
    """Evaporation leaves the mass behind (SWMM 5.2.4 raises the concentration of an
    evaporating conduit's water: 100 mg/L enters, 113 leaves): the control volume loses the
    water, no fraction of the contents is lost (`overflow_loss` 0) and no removal rate applies,
    and the concentration rises as ``V_start / V_end``. A seepage loss alone is a removal
    rate at the local concentration, ``seep / V_end``."""
    net, (model, state, drivers) = build(tmp_path, shape1=CIRCLE, seep1=100.0, quality=True,
                                         species=("tracer",))
    hyd = model.dynamic_wave
    start = torch.tensor([10.0, 20.0], dtype=F64)
    inflow = torch.tensor([0.0, -0.5], dtype=F64)              # a net loss at J2: evaporation
    out = hyd._quality_volume({"sewer.quality_volume": start}, inflow,
                              torch.zeros(2, dtype=F64), 5.0)
    torch.testing.assert_close(out["water_quality.overflow_loss"], torch.zeros(2, dtype=F64))
    torch.testing.assert_close(out["sewer.quality_volume"],
                               torch.tensor([10.0, 17.5], dtype=F64))
    from noodl.apps.sewer.dynwave import OverflowLoss
    kept = OverflowLoss(1).apply(torch.full((2,), 0.1, dtype=F64), 5.0, out)
    torch.testing.assert_close(kept, torch.full((2,), 0.1, dtype=F64))   # the mass is kept
    seeping = hyd._quality_volume({"sewer.quality_volume": start}, inflow,
                                  torch.zeros(2, dtype=F64), 5.0,
                                  torch.tensor([0.0, 0.5], dtype=F64))
    torch.testing.assert_close(seeping["water_quality.overflow_loss"], torch.zeros(2, dtype=F64))
    torch.testing.assert_close(seeping["water_quality.removal"],
                               torch.tensor([0.0, 0.5 / 17.5], dtype=F64))
    assert "water_quality.removal" not in out            # evaporation alone: none


def test_the_conduit_rate_is_split_into_seepage_and_evaporation(tmp_path):
    """An open conduit that seeps and evaporates reports both, and its seepage part alone
    leaves with the mass: zero without a seepage rate, the whole rate for a closed one
    (it cannot evaporate), and a gradient survives at the zero seepage rate of a dry link."""
    net, (model, state, drivers) = build(tmp_path, evap=1000.0, seep1=50.0, seep2=50.0)
    hyd = model.dynamic_wave
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=3)
    terms = hyd._terms(state, 5.0, drivers=drivers)
    assert terms.seep_loss_rate is not None
    assert bool((terms.seep_loss_rate > 0).all())
    assert float(terms.loss_rate[0] - terms.seep_loss_rate[0]) > 0.0      # C1 is open: evaporates
    assert float(terms.loss_rate[1]) == pytest.approx(float(terms.seep_loss_rate[1]), rel=1e-14)
    net, (model, state, drivers) = build(tmp_path, evap=1000.0)
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=3)
    terms = model.dynamic_wave._terms(state, 5.0, drivers=drivers)
    assert float(terms.loss_rate[0]) > 0.0 and float(terms.seep_loss_rate.abs().max()) == 0.0


def test_evaporation_takes_water_and_a_gradient_flows_through_it(tmp_path):
    """The loss is part of the solved network: less reaches the outfall than without it, and
    the outfall flow's sensitivity to an inflow (the autograd adjoint through the converged
    passes, which contain the loss) matches a finite difference."""
    def run(evap, bump=0.0, grad=False):
        net, (model, state, drivers) = build(tmp_path, evap=evap, iterate_tol=1e-11,
                                             iterate_max=500)
        inflow = drivers["inflow"].clone()
        inflow[0] = inflow[0] + bump
        leaf = inflow.clone().requires_grad_(grad)
        _, hist = simulate_dynamic_wave(model, state, drivers, steps=24, inflow=leaf,
                                        record=("water.q",))
        return hist["water.q"][-1, 1], leaf
    base, _ = run(0.0)
    lossy, _ = run(20000.0)
    assert lossy.item() < base.item() - 1e-5
    q, leaf = run(20000.0, grad=True)
    (g,) = torch.autograd.grad(q, leaf)
    h = 1e-6
    fd = (run(20000.0, bump=h)[0].item() - run(20000.0, bump=-h)[0].item()) / (2 * h)
    assert g[0].item() == pytest.approx(fd, rel=1e-4)


def test_seepage_beyond_the_dry_starts_contents_still_conserves_the_concentration(tmp_path):
    """From a dry start a junction holds a few litres and one step's seepage exceeds them: the
    removal rate has no cap, so the water that enters during the step and seeps out again
    takes its mass, and a uniform tracer stays uniform to round-off (the lost fraction capped
    at 1 left 2.5e-4 on the first step)."""
    rows = _tracer_steps(tmp_path, 100.0, steps=12, contents=None)
    for x, _, _ in rows:
        torch.testing.assert_close(x, torch.full_like(x, 0.1), rtol=1e-8, atol=0.0)
