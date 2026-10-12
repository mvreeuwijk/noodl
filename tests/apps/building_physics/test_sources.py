"""CONTAM contaminant source types (TN 1887r1 section 8.2.4) as source-term callables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.prj import PrjSource, project_to_model, read_prj
from noodl.apps.building_physics.simulation import simulate
from noodl.apps.building_physics.sources import (
    BurstSource,
    ConstantSource,
    CutoffSource,
    DecayingSource,
    DepositionRateSink,
    DepositionVelocitySink,
    PeakEmissionSource,
    PowerLawDecaySource,
    SpeciesSources,
    assemble_sources,
    sources_from_project,
)
from noodl.apps.building_physics.thermal import RHO_0, species_layer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "contam"


def test_constant_source_is_generation_minus_removal_times_concentration():
    s = ConstantSource(node=1, G=2e-6, R=1e-3)
    x = torch.tensor([0.0, 5e-4, 0.0], dtype=F64)
    out = s(0.0, x)
    assert out.shape == (3,)
    assert out[1].item() == pytest.approx(2e-6 - 1e-3 * 5e-4)
    assert out[0].item() == 0.0 and out[2].item() == 0.0


def test_cutoff_source_vanishes_at_the_cutoff_concentration():
    s = CutoffSource(node=2, G=1e-5, x_cut=1e-3)
    x = torch.tensor([0.0, 0.0, 1e-3], dtype=F64)
    assert s(0.0, x)[2].item() == pytest.approx(0.0)
    x[2] = 5e-4
    assert s(0.0, x)[2].item() == pytest.approx(5e-6)


def test_cutoff_source_below_cutoff_still_matches_the_unclamped_formula():
    s = CutoffSource(node=2, G=1e-5, x_cut=1e-3)
    x = torch.tensor([0.0, 0.0, 9e-4], dtype=F64)  # just below x_cut
    expected = 1e-5 * (1.0 - 9e-4 / 1e-3)
    assert s(0.0, x)[2].item() == pytest.approx(expected)


def test_decaying_source_follows_the_exponential_from_its_start():
    s = DecayingSource(node=0, G0=1e-5, tau=600.0, t0=100.0)
    x = torch.zeros(2, dtype=F64)
    assert s(50.0, x)[0].item() == 0.0
    assert s(700.0, x)[0].item() == pytest.approx(1e-5 * math.exp(-1.0))


def test_burst_source_delivers_its_mass_over_one_step():
    s = BurstSource(node=1, mass=0.5, t_burst=300.0, dt=60.0)
    x = torch.zeros(2, dtype=F64)
    assert s(299.0, x)[1].item() == 0.0
    assert s(300.0, x)[1].item() == pytest.approx(0.5 / 60.0)
    assert s(359.0, x)[1].item() == pytest.approx(0.5 / 60.0)
    assert s(360.0, x)[1].item() == 0.0


def test_assemble_sources_sums_contributions_in_full_node_order():
    srcs = [
        ConstantSource(node=1, G=1e-6),
        ConstantSource(node=1, G=2e-6),
        CutoffSource(node=2, G=1e-6, x_cut=1.0),
    ]
    x = torch.zeros(3, dtype=F64)
    # There is no node-count argument (the shape comes from `x_full`), so this is a
    # 3-argument call.
    out = assemble_sources(srcs, 0.0, x)
    torch.testing.assert_close(out, torch.tensor([0.0, 3e-6, 1e-6], dtype=F64))


# --------------------------------------------------------------------------------------
# `sources_from_project`: both of its error paths (a checked `KeyError` for an
# unresolvable zone, a checked `ValueError` for an unrecognised source type) get their own
# test here, plus one exercising the normal four-type translation.
# --------------------------------------------------------------------------------------


@dataclass
class _FakeProject:
    net: Network
    sources: list
    zone_nr_to_name: dict


def _network_with_zones(*names: str) -> Network:
    net = Network(dtype=F64)
    net.add_node("ambient")
    for name in names:
        net.add_node(name)
    return net


def test_sources_from_project_builds_each_type_at_its_resolved_zone():
    net = _network_with_zones("z1", "z2")
    sources = [
        PrjSource(
            nr=1, zone_nr=5, element_nr=1, source_type="ccf",
            params=[2e-6, 1e-3], mult=1.0,
        ),
        PrjSource(
            nr=2, zone_nr=7, element_nr=2, source_type="cut",
            params=[1e-5, 1e-3], mult=1.0,
        ),
        PrjSource(
            nr=3, zone_nr=5, element_nr=3, source_type="eds",
            params=[1e-5, 1.0 / 600.0], mult=1.0,
        ),
        PrjSource(nr=4, zone_nr=7, element_nr=4, source_type="brs", params=[0.5], mult=1.0),
    ]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1", 7: "z2"})
    out = sources_from_project(project, dt=60.0)
    assert [type(s) for s in out] == [ConstantSource, CutoffSource, DecayingSource, BurstSource]
    n_z1, n_z2 = net.node_index("z1"), net.node_index("z2")
    assert out[0].node == n_z1
    assert out[0].G == pytest.approx(2e-6) and out[0].R == pytest.approx(1e-3)
    assert out[1].node == n_z2 and out[1].x_cut == pytest.approx(1e-3)
    assert out[2].node == n_z1 and out[2].tau == pytest.approx(600.0)
    assert out[3].node == n_z2
    assert out[3].mass == pytest.approx(0.5) and out[3].dt == pytest.approx(60.0)


def test_sources_from_project_raises_keyerror_naming_source_and_zone_when_zone_is_absent():
    net = _network_with_zones("z1")
    sources = [
        PrjSource(
            nr=9, zone_nr=42, element_nr=1, source_type="ccf",
            params=[1e-6, 0.0], mult=1.0,
        ),
    ]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1"})
    with pytest.raises(KeyError, match="9") as exc:
        sources_from_project(project)
    assert "42" in str(exc.value)


def test_sources_from_project_raises_valueerror_naming_source_and_type_for_an_unknown_type():
    net = _network_with_zones("z1")
    sources = [PrjSource(nr=3, zone_nr=5, element_nr=1, source_type="wpc", params=[1e-6], mult=1.0)]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1"})
    with pytest.raises(ValueError, match="3") as exc:
        sources_from_project(project)
    assert "wpc" in str(exc.value)


def _layer(K):
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("a", volume=10.0)
    net.add_node("b", volume=20.0)
    net.add_edge("ambient", "a", kind="airpath")
    net.add_edge("a", "b", kind="airpath")
    return species_layer(net, n_species=K)


def _drivers(mult, K):
    return {"time": torch.tensor(0.0, dtype=F64),
            "species.source_multiplier": torch.as_tensor(mult, dtype=F64),
            "species.x_boundary": torch.zeros(1, K, dtype=F64)}


def test_each_source_lands_on_its_node_and_species_times_its_multiplier():
    layer = _layer(2)
    a, b = 1, 2
    srcs = SpeciesSources([ConstantSource(a, 1e-6), CutoffSource(b, 2e-6, 0.01)],
                          species=[1, 0], layer=layer, names=["gen", "cut"])
    x = torch.tensor([[0.0, 0.0], [0.005, 0.0]], dtype=F64)       # interior (a, b), K = 2
    got = srcs({"species.x": x}, _drivers([0.5, 1.0], 2))
    out = got["species.sources"]
    assert out.shape == (3, 2)
    expected = torch.zeros(3, 2, dtype=F64)
    expected[a, 1] = 0.5e-6
    expected[b, 0] = 2e-6                                   # generation only
    torch.testing.assert_close(out, expected, rtol=1e-15, atol=0)
    removal = got["species.removal"]                        # (n_i, K) = interior (a, b)
    assert removal.shape == (2, 2)
    assert removal[1, 0].item() == pytest.approx(2e-6 / 0.01 / (RHO_0 * 20.0), rel=1e-14)
    assert removal[0].abs().sum().item() == 0.0 and removal[1, 1].item() == 0.0


def test_a_single_species_state_without_a_species_axis_and_a_batched_multiplier():
    layer = _layer(1)
    srcs = SpeciesSources([ConstantSource(1, 1e-6)], species=[0], layer=layer)
    out = srcs({"species.x": torch.zeros(2, dtype=F64)},
               _drivers([[1.0], [0.0]], 1))["species.sources"]
    assert out.shape == (2, 3)
    assert out[0, 1].item() == 1e-6 and out[1].abs().sum().item() == 0.0


def test_a_hand_driven_model_without_time_gets_a_named_error():
    srcs = SpeciesSources([ConstantSource(1, 1e-6)], species=[0], layer=_layer(1))
    d = _drivers([1.0], 1)
    del d["time"]
    with pytest.raises(KeyError, match="'time'"):
        srcs({"species.x": torch.zeros(2, dtype=F64)}, d)


def test_project_to_model_attaches_the_projects_sources():
    p = read_prj(DATA / "doorway_damper_fan.prj")
    model, state, drivers = project_to_model(p)
    (srcs,) = [c for c in model.closures if isinstance(c, SpeciesSources)]
    assert drivers["species.source_multiplier"].tolist() == [1.0]
    assert drivers["time"].item() == 0.0
    assert p.sources[0].species == 0
    out = srcs(state, drivers)["species.sources"]
    zone = model.net.node_index(p.zone_nr_to_name[p.sources[0].zone_nr])
    assert out[zone].sum().item() == pytest.approx(1e-4)


def test_a_burst_delivers_its_mass_over_the_steps_that_overlap_it():
    b = BurstSource(1, mass=3.0, t_burst=0.0, dt=300.0)
    x = torch.zeros(3, dtype=F64)
    assert b(300.0, x, step=300.0)[1].item() == pytest.approx(0.01)        # the burst's own step
    assert b(300.0, x, step=900.0)[1].item() == pytest.approx(3.0 / 900.0)  # a longer step
    assert b(600.0, x, step=300.0)[1].item() == 0.0                         # after it
    assert b(150.0, x, step=300.0)[1].item() == pytest.approx(0.005)       # window [-150, 150]
    assert b(150.0, x, step=150.0)[1].item() == pytest.approx(0.01)


def test_burst_rate_times_step_sums_to_the_mass_over_any_contiguous_grid():
    b = BurstSource(1, mass=3.0, t_burst=100.0, dt=300.0)          # burst on [100, 400)
    x = torch.zeros(3, dtype=F64)
    for edges in ([0.0, 50.0, 250.0, 500.0], [0.0, 400.0, 1000.0], [0.0, 100.0, 400.0, 450.0],
                  [90.0, 130.0, 399.0, 401.0, 600.0]):
        total = sum(b(t1, x, step=t1 - t0)[1].item() * (t1 - t0)
                    for t0, t1 in zip(edges, edges[1:], strict=False))
        assert total == pytest.approx(3.0, rel=1e-14)


def test_burst_step_must_be_positive():
    with pytest.raises(ValueError, match="step length must be positive"):
        BurstSource(1, 1.0, 0.0, 60.0)(0.0, torch.zeros(3, dtype=F64), step=0.0)


def test_an_unscheduled_burst_at_t0_releases_its_full_mass_under_simulate():
    mass = 2.5e-3
    p = read_prj(DATA / "doorway_damper_fan.prj")
    p.sources[0].source_type, p.sources[0].params = "brs", [mass, 0.0]
    model, state, drivers = project_to_model(p)
    dt = 300.0
    out = simulate(model, state, drivers, [0.0, dt, 2 * dt])
    x, q = out["species.x"][:, 0, 0], out["air.q"]
    assert x[0].item() == 0.0 and x[1].item() > 0.0
    # Implicit Euler on the single zone, with the layer's own advection operator A (per unit
    # capacity, dx/dt = A x + N x_ambient, the ambient's sarin being zero): the mass released
    # in a step is M (x1 - x0) - dt M A x1, the same expression as the layer's update.
    A, _ = model.transport["species"].operator(q[1])
    zone = model.net.node_index(p.zone_nr_to_name[p.sources[0].zone_nr])
    M = (drivers["rho"] * model.closures[0].volumes)[zone].item()
    a0 = A[0, 0].item()                                          # sarin on the only zone
    released = [M * (x[k + 1] - x[k]).item() - dt * M * a0 * x[k + 1].item() for k in (0, 1)]
    assert released[0] == pytest.approx(mass, rel=1e-12)
    assert released[1] == pytest.approx(0.0, abs=1e-12 * mass)    # nothing in step 2


def test_a_project_with_sources_passes_the_setup_check():
    model, state, drivers = project_to_model(read_prj(DATA / "doorway_damper_fan.prj"))
    assert not model.check(state, drivers).errors


def _sealed(sources, volume=300.0):
    """One zone `z` (node 1), no airflow: what a sink or source alone does to it."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("z", volume=volume)
    net.add_edge("ambient", "z", kind="airpath")
    layer = species_layer(net, n_species=1)
    srcs = SpeciesSources(sources, species=[0] * len(sources), layer=layer)

    def still(state, drivers):
        return {"species.q": torch.zeros(1, dtype=F64)}

    return Model(net, {"species": layer}, closures=[still, srcs]), srcs


def test_a_deposition_velocity_sink_removes_at_v_a_over_v():
    srcs = SpeciesSources([DepositionVelocitySink(1, 1e-3, 10.0)], species=[0],
                          layer=_layer(1))
    out = srcs({"species.x": torch.zeros(2, dtype=F64)}, _drivers([1.0], 1))
    assert out["species.sources"].abs().sum().item() == 0.0
    torch.testing.assert_close(out["species.removal"],
                               torch.tensor([[1e-3 * 10.0 / 10.0], [0.0]], dtype=F64),
                               rtol=1e-14, atol=0)


def test_a_deposition_rate_sink_removes_at_its_rate_times_its_multiplier():
    srcs = SpeciesSources([DepositionRateSink(2, 5e-4)], species=[0], layer=_layer(1))
    out = srcs({"species.x": torch.zeros(2, dtype=F64)}, _drivers([0.5], 1))
    torch.testing.assert_close(out["species.removal"],
                               torch.tensor([[0.0], [2.5e-4]], dtype=F64), rtol=1e-14, atol=0)


def test_constant_and_cutoff_sources_split_into_generation_and_removal():
    srcs = SpeciesSources([ConstantSource(1, 2e-6, 0.05), CutoffSource(2, 1e-6, 2e-4)],
                          species=[0, 0], layer=_layer(1))
    out = srcs({"species.x": torch.tensor([1e-4, 5e-4], dtype=F64)}, _drivers([1.0, 1.0], 1))
    torch.testing.assert_close(out["species.sources"],
                               torch.tensor([0.0, 2e-6, 1e-6], dtype=F64))
    M = RHO_0 * torch.tensor([10.0, 20.0], dtype=F64)
    torch.testing.assert_close(out["species.removal"][:, 0],
                               torch.tensor([0.05, 1e-6 / 2e-4], dtype=F64) / M,
                               rtol=1e-14, atol=0)


def test_a_sink_is_implicit_in_the_species_step():
    """ContamX holds a deposition sink implicitly (measured): x1 = x0 / (1 + v A / V dt)."""
    model, _ = _sealed([DepositionVelocitySink(1, 2e-3, 10.0)])
    x1 = model.step({"species.x": torch.tensor([1e-3], dtype=F64)}, _drivers([1.0], 1),
                    300.0)["species.x"]
    assert x1.item() == pytest.approx(1e-3 / (1 + 2e-3 * 10.0 / 300.0 * 300.0), rel=1e-13)


def test_a_cutoff_source_above_its_cutoff_removes_as_contamx_does():
    """ContamX 3.4.1.7: a sealed 300 m3 zone at 2e-4 kg/kg with a cutoff source G = 1e-5
    kg/s, C_cut = 1e-4 falls to 1.9233185583e-4 in one 300 s step: the implicit, unclamped
    G (1 - C / C_cut)."""
    s = CutoffSource(node=1, G=1e-5, x_cut=1e-4)
    assert s(0.0, torch.tensor([0.0, 2e-4], dtype=F64))[1].item() == pytest.approx(-1e-5)
    model, _ = _sealed([s])
    x1 = model.step({"species.x": torch.tensor([2e-4], dtype=F64)}, _drivers([1.0], 1),
                    300.0)["species.x"].item()
    M = RHO_0 * 300.0
    assert x1 == pytest.approx((M * 2e-4 + 1e-5 * 300.0) / (M + 1e-5 * 300.0 / 1e-4),
                               rel=1e-13)
    assert x1 == pytest.approx(1.9233185583e-4, rel=1e-6)


def test_a_batched_multiplier_gives_a_batched_removal_and_a_batched_step():
    model, srcs = _sealed([DepositionRateSink(1, 1e-3)])
    d = _drivers([[1.0], [0.5]], 1)
    x0 = {"species.x": torch.tensor([1e-3], dtype=F64)}
    assert srcs(x0, d)["species.removal"].shape == (2, 1, 1)
    x1 = model.step(x0, d, 300.0)["species.x"]
    assert x1.shape == (2, 1)
    torch.testing.assert_close(x1[:, 0], 1e-3 / (1 + torch.tensor([1e-3, 0.5e-3],
                                                                  dtype=F64) * 300.0))


def test_the_power_law_source_counts_hours_and_holds_until_t_p():
    """ContamX: a = 1e-6 kg/s, b = 0.5, t_p = 600 s emits 2.449490e-6 kg/s over the steps
    ending 300 and 600 s and 2.0e-6 over the one ending 900 s: the power is taken of the
    time in hours."""
    s = PowerLawDecaySource(1, 1e-6, 0.5, 600.0)
    x = torch.zeros(3, dtype=F64)
    assert s.generation(300.0, x, None)[1].item() == pytest.approx(1e-6 * 6.0 ** 0.5,
                                                                   rel=1e-14)
    assert s.generation(900.0, x, None)[1].item() == pytest.approx(2e-6, rel=1e-14)
    assert s(1200.0, x)[1].item() == pytest.approx(1e-6 * 3.0 ** 0.5, rel=1e-14)
    with pytest.raises(ValueError, match="t_p must be positive"):
        PowerLawDecaySource(1, 1e-6, 0.5, 0.0)


def test_the_peak_source_peaks_at_its_peak_time():
    """ContamX: a = 1e-6 kg/s, b = 1, t_p = 900 s emits 5.469081e-7, 9.210868e-7 and
    1.0e-6 kg/s over the steps ending 300, 600 and 900 s."""
    s = PeakEmissionSource(1, 1e-6, 1.0, 900.0)
    x = torch.zeros(3, dtype=F64)
    for t, ref in ((300.0, 5.469081e-7), (600.0, 9.210868e-7), (900.0, 1e-6)):
        assert s.generation(t, x, None)[1].item() == pytest.approx(ref, rel=1e-6)
    assert s.generation(0.0, x, None).abs().sum().item() == 0.0


def test_sink_parameters_carry_gradients():
    v = torch.tensor(1e-3, dtype=F64, requires_grad=True)
    srcs = SpeciesSources([DepositionVelocitySink(1, v, 10.0)], species=[0], layer=_layer(1))
    r = srcs({"species.x": torch.zeros(2, dtype=F64)}, _drivers([1.0], 1))["species.removal"]
    (g,) = torch.autograd.grad(r.sum(), v)
    assert g.item() == pytest.approx(10.0 / 10.0, rel=1e-12)


def test_assemble_sources_refuses_a_sink_by_name():
    """A sink's removal needs the zone's air mass, which `assemble_sources` does not have."""
    x = torch.zeros(3, dtype=F64)
    out = assemble_sources([PowerLawDecaySource(1, 1e-6, 0.5, 600.0)], 300.0, x)
    assert out[1].item() == pytest.approx(1e-6 * 6.0 ** 0.5, rel=1e-14)
    with pytest.raises(TypeError, match=r"source 2 \(DepositionRateSink\).*SpeciesSources"):
        assemble_sources([ConstantSource(1, 1e-6), DepositionRateSink(1, 1e-3)], 0.0, x)


def test_a_source_without_removal_writes_no_removal_driver():
    srcs = SpeciesSources([ConstantSource(1, 1e-6), BurstSource(2, 1.0, 0.0, 60.0)],
                          species=[0, 0], layer=_layer(1))
    assert srcs.outputs == ("species.sources", "species.removal")   # ConstantSource has R
    plain = SpeciesSources([BurstSource(2, 1.0, 0.0, 60.0)], species=[0], layer=_layer(1))
    assert plain.outputs == ("species.sources",)
    d = _drivers([1.0], 1) | {"time": torch.tensor(60.0, dtype=F64), "time_step": 120.0}
    out = plain({"species.x": torch.zeros(2, dtype=F64)}, d)
    assert set(out) == {"species.sources"}
    assert out["species.sources"][2].item() == pytest.approx(1.0 / 120.0)  # step-integrated


def _layer_with_a_loose_zone():
    """`_layer(1)` plus a zone `c` (node 3) that no airflow path touches."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("a", volume=10.0)
    net.add_node("b", volume=20.0)
    net.add_node("c", volume=30.0)
    net.add_edge("ambient", "a", kind="airpath")
    net.add_edge("a", "b", kind="airpath")
    return species_layer(net, n_species=1)


@pytest.mark.parametrize("src", [DepositionRateSink(0, 1e-3), ConstantSource(0, 1e-6, 0.1),
                                 PowerLawDecaySource(0, 1e-6, 0.5, 600.0)])
def test_a_source_on_the_ambient_is_refused_by_name(src):
    with pytest.raises(ValueError, match=r"sink 1 .*boundary node 'ambient'"):
        SpeciesSources([src], species=[0], layer=_layer(1), names=["sink 1"])


@pytest.mark.parametrize("src", [DepositionRateSink(3, 1e-3), ConstantSource(3, 1e-6)])
def test_a_source_on_a_zone_no_flow_edge_touches_is_refused_by_name(src):
    with pytest.raises(ValueError, match=r"source 1 .*node 'c'.*no flow edge"):
        SpeciesSources([src], species=[0], layer=_layer_with_a_loose_zone())


def test_an_object_outside_the_source_protocol_is_refused_by_name():
    class Removal:
        node = 1

        def removal(self, t, x, air_mass, density):
            return torch.zeros_like(x)

    with pytest.raises(TypeError, match=r"odd \(Removal\).*generation"):
        SpeciesSources([Removal()], species=[0], layer=_layer(1), names=["odd"])


def test_a_velocity_sink_needs_a_positive_volume():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("z")                                   # no volume attribute
    net.add_edge("ambient", "z", kind="airpath")
    layer = TransportLayer(net, "species", capacity=torch.tensor([360.0], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"])
    with pytest.raises(ValueError, match=r"node 'z'.*positive volume"):
        SpeciesSources([DepositionVelocitySink(1, 1e-3, 10.0)], species=[0], layer=layer)
    SpeciesSources([DepositionRateSink(1, 1e-3)], species=[0], layer=layer)   # no density


def test_a_layer_with_its_own_removal_is_refused_when_sources_would_replace_it():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("z", volume=300.0)
    net.add_edge("ambient", "z", kind="airpath")
    layer = TransportLayer(net, "species", capacity=torch.tensor([360.0], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"],
                           removal=torch.tensor([1e-4], dtype=F64))
    with pytest.raises(ValueError, match=r"construction-time removal.*DepositionRateSink"):
        SpeciesSources([DepositionRateSink(1, 1e-3)], species=[0], layer=layer)
    SpeciesSources([PowerLawDecaySource(1, 1e-6, 0.5, 600.0)], species=[0], layer=layer)


def test_peak_and_cutoff_parameters_are_checked():
    with pytest.raises(ValueError, match="t_peak must be positive"):
        PeakEmissionSource(1, 1e-6, 1.0, 0.0)
    with pytest.raises(ValueError, match="width must be positive"):
        PeakEmissionSource(1, 1e-6, -1.0, 900.0)
    with pytest.raises(ValueError, match="x_cut must be positive"):
        CutoffSource(1, 1e-6, 0.0)

