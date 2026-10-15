"""Filters on airflow paths: constant efficiency, particle size, filters in series, UVGI."""

from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import gradcheck

from noodl.apps.building_physics.filters import (
    AirThroughFilter,
    ConstantEfficiencyFilter,
    FilterSeries,
    FilterTransmission,
    ParticleFilter,
    UVGIFilter,
    filter_transmission,
)
from noodl.apps.building_physics.species import Species
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
SARIN = Species("sarin", molar_mass=140.09)
PM = Species("pm", mean_diameter=2e-6, effective_density=1000.0)
# The curve measured against ContamX 3.4.1.7 (efficiencies at 0.7, 1.5 and 2.5 um below).
MERV = ParticleFilter([0.3e-6, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6],
                      [0.1, 0.1, 0.4, 0.65, 0.8, 0.8], name="merv")


def test_a_constant_efficiency_filter_acts_on_the_species_it_lists_only():
    f = ConstantEfficiencyFilter({"sarin": 0.3})
    torch.testing.assert_close(f.efficiency([SARIN, PM]), torch.tensor([0.3, 0.0], dtype=F64))


def test_a_constant_efficiency_filter_naming_an_unknown_species_is_refused():
    with pytest.raises(KeyError, match=r"\['ozone'\]"):
        ConstantEfficiencyFilter({"sarin": 0.3, "ozone": 0.1}).efficiency([SARIN])
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConstantEfficiencyFilter({"sarin": 1.5})


def test_the_particle_curve_is_contamx_s_clamped_spline_in_log_diameter():
    """ContamX: 0.203118324, 0.544660687, 0.757098794 (float32). The float64 references are
    scipy's CubicSpline(log10 d[1:-1], e[1:-1], bc_type='clamped')."""
    for d, ref64, contamx in ((0.7e-6, 0.20311833998640128, 0.203118324),
                              (1.5e-6, 0.5446606796932887, 0.544660687),
                              (2.5e-6, 0.7570988051763687, 0.757098794)):
        assert MERV.at(d).item() == pytest.approx(ref64, rel=1e-12)
        assert MERV.at(d).item() == pytest.approx(contamx, rel=2e-7)


def test_the_particle_curve_passes_its_points_and_is_flat_beyond_them():
    assert MERV.at(1e-6).item() == pytest.approx(0.4, rel=1e-14)
    assert MERV.at(2e-6).item() == pytest.approx(0.65, rel=1e-14)
    assert MERV.at(0.4e-6).item() == pytest.approx(0.1, rel=1e-14)      # flat first segment
    assert MERV.at(0.1e-6).item() == pytest.approx(0.1, rel=1e-14)      # below the curve
    assert MERV.at(1e-5).item() == pytest.approx(0.8, rel=1e-14)        # above it
    torch.testing.assert_close(MERV.efficiency([SARIN, PM]),
                               torch.tensor([0.0, 0.65], dtype=F64))   # a gas passes


def test_a_particle_curve_without_flat_ends_is_refused():
    with pytest.raises(ValueError, match=r"'steep'.*first two and the last two"):
        ParticleFilter([1e-6, 2e-6, 3e-6, 4e-6], [0.2, 0.4, 0.6, 0.8], name="steep")
    with pytest.raises(ValueError, match="increasing"):
        ParticleFilter([2e-6, 1e-6, 3e-6], [0.1, 0.1, 0.1])


def test_short_flat_curves_are_constant():
    assert ParticleFilter([1e-6, 2e-6], [0.3, 0.3]).at(5e-6).item() == 0.3
    assert ParticleFilter([1e-6, 2e-6, 3e-6], [0.3, 0.3, 0.3]).at(1.5e-6).item() == 0.3


def test_the_particle_curve_carries_gradients_to_its_efficiencies():
    """gradcheck over the curve's free values (each flat end is one value, given twice), at
    diameters on the flat ends, between points and beyond the curve."""
    d = [0.3e-6, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6]
    at = torch.tensor([0.1e-6, 0.4e-6, 0.7e-6, 1.5e-6, 2.5e-6, 4e-6, 1e-5], dtype=F64)

    def curve(free):
        e = torch.cat([free[:1], free[:1], free[1:3], free[3:], free[3:]])
        return ParticleFilter(d, e).at(at)

    free = torch.tensor([0.1, 0.4, 0.65, 0.8], dtype=F64, requires_grad=True)
    assert gradcheck(curve, (free,), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_filters_in_series_pass_the_product_of_their_stages():
    """ContamX: two constant-efficiency stages of 0.1 and 0.2 remove 0.2800000310."""
    two = FilterSeries([ConstantEfficiencyFilter({"sarin": 0.1}),
                        ConstantEfficiencyFilter({"sarin": 0.2})])
    assert two.efficiency([SARIN]).item() == pytest.approx(0.28, rel=1e-14)
    both = FilterSeries([ConstantEfficiencyFilter({"sarin": 0.3}), MERV])
    torch.testing.assert_close(both.efficiency([SARIN, PM]),
                               torch.tensor([0.3, 0.65], dtype=F64))


def test_filter_transmission_is_one_minus_the_efficiency_in_flow_order():
    net = Network(dtype=F64)
    for node in ("ambient", "a", "b"):
        net.add_node(node)
    door = net.add_edge("a", "b", kind="door")
    leak = net.add_edge("ambient", "a", kind="airpath")
    net.add_edge("b", "ambient", kind="airpath")
    t = filter_transmission(net, ("airpath", "door"), [SARIN, PM],
                            {door: MERV, leak: ConstantEfficiencyFilter({"sarin": 0.3})})
    expected = torch.tensor([[0.7, 1.0], [1.0, 1.0], [1.0, 0.35]], dtype=F64)
    torch.testing.assert_close(t, expected)
    with pytest.raises(KeyError, match=r"kind 'door'.*\('airpath',\)"):
        filter_transmission(net, ("airpath",), [SARIN], {door: MERV})


def test_a_batched_differentiable_transmission_steps_through_a_transport_layer():
    """Two filter designs as one batch of transmissions: each batch member's zone value is
    its own pass fraction times the unfiltered one, and the gradient reaches the filters'
    efficiencies through the step."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("Z")
    supply = net.add_edge("ambient", "Z", kind="airpath")
    net.add_edge("Z", "ambient", kind="airpath")
    capacity, q, dt = 360.0, 0.1, 300.0
    x_amb = torch.tensor([[1e-3, 2e-3]], dtype=F64)

    def transmission(designs):                       # rows: (eta_sarin, lo, hi)
        rows = []
        for eta, lo, hi in designs:
            flt = FilterSeries([ConstantEfficiencyFilter({"sarin": eta}),
                                ParticleFilter([0.5e-6, 1e-6, 3e-6, 5e-6],
                                               torch.stack([lo, lo, hi, hi]))])
            rows.append(filter_transmission(net, ("airpath",), [SARIN, PM], {supply: flt}))
        return torch.stack(rows)                     # (B, b_flow, K)

    def step(designs):
        layer = TransportLayer(net, "species", capacity=torch.tensor([capacity], dtype=F64),
                               flow_kind="airpath", boundary=["ambient"], n_species=2,
                               transmission=transmission(designs))
        return layer.step(torch.zeros(1, 2, dtype=F64), torch.tensor([q, q], dtype=F64),
                          torch.zeros(2, 2, dtype=F64), x_amb, dt)

    designs = torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.9]], dtype=F64, requires_grad=True)
    x = step(designs)
    assert x.shape == (2, 1, 2)
    passed = transmission(designs)[:, 0, :]          # the supply edge's pass fractions
    unfiltered = x_amb * (1.0 - math.exp(-q * dt / capacity))
    torch.testing.assert_close(x[:, 0, :], passed * unfiltered, rtol=1e-12, atol=0)
    assert gradcheck(step, (designs,), eps=1e-6, atol=1e-10, rtol=1e-6)


def test_a_filter_acts_on_air_flowing_against_its_edges_declared_direction():
    """The edge is declared zone -> ambient, the air flows ambient -> zone (q < 0): the
    filter still passes only its pass fraction of the incoming ambient air."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("Z")
    exhaust = net.add_edge("Z", "ambient", kind="airpath")      # declared outward
    net.add_edge("ambient", "Z", kind="airpath")
    capacity, q, dt = 360.0, 0.1, 300.0
    x_amb = torch.tensor([[1e-3]], dtype=F64)
    t = filter_transmission(net, ("airpath",), [SARIN],
                            {exhaust: ConstantEfficiencyFilter({"sarin": 0.3})})
    layer = TransportLayer(net, "species", capacity=torch.tensor([capacity], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"], transmission=t)
    # In through the filtered edge (against its direction), out through the other one.
    x = layer.step(torch.zeros(1, 1, dtype=F64), torch.tensor([-q, -q], dtype=F64),
                   torch.zeros(2, 1, dtype=F64), x_amb, dt)
    unfiltered = 1e-3 * (1.0 - math.exp(-q * dt / capacity))
    assert x.reshape(-1).item() == pytest.approx(0.7 * unfiltered, rel=1e-12)


def test_a_particle_curve_given_as_tensors_keeps_their_gradients():
    lo = torch.tensor(0.1, dtype=F64, requires_grad=True)
    hi = torch.tensor(0.8, dtype=F64, requires_grad=True)
    f = ParticleFilter([0.5e-6, 1e-6, 3e-6, 5e-6], [lo, lo, hi, hi])
    g_lo, g_hi = torch.autograd.grad(f.at(1e-7) + f.at(1e-5), (lo, hi))
    assert g_lo.item() == pytest.approx(1.0) and g_hi.item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must hold scalars"):
        ParticleFilter([1e-6, 2e-6], [lo, torch.tensor([0.1, 0.1], dtype=F64)])


# UVGI: the efficiencies ContamX 3.4.1.7 gave on a path from the ambient (probes of a
# `uv1` element with S_design 0.1, u_design 2.5 m/s, F_design 0.1 kg/s, k_design 0.5 m2/J and
# a species of UV susceptibility 0.2 m2/J; flows and temperatures as measured there). The
# inputs are float32 in ContamX, hence the 1e-6 tolerance.
UV_SPECIES = [Species("bug", uv_susceptibility=0.2), PM]
UV_MEASURED = [
    # (C0..C4, F (kg/s), T_ambient (K), efficiency)
    ((50.0, 0.0, 1.0, 0.0, 0.0), 0.10034079699252058, 293.15, 0.3824392263818549),
    ((50.0, 0.0, 0.0, 0.0, 1.0), 0.10034079699252058, 293.15, 0.4035219266820882),
    ((50.0, 1.0, 0.0, 0.0, 0.0), 0.09626620142262403, 313.15, 0.5533969182346686),
    ((50.0, 0.0, 0.0, 0.01, 0.0), 0.09626620142262403, 313.15, 0.4462982143436811),
    ((50.0, 0.0, 1.0, 0.0, 0.0), 0.09626620142262403, 313.15, 0.3755293180930377),
]


def _uvgi(c=(50.0, 0.0, 0.0, 0.0, 0.0), age=None, **kw):
    params = dict(design_survival=0.1, design_velocity=2.5, design_mass_flow=0.1,
                  design_rate_constant=0.5, output_coefficients=c, age_coefficients=age)
    params.update(kw)
    return UVGIFilter(**params, name="uv")


def _air(F, T=293.15, time=0.0):
    F = torch.as_tensor(F, dtype=F64)
    T = torch.as_tensor(T, dtype=F64)
    return AirThroughFilter(F, T, 101325.0 / (287.055 * T), time)


@pytest.mark.parametrize(("c", "F", "T", "measured"), UV_MEASURED)
def test_a_uvgi_filter_matches_contamx_s_efficiency(c, F, T, measured):
    eta = _uvgi(c).efficiency(UV_SPECIES, _air(F, T))
    assert eta[0].item() == pytest.approx(measured, rel=1e-6)
    assert eta[1].item() == 0.0                       # no UV susceptibility: untouched


def test_the_lamp_output_is_clamped_between_0_and_100_percent():
    """ContamX: C0 = -10 or 0 inactivates nothing; C0 = 100 and 150 give the full output."""
    f_air = _air(0.1)
    assert _uvgi((-10.0, 0, 0, 0, 0)).efficiency(UV_SPECIES, f_air)[0].item() == 0.0
    full = _uvgi((100.0, 0, 0, 0, 0)).output(f_air)
    assert full.item() == 1.0 == _uvgi((150.0, 0, 0, 0, 0)).output(f_air).item()


@pytest.mark.parametrize(("age", "hours", "factor"), [
    ((100.0, -10.0), 0.5, 1.0),                       # above 100 % before t_100 = 1 h
    ((100.0, -10.0), 3900 / 3600, 1 - 0.1 * math.log(3900 / 3600)),   # ContamX 0.99201
    ((100.0, -10.0), 2.0, 1 - 0.1 * math.log(2.0)),
    ((90.0, 10.0), 1 / 12, 0.9 + 0.1 * math.log(1 / 12)),             # rising: no hold
    ((50.0, 0.0), 1.0, 0.5),                          # K1 = 0: K0 / 100 from the start
    ((150.0, 0.0), 1.0, 1.0),
    ((5.0, 0.0), 1.0, 0.05),
    ((0.0, 0.0), 1.0, 0.0),
    ((-10.0, 0.0), 1.0, 0.1),                         # a negative output: ContamX's 0.1
    ((10.0, -50.0), 1.1667, 0.1 - 0.5 * math.log(1.1667)),            # 0.023: passes
    ((10.0, -50.0), 1.25, 0.1),
])
def test_the_lamp_age_factor_follows_contamx(age, hours, factor):
    got = _uvgi(age=age).age_factor(hours * 3600.0)
    assert got.item() == pytest.approx(factor, rel=1e-12, abs=1e-15)


def test_the_age_factor_at_the_start_is_the_formulas_limit():
    assert _uvgi(age=(100.0, -10.0)).age_factor(0.0).item() == 1.0
    assert _uvgi(age=(50.0, 10.0)).age_factor(0.0).item() == 0.1
    assert _uvgi(age=(50.0, 0.0)).age_factor(0.0).item() == 0.5


def test_air_against_the_edge_gets_contamx_s_fixed_dose():
    """ContamX gives a UVGI filter crossed against its path's direction a dose of 1 J/m2,
    whatever the filter: 1 - exp(-0.2) for a susceptibility of 0.2 m2/J, even with no lamp
    output at all."""
    for flt in (_uvgi(), _uvgi((0.0, 0, 0, 0, 0)), _uvgi(design_survival=0.05)):
        eta = flt.efficiency(UV_SPECIES, _air(-0.1))
        assert eta[0].item() == pytest.approx(1 - math.exp(-0.2), rel=1e-15)


def test_zero_flow_inactivates_everything_with_finite_gradients():
    params = {k: torch.tensor(v, dtype=F64, requires_grad=True)
              for k, v in dict(design_survival=0.1, design_velocity=2.5, design_mass_flow=0.1,
                               design_rate_constant=0.5).items()}
    c = torch.tensor([50.0, 1.0, 1.0, 0.0, 0.0], dtype=F64, requires_grad=True)
    F = torch.tensor([0.0, 0.1, -0.1], dtype=F64, requires_grad=True)
    flt = UVGIFilter(**params, output_coefficients=c, age_coefficients=(100.0, -10.0))
    eta = flt.efficiency(UV_SPECIES, _air(F, time=7200.0))
    assert eta[0, 0].item() == 1.0 and eta[0, 1].item() == 0.0
    grads = torch.autograd.grad(eta.sum(), [F, c, *params.values()])
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert grads[0][0].item() == 0.0


def test_the_efficiency_is_differentiable_in_the_filter_and_the_air():
    def eta(S, u, Fd, k, c, ages, F, T):
        flt = UVGIFilter(design_survival=S, design_velocity=u, design_mass_flow=Fd,
                         design_rate_constant=k, output_coefficients=c,
                         age_coefficients=ages)
        rho = 101325.0 / (287.055 * T)
        return flt.efficiency(UV_SPECIES, AirThroughFilter(F, T, rho, 7200.0))[..., 0]

    args = [torch.tensor(v, dtype=F64, requires_grad=True) for v in (0.1, 2.5, 0.1, 0.5)]
    args += [torch.tensor([40.0, 0.5, 2.0, -0.01, -0.5], dtype=F64, requires_grad=True),
             torch.tensor([100.0, -10.0], dtype=F64, requires_grad=True),
             torch.tensor([0.08, 0.12], dtype=F64, requires_grad=True),
             torch.tensor(298.15, dtype=F64, requires_grad=True)]
    assert gradcheck(eta, args, eps=1e-6, atol=1e-9, rtol=1e-6)


def test_a_uvgi_filter_has_no_fixed_transmission():
    net = Network(dtype=F64)
    for node in ("ambient", "a"):
        net.add_node(node)
    edge = net.add_edge("ambient", "a", kind="airpath")
    series = FilterSeries([ConstantEfficiencyFilter({"bug": 0.3}), _uvgi()])
    assert series.depends_on_flow and not MERV.depends_on_flow
    with pytest.raises(ValueError, match=r"filter 'uv'.*FilterTransmission"):
        filter_transmission(net, ("airpath",), UV_SPECIES, {edge: _uvgi()})
    with pytest.raises(ValueError, match="pass `air`"):
        _uvgi().efficiency(UV_SPECIES)


def _filtered_network():
    net = Network(dtype=F64)
    for node in ("ambient", "a", "b"):
        net.add_node(node)
    supply = net.add_edge("ambient", "a", kind="airpath")
    middle = net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "ambient", kind="airpath")
    return net, supply, middle


class _Air:
    """The airflow layer's flow lookup, as `FilterTransmission` reads it."""

    name = "air"

    @staticmethod
    def flows_of_kind(q, kinds):
        return q


def test_the_filter_transmission_closure_follows_the_flows():
    net, supply, middle = _filtered_network()
    uv = _uvgi((50.0, 0.0, 1.0, 0.0, 0.0))
    cef = ConstantEfficiencyFilter({"bug": 0.3})
    seen = []
    closure = FilterTransmission(net, ("airpath",), UV_SPECIES,
                                 {supply: uv, middle: FilterSeries([cef, uv], name="both")},
                                 airflow=_Air(), on_reverse=seen.append)
    T = torch.tensor([293.15, 303.15, 293.15], dtype=F64)
    drivers = {"T": T, "rho": 101325.0 / (287.055 * T), "time": torch.tensor(0.0)}
    assert closure({}, drivers) == {}                  # no flows yet: nothing written
    q = torch.tensor([0.10034079699252058, -0.05, 0.1], dtype=F64)
    tau = closure({"air.q": q}, drivers)["species.transmission"]
    assert tau.shape == (3, 2)
    assert tau[0, 0].item() == pytest.approx(1 - 0.3824392263818549, rel=1e-6)
    assert tau[1, 0].item() == pytest.approx(0.7 * math.exp(-0.2), rel=1e-14)   # reverse
    assert tau[2].tolist() == [1.0, 1.0] and tau[:, 1].tolist() == [1.0, 1.0, 1.0]
    closure({"air.q": q}, drivers)
    assert [f.name for f in seen] == ["both"]          # once per filter
