"""ContamX's duct laws, against ContamX 3.4.1.7's own numbers (probes c2b-c2i, 11 Oct 2026).

ContamX evaluates these laws in single precision from float32 inputs. The tolerance for its
numbers is a first-order budget of that: the density and viscosity it derives from a float32
temperature (1 u each), the area (1 u, squared), the length, diameter and loss coefficient
(1 u each), the flow (1 u, squared), and the float32 arithmetic of its friction-factor
iteration (about 16 u in f): 32 u (1.9e-6) in the pressure drop, at most the same in the
flow. The probes measured at most 28 u."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.drives import DuctStack
from noodl.elements import AirDuct, DuctTerminal, FixedFlow, air_viscosity
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
U = 2.0**-24
TOL = 32 * U
R, PB = 287.055, 101325.0


def rho(T):
    return PB / (R * T)


def _duct(roughness=9e-5, loss=0.0, length=10.0, diameter=0.2, temperature_key="T"):
    return AirDuct(torch.tensor([length], dtype=F64), torch.tensor([diameter], dtype=F64),
                   torch.tensor([roughness], dtype=F64), torch.tensor([loss], dtype=F64),
                   src=[0], tgt=[1], temperature_key=temperature_key)


def _drivers(T_src=293.15, T_tgt=293.15):
    T = torch.tensor([T_src, T_tgt], dtype=F64)
    return {"rho": PB / (R * T), "T": T}


def test_air_viscosity_is_contamx_s_linear_law():
    assert air_viscosity(293.15).item() == pytest.approx(1.81625e-5, rel=1e-12)
    assert air_viscosity(313.15).item() == pytest.approx(1.81625e-5 + 20 * 4.92859e-8,
                                                         rel=1e-12)


@pytest.mark.parametrize("label, dp, flow, roughness, loss, T", [
    ("laminar, Re 1052", 0.011523253284394741, 0.003000000026077032, 9e-5, 0.0, 293.15),
    ("transition, Re 2800", 0.045018408447504044, 0.007988277822732925, 1e-7, 0.0, 293.15),
    ("turbulent, Re 17500", 1.4536845684051514, 0.05000000074505806, 9e-5, 0.0, 293.15),
    ("rough, Re 70100", 26.661516189575195, 0.20000000298023224, 1e-3, 0.0, 293.15),
    ("dynamic losses 2", 3.557359218597412, 0.05000000074505806, 9e-5, 2.0, 293.15),
    ("laminar at 313 K", 0.012977481819689274, 0.003000000026077032, 9e-5, 0.0, 313.15),
    ("turbulent at 313 K", 1.5715057849884033, 0.05000000074505806, 9e-5, 0.0, 313.15),
])
def test_the_duct_law_reproduces_contamx(label, dp, flow, roughness, loss, T):
    """A 10 m, 0.2 m duct (area pi D^2 / 4) at junction temperature T."""
    q = _duct(roughness, loss).flow(torch.tensor([dp], dtype=F64), _drivers(T, T))
    assert q.item() == pytest.approx(flow, rel=TOL), label


def test_reversed_flow_takes_the_targets_density_and_viscosity():
    """c2i: junctions at 273.15 and 313.15 K; the loss uses the upstream one's rho and mu."""
    duct = _duct()
    d = _drivers(273.15, 313.15)
    forward = duct.flow(torch.tensor([1.337726], dtype=F64), d)
    backward = duct.flow(torch.tensor([-1.571506], dtype=F64), d)
    assert forward.item() == pytest.approx(0.05, rel=TOL)
    assert backward.item() == pytest.approx(-0.05, rel=TOL)
    torch.testing.assert_close(duct.flow(torch.tensor([-1.337726], dtype=F64),
                                         _drivers(313.15, 273.15)), -forward)


def test_without_a_temperature_driver_the_viscosity_is_at_293_k():
    q = _duct(temperature_key=None).flow(torch.tensor([0.011523253284394741], dtype=F64),
                                         {"rho": torch.full((2,), rho(293.15), dtype=F64)})
    assert q.item() == pytest.approx(0.003, rel=TOL)


def test_the_duct_flow_is_differentiable_and_its_slope_is_the_laminar_one_at_zero():
    duct = _duct()
    d = _drivers()
    slope = duct.dflow(torch.zeros(1, dtype=F64), d)
    A = math.pi * 0.04 / 4
    a = 32 * air_viscosity(293.15).item() * 10.0 / (rho(293.15) * A * 0.04)
    assert slope.item() == pytest.approx(1 / a, rel=1e-10)
    D = torch.tensor([0.2], dtype=F64, requires_grad=True)
    q = AirDuct(torch.tensor([10.0], dtype=F64), D, torch.tensor([9e-5], dtype=F64),
                src=[0], tgt=[1]).flow(torch.tensor([1.45], dtype=F64), d)
    (g,) = torch.autograd.grad(q.sum(), D)
    eps = 1e-6
    hi = AirDuct(torch.tensor([10.0], dtype=F64), torch.tensor([0.2 + eps], dtype=F64),
                 torch.tensor([9e-5], dtype=F64), src=[0], tgt=[1])
    lo = AirDuct(torch.tensor([10.0], dtype=F64), torch.tensor([0.2 - eps], dtype=F64),
                 torch.tensor([9e-5], dtype=F64), src=[0], tgt=[1])
    dp = torch.tensor([1.45], dtype=F64)
    fd = (hi.flow(dp, d) - lo.flow(dp, d)).item() / (2 * eps)
    assert g.item() == pytest.approx(fd, rel=1e-6)


def test_a_duct_in_a_layer_carries_a_fan_s_flow_at_contamx_s_pressure_drop():
    """ambient -> room through the duct; a fixed 0.05 kg/s fan draws it out again."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("room")
    net.add_edge("ambient", "room", kind="duct")
    net.add_edge("room", "ambient", kind="fan")
    src, tgt = net.endpoints("duct")
    duct = AirDuct(torch.tensor([10.0], dtype=F64), torch.tensor([0.2], dtype=F64),
                   torch.tensor([9e-5], dtype=F64), src=src, tgt=tgt)
    layer = PotentialFlowLayer(net, "air", [duct, FixedFlow(torch.tensor([0.05], dtype=F64),
                                                             kind="fan")],
                               boundary=["ambient"])
    d = {"rho": torch.full((2,), rho(293.15), dtype=F64)}
    phi, q = layer.solve(torch.zeros(1, dtype=F64), d, atol=1e-14, rtol=1e-14)
    assert q[0].item() == pytest.approx(0.05, rel=1e-12)
    assert -phi[1].item() == pytest.approx(1.4536845684051514, rel=TOL)


@pytest.mark.parametrize("label, dp, flow, area, loss, opening, T", [
    ("Ct 0.125", 0.13147981464862823, 0.05000000074505806, 0.0314159, 0.125, None, 293.15),
    ("Ct + Cb 0.625", 0.65739905834198, 0.05000000074505806, 0.0314159, 0.625, None, 293.15),
    ("Ad 0.02", 0.3244127035140991, 0.05000000074505806, 0.02, 0.125, None, 293.15),
    ("S 0.5", 1.183319330215454, 0.05000000074505806, 0.0314159, 0.125, 0.5, 293.15),
    ("laminar 1e-6 kg/s", 5.0791477690381726e-09, 1.0000001111620804e-06, 0.0314159, 0.125,
     None, 293.15),
    ("laminar below F_t", 4.5712334895142703e-07, 9.00000159163028e-05, 0.0314159, 0.125,
     None, 293.15),
    ("quadratic above F_t", 5.25919688243448e-07, 9.999999747378752e-05, 0.0314159, 0.125,
     None, 293.15),
    ("laminar at 313 K", 5.7201354763947165e-09, 9.999999974752427e-07, 0.0314159, 0.125,
     None, 313.15),
])
def test_the_terminal_law_reproduces_contamx(label, dp, flow, area, loss, opening, T):
    term = DuctTerminal(torch.tensor([area], dtype=F64), torch.tensor([loss], dtype=F64),
                        src=[0], tgt=[1], temperature_key="T",
                        opening_key=None if opening is None else "open")
    d = _drivers(T, 293.15)
    if opening is not None:
        d["open"] = torch.tensor([opening], dtype=F64)
    assert term.flow(torch.tensor([dp], dtype=F64), d).item() == pytest.approx(flow, rel=TOL)


def test_the_terminal_opening_is_clamped_as_contamx_clamps_it():
    """S = clamp(s, 1e-6, 1): s = 0 gives Ce = Ct + (1 - 1e-6) / 1e-6 (ContamX: 1051838.75 Pa
    at 0.05 kg/s); s = 2 gives Ce = Ct (ContamX: unchanged); s < 0 is as closed as s = 0
    (TN 1887's S = max(1e-6, min(1, s)); not probed)."""
    term = DuctTerminal(torch.tensor([0.0314159], dtype=F64), torch.tensor([0.125], dtype=F64),
                        src=[0], tgt=[1], temperature_key="T", opening_key="open")
    d = _drivers()
    for s, dp in ((0.0, 1051838.75), (-0.5, 1051838.75), (2.0, 0.13147981464862823)):
        d["open"] = torch.tensor([s], dtype=F64)
        q = term.flow(torch.tensor([dp], dtype=F64), d)
        assert q.item() == pytest.approx(0.05, rel=TOL), s


def test_a_terminal_is_odd_and_continuous_at_its_laminar_switch():
    term = DuctTerminal(torch.tensor([0.0314159], dtype=F64), torch.tensor([0.125], dtype=F64),
                        src=[0], tgt=[1], temperature_key="T")
    d = _drivers()
    A = 0.0314159
    c = 0.125 / (2 * rho(293.15) * A * A)
    F_t = 30 * air_viscosity(293.15).item() * math.sqrt(A)
    dp_t = c * F_t**2
    for x in (dp_t * (1 - 1e-12), dp_t * (1 + 1e-12)):
        assert term.flow(torch.tensor([x], dtype=F64), d).item() == pytest.approx(F_t, rel=1e-9)
    dp = torch.tensor([-3e-7, 3e-7, -0.2, 0.2], dtype=F64)
    q = term.flow(dp, {"rho": torch.full((2,), rho(293.15), dtype=F64),
                       "T": torch.full((2,), 293.15, dtype=F64)})
    torch.testing.assert_close(q[0::2], -q[1::2])
    assert term.dflow(torch.zeros(1, dtype=F64), d).item() == pytest.approx(1 / (c * F_t),
                                                                            rel=1e-12)


def test_the_duct_stack_takes_the_mean_temperature_column():
    """c2h: junctions at 1.5 m and 273.15 K and at 4.5 m and 313.15 K: ContamX's column is at
    293.15 K, the harmonic mean of the two densities."""
    net = Network(dtype=F64)
    net.add_node("j1", z_ref=1.5)
    net.add_node("j2", z_ref=4.5)
    net.add_edge("j1", "j2", kind="duct")
    drive = DuctStack.from_network(net, "duct", g=9.8055)
    value = drive({"rho": torch.tensor([rho(273.15), rho(313.15)], dtype=F64)})
    assert value.item() == pytest.approx(9.8055 * rho(293.15) * (1.5 - 4.5), rel=1e-14)


def test_the_duct_flow_is_odd_zero_at_zero_and_accurate_for_tiny_flows():
    duct = _duct()
    d = _drivers()
    dp = torch.tensor([-1e-9, 1e-9, -0.04, 0.04, -3.0, 3.0, 0.0], dtype=F64)
    q = duct.flow(dp, d)
    torch.testing.assert_close(q[0:6:2], -q[1:6:2], rtol=0.0, atol=0.0)
    assert q[6].item() == 0.0
    A = math.pi * 0.04 / 4
    a = 32 * air_viscosity(293.15).item() * 10.0 / (rho(293.15) * A * 0.04)
    assert q[1].item() == pytest.approx(1e-9 / a, rel=1e-12)


def test_the_duct_law_is_continuous_at_both_reynolds_switches():
    """dp(F) is continuous at Re 2300 (64/Re meets the linear piece) and at Re 3500 (the
    linear piece meets Colebrook), so F(dp) is too."""
    duct = _duct(roughness=1e-4)
    d = _drivers()
    A = math.pi * 0.04 / 4
    mu = air_viscosity(293.15).item()
    for Re in (2300.0, 3500.0):
        F = Re * mu * A / 0.2
        F_lo = torch.tensor([F * (1 - 1e-9)], dtype=F64)
        F_hi = torch.tensor([F * (1 + 1e-9)], dtype=F64)
        dp_lo = duct.pressure_drop(F_lo, d)
        dp_hi = duct.pressure_drop(F_hi, d)
        assert dp_hi.item() == pytest.approx(dp_lo.item(), rel=1e-7), Re
        assert duct.flow(dp_lo, d).item() == pytest.approx(F_lo.item(), rel=1e-12), Re


def test_the_duct_and_terminal_work_batched_and_match_unbatched():
    duct = AirDuct(torch.tensor([10.0, 4.0], dtype=F64), torch.tensor([0.2, 0.1], dtype=F64),
                   torch.tensor([9e-5, 9e-5], dtype=F64), src=[0, 1], tgt=[1, 2],
                   temperature_key="T")
    term = DuctTerminal(torch.tensor([0.0314159, 0.02], dtype=F64),
                        torch.tensor([0.125, 0.6], dtype=F64), src=[0, 1], tgt=[1, 2],
                        temperature_key="T", opening_key="open")
    T = torch.tensor([[293.15, 303.15, 283.15], [273.15, 313.15, 293.15]], dtype=F64)
    d = {"rho": PB / (R * T), "T": T,
         "open": torch.tensor([[1.0, 0.5], [0.2, 1.0]], dtype=F64)}
    dp = torch.tensor([[0.5, -2.0], [-0.01, 3e-7]], dtype=F64)
    for el in (duct, term):
        q = el.flow(dp, d)
        assert q.shape == (2, 2)
        for b in range(2):
            row = {k: v[b] for k, v in d.items()}
            torch.testing.assert_close(q[b], el.flow(dp[b], row), rtol=1e-14, atol=0.0)
        torch.testing.assert_close(el.dflow(dp, d), _fd(el, dp, d), rtol=1e-6, atol=0.0)


def _fd(el, dp, d, h=1e-7):
    step = h * dp.abs().clamp(min=1e-3)
    return (el.flow(dp + step, d) - el.flow(dp - step, d)) / (2 * step)


def test_gradcheck_of_the_duct_and_terminal_in_their_parameters_and_drivers():
    T = torch.tensor([283.15, 303.15], dtype=F64, requires_grad=True)

    def duct_flow(L, D, eps, C, T_):
        el = AirDuct(L, D, eps, C, src=[0, 0, 0], tgt=[1, 1, 1], temperature_key="T")
        return el.flow(torch.tensor([0.003, 0.05, -2.0], dtype=F64),
                       {"rho": PB / (R * T_), "T": T_})

    args = tuple(torch.tensor([v], dtype=F64, requires_grad=True)
                 for v in (10.0, 0.2, 9e-5, 0.5)) + (T,)
    assert torch.autograd.gradcheck(duct_flow, args, eps=1e-7, atol=1e-8, rtol=1e-5)

    def term_flow(A, C, S, T_):
        el = DuctTerminal(A, C, src=[0, 0, 0], tgt=[1, 1, 1], temperature_key="T",
                          opening_key="open")
        return el.flow(torch.tensor([1e-7, 0.13, -0.5], dtype=F64),
                       {"rho": PB / (R * T_), "T": T_, "open": S})

    args = (torch.tensor([0.0314159], dtype=F64, requires_grad=True),
            torch.tensor([0.125], dtype=F64, requires_grad=True),
            torch.tensor([0.5], dtype=F64, requires_grad=True), T)
    assert torch.autograd.gradcheck(term_flow, args, eps=1e-7, atol=1e-8, rtol=1e-5)


def test_the_duct_and_terminal_refuse_bad_geometry_and_missing_drivers():
    with pytest.raises(ValueError, match="diameter must be strictly positive"):
        _duct(diameter=0.0)
    with pytest.raises(ValueError, match="roughness must be below 3.7 diameters"):
        _duct(roughness=0.75)
    with pytest.raises(ValueError, match="loss must be non-negative"):
        _duct(loss=-1.0)
    with pytest.raises(ValueError, match="area must be strictly positive"):
        DuctTerminal(torch.tensor([0.0], dtype=F64), torch.tensor([0.1], dtype=F64),
                     src=[0], tgt=[1])
    with pytest.raises(KeyError, match="'T' not found"):
        _duct().flow(torch.ones(1, dtype=F64), {"rho": torch.ones(2, dtype=F64)})
    term = DuctTerminal(torch.tensor([0.02], dtype=F64), torch.tensor([0.1], dtype=F64),
                        src=[0], tgt=[1], opening_key="open")
    with pytest.raises(KeyError, match="'open' not found"):
        term.flow(torch.ones(1, dtype=F64), {"rho": torch.ones(2, dtype=F64)})


def test_the_duct_stack_works_batched():
    stack = DuctStack("duct", src=[0, 1], tgt=[1, 2], z_ref=torch.tensor([0.0, 3.0, 1.0]))
    rho_ = torch.tensor([[1.2, 1.1, 1.3], [1.0, 1.0, 1.0]], dtype=F64)
    v = stack({"rho": rho_})
    assert v.shape == (2, 2)
    torch.testing.assert_close(v[1], 9.80665 * torch.tensor([-3.0, 2.0], dtype=F64))
    assert v[0, 0].item() == pytest.approx(9.80665 * 2 / (1 / 1.2 + 1 / 1.1) * -3.0,
                                           rel=1e-14)
    with pytest.raises(KeyError, match="DuctStack"):
        stack({})


def test_the_differentiable_layer_solve_reaches_a_learnable_duct_diameter():
    def room_pressure(diameter, differentiable):
        net = Network(dtype=F64)
        net.add_node("ambient")
        net.add_node("room")
        net.add_edge("ambient", "room", kind="duct")
        net.add_edge("room", "ambient", kind="fan")
        src, tgt = net.endpoints("duct")
        duct = AirDuct(torch.tensor([10.0], dtype=F64), torch.tensor([diameter], dtype=F64),
                       torch.tensor([9e-5], dtype=F64), src=src, tgt=tgt, learnable=True)
        fan = FixedFlow(torch.tensor([0.05], dtype=F64), kind="fan")
        layer = PotentialFlowLayer(net, "air", [duct, fan], boundary=["ambient"])
        phi, _ = layer.solve(torch.zeros(1, dtype=F64), {"rho": torch.full((2,), 1.2, dtype=F64)},
                             differentiable=differentiable, atol=1e-14, rtol=1e-14)
        return phi[1], duct

    p, duct = room_pressure(0.2, True)
    p.backward()
    eps = 1e-6
    with torch.no_grad():
        fd = (room_pressure(0.2 + eps, False)[0] - room_pressure(0.2 - eps, False)[0]).item()
    assert duct.diameter.grad.item() == pytest.approx(fd / (2 * eps), rel=1e-6)


def test_autograd_at_zero_pressure_difference_is_finite_and_correct():
    """At dp = 0: dF/ddp = 1 / a (the laminar slope) and dF/dD = 0 (no flow for any D);
    the terminal's dF/ddp is its laminar slope and dF/dA = 0."""
    d = _drivers()
    dp = torch.zeros(1, dtype=F64, requires_grad=True)
    D = torch.tensor([0.2], dtype=F64, requires_grad=True)
    duct = AirDuct(torch.tensor([10.0], dtype=F64), D, torch.tensor([9e-5], dtype=F64),
                   src=[0], tgt=[1], learnable=True)
    g_dp, g_D = torch.autograd.grad(duct.flow(dp, d).sum(), (dp, duct.diameter))
    A = math.pi * 0.04 / 4
    a = 32 * air_viscosity(293.15).item() * 10.0 / (rho(293.15) * A * 0.04)
    assert g_dp.item() == pytest.approx(1 / a, rel=1e-12)
    assert g_D.item() == 0.0
    term = DuctTerminal(torch.tensor([0.0314159], dtype=F64), torch.tensor([0.125], dtype=F64),
                        src=[0], tgt=[1], learnable=True)
    g_dp, g_A = torch.autograd.grad(term.flow(dp, d).sum(), (dp, term.area))
    assert torch.isfinite(g_dp).all() and g_A.item() == 0.0
    assert g_dp.item() == pytest.approx(term.dflow(torch.zeros(1, dtype=F64), d).item(),
                                        rel=1e-14)
