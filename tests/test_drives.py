"""Tests for the Drive protocol and ConstantDrive (milestone 2 spec section 4.1)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tellegen.drives import ConstantDrive, Drive, Stack, Wind, WindProfile, check_drive_signature
from tellegen.elements import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network


def test_constant_drive_returns_the_named_driver():
    drive = ConstantDrive("airpath", "wind")
    assert drive.kind == "airpath"
    drivers = {"wind": torch.tensor([1.0, 2.0, 3.0])}
    torch.testing.assert_close(drive(drivers), drivers["wind"])


def test_constant_drive_raises_key_error_naming_the_missing_key():
    drive = ConstantDrive("airpath", "gust")
    with pytest.raises(KeyError, match="gust"):
        drive({})


def test_drive_call_takes_the_drivers_mapping_only():
    params = list(inspect.signature(ConstantDrive.__call__).parameters)
    assert params == ["self", "drivers"]
    assert isinstance(ConstantDrive("airpath", "wind"), Drive)


def test_check_drive_signature_rejects_the_old_two_argument_form():
    class OldStyle:
        kind = "airpath"

        def __call__(self, phi, drivers):
            return drivers["wind"]

    with pytest.raises(TypeError, match=r"OldStyle.*drivers"):
        check_drive_signature(OldStyle(), where="test")
    check_drive_signature(ConstantDrive("airpath", "wind"), where="test")  # no raise


def test_layer_construction_rejects_a_two_argument_drive(two_zone_layer):
    net, elements, _, boundary = two_zone_layer

    class OldStyle:
        kind = "airpath"

        def __call__(self, phi, drivers):
            return drivers["wind"]

    with pytest.raises(TypeError, match=r"PotentialFlowLayer 'zones'.*OldStyle"):
        PotentialFlowLayer(net, "zones", elements, drives=[OldStyle()], boundary=boundary)


def test_layer_solve_feeds_drives_the_drivers_mapping_only(two_zone_layer):
    net, elements, _, boundary = two_zone_layer
    seen: dict = {}

    class Recorder:
        kind = "airpath"

        def __call__(self, drivers):
            seen["keys"] = sorted(drivers)
            return drivers["wind"]

    layer = PotentialFlowLayer(net, "zones", elements, drives=[Recorder()], boundary=boundary)
    wind = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
    phi_b = torch.zeros(1, dtype=torch.float64)
    phi, q = layer.solve(phi_b, {"wind": wind}, differentiable=False)
    assert seen["keys"] == ["wind"]
    assert torch.isfinite(q).all()
    # The same solve on the differentiable path (the functional drive call site).
    phi2, q2 = layer.solve(phi_b, {"wind": wind}, differentiable=True)
    torch.testing.assert_close(q2, q, rtol=1e-9, atol=1e-12)


F64 = torch.float64
G = 9.80665


def _two_opening_net():
    """ambient (z_ref 0) and zone (z_ref 0), joined by a low (-1 m) and a high (+1 m)
    opening, both edges ambient -> zone, kind airpath; node order ambient, zone."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    net.add_node("zone", z_ref=0.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=-1.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=1.0)
    return net


def test_stack_value_is_the_hydrostatic_difference_at_the_path_elevation():
    net = Network(dtype=F64)
    net.add_node("a", z_ref=0.0)
    net.add_node("b", z_ref=2.0)
    net.add_edge("a", "b", kind="airpath", z_path=0.5)
    stack = Stack.from_network(net, "airpath")
    rho = torch.tensor([1.2, 1.1], dtype=F64)
    expected = G * (1.2 * (0.0 - 0.5) - 1.1 * (2.0 - 0.5))
    torch.testing.assert_close(stack({"rho": rho}), torch.tensor([expected], dtype=F64))


def test_stack_is_antisymmetric_under_edge_reversal():
    fwd = Network(dtype=F64)
    rev = Network(dtype=F64)
    for net in (fwd, rev):
        net.add_node("a", z_ref=0.0)
        net.add_node("b", z_ref=3.0)
    fwd.add_edge("a", "b", kind="airpath", z_path=1.0)
    rev.add_edge("b", "a", kind="airpath", z_path=1.0)
    rho = {"rho": torch.tensor([1.25, 1.15], dtype=F64)}
    torch.testing.assert_close(
        Stack.from_network(fwd, "airpath")(rho), -Stack.from_network(rev, "airpath")(rho)
    )


def test_warm_zone_draws_air_in_low_and_pushes_it_out_high():
    net = _two_opening_net()
    layer = PotentialFlowLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.01], dtype=F64), 0.5)],
        drives=[Stack.from_network(net, "airpath")], boundary=["ambient"],
    )
    rho = torch.tensor([1.2, 1.1], dtype=F64)          # ambient dense, zone warm
    phi, q = layer.solve(torch.zeros(1, dtype=F64), {"rho": rho}, differentiable=False)
    assert q[0] > 0 and q[1] < 0                        # in low, out high
    assert (q[0] + q[1]).abs().item() < 1e-10           # conservation
    assert abs(phi[1].item()) < 1e-9                    # symmetric openings: neutral at 0
    expected = 0.01 * (0.1 * G * 1.0) ** 0.5            # C (g (rho_a - rho_z) 1 m)^0.5
    assert q[0].item() == pytest.approx(expected, rel=1e-8)


def test_stack_equals_a_constant_drive_fed_the_same_values_on_a_shaft():
    net = Network(dtype=F64)
    for i, z in enumerate((0.0, 3.0, 6.0, 9.0)):
        net.add_node(f"n{i}", z_ref=z)
    for i in range(3):
        net.add_edge(f"n{i}", f"n{i + 1}", kind="airpath", z_path=1.5 + 3.0 * i)
    rho = torch.tensor([1.20, 1.18, 1.16, 1.14], dtype=F64)
    stack = Stack.from_network(net, "airpath")
    values = stack({"rho": rho})
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)
    a = PotentialFlowLayer(net, "a", [el], drives=[stack], boundary=["n0", "n3"])
    b = PotentialFlowLayer(
        net, "b", [el], drives=[ConstantDrive("airpath", "s")], boundary=["n0", "n3"]
    )
    pb = torch.tensor([0.0, -5.0], dtype=F64)
    phi_a, q_a = a.solve(pb, {"rho": rho}, differentiable=False)
    phi_b, q_b = b.solve(pb, {"s": values}, differentiable=False)
    torch.testing.assert_close(phi_a, phi_b, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(q_a, q_b, rtol=1e-12, atol=1e-12)


def test_stack_broadcasts_a_batch_of_densities():
    net = _two_opening_net()
    stack = Stack.from_network(net, "airpath")
    rho = torch.tensor([[1.2, 1.1], [1.2, 1.2], [1.1, 1.2]], dtype=F64)
    v = stack({"rho": rho})
    assert v.shape == (3, 2)
    torch.testing.assert_close(v[1], torch.zeros(2, dtype=F64))
    torch.testing.assert_close(v[2], -v[0])


def test_stack_gradient_wrt_rho_through_the_differentiable_solve():
    net = _two_opening_net()
    layer = PotentialFlowLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.01], dtype=F64), 0.5)],
        drives=[Stack.from_network(net, "airpath")], boundary=["ambient"],
    )
    rho = torch.tensor([1.2, 1.1], dtype=F64, requires_grad=True)
    pb = torch.zeros(1, dtype=F64)
    assert torch.autograd.gradcheck(
        lambda r: layer.solve(pb, {"rho": r})[1], (rho,), eps=1e-6, atol=1e-6
    )


def test_from_network_names_a_missing_path_elevation():
    net = Network(dtype=F64)
    net.add_node("a", z_ref=0.0)
    net.add_node("b", z_ref=1.0)
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(KeyError, match="z_path"):
        Stack.from_network(net, "airpath")


def test_stack_reports_a_missing_density_driver_by_key():
    net = _two_opening_net()
    with pytest.raises(KeyError, match="rho"):
        Stack.from_network(net, "airpath")({})


def test_wind_profile_interpolates_periodically_like_numpy():
    ang = [0.0, 90.0, 180.0, 270.0]
    cp = [0.6, -0.3, -0.4, -0.3]
    prof = WindProfile(ang, cp)
    theta = torch.tensor([-30.0, 0.0, 45.0, 135.0, 359.0, 400.0], dtype=F64)
    expected = np.interp(np.mod(theta.numpy(), 360.0), ang + [360.0], cp + [cp[0]])
    torch.testing.assert_close(prof(theta), torch.tensor(expected, dtype=F64))


def test_wind_profile_accepts_a_contam_style_closing_360_row_and_refuses_bad_tables():
    prof = WindProfile([0.0, 180.0, 360.0], [0.6, -0.4, 0.6])
    assert prof(torch.tensor([360.0], dtype=F64)).item() == pytest.approx(0.6)
    with pytest.raises(ValueError, match="WindProfile.*360"):
        WindProfile([0.0, 180.0, 360.0], [0.6, -0.4, 0.1])
    with pytest.raises(ValueError, match="WindProfile.*start at 0"):
        WindProfile([10.0, 180.0], [0.6, -0.4])
    with pytest.raises(ValueError, match="WindProfile.*increasing"):
        WindProfile([0.0, 180.0, 90.0], [0.6, -0.4, 0.1])


def _wind_net():
    """ambient -> z1 (envelope, azimuth 0, Ch 0.5), z1 -> z2 (interior), z2 -> ambient
    (envelope, azimuth 180, Ch 0.5); node order ambient, z1, z2."""
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", azimuth=0.0, Ch=0.5, profile=1)
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath", azimuth=180.0, Ch=0.5, profile=1)
    return net


def test_wind_pressure_formula_sign_and_envelope_mask():
    net = _wind_net()
    prof = WindProfile([0.0, 90.0, 180.0, 270.0], [0.6, -0.3, -0.4, -0.3])
    wind = Wind.from_network(net, "airpath", ambient="ambient", profiles=[prof])
    drivers = {
        "rho_amb": torch.tensor(1.2, dtype=F64),
        "V_met": torch.tensor(4.0, dtype=F64),
        "theta_w": torch.tensor(0.0, dtype=F64),
    }
    v = wind(drivers)
    q = 0.5 * 1.2 * 16.0 * 0.5
    # edge 0: ambient is src -> +; theta_rel = 0 -> Cp 0.6.  edge 2: ambient is tgt -> -;
    # theta_rel = -180 -> Cp -0.4, so value = -(q * -0.4) = +0.4 q.  interior edge: 0.
    torch.testing.assert_close(v, torch.tensor([0.6 * q, 0.0, 0.4 * q], dtype=F64))


def test_wind_constant_cp_edges_and_batched_speed():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="airpath", Cp=0.7, Ch=1.0)
    net.add_edge("z", "ambient", kind="airpath", Cp=-0.2, Ch=1.0)
    wind = Wind.from_network(net, "airpath", ambient="ambient")
    V = torch.tensor([0.0, 2.0, 4.0], dtype=F64)
    v = wind({
        "rho_amb": torch.full((3,), 1.2, dtype=F64),
        "V_met": V,
        "theta_w": torch.zeros(3, dtype=F64),
    })
    assert v.shape == (3, 2)
    torch.testing.assert_close(v[:, 0], 0.5 * 1.2 * V**2 * 0.7)
    torch.testing.assert_close(v[:, 1], 0.5 * 1.2 * V**2 * 0.2)   # tgt is ambient: -(-0.2)


def test_wind_drives_a_through_flow_and_is_differentiable_wrt_speed_and_direction():
    net = _wind_net()
    prof = WindProfile([0.0, 90.0, 180.0, 270.0], [0.6, -0.3, -0.4, -0.3])
    layer = PotentialFlowLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.05, 0.01], dtype=F64), 0.5)],
        drives=[Wind.from_network(net, "airpath", ambient="ambient", profiles=[prof])],
        boundary=["ambient"],
    )
    pb = torch.zeros(1, dtype=F64)
    rho = torch.tensor(1.2, dtype=F64)
    phi, q = layer.solve(
        pb, {"rho_amb": rho, "V_met": torch.tensor(4.0, dtype=F64),
             "theta_w": torch.tensor(20.0, dtype=F64)}, differentiable=False,
    )
    assert q[0] > 0 and q[1] > 0 and q[2] > 0        # windward in, leeward out
    V = torch.tensor(4.0, dtype=F64, requires_grad=True)
    th = torch.tensor(20.0, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v, t: layer.solve(pb, {"rho_amb": rho, "V_met": v, "theta_w": t})[1],
        (V, th), eps=1e-6, atol=1e-6,
    )


def test_wind_from_network_names_an_unknown_ambient_and_a_missing_driver():
    net = _wind_net()
    with pytest.raises(KeyError, match="outside"):
        Wind.from_network(net, "airpath", ambient="outside")
    # _wind_net()'s envelope edges reference profile 1 (Ruling R22: validation is strict,
    # so this construction must be given that profile to succeed) -- the point of this
    # test is the ambient and driver errors, not profile configuration.
    prof = WindProfile([0.0, 180.0], [0.6, -0.4])
    wind = Wind.from_network(net, "airpath", ambient="ambient", profiles=[prof])
    with pytest.raises(KeyError, match="V_met"):
        wind({"rho_amb": torch.tensor(1.2, dtype=F64), "theta_w": torch.tensor(0.0, dtype=F64)})


def test_wind_refuses_a_negative_profile_index_naming_the_edges():
    """The valid range is 0 (constant Cp) to len(profiles). A NEGATIVE index matches no
    `profile_index == i` in `__call__` (i runs from 1), so it would silently fall back to
    `cp_const` rather than being refused; the upper bound was already checked, this is the
    lower one."""
    prof = WindProfile([0.0, 180.0], [0.6, -0.4])
    kw = dict(
        sign=torch.tensor([1.0, -1.0], dtype=F64),
        envelope=torch.tensor([1.0, 1.0], dtype=F64),
        azimuth=torch.tensor([0.0, 180.0], dtype=F64),
        ch=torch.tensor([0.5, 0.5], dtype=F64),
        cp_const=torch.tensor([0.0, 0.0], dtype=F64),
        profiles=[prof],
    )
    with pytest.raises(ValueError, match=r"edges \[1\] reference profile -1.*0 \(the constant"):
        Wind("airpath", profile_index=torch.tensor([1, -1]), **kw)
    # The upper bound still refuses on its own terms, and the valid range is accepted.
    with pytest.raises(ValueError, match="profile 2 but only 1 were given"):
        Wind("airpath", profile_index=torch.tensor([0, 2]), **kw)
    assert Wind("airpath", profile_index=torch.tensor([0, 1]), **kw).profiles == [prof]


def test_wind_from_network_accepts_a_profile_number_mapping_with_gaps():
    """Ruling R3: `profiles` may be a Mapping[int, WindProfile] keyed by CONTAM's own,
    possibly non-contiguous, profile number -- here 5 and 2, not 1 and 2."""
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", azimuth=0.0, Ch=0.5, profile=5)
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath", azimuth=180.0, Ch=0.5, profile=2)
    prof_a = WindProfile([0.0, 180.0], [0.6, -0.4])
    prof_b = WindProfile([0.0, 180.0], [0.9, -0.9])
    wind = Wind.from_network(
        net, "airpath", ambient="ambient", profiles={5: prof_a, 2: prof_b}
    )
    drivers = {
        "rho_amb": torch.tensor(1.2, dtype=F64),
        "V_met": torch.tensor(4.0, dtype=F64),
        "theta_w": torch.tensor(0.0, dtype=F64),
    }
    v = wind(drivers)
    q = 0.5 * 1.2 * 16.0 * 0.5
    # edge 0 (profile 5 -> prof_a): theta_rel = 0 -> Cp 0.6, ambient is src -> +.
    # edge 2 (profile 2 -> prof_b): theta_rel = -180 -> Cp -0.9, ambient is tgt ->
    # -(q * -0.9) = +0.9 q.
    torch.testing.assert_close(v, torch.tensor([0.6 * q, 0.0, 0.9 * q], dtype=F64))


def test_wind_from_network_names_the_edge_and_number_for_an_absent_profile():
    """Ruling R3: a profile number an edge references but that is absent from a `profiles`
    Mapping is a KeyError naming the edge and the number, not a silent mismatch."""
    net = _wind_net()
    prof = WindProfile([0.0, 180.0], [0.6, -0.4])
    with pytest.raises(KeyError, match="profile number 1") as excinfo:
        Wind.from_network(net, "airpath", ambient="ambient", profiles={2: prof})
    assert "ambient" in str(excinfo.value) and "z1" in str(excinfo.value)
