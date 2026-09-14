"""Tests for the Drive protocol and ConstantDrive (milestone 2 spec section 4.1)."""

from __future__ import annotations

import inspect

import pytest
import torch

from tellegen.drives import ConstantDrive, Drive, Stack, check_drive_signature
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
