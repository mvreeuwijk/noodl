"""Tests for the Drive protocol and ConstantDrive (milestone 2 spec section 4.1)."""

from __future__ import annotations

import inspect

import pytest
import torch

from tellegen.drives import ConstantDrive, Drive, check_drive_signature
from tellegen.layers.potential import PotentialFlowLayer


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
