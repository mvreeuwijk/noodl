"""Tests for the Drive protocol and ConstantDrive."""

import pytest
import torch

from tellegen.drives import ConstantDrive


def test_constant_drive_returns_the_named_driver():
    drive = ConstantDrive("airpath", "wind")
    assert drive.kind == "airpath"
    drivers = {"wind": torch.tensor([1.0, 2.0, 3.0])}
    torch.testing.assert_close(drive(torch.zeros(3), drivers), drivers["wind"])


def test_constant_drive_raises_key_error_naming_the_missing_key():
    drive = ConstantDrive("airpath", "gust")
    with pytest.raises(KeyError, match="gust"):
        drive(torch.zeros(3), {})
