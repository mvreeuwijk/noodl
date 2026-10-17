"""Every application exposes `build_model`."""
import importlib

import pytest


@pytest.mark.parametrize("package", ["building_physics", "street_aq", "sewer", "water", "wsimod"])
def test_every_application_exposes_build_model(package):
    module = importlib.import_module(f"noodl.apps.{package}")
    assert callable(module.build_model)
    assert "build_model" in module.__all__
