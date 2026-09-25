"""Every application exposes `build_model`; the SWMM reader carries its tool name."""
import importlib

import pytest


@pytest.mark.parametrize("package", ["building_physics", "street_aq", "sewer", "water"])
def test_every_application_exposes_build_model(package):
    module = importlib.import_module(f"noodl.apps.{package}")
    assert callable(module.build_model)
    assert "build_model" in module.__all__


@pytest.mark.parametrize(
    ("package", "old"),
    [
        ("street_aq", "build_street_model"),
        ("sewer", "build_sewer_model"),
        ("water", "build_water_model"),
        ("sewer", "read_inp"),
    ],
)
def test_the_old_names_are_gone(package, old):
    module = importlib.import_module(f"noodl.apps.{package}")
    assert not hasattr(module, old)


def test_the_swmm_reader_is_named_for_its_tool():
    from noodl.apps.sewer import read_swmm_inp

    assert callable(read_swmm_inp)
