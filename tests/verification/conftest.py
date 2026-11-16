"""WSIMOD's mutable constructor defaults are restored after every test of a module that runs
WSIMOD (L14): an override of a dict parameter mutates the shared default in place
(``land.py:397``), and any surface built later without it would inherit the change, in the same
process or xdist worker. ``tests/apps/wsimod`` takes the same fixture."""

from __future__ import annotations

import pytest

from tests.verification._wsimod_reference import (
    _restore_constants,
    _restore_constructor_defaults,
    _snapshot_constants,
)


@pytest.fixture(autouse=True)
def _wsimod_constructor_defaults(request):
    yield
    if "wsimod" in request.module.__name__:
        _restore_constructor_defaults()


@pytest.fixture(autouse=True)
def _wsimod_constants(request):
    """A test that sets ``wsimod.core.constants``' pollutant lists (as a unit test of a WSIMOD
    object does) leaves them as they were: the lists are refilled in place (Y84)."""
    if "wsimod" not in request.module.__name__:
        yield
        return
    snapshot = _snapshot_constants()
    yield
    _restore_constants(snapshot)
