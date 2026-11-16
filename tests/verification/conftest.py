"""WSIMOD's mutable constructor defaults are restored after every test of a module that runs
WSIMOD (L14): an override of a dict parameter mutates the shared default in place
(``land.py:397``), and any surface built later without it would inherit the change, in the same
process or xdist worker. ``tests/apps/wsimod`` takes the same fixture."""

from __future__ import annotations

import pytest

from tests.verification._wsimod_reference import _restore_constructor_defaults


@pytest.fixture(autouse=True)
def _wsimod_constructor_defaults(request):
    yield
    if "wsimod" in request.module.__name__:
        _restore_constructor_defaults()
