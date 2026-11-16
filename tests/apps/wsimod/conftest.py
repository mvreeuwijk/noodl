"""See ``tests/verification/conftest.py``: every test of this package restores WSIMOD's mutable
constructor defaults (L14)."""

from __future__ import annotations

import pytest

from tests.verification._wsimod_reference import _restore_constructor_defaults


@pytest.fixture(autouse=True)
def _wsimod_constructor_defaults():
    yield
    _restore_constructor_defaults()
