"""See ``tests/verification/conftest.py``: every test of this package restores WSIMOD's mutable
constructor defaults (L14)."""

from __future__ import annotations

import pytest

from tests.verification._wsimod_reference import (
    _restore_constants,
    _restore_constructor_defaults,
    _snapshot_constants,
)


@pytest.fixture(autouse=True)
def _wsimod_constructor_defaults():
    yield
    _restore_constructor_defaults()


@pytest.fixture(autouse=True)
def _wsimod_constants():
    """See ``tests/verification/conftest.py`` (Y84): the pollutant lists are restored in place,
    and ``FLOAT_ACCURACY`` (M7)."""
    snapshot = _snapshot_constants()
    yield
    _restore_constants(snapshot)
