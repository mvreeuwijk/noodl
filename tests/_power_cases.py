"""MATPOWER's standard cases for tests: fetched on first use, the test skipped when offline.

Tests calling `power_case` carry the `external` marker.
"""

from pathlib import Path

import pytest

from noodl.apps.electrical.power import MatpowerCaseUnavailableError, matpower_case


def power_case(name: str) -> Path:
    """The cached, verified MATPOWER case `name`; skips the calling test without a network."""
    try:
        return matpower_case(name)
    except MatpowerCaseUnavailableError as exc:
        pytest.skip(f"MATPOWER case not available: {exc}")
