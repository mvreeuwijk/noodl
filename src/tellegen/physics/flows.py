"""Compatibility wrapper: branch flows and the orientation check now live in ``tellegen.cycles``."""

from __future__ import annotations

from tellegen.cycles import assert_forward_oriented, branch_flows

__all__ = ["assert_forward_oriented", "branch_flows"]
