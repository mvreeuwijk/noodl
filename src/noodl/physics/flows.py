"""Compatibility wrapper: branch flows and the orientation check now live in ``noodl.cycles``."""

from __future__ import annotations

from noodl.cycles import assert_forward_oriented, branch_flows

__all__ = ["assert_forward_oriented", "branch_flows"]
