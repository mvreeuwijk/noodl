"""Comparing a run with a record written to `tests/data`."""

from __future__ import annotations


def assert_matches_record(got, ref, what, tol=1e-12):
    """`got` equals the recorded `ref` to `tol` of its largest magnitude. The records were
    written on Windows and Linux's libm differs in the last bits, so a record holds the
    values to rounding, not bit for bit; comparisons within one run stay bit-exact."""
    scale = float(ref.abs().max())
    assert scale > 0, what
    err = float((got - ref).abs().max()) / scale
    assert err <= tol, (what, err)
