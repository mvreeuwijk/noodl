"""The whole-model comparison of a noodl WSIMOD run against WSIMOD 0.8.1's own (spec,
Verification 2).

Test-only. ``assert_matches_wsimod`` compares every arc's flow and (with ``quality``) every
pollutant entry, every recorded tank and every surface field, on every day, against
``WsimodReference``, the record of WSIMOD's ``Model.run(record_all=True)``. A series' ``scale`` is
its largest magnitude over the run; a difference must lie within ``BOUND * scale`` (W19). A larger
one is admitted only when it is named by an :class:`Admission` and the run's ``"decisions"``
record holds a branch at that site with ``|margin| <= ADMISSION_MARGIN * scale`` that day.
Nothing here loosens a bound: the demos match bitwise (worst exactly 0), the bound is a ceiling.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl.apps.wsimod import WsimodResults
from tests.verification._wsimod_reference import WsimodReference

__all__ = ["ADMISSION_MARGIN", "BOUND", "Admission", "assert_matches_wsimod", "mass_balance_worst"]

BOUND = 1e-12
ADMISSION_MARGIN = 1e-9

_DEFICIT = re.compile(r"demand deficit of \S+ at (\S+) on (.+)")
_MAXITER = re.compile(r"Maxiter reached in (\S+) at (.+)")


@dataclass(frozen=True)
class Admission:
    """A named difference beyond the bound: the case, the element's label (see
    :func:`assert_matches_wsimod`), the day index and the decision site that flipped."""

    case: str
    element: str
    day: int
    site: str


def _series(
    group: str,
    ours: Tensor,
    theirs: Tensor,
    labels: Sequence[str],
) -> tuple[Tensor, Tensor, list[str]]:
    """``(difference, scale, labels)`` of one ``(T, ...)`` group, flattened to ``(T, M)``.

    NaN (WSIMOD records nothing there) must sit in the same places in both; it is not
    compared."""
    ours = ours.reshape(ours.shape[0], -1)
    theirs = theirs.reshape(theirs.shape[0], -1)
    nan = torch.isnan(theirs)
    if not torch.equal(torch.isnan(ours), nan):
        raise AssertionError(f"{group}: NaN (not recorded) differs between noodl and WSIMOD")
    ours = torch.where(nan, 0.0, ours)
    theirs = torch.where(nan, 0.0, theirs)
    return (ours - theirs).abs(), theirs.abs().amax(dim=0), [f"{group}:{x}" for x in labels]


def _pollutant_labels(names: Sequence[str], pollutants: Sequence[str]) -> list[str]:
    return [f"{name}:{p}" for name in names for p in pollutants]


def _decision_margin(decisions: list, day: int, site: str) -> float | None:
    """The smallest ``|margin|`` of an active decision at ``site`` on ``day``."""
    margins = [
        float(d.margin.abs().min())
        for d in decisions[day]
        if d.site == site and bool(d.active.any())
    ]
    return min(margins) if margins else None


def assert_matches_wsimod(
    ours: WsimodResults,
    theirs: WsimodReference,
    *,
    diagnostics: dict,
    case: str,
    quality: bool,
    admitted: Sequence[Admission] = (),
) -> dict:
    """Assert noodl's run equals WSIMOD's within ``BOUND * scale`` on every day (module text).

    Element labels, for the admissions and the failure message: ``"arc_flow:<arc>"``,
    ``"arc_quality:<arc>:<pollutant>"``, ``"tank_storage:<node>.<attribute>"``,
    ``"tank_quality:<node>.<attribute>:<pollutant>"``, ``"surface.<field>:<node>.<surface>"``,
    ``"surface_quality:<node>.<surface>:<pollutant>"``. ``quality=False`` (a quantity-only noodl
    run) compares the volumes alone.

    ``diagnostics`` is the dict ``wsimod_run`` filled, plus ``"node_names"`` (``model.wsimod_day
    .node_names``), and ``"decisions"`` when an admission needs one. The prints of WSIMOD that
    carry the node and the date (D13: ``demand deficit of ... at <node> on <date>``, ``Maxiter
    reached in <node> at <date>``) must equal noodl's ``"demand_deficit"`` and ``"maxiter"``
    records day by day, and no ``mass balance error`` may be printed (ruling W86: where WSIMOD
    prints one, it is Land's evaporation-clamp gap, asserted equal to noodl's residual by
    ``test_the_evaporation_clamp_residual_is_wsimods_printed_node_gap``; the demos print none).

    Returns ``{"worst": the largest absolute difference, "scale": the scale of its series,
    "admissions": the admissions used}``.
    """
    if ours.arc_names != theirs.arc_names:
        raise AssertionError(f"{case}: arc names differ")
    if ours.tank_names != theirs.tank_names:
        raise AssertionError(f"{case}: tank names differ")
    if ours.surface_names != theirs.surface_names:
        raise AssertionError(f"{case}: surface names differ")
    if len(ours.dates) != len(theirs.dates):
        raise AssertionError(f"{case}: {len(ours.dates)} days against {len(theirs.dates)}")
    groups = [
        _series("arc_flow", ours.arc_flow, theirs.arc_flow, ours.arc_names),
        _series(
            "tank_storage", ours.tank_storage, theirs.tank_storage,
            [f"{n}.{a}" for n, a in ours.tank_names],
        ),
    ]  # fmt: skip
    surfaces = [f"{n}.{s}" for n, s in ours.surface_names]
    for field in theirs.surface:
        groups.append(
            _series(f"surface.{field}", ours.surface[field], theirs.surface[field], surfaces)
        )
    if quality:
        p = theirs.pollutants
        groups += [
            _series(
                "arc_quality", ours.arc_quality, theirs.arc_quality,
                _pollutant_labels(ours.arc_names, p),
            ),
            _series(
                "tank_quality", ours.tank_quality, theirs.tank_quality,
                _pollutant_labels([f"{n}.{a}" for n, a in ours.tank_names], p),
            ),
            _series(
                "surface_quality", ours.surface_quality, theirs.surface_quality,
                _pollutant_labels(surfaces, p),
            ),
        ]  # fmt: skip

    allowed = {(a.case, a.element, a.day, a.site): a for a in admitted}
    used: list[Admission] = []
    worst, worst_scale = 0.0, max(float(s.max()) for _, s, _ in groups if s.numel())
    for diff, scale, labels in groups:
        bound = BOUND * scale
        for day, column in (diff > bound).nonzero().tolist():
            label = labels[column]
            decisions = diagnostics.get("decisions")
            sites = [k for k in allowed if k[0] == case and k[1] == label and k[2] == day]
            margin = None
            for key in sites:
                if decisions is None:
                    break
                margin = _decision_margin(decisions, day, key[3])
                if margin is not None and margin <= ADMISSION_MARGIN * float(scale[column]):
                    used.append(allowed[key])
                    break
            else:
                raise AssertionError(
                    f"{case}: first divergence {label} on day {day} ({ours.dates[day]}): noodl "
                    f"differs from WSIMOD by {float(diff[day, column]):.3e} against the bound "
                    f"{float(bound[column]):.3e} (scale {float(scale[column]):.3e}); no "
                    f"admission with a decision margin <= {ADMISSION_MARGIN:g} * scale"
                )
        if diff.numel():
            at = int(diff.amax(dim=0).argmax())
            top = float(diff.amax())
            if top > worst:
                worst, worst_scale = top, float(scale[at])

    names = diagnostics["node_names"]
    errors = [line for line in theirs.printed if "mass balance error" in line]
    if errors:
        raise AssertionError(f"{case}: WSIMOD printed {errors[0]!r}")
    flags = diagnostics["events"]["demand_deficit"]
    deficits = {
        (names[i], theirs.dates[d])
        for d in range(flags.shape[0])
        for i in range(flags.shape[1])
        if bool(flags[d, i])
    }
    printed = {m.groups() for line in theirs.printed if (m := _DEFICIT.fullmatch(line))}
    if deficits != printed:
        raise AssertionError(
            f"{case}: demand deficits differ: noodl-only {sorted(deficits - printed)[:3]}, "
            f"WSIMOD-only {sorted(printed - deficits)[:3]}"
        )
    maxiter = {
        (site[0], theirs.dates[d])
        for site, hit in diagnostics["maxiter"].items()
        for d in range(len(theirs.dates))
        if bool(hit[d])
    }
    printed = {m.groups() for line in theirs.printed if (m := _MAXITER.fullmatch(line))}
    if maxiter != printed:
        raise AssertionError(
            f"{case}: Maxiter days differ: noodl-only {sorted(maxiter - printed)[:3]}, "
            f"WSIMOD-only {sorted(printed - maxiter)[:3]}"
        )
    return {"worst": worst, "scale": worst_scale, "admissions": used}


def mass_balance_worst(ours: WsimodResults) -> float:
    """The largest daily residual of the system balance in units of its column's scale: the
    largest magnitude of the inflow, outflow and storage change over the run (1 for a column
    that never moves). The balance closes within ``BOUND * scale`` on every day when this is at
    most ``BOUND``."""
    balance = ours.mass_balance
    scale = torch.stack(
        [balance.inflow.abs(), balance.outflow.abs(), balance.storage_change.abs()]
    ).amax(dim=(0, 1))
    scale = torch.where(scale > 0, scale, 1.0)
    return float((balance.residual.abs() / scale).max())
