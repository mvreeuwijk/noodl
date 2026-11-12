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

import io
import re
import types
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl.apps.wsimod import WsimodResults
from tests.verification import _wsimod_reference
from tests.verification._wsimod_reference import WsimodReference, record_wsimod_run, wsimod_model

__all__ = [
    "ADMISSION_MARGIN",
    "BOUND",
    "Admission",
    "assert_matches_wsimod",
    "mass_balance_worst",
    "wsimod_reference_by_day",
]

BOUND = 1e-12
ADMISSION_MARGIN = 1e-9

_DEFICIT = re.compile(r"demand deficit of \S+ at (\S+) on (.+)")
_MAXITER = re.compile(r"Maxiter reached in (\S+) at (.+)")
_MASS_BALANCE = re.compile(r"mass balance error for (\S+) of (\S+) in (\S+)")
# The prints that name their node and date: matched node by node and day by day (D13).
_DATED = {
    "demand_deficit": _DEFICIT,
    "leakage_not_going_to_gw": re.compile(
        r"warning, distribution leakage not going to GW in (\S+) at (.+)"
    ),
    "service_reservoirs_not_filled": re.compile(r"Service reservoirs not filled at (\S+) on (.+)"),
}
# The prints without a node or a date: lines of one kind exist exactly when noodl flags the event
# on some day and node, and there are at least as many lines as flags (a node can print more than
# once a day, an event is one flag).
_UNDATED = {
    "storage_unable_to_push": re.compile(r"Storage unable to push"),
    "groundwater_couldnt_push_all": re.compile(r"Groundwater couldnt push all"),
    "miscalculated_tank_storage": re.compile(r"Miscalculated tank storage in discharge"),
    "environmental_not_able_to_push": re.compile(r"warning: environmental not able to push"),
    "demand_not_able_to_push": re.compile(r"Demand not able to push"),
    "wwtw_couldnt_push": re.compile(r"WWTW couldnt push"),
    "more_treated_than_input": re.compile(r"more treated than input"),
    "sludge_nowhere": re.compile(r"nowhere for sludge to go"),
    "excess_treated_water": re.compile(r"excess treated water"),
    "reservoir_spill": re.compile(r"Spill at reservoir by \S+"),
    "river_cant_push": re.compile(r"river cant push: \S+"),
    "ponded_water_cant_reenter": re.compile(r"ponded water cant reenter"),
    "more_backflow_than_vqip": re.compile(r"more backflow than vqip\.\.\."),
    "inaccurate_evaporation": re.compile(r"inaccurate evaporation calculation of \S+"),
}
# WSIMOD's deny handlers (``nodes.py:560-608``) print and return a zero reply; noodl's are silent
# and record nothing, the reply being all that has an effect (and replies are compared bitwise).
_DENY = re.compile(r"Attempted (push|pull) (set|check) (to|from) deny")


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
        float(d.margin[d.active].abs().min())
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
    print_days: Sequence[int] | None = None,
    expected_mass_balance_prints: Sequence[str] = (),
) -> dict:
    """Assert noodl's run equals WSIMOD's within ``BOUND * scale`` on every day (module text).

    Element labels, for the admissions and the failure message: ``"arc_flow:<arc>"``,
    ``"arc_quality:<arc>:<pollutant>"``, ``"tank_storage:<node>.<attribute>"``,
    ``"tank_quality:<node>.<attribute>:<pollutant>"``, ``"surface.<field>:<node>.<surface>"``,
    ``"surface_quality:<node>.<surface>:<pollutant>"``. ``quality=False`` (a quantity-only noodl
    run) compares the volumes alone. ``print_days`` is the day of each line of ``theirs.printed``
    (:func:`wsimod_reference_by_day`): the undated prints are then checked day by day.

    ``diagnostics`` is the dict ``wsimod_run`` filled, plus ``"node_names"`` (``model.wsimod_day
    .node_names``), and ``"decisions"`` when an admission needs one.
    Every print of WSIMOD must be one of the kinds in ``_DATED``, ``_UNDATED``, ``_DENY``,
    Maxiter or mass balance error, and agree with
    noodl's records (``_check_prints``): the dated ones (demand deficit, Maxiter, leakage not
    going to the groundwater, service reservoirs not filled) equal noodl's flags node by node and
    day by day, the undated ones exist exactly when noodl flags the event, and the node check's
    ``mass balance error`` lines equal, in order, noodl's ``residual + dropped`` on the days it
    exceeds the bound (ruling W86: the evaporation-clamp gap and the water a node drops below
    ``FLOAT_ACCURACY``). The demos print nothing.

    ``expected_mass_balance_prints`` (D17, X32) names the nodes whose ``mass balance error``
    lines WSIMOD is expected to print on **every** day while noodl's own balance closes: a
    ``River`` whose ``calculate_discharge`` runs twice a day (``river_double_discharge``), where
    WSIMOD's assigned ``bio_in``/``bio_out`` drop the first call's terms. Those nodes' pollutant
    lines are asserted present on every day (``print_days`` is required) and are left out of the
    residual comparison; a volume line of theirs, and every other print, is checked as before.

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
    # ours keeps the labels as written ("2009-03-03 00:00:00"), WSIMOD records str(date)
    if [d[:10] for d in ours.dates] != [d[:10] for d in theirs.dates]:
        raise AssertionError(f"{case}: the dates differ")
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
            decisions = diagnostics.get("decisions") or None
            sites = [k for k in allowed if k[0] == case and k[1] == label and k[2] == day]
            admitted_key = None
            for key in sites if decisions is not None else ():
                margin = _decision_margin(decisions, day, key[3])
                if margin is not None and margin <= ADMISSION_MARGIN * float(scale[column]):
                    admitted_key = key
                    break
            if admitted_key is None:
                raise AssertionError(
                    f"{case}: first divergence {label} on day {day} ({ours.dates[day]}): noodl "
                    f"differs from WSIMOD by {float(diff[day, column]):.3e} against the bound "
                    f"{float(bound[column]):.3e} (scale {float(scale[column]):.3e}); no "
                    f"admission with a decision margin <= {ADMISSION_MARGIN:g} * scale"
                )
            used.append(allowed[admitted_key])
        if diff.numel():
            at = int(diff.amax(dim=0).argmax())
            top = float(diff.amax())
            if top > worst:
                worst, worst_scale = top, float(scale[at])

    names = diagnostics["node_names"]
    _check_prints(
        ours, theirs, diagnostics, case, names, print_days, tuple(expected_mass_balance_prints)
    )
    return {"worst": worst, "scale": worst_scale, "admissions": used}


def _check_prints(
    ours: WsimodResults,
    theirs: WsimodReference,
    diagnostics: dict,
    case: str,
    names,
    print_days: Sequence[int] | None,
    expected_mass_balance_prints: tuple[str, ...] = (),
):
    """Every print of WSIMOD is one of the kinds above and agrees with noodl's records (module
    text, D13); an unknown print is refused.

    ``mass balance error for volume of <x> in <node>`` (``core.py:420-429``) is WSIMOD's node
    check. Its lines, in order, equal the days' ``residual + dropped`` that exceed the bound
    (ruling W86): the evaporation-clamp gap, which noodl leaves in the residual, and the water a
    node drops below ``FLOAT_ACCURACY`` (``Land.run``'s percolation push-back), which noodl
    counts as ``"dropped"`` and so closes. A ``QueueArc``'s drop is inside an arc, in no node's
    balance: it is left out (``dropped - dropped_arcs``). With ``print_days`` each error line must
    also fall on its day. Any other line, or a missing one, fails. The lines of a node in
    ``expected_mass_balance_prints`` (D17) are the exception: WSIMOD's own balance of that node
    is wrong by construction (X16, X32), so its pollutant lines are required on every day and
    compared with nothing; its volume lines, which the biochemistry cannot cause, go through
    the residual comparison like any other.

    Limits of the comparison: WSIMOD's check is per node and relative (the values normalised by
    the largest power of ten, against ``1e-11``), noodl's expectation is the system sum against
    ``1e-12 * scale``. They agree on single-node error cases, which is all the corpus has; two
    nodes with opposite errors on one day could cancel in noodl's sum.

    The undated prints are checked per day when ``print_days`` is given: on each day, lines of a
    kind exist exactly when noodl flags the event at some node that day (and over the run there
    are at least as many lines as flags); without it, over the whole run only."""
    events = diagnostics["events"]
    printed = list(theirs.printed)
    kinds: dict[str, list[str]] = {}
    for line in printed:
        for name, pattern in {**_DATED, **_UNDATED}.items():
            if pattern.fullmatch(line):
                kinds.setdefault(name, []).append(line)
                break
        else:
            if not (
                _MAXITER.fullmatch(line) or _MASS_BALANCE.fullmatch(line) or _DENY.fullmatch(line)
            ):
                raise AssertionError(f"{case}: WSIMOD printed a line no check covers: {line!r}")
    for name, pattern in _DATED.items():
        flags = events[name]
        flagged = {
            (names[i], theirs.dates[d])
            for d in range(flags.shape[0])
            for i in range(flags.shape[1])
            if bool(flags[d, i])
        }
        got = {m.groups() for line in kinds.get(name, []) if (m := pattern.fullmatch(line))}
        if name == "demand_deficit":
            got = {m.groups() for line in printed if (m := _DEFICIT.fullmatch(line))}
        if flagged != got:
            what = "demand deficits" if name == "demand_deficit" else name
            raise AssertionError(
                f"{case}: {what} differ: noodl-only {sorted(flagged - got)[:3]}, "
                f"WSIMOD-only {sorted(got - flagged)[:3]}"
            )
    for name, pattern in _UNDATED.items():
        lines, flagged = len(kinds.get(name, [])), int(events[name].sum())
        if (lines > 0) != (flagged > 0) or lines < flagged:
            raise AssertionError(
                f"{case}: WSIMOD printed {lines} {name!r} lines, noodl flagged {flagged} "
                "(node, day) pairs"
            )
        if print_days is not None:
            days = {
                d for d, line in zip(print_days, printed, strict=True) if pattern.fullmatch(line)
            }
            flagged_days = {d for d in range(events[name].shape[0]) if bool(events[name][d].any())}
            if days != flagged_days:
                raise AssertionError(
                    f"{case}: {name!r} printed on days {sorted(days - flagged_days)[:3]} that "
                    f"noodl does not flag, not on {sorted(flagged_days - days)[:3]} that it does"
                )
    maxiter = {
        (site[0], theirs.dates[d])
        for site, hit in diagnostics["maxiter"].items()
        for d in range(len(theirs.dates))
        if bool(hit[d])
    }
    got = {m.groups() for line in printed if (m := _MAXITER.fullmatch(line))}
    if maxiter != got:
        raise AssertionError(
            f"{case}: Maxiter days differ: noodl-only {sorted(maxiter - got)[:3]}, "
            f"WSIMOD-only {sorted(got - maxiter)[:3]}"
        )
    balance = ours.mass_balance
    volume = balance.residual[:, 0] + (balance.terms["dropped"] - balance.dropped_arcs)[:, 0]
    scale = float(
        torch.stack([balance.inflow.abs(), balance.outflow.abs(), balance.storage_change.abs()])[
            ..., 0
        ].max()
    )
    significant = volume.abs() > BOUND * scale
    expected = volume[significant]
    errors, error_days = [], []
    unknown = [n for n in expected_mass_balance_prints if n not in names]
    if unknown:
        raise AssertionError(f"{case}: expected mass balance prints of unknown nodes {unknown}")
    if expected_mass_balance_prints and print_days is None:
        raise ValueError(f"{case}: expected_mass_balance_prints needs print_days (D17)")
    expected_days: dict[str, set[int]] = {n: set() for n in expected_mass_balance_prints}
    for k, line in enumerate(printed):
        if m := _MASS_BALANCE.fullmatch(line):
            if m.group(3) in expected_days and m.group(1) != "volume":
                expected_days[m.group(3)].add(print_days[k])  # D17: a pollutant line
                continue
            error_days.append(-1 if print_days is None else print_days[k])
            if m.group(1) != "volume" or m.group(3) not in names:
                raise AssertionError(f"{case}: WSIMOD printed {line!r}")
            errors.append(float(m.group(2)))
    for node, days in expected_days.items():
        missing = sorted(set(range(len(theirs.dates))) - days)
        if missing:
            raise AssertionError(
                f"{case}: WSIMOD printed no mass balance error for {node!r} on day "
                f"{missing[0]} ({theirs.dates[missing[0]]}); one is expected on every day (D17)"
            )
    if len(errors) != len(expected) or (
        errors
        and float((expected - torch.tensor(errors, dtype=torch.float64)).abs().max())
        > BOUND * scale
    ):
        raise AssertionError(
            f"{case}: WSIMOD's mass balance errors {errors[:3]} differ from noodl's residual plus "
            f"dropped water {expected.tolist()[:3]} ({len(errors)} lines, {len(expected)} days)"
        )
    if print_days is not None and error_days != significant.nonzero().flatten().tolist():
        raise AssertionError(
            f"{case}: WSIMOD's mass balance errors fall on days {error_days[:3]}, noodl's on "
            f"{significant.nonzero().flatten().tolist()[:3]}"
        )


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


class _DayTaggedStdout(io.StringIO):
    """A ``StringIO`` that also records the day (``days[0]``) each completed line was written on."""

    def __init__(self, days: list[int], lines: list[tuple[int, str]]):
        super().__init__()
        self._days, self._lines, self._pending = days, lines, ""

    def write(self, text: str) -> int:
        self._pending += text
        *complete, self._pending = self._pending.split("\n")
        self._lines.extend((self._days[0], line) for line in complete)
        return super().write(text)


def wsimod_reference_by_day(
    source, *, days: int | None = None
) -> tuple[WsimodReference, list[int]]:
    """``wsimod_reference(source, days=days)`` plus the day (0-based) of each line of its
    ``printed``.

    The run is WSIMOD's own (``record_wsimod_run``), its stdout going to a writer that tags each
    line with a day counter; the counter is bumped by the first node's ``end_timestep``, and every
    print of a day (the node checks included) comes before it (``model.py:995-1060``). The
    reference module is not edited: its ``io`` is swapped for the run's duration."""
    model = wsimod_model(source)
    day, lines = [0], []
    first = next(iter(model.nodes.values()))
    original = first.end_timestep

    def end_timestep(*args, **kwargs):
        day[0] += 1
        return original(*args, **kwargs)

    first.end_timestep = end_timestep
    saved = _wsimod_reference.io
    _wsimod_reference.io = types.SimpleNamespace(StringIO=lambda: _DayTaggedStdout(day, lines))
    try:
        reference = record_wsimod_run(model, days=days)
    finally:
        _wsimod_reference.io = saved
    if [line for _, line in lines] != reference.printed:
        raise AssertionError("wsimod_reference_by_day: the tagged lines differ from the printed")
    return reference, [d for d, _ in lines]
