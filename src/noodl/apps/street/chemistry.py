"""Photostationary chemistry for a street model, and the steady state that includes it."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from noodl.layers.reaction import Photostationary, Reaction
from noodl.model import Drivers, Model, State


def photostationary_for_streets(
    species: Sequence[str], *, j_key: str = "J_NO2"
) -> Photostationary:
    """The Leighton reaction wired to the `"no"`, `"no2"` and `"o3"` columns of `species`.

    Names are matched case-insensitively and exactly; a missing one is named rather than
    guessed at, because a silently mis-wired species column produces a plausible-looking
    answer that is simply wrong.
    """
    lowered = [str(name).lower() for name in species]
    columns = []
    for wanted in ("no", "no2", "o3"):
        if wanted not in lowered:
            raise ValueError(
                f"photostationary_for_streets: species {wanted!r} is not among "
                f"{tuple(species)}; the Leighton state needs all of 'no', 'no2' and 'o3'"
            )
        columns.append(lowered.index(wanted))
    return Photostationary(columns[0], columns[1], columns[2], j_key=j_key)


def street_steady(
    model: Model,
    state: State,
    drivers: Drivers,
    *,
    reaction: Reaction | None = None,
    tol: float = 1e-14,
    max_iter: int = 50,
    layer_name: str = "street",
    **solve_kwargs,
) -> State:
    """The steady state of transport AND chemistry, by an explicit fixed point.

    `Model.steady` does NOT apply reactions -- `Model._pass` applies them only on its
    stepping branch, and `Model.steady`'s own docstring records that as deliberate. The
    milestone 3 spec's section 5 says otherwise; it is wrong, and this function is the
    correction. With `reaction=None` this is exactly `model.steady(...)`, returned
    unchanged.

    The iteration is plain successive substitution -- solve transport at the current
    composition, relax the composition to its photostationary state, repeat -- and it is
    differentiable by unrolling, like `Model`'s own iterated coupling. `tol` is an ABSOLUTE
    tolerance on the state in its own units (kg/m3), tested on the largest change over all
    streets and species; a budget it cannot meet raises, naming the pass count and the
    change that was left.
    """
    if reaction is None:
        return model.steady(state, drivers, **solve_kwargs)
    key = f"{layer_name}.x"
    current = dict(state)
    change = float("inf")
    # `_passes` rather than `passes`: the count is not read inside the body (ruff B007),
    # only the budget matters, and the failure message names `max_iter` itself.
    for _passes in range(int(max_iter)):
        solved = model.steady(current, drivers, **solve_kwargs)
        relaxed = dict(solved)
        relaxed[key] = reaction.apply(solved[key], None, drivers)
        with torch.no_grad():
            if key in current:
                change = float((relaxed[key] - current[key]).abs().max())
        current = relaxed
        if change <= tol:
            return current
    raise RuntimeError(
        f"street_steady: the transport-and-chemistry fixed point did not converge within "
        f"{max_iter} passes; the largest change on the last pass was {change} against an "
        f"absolute tolerance of {tol} (kg/m3)"
    )


J_NO2_ZENITH_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 78.0, 86.0, 90.0)
"""The solar zenith angles MUNICH's Leighton mechanism tabulates `J_NO2` at.
`include/modules/chemistry/Leighton/reactions`, `SET TABULATION 11 DEGREES ...`."""

J_NO2_CLEAR_SKY = (
    9.31026e-3, 9.21901e-3, 8.90995e-3, 8.37928e-3, 7.60031e-3, 6.52988e-3,
    5.10803e-3, 3.29332e-3, 1.74121e-3, 5.11393e-4, 1.63208e-4,
)
"""Clear-sky `J_NO2` [1/s] at those angles, from the same file's `KINETIC PHOTOLYSIS`
line (a RACM tabulation, per the comment there)."""


def j_no2(zenith_deg, attenuation=1.0) -> torch.Tensor:
    """Clear-sky `J_NO2` [1/s] at `zenith_deg`, times `attenuation`.

    Piecewise-linear in the zenith angle between MUNICH's eleven tabulated points, and
    held at the end values outside them -- which is the interpolation MUNICH's own
    `Photolysis_tabulation_option: 2` performs on the same table. `attenuation` is
    MUNICH's per-street `Attenuation` field, the only modulation its chemistry applies:
    `include/modules/chemistry/Common/chem.f:232-234` computes
    `J = Zatt * J_tabulated` and nothing else -- there is no separate canyon shading
    factor. The SOLAR GEOMETRY is deliberately not computed here: a zenith angle wants a
    date, a latitude and a longitude, none of which the street application carries, and
    the milestone exercises chemistry on synthetic cases only (the Leiden data is
    NOx-only). Pass the angle you want.
    """
    zenith = torch.as_tensor(zenith_deg, dtype=torch.float64)
    angles = torch.tensor(J_NO2_ZENITH_DEG, dtype=torch.float64)
    values = torch.tensor(J_NO2_CLEAR_SKY, dtype=torch.float64)
    clamped = torch.clamp(zenith, min=float(angles[0]), max=float(angles[-1]))
    upper = torch.clamp(torch.searchsorted(angles, clamped, right=True), 1,
                        len(angles) - 1)
    lower = upper - 1
    span = angles[upper] - angles[lower]
    weight = (clamped - angles[lower]) / span
    interpolated = values[lower] + weight * (values[upper] - values[lower])
    return interpolated * torch.as_tensor(attenuation, dtype=torch.float64)
