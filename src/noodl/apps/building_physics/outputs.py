"""What a building simulation is run for, computed from `simulate`'s histories: exposure,
inhaled dose and infection risk.

Exposure follows CONTAM: `occupant_exposure` is, for every step, the mean over the step of
the mass fraction in the zone the occupant is in, interpolated linearly between the step's
start and end values and split where the occupant moves (as ContamX's exposure file; 0 out
of the building). `cumulative_exposure` integrates it over time. `inhaled_dose` and the
Wells-Riley infection probability are additions of noodl: ContamX has no infection model,
and CONTAM 3.2 removed its inhalation fields. A quanta species is an ordinary species whose
"kg" are quanta: generation in quanta/s, mass fraction in quanta per kg of air, so that
the inhaled dose p rho int x dt is p int c_q dt and P = 1 - exp(-dose) is the Wells-Riley
probability in its quanta-concentration form.

Every function is differentiable in its tensor arguments and keeps leading batch dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.occupants import Occupant, _occupant_nodes
from noodl.apps.building_physics.schedules import Calendar
from noodl.apps.building_physics.thermal import RHO_0

Tensor = torch.Tensor
F64 = torch.float64


def occupant_exposure(history: Mapping[str, Tensor], occupant: Occupant, model, *,
                      calendar: Calendar, layer: str = "species") -> Tensor:
    """The occupant's exposure over each step of `history` (from `simulate`): the step mean
    of the occupied zone's `"<layer>.x"`, linearly interpolated within the step and split at
    every move; zero out of the building. Shape (N - 1, ..., K), in the layer's unit (kg/kg
    for mass fractions), row k for the step from `history["time"][k]` to `[k + 1]`.

    `calendar` places t = 0 in the year, so it picks each day's occupancy day and the time
    of every move: pass the run's own (`simulate`'s `calendar=`). `"<layer>.x"` is
    `(N, ..., n_i, K)`, the layer's interior rows with the species axis, as `simulate`
    records it.

    The interpolation is between output rows: a run that splits its steps at input
    breakpoints (`simulate`'s `land_on_breakpoints=True`) keeps one row per output time, so
    the history cannot show those sub-steps. The result is exact for steps the solver takes
    whole, as in ContamX's runs (`simulate_project`).

    Refused by name, as for the occupant's generation: any zone of any of the occupant's
    day schedules that is not a node (KeyError), is the layer's boundary (the ambient), is
    a node no flow edge of the layer touches, or is a storage-free node (ValueError): the
    layer holds no mass fraction to breathe there. The weights are built per step and
    occupant in Python, which is cheap next to the run for day-long histories.
    """
    lay = model.transport[layer]
    nodes = _occupant_nodes(occupant, lay, "occupant_exposure")
    interior = lay.interior_idx.tolist()
    pos = {zone: interior.index(node) for zone, node in nodes.items()}
    key = f"{layer}.x"
    x = torch.as_tensor(history[key], dtype=F64)
    if x.dim() < 3 or tuple(x.shape[-2:]) != (lay.n_i, lay.n_species):
        raise ValueError(f"occupant_exposure: {key!r} must be (N, ..., {lay.n_i}, "
                         f"{lay.n_species}), the layer's interior rows and species, got "
                         f"{tuple(x.shape)}")
    times = [float(v) for v in torch.as_tensor(history["time"], dtype=F64).tolist()]
    n = len(times)
    if x.shape[0] != n:
        raise ValueError(f"occupant_exposure: {key!r} has {x.shape[0]} rows, "
                         f"'time' {n}")
    w0 = torch.zeros(n - 1, lay.n_i, dtype=F64)
    w1 = torch.zeros(n - 1, lay.n_i, dtype=F64)
    for k in range(1, n):
        t0, t1 = times[k - 1], times[k]
        span = t1 - t0
        for a, b, zone in occupant.occupancy.segments(t0, t1, calendar):
            if zone is None:
                continue
            frac, s = (b - a) / span, (0.5 * (a + b) - t0) / span
            w0[k - 1, pos[zone]] += frac * (1.0 - s)
            w1[k - 1, pos[zone]] += frac * s
    shape = (n - 1,) + (1,) * (x.dim() - 3) + (lay.n_i, 1)
    return (w0.reshape(shape) * x[:-1] + w1.reshape(shape) * x[1:]).sum(-2)


def cumulative_exposure(exposure: Tensor, times) -> Tensor:
    """The time integral of `occupant_exposure` from the start: shape (N, ..., K), row 0
    zero, in the exposure's unit times s (kg/kg s for mass fractions)."""
    exposure = torch.as_tensor(exposure, dtype=F64)
    dt = torch.diff(torch.as_tensor(times, dtype=F64))
    dt = dt.reshape((-1,) + (1,) * (exposure.dim() - 1))
    steps = torch.cumsum(exposure * dt, dim=0)
    return torch.cat([torch.zeros_like(steps[:1]), steps], dim=0)


def inhaled_dose(integrated_exposure, breathing_rate, *, density=RHO_0) -> Tensor:
    """What was breathed in: breathing rate (m3/s) x air density (kg/m3) x the time
    integral of the mass fraction (kg/kg s; `cumulative_exposure`, not the per-step
    `occupant_exposure`), in kg, or quanta for a quanta species. The density defaults to
    1.2041 kg/m3, CONTAM's standard air density."""
    return (torch.as_tensor(breathing_rate, dtype=F64) * torch.as_tensor(density, dtype=F64)
            * torch.as_tensor(integrated_exposure, dtype=F64))


def infection_probability(dose) -> Tensor:
    """Wells-Riley: P = 1 - exp(-dose), the dose in quanta (accurate for small doses);
    dimensionless."""
    return -torch.expm1(-torch.as_tensor(dose, dtype=F64))


def wells_riley(infectors, quanta_rate, breathing_rate, ventilation_rate, duration) -> Tensor:
    """The classic steady single-zone form P = 1 - exp(-I q p t / Q): `infectors` I,
    `quanta_rate` q (quanta/s per infector), `breathing_rate` p (m3/s), `ventilation_rate` Q
    (m3/s of clean air), `duration` t (s); dimensionless."""
    n, q, p, flow, t = (torch.as_tensor(v, dtype=F64) for v in (
        infectors, quanta_rate, breathing_rate, ventilation_rate, duration))
    return infection_probability(n * q * p * t / flow)
