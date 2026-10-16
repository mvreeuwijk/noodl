"""What a building simulation is run for, computed from `simulate`'s histories: exposure,
inhaled dose, infection risk, air change rates and the local mean age of air.

Exposure follows CONTAM: `occupant_exposure` is, for every step, the mean over the step of
the mass fraction in the zone the occupant is in, interpolated linearly between the step's
start and end values and split where the occupant moves (as ContamX's exposure file; 0 out
of the building). `cumulative_exposure` integrates it over time. `inhaled_dose` and the
Wells-Riley infection probability are additions of noodl: ContamX has no infection model,
and CONTAM 3.2 removed its inhalation fields. A quanta species is an ordinary species whose
"kg" are quanta: generation in quanta/s, mass fraction in quanta per kg of air, so that
the inhaled dose p rho int x dt is p int c_q dt and P = 1 - exp(-dose) is the Wells-Riley
probability in its quanta-concentration form.

The air change rates and the age of air take a history with the flows of the airflow layer
that advects the species layer (`"air.q"` as `simulate` records it, any leading dimensions:
the rows of a run, a batch) and work on the edges the species layer advects on. The age of
air and the zone air change rate divide by each zone's air mass, which changes with the
zone's temperature and pressure: they take it from the history's `"<layer>.capacity"`,
which `simulate` records whenever a closure writes it (a `.prj` model's always), or from
`capacity=`. The local mean age of air and the building air change rate are what ContamX
writes to its `.age` and `.ach` files, measured against ContamX; the zone air change rate is
an addition of noodl.

Every function is differentiable in its tensor arguments and keeps leading batch dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.occupants import Occupant, _occupant_nodes
from noodl.apps.building_physics.schedules import Calendar
from noodl.apps.building_physics.thermal import RHO_0
from noodl.layers.transport import TransportLayer

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
    nodes = _occupant_nodes(occupant, lay, "occupant_exposure", "breathe")
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
    w0 = torch.zeros(n - 1, lay.n_i, dtype=F64, device=x.device)
    w1 = torch.zeros(n - 1, lay.n_i, dtype=F64, device=x.device)
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
    1.2041 kg/m3, CONTAM's standard air density; pass the density the species layer's mass
    fractions were computed with (its capacity over the zone volume) to turn them back into
    the inhaled amount exactly."""
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


def _layer_flows(history: Mapping[str, Tensor], model, layer: str):
    """The species layer and its flows from `history`: `"<owner>.q"` of the airflow layer
    that advects it, cut to its edge kinds, or `"<layer>.q"` for a layer whose flows are a
    driver (no airflow layer)."""
    lay = model.transport[layer]
    owner = model.flow_layer_of[layer]
    key = f"{layer}.q" if owner is None else f"{owner}.q"
    if key not in history:
        raise KeyError(f"the history has no {key!r}, the flows layer {layer!r} is advected by")
    q = torch.as_tensor(history[key], dtype=F64)
    if owner is not None:
        q = model.kirchhoff[owner].flows_of_kind(q, lay.flow_kinds)
    return lay, q


def _endpoints(model, lay) -> tuple[Tensor, Tensor]:
    src, tgt = zip(*(model.net.endpoints(k) for k in lay.flow_kinds), strict=True)
    return torch.cat(src), torch.cat(tgt)


def _air_mass(history: Mapping[str, Tensor], model, lay, capacity, where: str) -> Tensor:
    """Each interior row's air mass (kg, `(..., n_i)`), zero at the storage-free rows: the
    given `capacity`, else the history's `"<layer>.capacity"`, else the layer's own when
    nothing can change it. A model with a closure that writes (or may write: it declares no
    outputs) `"<layer>.capacity"` changes the air mass with the temperature and pressure, so
    its construction-time value is never used for it silently. A capacity the caller passes
    as a driver is not seen here: give it as `capacity=` (or use `simulate`'s history)."""
    key = f"{lay.name}.capacity"
    if capacity is not None:
        cap = torch.as_tensor(capacity, dtype=F64)
        if cap.dim() == 0 or cap.shape[-1] != lay.n_i:
            raise ValueError(
                f"{where}: capacity has shape {tuple(cap.shape)}; it must end in the layer "
                f"{lay.name!r}'s {lay.n_i} interior rows (its interior order, the "
                f"storage-free rows included)"
            )
    elif key in history:
        cap = torch.as_tensor(history[key], dtype=F64)
    else:
        varying = [c for c in model.closures
                   if not hasattr(c, "outputs") or key in tuple(c.outputs)]
        if varying:
            raise ValueError(
                f"{where}: the history has no {key!r}, and the model's closures write it (or "
                f"declare no outputs: {[type(c).__name__ for c in varying]}): the zones' air "
                f"mass changes with their temperature and pressure. Pass a history from "
                f"simulate (which records it), or capacity= (Model.initial_capacities gives "
                f"it at a state)"
            )
        cap = lay.capacity
    free = lay.storage_free_mask.to(cap.device)
    return torch.where(free, torch.zeros_like(cap), cap)


def zone_air_change_rate(history: Mapping[str, Tensor], model, *, layer: str = "species",
                         capacity=None) -> tuple[Tensor, list]:
    """Each zone's air change rate (1/h): all the air flowing into it, from the ambient and
    from other zones alike, over its air mass, times 3600; the inverse of its nominal time
    constant, not an outdoor air change rate. An addition of noodl: ContamX writes no zone
    rate. Returns `(rate, zones)`: `rate` `(..., n_z)` over the history's leading dimensions,
    and `zones` the node names of its last axis, the layer's interior rows in their order
    without the storage-free ones (a zero-volume plenum holds no air to change). A duct
    junction with volume is a row like a zone.

    The air mass is `capacity` (kg, `(..., n_i)` in the layer's interior order), else the
    history's `"<layer>.capacity"` (each row's own, as `simulate` records it); refused by
    name when neither is there and a closure of the model writes it. Otherwise the layer's
    construction-time capacity is used: a capacity given only as a driver to a hand-run
    model is not known here, so pass it as `capacity=`. The flows are those of `history`
    (see the module docstring)."""
    lay, f = _layer_flows(history, model, layer)
    mass = _air_mass(history, model, lay, capacity, "zone_air_change_rate")
    src, tgt = _endpoints(model, lay)
    into = torch.zeros(f.shape[:-1] + (model.net.n,), dtype=F64, device=f.device)
    into = into.index_add(-1, tgt.to(f.device), torch.relu(f)).index_add(
        -1, src.to(f.device), torch.relu(-f))
    into = into.index_select(-1, lay.interior_idx.to(f.device))
    kept = torch.nonzero(~lay.storage_free_mask).flatten()
    rate = (into.index_select(-1, kept.to(f.device))
            / mass.index_select(-1, kept.to(mass.device)) * 3600.0)
    zones = [model.net.nodes[int(lay.interior_idx[k])] for k in kept.tolist()]
    return rate, zones


def building_air_change_rate(history: Mapping[str, Tensor], model, *, conditioned=None,
                             layer: str = "species", density=RHO_0) -> Tensor:
    """The building air change rate (1/h), as ContamX writes it to its `.ach` file: every
    flow entering the conditioned space from outside it, from the ambient or from a zone
    left out of it, over `density` (1.2041 kg/m3, ContamX's) times the space's volume
    (the nodes' `"volume"`), times 3600. Shape: the history's leading dimensions.

    `conditioned` names the nodes of the conditioned space; by default every interior row
    of the layer except the nodes whose `"in_building_volume"` attribute is 0 (the CONTAM
    reader sets it from a zone's "outside the building volume" flag), which for a `.prj`
    project is `project.building_zones`. ContamX counts in it every zone without that flag
    and the whole mechanical system: each duct junction (its share of the duct segments'
    volume) and each air handler's plenums. Air a duct terminal or an air handler
    delivers to a zone then moves within the space, and what enters is the outdoor air drawn
    in at an ambient terminal or an air handler's intake (ContamX's `duct` column) and
    through the envelope (its `path` column); only the total is returned, not that split.
    Refused by name: a
    node that is not an interior row of the layer (KeyError), an empty space, and a space
    that leaves out a storage-free node (a zero-volume plenum: ContamX counts the air
    handler in the space, and air recirculated from it would enter as if from outside)."""
    lay, f = _layer_flows(history, model, layer)
    net = model.net
    interior = lay.interior_idx.tolist()
    if conditioned is None:
        flagged_in = net.node_attr("in_building_volume", default=1.0)
        names = [net.nodes[i] for i in interior if flagged_in[i] != 0]
    else:
        names = list(conditioned)
    if not names:
        raise ValueError("building_air_change_rate: the conditioned space has no nodes")
    inside = torch.zeros(net.n, dtype=torch.bool)
    for name in names:
        i = int(net.node_index(name))
        if i not in interior:
            raise KeyError(f"building_air_change_rate: {name!r} is not an interior node of "
                           f"layer {layer!r}")
        inside[i] = True
    left_out = [net.nodes[i] for i in lay.interior_idx[lay.storage_free_mask].tolist()
                if not inside[i]]
    if left_out:
        raise ValueError(
            f"building_air_change_rate: the conditioned space leaves out the storage-free "
            f"node(s) {left_out} (an air handler's plenum): ContamX counts the mechanical "
            f"system in the conditioned space"
        )
    src, tgt = _endpoints(model, lay)
    crossing_in = (inside[tgt] & ~inside[src]).to(f.device)
    crossing_out = (inside[src] & ~inside[tgt]).to(f.device)
    entering = (torch.relu(f) * crossing_in + torch.relu(-f) * crossing_out).sum(-1)
    volume = torch.as_tensor(net.node_attr("volume", default=0.0), dtype=F64)[inside].sum()
    return entering / (torch.as_tensor(density, dtype=F64) * volume.to(f.device)) * 3600.0


def local_mean_age(history: Mapping[str, Tensor], model, *, layer: str = "species",
                   capacity=None) -> Tensor:
    """The local mean age of air of every interior row of the layer (s; ContamX's `.age`
    file gives it in hours): the steady age equation sum_in q tau_upstream - sum_out q tau
    + M = 0 at every row, tau = 0 at the boundary (the outdoor air), with M the row's air
    mass: a transport problem with a uniform source of one second per second in every
    kilogram of air (Sandberg's definition; the method of AIVC Technical Note 34, which
    ContamX cites). Shape `(..., n_i)` over the history's leading dimensions, in the layer's
    interior order (`model.transport[layer].interior_idx`); evaluated at every row, with
    that row's flows and air mass.

    The air mass is as in `zone_air_change_rate`. A storage-free row (a zero-volume
    plenum) holds none: its row is the algebraic balance of what passes through, and its age
    is the flow-weighted mix of the ages flowing in, whatever capacity is given there.
    Solved on a fresh single-species layer over the same flow edges, so filters, sinks and
    reactions of the species layer do not enter. Differentiable in the flows and the air
    mass."""
    lay, f = _layer_flows(history, model, layer)
    mass = _air_mass(history, model, lay, capacity, "local_mean_age")
    free = [model.net.nodes[i] for i in lay.interior_idx[lay.storage_free_mask].tolist()]
    # The layer's own capacity (the stand-in 1 kg at a storage-free row is never read: the
    # solve takes `mass`, zero there).
    age = TransportLayer(model.net, f"{lay.name} age", capacity=lay.capacity,
                         flow_kind=lay.flow_kinds, boundary=list(lay.boundary),
                         storage_free=free, scheme="implicit", quantity="age", unit="s",
                         source_unit="kg/s", capacity_unit="kg", flow_unit="kg/s")
    batch = torch.broadcast_shapes(f.shape[:-1], mass.shape[:-1])
    source = torch.zeros(batch + (model.net.n,), dtype=F64, device=f.device).index_copy(
        -1, lay.interior_idx.to(f.device), mass.expand(batch + (lay.n_i,)))
    return age.steady(f, source, torch.zeros(lay.n_b, dtype=F64, device=f.device),
                      capacity=mass)
