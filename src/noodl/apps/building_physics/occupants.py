"""Occupants: where people are over the day.

An `OccupancyDay` gives an occupant's zone at times of day (seconds after midnight, from
00:00:00 to 24:00:00); `None` means out of the building. An `Occupancy` holds one occupancy
day per CONTAM day type (1-12) and follows a `Calendar`, like a week schedule. The occupant
is in a listed zone FROM the listed time, also in the middle of a step: a move at 01:02:30
splits the step from 01:00 to 01:05 there. This is how ContamX accounts for it, both in the
exposure it reports and in the contaminants an occupant generates (measured on ContamX
3.4.1.7).

`Presence(occupancy, net)` is the occupant's location as a signal over the network's nodes:
1 at the occupied node at an instant (the left limit at a move, as every schedule), and
over a step (t0, t1] the fraction of the step spent at each node (`step_mean`), which
`apply_inputs` writes when it is given the step's start.

An `Occupant` follows an occupancy and generates contaminants at fixed rates (kg/s per
person, times `multiplier` persons, times an optional schedule). `occupant_inputs` gives the
inputs that write its presence and schedules for `simulate`, and the `OccupantGeneration`
closure adds the generation to the species sources.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from noodl.apps.building_physics.inputs import Input
from noodl.apps.building_physics.schedules import (
    N_DAY_TYPES,
    Calendar,
    _check_day_times,
    _check_time_of_day,
    _DayTypeCycle,
    _piecewise,
)
from noodl.apps.building_physics.sources import _add_sources, _check_emitting_node

Tensor = torch.Tensor
F64 = torch.float64


class OccupancyDay:
    """An occupant's zone over one day: `zones[i]` from `times[i]` (s after midnight) until
    the next time. `times` run from 0 (00:00:00) to 86400 (24:00:00), strictly increasing.
    A zone is a node name, or None for out of the building. ContamX needs the 24:00:00 point
    too: without it the run stops."""

    def __init__(self, times, zones: Sequence[Hashable | None], *, name: str = "") -> None:
        t = [float(v) for v in times]
        z = list(zones)
        label = f"OccupancyDay {name!r}" if name else "OccupancyDay"
        _check_day_times(label, t, len(z), "zones")
        self.times, self.zones, self.name = t, z, name

    def __repr__(self) -> str:
        return f"OccupancyDay({self.name!r}, {len(self.times)} points)"

    def zone_at(self, seconds: float, *, left: bool) -> Hashable | None:
        """The zone at `seconds` after midnight (0..86400); `left` takes the one before a move
        listed at exactly that time."""
        s = _check_time_of_day(f"OccupancyDay {self.name!r}", seconds)
        return _piecewise(self.times, self.zones, s, linear=False, left=left)

    def knots(self) -> list[float]:
        """The interior times (s after midnight) where the occupant may move."""
        return self.times[1:-1]


class Occupancy(_DayTypeCycle):
    """An occupant's movements: one `OccupancyDay` per CONTAM day type (1-12).

    `day_types` has 12 entries; entry k is the 1-based position in `day_schedules` used on
    day type k + 1 (CONTAM's twelve occupancy-schedule numbers). The default uses the first
    occupancy day every day.
    """

    _KIND = "Occupancy"
    _DAY = "occupancy day"

    def __init__(self, day_schedules: Sequence[OccupancyDay],
                 day_types: Sequence[int] = (1,) * N_DAY_TYPES, *, name: str = "") -> None:
        super().__init__(day_schedules, day_types, name=name)

    @property
    def zones(self) -> set[Hashable]:
        """Every zone of any of its day schedules, also of a day type a run does not reach
        (None, out of the building, excluded)."""
        return {z for d in self.day_schedules for z in d.zones if z is not None}

    def zone_at(self, t: float, calendar: Calendar, *, left: bool) -> Hashable | None:
        """The zone at `t` s since the start; `left` (ignored at t = 0) takes the zone before
        a move at exactly `t`, and at midnight the previous day's last zone. A time within
        the calendar's round-off of a move is that move."""
        left = left and float(t) > 0.0
        day, tod = self._day_and_time(t, calendar, left=left)
        return day.zone_at(tod, left=left)

    def segments(self, t0: float, t1: float, calendar: Calendar
                 ) -> list[tuple[float, float, Hashable | None]]:
        """(a, b, zone) for every piece of (t0, t1] (s since the start) spent in one zone, in
        time order; cut at every move and every midnight."""
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError(f"Occupancy {self.name!r}: segments need t1 > t0, got "
                             f"({t0}, {t1})")
        cuts = [t0, *self.breakpoints(t0, t1, calendar), t1]
        return [(a, b, self.zone_at(a, calendar, left=False))
                for a, b in zip(cuts, cuts[1:], strict=False)]


def _zone_nodes(occupancy: Occupancy, net, what: str) -> dict[Hashable, int]:
    """{zone: its index into `net.nodes`} for every zone of any of `occupancy`'s day
    schedules; a zone that is not a node is refused (KeyError, `what` names the owner)."""
    out: dict[Hashable, int] = {}
    for zone in sorted(occupancy.zones, key=str):
        try:
            out[zone] = int(net.node_index(zone))
        except KeyError as exc:
            raise KeyError(f"{what}: zone {zone!r} is not a node of the network") from exc
    return out


def _scalar_rate(value, what: str) -> Tensor:
    """`value` as a finite float64 scalar tensor (gradients kept), else ValueError."""
    t = torch.as_tensor(value, dtype=F64)
    if t.dim() != 0:
        raise ValueError(f"{what} must be a scalar, got shape {tuple(t.shape)}")
    if not bool(torch.isfinite(t)):
        raise ValueError(f"{what} must be finite, got {t.item()}")
    return t


class Presence:
    """An occupant's location as a signal over `net`'s nodes (full node order, float64,
    dimensionless).

    `value(t, calendar)` is 1 at the occupied node (the left limit at a move for t > 0, as
    every schedule) and all zeros out of the building. `step_mean(t0, t1, calendar)` is the
    fraction of the step (t0, t1] spent at each node; `apply_inputs` writes it when it knows
    the step's start, which is how an occupant's generation is split at a mid-step move.

    Occupancy is a step-mean signal, not an instantaneous one: ContamX averages an occupant's
    location over each step, as it averages its schedules, and splits the step at a move
    inside it. The instant at the step's end would put the whole step in the new zone.

    A zone the occupancy names that is not a node of `net` is refused (KeyError).
    """

    def __init__(self, occupancy: Occupancy, net, *, name: str = "") -> None:
        self.occupancy, self.name, self.n = occupancy, name, int(net.n)
        self.index = _zone_nodes(occupancy, net, f"Presence {name!r}")

    def __repr__(self) -> str:
        return f"Presence({self.name!r})"

    def value(self, t: float, calendar: Calendar) -> Tensor:
        """1 at the occupied node at `t` s since the start, 0 elsewhere, `(n,)`."""
        out = torch.zeros(self.n, dtype=F64)
        zone = self.occupancy.zone_at(t, calendar, left=float(t) > 0.0)
        if zone is not None:
            out[self.index[zone]] = 1.0
        return out

    def step_mean(self, t0: float, t1: float, calendar: Calendar) -> Tensor:
        """The fraction of (t0, t1] (s since the start) spent at each node, `(n,)`; sums to
        the fraction spent in the building."""
        out = torch.zeros(self.n, dtype=F64)
        span = float(t1) - float(t0)
        for a, b, zone in self.occupancy.segments(t0, t1, calendar):
            if zone is not None:
                out[self.index[zone]] += (b - a) / span
        return out

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]:
        """The times in (lo, hi) (s since the start) where the occupant may move."""
        return self.occupancy.breakpoints(lo, hi, calendar)


class Occupant:
    """A person (or `multiplier` people) following an `Occupancy`, generating each species in
    `generation` (kg/s per person; quanta/s for a quanta species) while in the building.

    `schedules` maps a generated species to a signal (a `WeekSchedule`, a `TimeSeries`)
    multiplying its rate, taken at the end of each step as every schedule. The generation
    goes to the zone the occupant is in, split by the time spent in each zone over the step
    (measured on ContamX). `name` keys the occupant's drivers: `"presence.<name>"` and
    `"generation_multiplier.<name>"`; `description` is free text. A generation rate or a
    multiplier that is not one finite number is refused, naming the occupant.
    """

    def __init__(self, occupancy: Occupancy, *, generation=None, multiplier=1.0,
                 schedules=None, name: str = "occupant", description: str = "") -> None:
        self.occupancy, self.name, self.description = occupancy, str(name), str(description)
        self.generation = {
            str(k): _scalar_rate(v, f"Occupant {self.name!r}: the generation of {str(k)!r}")
            for k, v in dict(generation or {}).items()}
        self.multiplier = _scalar_rate(multiplier, f"Occupant {self.name!r}: the multiplier")
        self.schedules = dict(schedules or {})
        extra = sorted(set(self.schedules) - set(self.generation))
        if extra:
            raise ValueError(f"Occupant {self.name!r}: schedules for {extra}, which it does "
                             f"not generate")

    def __repr__(self) -> str:
        return f"Occupant({self.name!r}, generation={sorted(self.generation)})"

    @property
    def presence_key(self) -> str:
        """The driver of the fraction of each step the occupant spends at each node."""
        return f"presence.{self.name}"

    @property
    def multiplier_key(self) -> str:
        """The driver of the schedule multiplier of each species the occupant generates."""
        return f"generation_multiplier.{self.name}"


class _Stacked:
    """Scalar signals (None: always 1) as one vector signal."""

    def __init__(self, signals) -> None:
        self.signals = list(signals)

    def value(self, t: float, calendar: Calendar) -> Tensor:
        one = torch.ones((), dtype=F64)
        return torch.stack([one if s is None else
                            torch.as_tensor(s.value(t, calendar), dtype=F64).reshape(())
                            for s in self.signals])

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]:
        pts: set[float] = set()
        for s in self.signals:
            if s is not None:
                pts.update(s.breakpoints(lo, hi, calendar))
        return sorted(pts)


def occupant_inputs(occupants: Sequence[Occupant], net) -> list[Input]:
    """The inputs `simulate` needs for the occupants that generate something: each one's
    presence on `"presence.<name>"` (a step mean, the fraction of the step at each node of
    `net`) and, when it has schedules, the schedule values (1 for an unscheduled species,
    in the order of its `generation`) on `"generation_multiplier.<name>"`. Occupants whose
    names repeat are refused: their drivers would collide."""
    _refuse_repeated_names(occupants, "occupant_inputs")
    out: list[Input] = []
    for occ in occupants:
        if not occ.generation:
            continue
        out.append(Input(Presence(occ.occupancy, net, name=occ.name), occ.presence_key,
                         name=f"{occ.name} location"))
        if occ.schedules:
            signals = [occ.schedules.get(s) for s in occ.generation]
            out.append(Input(_Stacked(signals), occ.multiplier_key,
                             name=f"{occ.name} generation schedules"))
    return out


def _refuse_repeated_names(occupants: Sequence[Occupant], what: str) -> None:
    names = [o.name for o in occupants]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{what}: occupant names {dup} repeat; each needs its own")


def _occupant_nodes(occupant: Occupant, layer, what: str) -> dict[Hashable, int]:
    """{zone: its index into `layer.net.nodes`} for every zone of any of `occupant`'s day
    schedules, each refused by name where `layer` holds no air: not a node (KeyError), the
    boundary, a node no flow edge of the layer touches, or a storage-free node
    (ValueError)."""
    who = f"{what}: occupant {occupant.name!r}"
    nodes = _zone_nodes(occupant.occupancy, layer.net, who)
    for node in nodes.values():
        _check_emitting_node(layer, node, who)
    return nodes


class OccupantGeneration:
    """Closure: the occupants' generation added to `"<layer>.sources"` (kg/s, full node
    order) of the species `layer` (a `TransportLayer`).

    An occupant's rate per species (generation x multiplier x its entry of
    `"generation_multiplier.<name>"`, 1 when that driver is absent) is spread over the nodes
    by `"presence.<name>"`, the fraction of the step spent at each node, which
    `occupant_inputs` writes. The result is added to what the sources already hold (a
    driver, `SpeciesSources`), so the order of the two closures does not matter. With one
    species the sum keeps the layout of what was there (with or without the species axis);
    alone it is `(..., n, K)`.

    `species` are the layer's species names, in its order: `layer.species_names`, or given
    for a layer that names none. Refused by name at construction: occupants whose names
    repeat, a generated species the layer does not carry (KeyError), and any zone of any of
    a generating occupant's day schedules (also of a day type the run does not reach) that
    is not a node (KeyError), is the layer's boundary (the ambient), is a node no flow edge
    of the layer touches, or is a storage-free node: the layer holds no air there to
    generate into. Occupants who generate nothing are ignored.
    """

    def __init__(self, occupants: Sequence[Occupant], *, layer,
                 species: Sequence[str] | None = None) -> None:
        _refuse_repeated_names(occupants, "OccupantGeneration")
        self.layer = layer
        self.species = self._species(layer, species)
        self.occupants = [o for o in occupants if o.generation]
        for o in self.occupants:
            unknown = sorted(set(o.generation) - set(self.species))
            if unknown:
                raise KeyError(f"OccupantGeneration: occupant {o.name!r} generates {unknown}, "
                               f"which layer {layer.name!r} does not carry (species "
                               f"{self.species})")
            _occupant_nodes(o, layer, "OccupantGeneration")
        self._columns = [torch.tensor([self.species.index(s) for s in o.generation])
                         for o in self.occupants]
        self.key = f"{layer.name}.sources"
        self.inputs = tuple(k for o in self.occupants
                            for k in (o.presence_key, o.multiplier_key)) + (self.key,)
        self.outputs = (self.key,)

    @staticmethod
    def _species(layer, species) -> list[str]:
        named = None if layer.species_names is None else list(layer.species_names)
        if species is None:
            if named is None:
                raise ValueError(f"OccupantGeneration: layer {layer.name!r} names no species; "
                                 f"give them with species=[...], in the layer's order")
            return named
        given = [str(s) for s in species]
        if named is not None and given != named:
            raise ValueError(f"OccupantGeneration: species {given} differ from layer "
                             f"{layer.name!r}'s {named}")
        if len(given) != layer.n_species:
            raise ValueError(f"OccupantGeneration: layer {layer.name!r} carries "
                             f"{layer.n_species} species, got the names {given}")
        return given

    def __repr__(self) -> str:
        return f"OccupantGeneration({[o.name for o in self.occupants]})"

    @property
    def input_specs(self) -> dict[str, dict]:
        """Each generating occupant's presence and schedule multipliers."""
        specs: dict[str, dict] = {}
        for o in self.occupants:
            specs[o.presence_key] = dict(
                description=f"fraction of the step occupant {o.name!r} spends at each node",
                unit="-", over="nodes")
            specs[o.multiplier_key] = dict(
                description=f"schedule multiplier of each species occupant {o.name!r} "
                            f"generates", unit="-", over=list(o.generation), required=False)
        return specs

    def __call__(self, state, drivers):
        K = len(self.species)
        total = None
        for o, cols in zip(self.occupants, self._columns, strict=True):
            try:
                presence = torch.as_tensor(drivers[o.presence_key], dtype=F64)
            except KeyError as exc:
                raise KeyError(f"OccupantGeneration: driver {o.presence_key!r} is missing; "
                               f"simulate writes it from occupant_inputs(...)") from exc
            rate = torch.stack(list(o.generation.values())) * o.multiplier
            mult = drivers.get(o.multiplier_key)
            if mult is not None:
                rate = rate * torch.as_tensor(mult, dtype=F64)
            per_species = torch.zeros(rate.shape[:-1] + (K,), dtype=F64).index_copy(
                -1, cols, rate)
            g = presence.unsqueeze(-1) * per_species.unsqueeze(-2)
            total = g if total is None else total + g
        if total is None:
            return {}
        return {self.key: _add_sources(drivers.get(self.key), total, own_axis=True,
                                       n_species=K)}
