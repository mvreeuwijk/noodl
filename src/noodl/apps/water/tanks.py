"""Simple `[CONTROLS]` on tank levels, as one closure carrying the link status.

`state_keys = ("water.link_status",)`: the open/closed status of every controlled link,
which persists across steps and belongs to no layer. A tank's level is not carried here:
it is the tank node's head less its bottom, read from `"water.phi"` (`TankLevels.levels`).
The tank itself is a `noodl.LinearStorage` of its cross-section area in the water layer
(`build_model(tank_scheme=...)`): under `"forward_euler"` (EPANET's method) a boundary node
the solve holds at its step-start head and the step then moves by `dt * inflow / A`
(`KirchhoffLayer.advance_storage`, `water_event_step`); under `"backward_euler"` or
`"trapezoidal"` an interior node stepped inside the solve.

Per call the closure (1) applies every control's level test with EPANET's semantics, under
`no_grad` -- the switch itself is a combinatorial decision; and (2) writes the reservoir heads
into `"water.phi_boundary"` and the per-link status into `"water.status"`: every pump's from
the carried `"water.link_status"`, every other link's from the drivers' own `"water.status"`
(or, when they carry none, the build's). A tank held at a
boundary node has a column of `"water.phi_boundary"` too, which the closure fills with zeros:
the layer holds the tank at its step-start head there on every path (`Model` steps, steady
solves, `residuals`, `current_flows`; `KirchhoffLayer.hold_storage`), so it is the only
authority for the tank heads. During a step the controls read the step-start levels: a
`Model` pass shows a closure the held head of such a node, not the level a previous pass of
the same step has advanced it to (`coupling="iterate"`).

`event_step` is EPANET's ADAPTIVE shortening, and it is not optional: with a fixed step
the pump switches up to a whole step late and the tank level diverges from EPANET; the step
is shortened to the linearly-projected trigger crossing instead (numbers in
`docs/design/water_street.md#event-step`; checked in
`tests/verification/test_water_parity.py::test_net1_extended_period_tank_level`). It
implements only the CONTROL-CROSSING half of EPANET's rule
(Manual section 13.1 item 17, p.113: the next step is the minimum of the nominal step and
the time to the next control crossing) -- not the demand-PERIOD boundary half, which the
Net1 driver does not need: Net1's pattern step (2 h) is an exact multiple of its
hydraulic step (1 h), so no demand-period boundary ever falls strictly inside a step and the
omission is masked on every fixture the application is measured against.

FULL AND EMPTY TANKS. Under `"forward_euler"` the closure applies EPANET's `tankstatus`
(`hydstatus.c`): a tank whose head is within `HTOL` of its maximum is full and one within
`HTOL` of its minimum is empty, and every link to it is given the one direction that cannot
fill a full tank or drain an empty one (`"water.one_way"`); a pump discharging into a full
tank or drawing from an empty one is CLOSED for the step, as is a link both rules forbid (a
check valve into a full tank). A tank that may overflow (`Tank.overflow`) is never full.
The step itself is EPANET's `tanklevels` (`water_event_step`): it is shortened to the time a
tank fills or empties (`TankLevels.time_to_limit`), and a tank within one second of its
limit is put on it. Under the implicit schemes the tank moves inside the solve, so these
rules do not apply there and a level outside `[y_min, y_max]` is refused by name instead.

A tank that is not a cylinder is a `TankVolumeCurve`: its volume a piecewise linear
function of its level (`[CURVES]`), stepped by volume (`NodeStorage.potential_of`).
"""

from __future__ import annotations

import torch

from noodl.apps.water.controls import CLOSED, OPEN
from noodl.apps.water.elements import M_PER_FT, ONE_WAY_KEY, STATUS_KEY
from noodl.elements.base import ConstitutiveLaw
from noodl.storage import NodeStorage

Tensor = torch.Tensor
F64 = torch.float64

#: EPANET 2.2's head tolerance `Htol` (0.0005 ft, `hydraul.c`, `inithyd`) in metres: a tank
#: within it of its maximum (minimum) level is full (empty) (`hydstatus.c`, `tankstatus`).
HTOL = 0.0005 * M_PER_FT

#: EPANET's `TINY` (`types.h`): a curve segment narrower than this takes its end value
#: (`project.c`, `interp`). EPANET applies it in the curve's FILE units (ft or m, ft3 or m3);
#: here it is applied in SI (m, m3). Every curve `TankVolumeCurve` accepts has strictly
#: increasing levels and volumes that do not decrease; `interp` on the volumes never meets
#: a flat segment's zero width (it takes the first point at or above the volume).
TINY = 1e-6

#: EPANET's `QZERO` (`types.h`, 1e-6 cfs) in m3/s: `tanktimestep` (`hydraul.c`) skips a tank
#: whose net inflow is no larger.
QZERO = 1e-6 * M_PER_FT**3


def _interp(x: Tensor, y: Tensor, last: Tensor, xx: Tensor, *,
            right: bool = False) -> tuple[Tensor, Tensor]:
    """EPANET 2.2's `interp` (`project.c`), per curve, and the slope it takes.

    `x`, `y`: `(n, width)`, one curve per row, `x` ascending, padded beyond each row's
    `last` index (at least 1: two points); `xx`: `(..., n)`, in the curves' SI units, with
    `TINY` applied in them too (EPANET applies it in the file's units). Returns
    `(y(xx), dy/dx)`: `y[0]` at or below the first point, `y[last]` beyond the last (no
    extrapolation, slope 0), and otherwise the segment
    whose right end is the first point with `x >= xx`, so the LEFT segment at a point, as
    EPANET; `right=True` takes the right one there instead (the first point with `x > xx`;
    used only for the slope a rising level moves onto)."""
    width = x.shape[-1]
    valid = torch.arange(width) <= last[:, None]
    xd, xxd = x.detach(), xx.detach()[..., None]
    k = (((xxd >= xd) if right else (xxd > xd)) & valid).sum(-1)
    hi = torch.minimum(k.clamp(min=1), last.expand(k.shape))
    batch = xx.shape[:-1]
    x_full, y_full = x.expand(batch + x.shape), y.expand(batch + y.shape)
    x1 = x_full.gather(-1, hi[..., None]).squeeze(-1)
    x0 = x_full.gather(-1, (hi - 1)[..., None]).squeeze(-1)
    y1 = y_full.gather(-1, hi[..., None]).squeeze(-1)
    y0 = y_full.gather(-1, (hi - 1)[..., None]).squeeze(-1)
    dx, dy = x1 - x0, y1 - y0
    flat = dx.abs() < TINY
    slope = torch.where(flat, torch.zeros_like(dy), dy / torch.where(flat, 1.0, dx))
    inside = torch.where(flat, y1, y1 - (x1 - xx) * slope)
    first = y_full[..., 0]
    end = y_full.gather(-1, last.expand(k.shape)[..., None]).squeeze(-1)
    below, above = k == 0, k > last
    value = torch.where(below, first, torch.where(above, end, inside))
    slope = torch.where(below | above, torch.zeros_like(slope), slope)
    return value, slope


class TankVolumeCurve(NodeStorage):
    """Tanks whose volume is a piecewise linear function of the level: EPANET's volume
    curve (`[TANKS]` VolCurve, `[CURVES]` level against volume).

    `nodes` are the tank nodes, `bottom` (m) their bottom elevations, and `levels` (m
    above the bottom) and `volumes` (m3) each tank's curve points (one sequence per tank,
    of at least two points, levels strictly increasing and volumes not decreasing). The
    layer's potential is the head times `head_scale` (`rho g` on the Darcy-Weisbach path).

    FLAT SEGMENTS. A segment whose two volumes are equal has zero area; EPANET 2.2 accepts it
    and so does this class. Its level is a jump: `tankgrade` gives the flat volume the
    segment's LOWER level and a volume just above it a level just above its upper one
    (`interp` takes the first point at or above the volume), and `potential_of` is the same
    transcription. A maximum (minimum) level on a flat stretch has the stretch's volume as
    `Vmax` (`Vmin`), and a tank put on it is put at `tankgrade(Vmax)`, the stretch's lower
    end (`_settle_tanks`); its controls are tested by volume (`TankLevels._apply_controls`).
    A DECREASING volume is refused by name: EPANET accepts it too (wntr 1.5.0 solves Net1
    with `(110, 80000)` after `(60, 90000)`), but a volume that falls as the level rises is
    not physical.

    `amount` is EPANET's `tankvolume` and `potential_of` its `tankgrade` (`hydraul.c`), both
    through `interp` (`project.c`): linear between the points, the end values beyond the
    ends. `capacity` is the slope of the segment (`d volume / d phi`, the left one at a
    point). Under `tank_scheme="forward_euler"` the layer moves a tank to
    `potential_of(amount + dt * inflow)` (`KirchhoffLayer.advance_storage`), EPANET's
    `V += q dt; H = tankgrade(V)`, so the volume changes by exactly `dt * inflow`.
    With `learnable=True` the volumes are a trainable `nn.Parameter`; the gradient reaches
    them through the steps, not through the solve, which holds the tank at its head.
    """

    linear = False

    def __init__(self, nodes, bottom, levels, volumes, *, head_scale: float = 1.0,
                 learnable: bool = False) -> None:
        super().__init__(nodes)
        rows_x = [[float(v) for v in row] for row in levels]
        rows_y = [[float(v) for v in row] for row in volumes]
        n = self.nodes.numel()
        if len(rows_x) != n or len(rows_y) != n:
            raise ValueError(
                f"TankVolumeCurve: {n} nodes need {n} curves, got {len(rows_x)} level and "
                f"{len(rows_y)} volume rows")
        for i, (xs, ys) in enumerate(zip(rows_x, rows_y, strict=True)):
            if len(xs) != len(ys):
                raise ValueError(
                    f"TankVolumeCurve: curve {i} has {len(xs)} levels and {len(ys)} volumes")
            if len(xs) < 2:
                raise ValueError(
                    f"TankVolumeCurve: curve {i} has {len(xs)} point(s); a volume curve needs "
                    f"at least two")
            if any(b <= a for a, b in zip(xs, xs[1:], strict=False)) or any(
                    b < a for a, b in zip(ys, ys[1:], strict=False)):
                raise ValueError(
                    f"TankVolumeCurve: curve {i} must have strictly increasing levels and "
                    f"volumes that do not decrease (a decreasing volume, which EPANET "
                    f"accepts, is not physical), got levels {xs} and volumes {ys}")
        width = max(len(xs) for xs in rows_x)
        pad = [(xs + [xs[-1]] * (width - len(xs)), ys + [ys[-1]] * (width - len(ys)))
               for xs, ys in zip(rows_x, rows_y, strict=True)]
        self.register_buffer("levels", torch.tensor([x for x, _ in pad], dtype=F64))
        self.register_buffer("last", torch.tensor([len(xs) - 1 for xs in rows_x]))
        self.register_buffer("bottom", torch.as_tensor(bottom, dtype=F64).clone())
        self.volumes = ConstitutiveLaw._param(torch.tensor([y for _, y in pad], dtype=F64),
                                              learnable)
        self.head_scale = float(head_scale)

    def _level(self, phi_nodes: Tensor) -> Tensor:
        return phi_nodes / self.head_scale - self.bottom

    def amount(self, phi_nodes, drivers=None):
        return _interp(self.levels, self.volumes, self.last, self._level(phi_nodes))[0]

    def capacity(self, phi_nodes, phi_start, drivers=None):
        return _interp(self.levels, self.volumes, self.last,
                       self._level(phi_nodes))[1] / self.head_scale

    def potential_of(self, amount, drivers=None):
        level = _interp(self.volumes, self.levels, self.last, amount)[0]
        return (self.bottom + level) * self.head_scale

    def area_toward(self, phi_nodes: Tensor, flow: Tensor) -> Tensor:
        """Each tank's cross-section area (m2, `d volume / d level`) on the segment its
        level moves onto at inflow `flow`: the right one at a point when rising, the left
        one otherwise; where that area is zero (no segment beyond the last point rising or
        the first falling, or a segment of flat volumes) the other one's, and 1 where
        neither has one (inside a flat segment, a level only an initial level can have).
        Only the level rate's sign is read from it (`water_event_step`):
        the times to a limit or a control are by volume at the inflow
        (`TankLevels._volume_time`) and the advance is by volume (`potential_of`), as
        EPANET's `tanktimestep` and `tanklevels`, so no area enters them. Across a flat
        segment EPANET's level jumps: `tankgrade` puts the flat volume at the segment's
        lower end and any more at its upper end; `potential_of` does the same."""
        level = self._level(phi_nodes)
        left = _interp(self.levels, self.volumes, self.last, level)[1]
        right = _interp(self.levels, self.volumes, self.last, level, right=True)[1]
        rising = flow.detach() > 0
        toward = torch.where(rising, right, left)
        other = torch.where(rising, left, right)
        return torch.where(toward != 0, toward, torch.where(other != 0, other, 1.0))

    def extra_repr(self) -> str:
        return f"nodes={self.nodes.tolist()}"


class TankLevels:
    """Simple controls on tank levels, and the boundary heads of a water layer whose tanks
    are held at boundary nodes (`build_model(tank_scheme="forward_euler")`).

    A tank's level is its head in `"water.phi"` at `tank_nodes` less its bottom (`levels`).
    Per call it applies the controls to those levels and writes the reservoir heads into
    `"water.phi_boundary"`, then zeros in the tanks' columns, which the layer fills with
    their step-start heads (`KirchhoffLayer.hold_storage`). Only the link status is
    carried. `event_step` is EPANET's shortening of the step to the next control crossing,
    `time_to_limit` the time a tank takes to fill or empty.

    FULL AND EMPTY TANKS (`hydstatus.c`, `tankstatus`). `tank_links` lists, per link that
    touches a tank, `(link, tank, at_start, kind)`: the link's position in `link_names`, the
    tank EPANET examines for it (the start node when that is a tank or reservoir -- a link
    from a reservoir is never examined -- else the end node), whether that tank is the
    link's start node, and `"pipe"`, `"pump"` or `"valve"`. When a tank is full (level
    `>= max - HTOL`, and not `overflow`) or empty (`<= min + HTOL`) the closure also writes
    `"water.one_way"`: a pipe to it takes the one direction out of a full tank or into an
    empty one (on top of the drivers' own, `one_way` as built); a pump discharging into a
    full tank or drawing from an empty one, and a pipe both directions are then forbidden
    (a check valve into a full tank), are CLOSED in `"water.status"` for the call. A valve
    at a full or empty tank is refused by name (the valves take no direction).

    A pump's entry of `"water.status"` is written from the carried `"water.link_status"` on
    every call, so a pump is closed by setting that state, not the driver.

    The closure remembers its last answer with the values it read (the tank levels,
    `"water.link_status"`, the drivers' `"water.status"`): a call that reads the same
    values returns it again, which is what lets `water_steady` read the status before the
    solve without running the controls twice.
    """

    state_keys = ("water.link_status",)
    # The tank levels are stepped by the water layer's storage, not here, so this closure
    # does not integrate and takes no `StepContext`.
    integrates = False

    def __init__(
        self,
        net,
        tanks,
        controls,
        pump_names,
        *,
        tank_nodes,
        dt: float,
        reservoir_heads: Tensor,
        head_scale: float = 1.0,
        link_names,
        link_status: Tensor,
        tank_links=(),
        one_way: Tensor | None = None,
        storage: NodeStorage | None = None,
    ) -> None:
        self.net = net
        self.tanks = list(tanks)
        self.controls = list(controls)
        self.pump_names = list(pump_names)
        # The labels of the closure-carried state, in its order (`noodl.refs`).
        self.key_labels = {"water.link_status": list(self.pump_names)}
        self.tank_nodes = torch.as_tensor(tank_nodes, dtype=torch.long)
        self.dt = float(dt)
        self.reservoir_heads = torch.as_tensor(reservoir_heads, dtype=F64)
        # 1.0 on the Hazen-Williams path (the potential IS metres of head) and `rho g` on
        # the Darcy-Weisbach path, where the layer's potential is PRESSURE: the levels and
        # the trigger elevations stay in metres either way and are converted here, once.
        self.head_scale = float(head_scale)
        self.notes: dict[str, str] = {}
        names = [t.name for t in self.tanks]
        for control in self.controls:
            if control.node not in names:
                raise ValueError(
                    f"TankLevels: control on link {control.link!r} tests node "
                    f"{control.node!r}, which is not a tank of this network ({names})"
                )
            if control.link not in self.pump_names:
                raise ValueError(
                    f"TankLevels: control names link {control.link!r}, which is not a pump "
                    f"of this network ({self.pump_names}); pipe controls are not "
                    f"implemented"
                )
            if control.trigger != "node" or control.status is None or control.above is None:
                raise ValueError(
                    f"TankLevels: control on link {control.link!r} is not a tank-level "
                    f"OPEN/CLOSED control; only 'LINK <id> OPEN|CLOSED IF NODE <tank> "
                    f"BELOW|ABOVE <level>' is applied"
                )
        self.tank_of = {t.name: i for i, t in enumerate(self.tanks)}
        self.pump_of = {name: i for i, name in enumerate(self.pump_names)}
        # Every link's status as built (`model.link_names` order), and the pumps' places in it.
        self.link_names = list(link_names)
        self.link_status = torch.as_tensor(link_status, dtype=F64)
        self.pump_links = torch.tensor([self.link_names.index(n) for n in self.pump_names],
                                       dtype=torch.long)
        self.bottom = torch.tensor([t.elevation for t in self.tanks], dtype=F64)
        self.y_min = torch.tensor([t.min_level for t in self.tanks], dtype=F64)
        self.y_max = torch.tensor([t.max_level for t in self.tanks], dtype=F64)
        self.overflow = torch.tensor([bool(getattr(t, "overflow", False))
                                      for t in self.tanks])
        # the tanks' storage, read for their volumes when it is not a cylinder's
        self.storage = storage
        # EPANET's `tankstatus` rules, per link that touches a tank (see the class docstring)
        rules = list(tank_links)
        self.one_way = (torch.zeros(len(self.link_names), dtype=F64) if one_way is None
                        else torch.as_tensor(one_way, dtype=F64))
        self._rule_links = torch.tensor([r[0] for r in rules], dtype=torch.long)
        self._rule_tanks = torch.tensor([r[1] for r in rules], dtype=torch.long)
        self._rule_out = torch.tensor([bool(r[2]) for r in rules])   # tank at the start
        self._rule_valve = torch.tensor([r[3] == "valve" for r in rules])
        self._is_pump = torch.zeros(len(self.link_names), dtype=torch.bool)
        self._is_pump[self.pump_links] = True

    def levels(self, state) -> Tensor:
        """Every tank's level (m above its bottom) in `state`."""
        return state["water.phi"][..., self.tank_nodes] / self.head_scale - self.bottom

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        status = state.get("water.link_status")
        if status is None or "water.phi" not in state:
            raise KeyError(
                f"{type(self).__name__}: states 'water.phi' and 'water.link_status' are "
                f"both required; build them with initial_state(model)"
            )
        level = self.levels(state)
        # a tank on a volume curve tests its controls by volume, with a margin of one
        # second of the last solve's inflow (`_apply_controls`)
        inflow = state.get("water.storage_flow") if self._curved else None
        read = (level.detach(), status.detach(), drivers.get(STATUS_KEY),
                drivers.get(ONE_WAY_KEY), None if inflow is None else inflow.detach())
        memo = getattr(self, "_memo", None)
        if memo is not None and all(
                (a is None and b is None) or (a is not None and b is not None
                                              and a.shape == b.shape and torch.equal(a, b))
                for a, b in zip(read, memo[0], strict=True)):
            return dict(memo[1])
        with torch.no_grad():
            status = self._apply_controls(level, status, inflow)
        self._refuse_outside(level)
        out = {"water.phi_boundary": self._boundary(state, level),
               STATUS_KEY: self._link_status(drivers, status),
               "water.link_status": status}
        limited = self._full_and_empty(level, drivers, out[STATUS_KEY])
        if limited is not None:
            out[STATUS_KEY], out[ONE_WAY_KEY] = limited
        self._memo = (tuple(None if t is None else t.clone() for t in read), out)
        return dict(out)

    def _link_status(self, drivers, pump_status: Tensor) -> Tensor:
        """Every link's status: the drivers' `"water.status"` (the build's when they carry
        none), with the pumps' entries from the carried `pump_status`."""
        base = drivers.get("water.status", self.link_status).to(pump_status.dtype)
        batch = pump_status.new_zeros(pump_status.shape[:-1] + (1,))   # its batch shape only
        base, batch = torch.broadcast_tensors(base, batch)
        out = base.clone()
        out[..., self.pump_links] = pump_status.expand(batch.shape[:-1] + pump_status.shape[-1:])
        return out

    def _boundary(self, state, level: Tensor) -> Tensor:
        """The reservoir heads, then zeros in the tanks' columns (boundary nodes here, whose
        heads the layer holds: `KirchhoffLayer.hold_storage`)."""
        batch = level.shape[:-1]
        return torch.cat(
            [self.reservoir_heads.expand(batch + self.reservoir_heads.shape),
             torch.zeros(batch + level.shape[-1:], dtype=level.dtype, device=level.device)],
            dim=-1,
        )

    def _full_and_empty(self, level: Tensor, drivers, link_status: Tensor):
        """`(status, one_way)` with EPANET's `tankstatus` rules applied for every full and
        empty tank (see the class docstring), or `None` when no tank is either (the usual
        case, which adds no driver). Computed without a graph: it only selects."""
        if not self._rule_links.numel():
            return None
        with torch.no_grad():
            level = level.detach()
            full = (level >= self.y_max - HTOL) & ~self.overflow
            empty = level <= self.y_min + HTOL
            if not bool((full | empty).any()):
                return None
            f, e = full[..., self._rule_tanks], empty[..., self._rule_tanks]
            start = self._rule_out
            # the direction left open: out of a full tank, into an empty one
            forward = (f & start) | (e & ~start)
            reverse = (f & ~start) | (e & start)
            if not bool((forward | reverse).any()):
                return None
            n_links = len(self.link_names)
            way = drivers.get(ONE_WAY_KEY, self.one_way).detach().to(F64)
            batch = torch.broadcast_shapes(level.shape[:-1], link_status.shape[:-1],
                                           way.shape[:-1])
            open_ = link_status.expand(batch + (n_links,))[..., self._rule_links] != CLOSED
            valve = (forward | reverse) & self._rule_valve & open_
            if bool(valve.any()):
                links = self._rule_links[valve.reshape(-1, valve.shape[-1]).any(0)]
                names = [self.link_names[j] for j in links.tolist()]
                raise ValueError(
                    f"TankLevels: valve(s) {names} are at a full or empty tank, which "
                    f"EPANET closes against filling or draining it (`tankstatus`); a valve "
                    f"takes no flow direction here")
            pos = torch.zeros(batch + (n_links,), dtype=torch.bool)
            neg = torch.zeros(batch + (n_links,), dtype=torch.bool)
            pos[..., self._rule_links] = forward.expand(batch + forward.shape[-1:])
            neg[..., self._rule_links] = reverse.expand(batch + reverse.shape[-1:])
            way = way.expand(batch + (n_links,))
            pump = self._is_pump
            # a pump runs forward only: one that may only run in reverse is closed
            closed = (pump & neg) | (~pump & ((pos & (neg | (way < 0)))
                                              | (neg & (pos | (way > 0)))))
            pipe = ~pump & ~closed
            way = torch.where(pipe & pos, 1.0, torch.where(pipe & neg, -1.0, way))
            status = torch.where(closed, float(CLOSED),
                                 link_status.expand(batch + (n_links,)))
        return status, way

    def time_to_limit(self, level: Tensor, rate: Tensor,
                      inflow: Tensor | None = None) -> Tensor:
        """Seconds each tank takes to reach its maximum level (rising, `rate > 0`) or its
        minimum (falling) at the constant rate `rate` (dy/dt, m/s), `inf` where it stands
        still or is at or beyond the limit it moves towards. EPANET's `tanktimestep`
        (`hydraul.c`): `(Vmax - V) / q` or `(Vmin - V) / q` by volume, which a tank on a
        volume curve (`TankVolumeCurve`) reads from its curve, its inflow `q` being
        `rate` times its area on the segment it moves onto (`area_toward`, the area
        `water_event_step` divides the inflow by). Given `inflow` (m3/s per tank), a
        tank whose net inflow is at most `QZERO` is skipped (`inf`), as `tanktimestep` skips
        it. EPANET then rounds to whole seconds; this is the exact time. Not
        differentiable: it only places a step."""
        with torch.no_grad():
            level, rate = level.detach(), rate.detach()
            if inflow is not None:
                rate = torch.where(inflow.detach().abs() <= QZERO, torch.zeros_like(rate),
                                   rate)
            rising, falling = rate > 0, rate < 0
            moving = (rising & (level < self.y_max)) | (falling & (level > self.y_min))
            target = torch.where(rising, self.y_max, self.y_min)
            safe = torch.where(moving, rate, torch.ones_like(rate))
            if self._curved:
                time = self._volume_time(target, level, safe, inflow)
            else:
                time = (target - level) / safe
            return torch.where(moving, time, torch.full_like(time, torch.inf))

    @property
    def _curved(self) -> bool:
        """True when the tanks are on volume curves (their storage is not linear)."""
        return self.storage is not None and not self.storage.linear

    def _volume_time(self, target: Tensor, level: Tensor, rate: Tensor,
                     inflow: Tensor | None = None) -> Tensor:
        """Seconds from `level` to `target` (m, per tank) by volume, EPANET's
        `(V(target) - V(level)) / q` (`tanktimestep`, `controltimestep`): `q` the tanks'
        inflow `inflow` (m3/s), or, when not given, `rate * area_toward(level)`, the level
        rate `rate` times the area it was divided by. No area enters with `inflow` given,
        so a segment of zero area (flat volumes) needs none."""
        phi = (self.bottom + level) * self.head_scale
        volume = self.storage.amount(phi)
        to = self.storage.amount((self.bottom + target) * self.head_scale)
        flow = (rate * self.storage.area_toward(phi, rate) if inflow is None
                else inflow.detach())
        return (to - volume) / flow

    def _refuse_outside(self, level: Tensor) -> None:
        """Nothing: a tank held at a boundary node beyond a limit is full or empty
        (`_full_and_empty`), as EPANET treats it. `TankStorageControls` refuses it."""

    def _refuse_level_outside(self, level: Tensor) -> None:
        """Refuse, by name, a tank outside its `[min, max]` levels."""
        outside = (level < self.y_min - 1e-9) | (level > self.y_max + 1e-9)
        if bool(outside.any()):
            bad = [
                self.tanks[i].name
                for i, flag in enumerate(
                    outside.reshape(-1, len(self.tanks)).any(0).tolist()
                )
                if flag
            ]
            raise ValueError(
                f"TankLevels: tank(s) {bad} are outside their [min, max] levels; EPANET "
                f"closes the attached links instead, which this model does not do -- widen "
                f"the levels or shorten the run"
            )

    def event_step(self, level: Tensor, rate: Tensor, dt: float,
                   inflow: Tensor | None = None) -> float:
        """EPANET's shortened hydraulic step: the time to the next control crossing.

        Manual section 13.1 item 17, p.113: the next step is the minimum of the nominal
        step and the time until a tank level "reaches a point that triggers a change in
        status for some link", computed on the assumption that the level changes LINEARLY
        at the current solution's rate.

        `level` and `rate` (dy/dt, m/s) are given for EVERY tank, not just the one the
        caller happens to be tracking: each control tests its OWN tank (`control.node`)
        against that tank's own level and rate. A single scalar pair, tested against
        every control regardless of which tank it names, is wrong the moment a second
        controlled tank exists -- it silently uses the wrong tank's level.

        On a volume curve the crossing is by volume (`_volume_time`), at the tanks' inflow
        `inflow` (m3/s) when given, as EPANET's `controltimestep` (`hydraul.c`).
        """
        step = dt
        level, rate = level.detach(), rate.detach()
        for control in self.controls:
            i = self.tank_of[control.node]
            r = float(rate[i])
            if r == 0.0:
                continue
            if self._curved:   # by volume, as EPANET's `controltimestep`
                target = level.clone()
                target[i] = control.value
                crossing = float(self._volume_time(target, level, rate, inflow)[i].detach())
            else:
                crossing = (control.value - float(level[i])) / r
            if 1e-9 < crossing < step:
                step = crossing + 1e-9
        return step

    def _apply_controls(self, level: Tensor, status: Tensor,
                        inflow: Tensor | None = None) -> Tensor:
        """`status` with every control whose condition holds applied, in file order (a
        later control on the same pump wins). A cylinder's level is tested against the
        control's: ``level > value`` (ABOVE), ``level < value`` (BELOW). A tank on a volume
        curve is tested by volume, as EPANET 2.2's `controls` (`hydraul.c`) tests every
        tank: ``V(level) >= V(value) - |q|`` (ABOVE), ``V(level) <= V(value) + |q|``
        (BELOW), ``q`` the last solve's inflow over one second (`inflow`, m3/s; 0 when not
        given). On a curve whose volume is flat between two levels the two differ: at the
        flat stretch's volume every control level on it is reached (a tank held at
        `tankgrade(Vmax)` below its maximum level fires an ABOVE control set higher).

        A KNOWN DIFFERENCE between the two kinds, to be settled in Task 10: EPANET tests a
        cylinder by volume too, with the same ``|q|`` margin (one second of inflow, so it
        fires that much early, and at equality), where a cylinder here keeps the strict
        level test."""
        out = status.clone()
        if self._curved:
            phi = (self.bottom + level) * self.head_scale
            volume = self.storage.amount(phi)
            margin = (torch.zeros_like(volume) if inflow is None
                      else inflow.detach().abs().expand(volume.shape))
        for control in self.controls:
            i = self.tank_of[control.node]
            j = self.pump_of[control.link]
            if self._curved:
                at = level.new_full(level.shape, control.value)
                v2 = self.storage.amount((self.bottom + at) * self.head_scale)[..., i]
                v1, vplus = volume[..., i], margin[..., i]
                fires = v1 >= v2 - vplus if control.above else v1 <= v2 + vplus
            else:
                value = level[..., i]
                fires = value > control.value if control.above else value < control.value
            new = 1.0 if control.status == OPEN else 0.0
            out[..., j] = torch.where(fires, torch.full_like(out[..., j], new), out[..., j])
        return out


class TankStorageControls(TankLevels):
    """`TankLevels` for tanks stepped inside the solve, as interior storage nodes
    (`build_model(tank_scheme="backward_euler")` or `"trapezoidal"`): it writes only the
    reservoir heads into `"water.phi_boundary"`, refuses a level outside `[min, max]` by
    name instead of applying the full and empty rules, and its `event_step` never shortens
    a step below `min_step`.
    """

    def _boundary(self, state, level: Tensor) -> Tensor:
        """The reservoir heads (the tanks are interior nodes here)."""
        return self.reservoir_heads.expand(level.shape[:-1] + self.reservoir_heads.shape)

    def _refuse_outside(self, level: Tensor) -> None:
        """Refuse, by name, a tank outside its `[min, max]` levels: stepped inside the solve,
        a tank has no full or empty rule."""
        self._refuse_level_outside(level)

    def _full_and_empty(self, level: Tensor, drivers, link_status: Tensor):
        """`None`: the full and empty rules act at the step start on a held tank; here the
        tank moves inside the solve, and a level beyond a limit is refused instead."""
        return None

    def event_step(self, level: Tensor, rate: Tensor, dt: float,
                   min_step: float = 1.0) -> float:
        """`TankLevels.event_step`, never below `min_step` (s). The implicit step lands a
        little short of the projected crossing (its rate is the step's end rate, not the
        start's), and the next projection is then a fraction of a second away: a step that
        short makes the storage equation's `A / dt` large enough to stall Newton, while
        `min_step` carries the level past the trigger, so the control acts at the next
        step."""
        return max(super().event_step(level, rate, dt), min(min_step, dt))
