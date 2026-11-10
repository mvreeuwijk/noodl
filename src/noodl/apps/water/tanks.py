"""Simple `[CONTROLS]` (time, clock time, tank level, junction pressure), as
one closure carrying the link status and setting.

`state_keys = ("water.link_status", "water.link_setting")`: the status and setting of every
link (`model.link_names` order), which persist across steps and belong to no layer. The
controls (`noodl.apps.water.controls.apply_controls`) change them; the closure writes them
to `"water.status"` for every pump and every link a control names, and to `"water.setting"`
for the links whose setting a control has changed from the built one (the drivers' own are
used otherwise). `AT TIME` and `AT CLOCKTIME` read the driver `"water.time"` (s since the
start; the driver is built when the network has such a control). A junction-pressure control
acts after a solve, not before: `pressure_switch`, which `water_steady` repeats the solve for
(EPANET's `pswitch`). A tank's level is not carried here:
it is the tank node's head less its bottom, read from `"water.phi"` (`TankLevels.levels`).
The tank itself is a `noodl.LinearStorage` of its cross-section area in the water layer
(`build_model(tank_scheme=...)`): under `"forward_euler"` (EPANET's method) a boundary node
the solve holds at its step-start head and the step then moves by `dt * inflow / A`
(`KirchhoffLayer.advance_storage`, `noodl.apps.water.simulate.water_run`); under
`"backward_euler"` or `"trapezoidal"` an interior node stepped inside the solve.

Per call the closure (1) applies every control with EPANET's semantics, under `no_grad` --
the switch itself is a combinatorial decision; and (2) writes the reservoir heads
into `"water.phi_boundary"` and the per-link status into `"water.status"`: every pump's and
every controlled link's from the carried `"water.link_status"`, every other link's from the
drivers' own `"water.status"` (or, when they carry none, the build's). A tank held at a
boundary node has no column there: the layer holds it at its head in the step-start state
`"water.phi"`, which `Model` passes to every solve as `history` (`KirchhoffLayer.solve`),
so that state is the only authority for the tank heads. During a step the controls read the
step-start levels: a `Model` pass shows a closure the held head of such a node, not the level
a previous pass of the same step has advanced it to (`coupling="iterate"`).

`control_time` (`next_control_time`, EPANET's `controltimestep`) and `time_to_limit`
(`tanktimestep`) are the two tank halves of EPANET's step rule, which `water_run` composes
with the hydraulic, pattern and report steps (`hydraul.c`, `timestep`). They are not
optional: with a fixed step the pump switches up to a whole step late and the tank level
diverges from EPANET (numbers in `docs/design/water_street.md#event-step`; checked in
`tests/verification/test_water_parity.py::test_net1_extended_period_tank_level`).

FULL AND EMPTY TANKS. Under `"forward_euler"` the closure applies EPANET's `tankstatus`
(`hydstatus.c`): a tank whose head is within `HTOL` of its maximum is full and one within
`HTOL` of its minimum is empty, and every link to it is given the one direction that cannot
fill a full tank or drain an empty one (`"water.one_way"`); a pump discharging into a full
tank or drawing from an empty one is CLOSED for the step, as is a link both rules forbid (a
check valve into a full tank). A tank that may overflow (`Tank.overflow`) is never full.
The step itself is EPANET's `tanklevels` (`water_run`): it is shortened to the time a
tank fills or empties (`TankLevels.time_to_limit`), and a tank within one second of its
limit is put on it. Under the implicit schemes the tank moves inside the solve, so these
rules do not apply there and a level outside `[y_min, y_max]` is refused by name instead.

A tank that is not a cylinder is a `TankVolumeCurve`: its volume a piecewise linear
function of its level (`[CURVES]`), stepped by volume (`NodeStorage.potential_of`).
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from noodl.apps.water.controls import (
    CLOSED,
    HTOL,
    OPEN,
    apply_controls,
    next_control_time,
)
from noodl.apps.water.elements import (
    ONE_WAY_KEY,
    RESERVOIR_HEAD_KEY,
    SETTING_KEY,
    STATUS_KEY,
    TANK_INFLOW_KEY,
    TIME_KEY,
)
from noodl.apps.water.rules import SETTING_TOLERANCE
from noodl.apps.water.units import M_PER_FT, TINY
from noodl.elements.base import ConstitutiveLaw
from noodl.storage import NodeStorage

Tensor = torch.Tensor
F64 = torch.float64

# `HTOL` (`noodl.apps.water.controls`): a tank within it of its maximum (minimum) level is
# full (empty) (`hydstatus.c`, `tankstatus`).

# EPANET's `TINY` (`noodl.apps.water.units`): a curve segment narrower than it takes its end
# value (`project.c`, `interp`). EPANET applies it in the curve's FILE units (ft or m, ft3 or
# m3); here it is applied in SI (m, m3). Every curve `TankVolumeCurve` accepts has strictly
# increasing levels and volumes that do not decrease; `interp` on the volumes never meets a
# flat segment's zero width (it takes the first point at or above the volume).

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
        Only the level rate's sign is read from it (`water_run`):
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


def _identity(values) -> tuple | None:
    """What identifies `values` (tensors, numbers or `None`) for `_same`: a tensor by weak
    reference and version counter, `None` for an inference-mode tensor (no counter), any
    other value as it is."""
    out = []
    for v in values:
        if isinstance(v, torch.Tensor):
            try:
                out.append((weakref.ref(v), v._version))
            except RuntimeError:
                return None
        else:
            out.append(v)
    return tuple(out)


def _same(a: tuple, b: tuple) -> bool:
    """`_identity` keys of the same objects at the same versions (values equal)."""
    for x, y in zip(a, b, strict=True):
        if isinstance(x, tuple) and isinstance(y, tuple):
            if x[0]() is None or x[0]() is not y[0]() or x[1] != y[1]:
                return False
        elif isinstance(x, tuple) or isinstance(y, tuple):
            return False
        elif x is not y and not (x == y):
            return False
    return True


@dataclass(frozen=True)
class LinkData:
    """Every link of the network as `TankLevels` carries it, each field one value per link in
    `names` order (`model.link_names`).

    `status`: the built `"water.status"` (`CLOSED`, `OPEN`, `ACTIVE`); `kinds`: `"PIPE"`,
    `"PUMP"` or `"VALVE"` (default: every link a pump among `TankLevels`' `pump_names`,
    every other a pipe, which a control or a rule on a valve refuses); `setting`: the built
    `"water.setting"` (default 0); `setting_scale`: the setting's unit per SI unit (a PRV's,
    PSV's or PBV's head in the layer's potential unit; default 1); `setting_tolerance`:
    EPANET's tolerance on a setting a rule sets, in the carried unit (default
    `SETTING_TOLERANCE`); `one_way`: the built `"water.one_way"` (default 0)."""

    names: Sequence[str]
    status: Tensor
    kinds: Sequence[str] | None = None
    setting: Tensor | None = None
    setting_scale: Sequence[float] | None = None
    setting_tolerance: Sequence[float] | None = None
    one_way: Tensor | None = None


class TankLevels:
    """Every simple `[CONTROLS]` form (time, clock time, tank level, junction
    pressure), and the boundary heads of a water layer whose tanks are held at boundary nodes
    (`build_model(tank_scheme="forward_euler")`); on a network without tanks it carries the
    controls alone.

    A tank's level is its head in `"water.phi"` at `tank_nodes` less its bottom (`levels`).
    Per call it applies the controls to those levels and writes the reservoir heads into
    `"water.phi_boundary"` (the layer holds the tanks at their step-start heads,
    `KirchhoffLayer.solve`'s `history`). Only the link status and setting are
    carried. `control_time` is the time to the next control that fires (EPANET's
    `controltimestep`), `time_to_limit` the time a tank takes to fill or empty (`tanktimestep`).
    `links` (`LinkData`) is every link as built: its name, kind, status, setting, the
    setting's unit and a rule's tolerance on it, and its allowed direction.

    RESERVOIR HEADS. The boundary heads are the driver `"water.reservoir_head"` (one per
    reservoir, in the layer's potential unit) when the drivers carry it -- `water_run`
    writes it from the reservoirs' head patterns every step (`hydraul.c`, `demands`) -- else
    the heads as built.

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
    solve without running the controls twice. The entries built from a driver that requires
    grad (`"water.setting"`, `"water.reservoir_head"`) are not remembered but built again
    from the call's own drivers, so a second `backward()` and a new leaf of equal value
    each get a graph of their own.
    """

    state_keys = ("water.link_status", "water.link_setting")
    # The tank levels are stepped by the water layer's storage, not here, so this closure
    # does not integrate and takes no `StepContext`.
    integrates = False

    def __init__(
        self,
        tanks,
        controls,
        pump_names,
        *,
        links: LinkData,
        tank_nodes,
        dt: float,
        reservoir_heads: Tensor,
        head_scale: float = 1.0,
        tank_links=(),
        storage: NodeStorage | None = None,
        junctions=None,
        start_clocktime: float = 0.0,
        rules=(),
        node_info=None,
        speed_patterns=(),
    ) -> None:
        self.tanks = list(tanks)
        self.controls = list(controls)
        # `[RULES]`: tested by `apply_rules` (`noodl.apps.water.network`) after a solve, which
        # changes the same carried status and setting the controls do. `node_info` is
        # `{node: (index in "water.phi", elevation m)}` for every node a premise may read.
        self.rules = list(rules)
        self.node_info = {} if node_info is None else dict(node_info)
        rule_links = {a.link for r in self.rules for a in (*r.then_actions, *r.else_actions)}
        self.pump_names = list(pump_names)
        self.tank_nodes = torch.as_tensor(tank_nodes, dtype=torch.long)
        self.dt = float(dt)
        self.reservoir_heads = torch.as_tensor(reservoir_heads, dtype=F64)
        # 1.0 on the Hazen-Williams path (the potential IS metres of head) and `rho g` on
        # the Darcy-Weisbach path, where the layer's potential is PRESSURE: the levels and
        # the trigger elevations stay in metres either way and are converted here, once.
        self.head_scale = float(head_scale)
        self.start_clocktime = float(start_clocktime)
        self.notes: dict[str, str] = {}
        # `{junction: (node index, elevation m)}`: the nodes a pressure control reads
        self.junctions = {} if junctions is None else dict(junctions)
        names = [t.name for t in self.tanks]
        self.link_names = list(links.names)
        # The labels of the closure-carried state, in its order (`noodl.refs`).
        self.key_labels = {"water.link_status": list(self.link_names),
                           "water.link_setting": list(self.link_names)}
        for control in self.controls:
            if control.link not in self.link_names:
                raise ValueError(
                    f"TankLevels: control names link {control.link!r}, which is not a link "
                    f"of this network")
            if control.trigger == "node" and control.node not in (
                    names + list(self.junctions)):
                raise ValueError(
                    f"TankLevels: control on link {control.link!r} tests node "
                    f"{control.node!r}, which is not a tank or junction of this "
                    f"network (tanks {names})"
                )
        self.tank_of = {t.name: i for i, t in enumerate(self.tanks)}
        self.pump_of = {name: i for i, name in enumerate(self.pump_names)}
        # Every link's status and setting as built (`model.link_names` order), the pumps'
        # places in it and every link's kind for the controls' actions.
        self.link_status = torch.as_tensor(links.status, dtype=F64)
        self.link_setting = (torch.zeros(len(self.link_names), dtype=F64)
                             if links.setting is None
                             else torch.as_tensor(links.setting, dtype=F64))
        self.pump_links = torch.tensor([self.link_names.index(n) for n in self.pump_names],
                                       dtype=torch.long)
        link_kinds = links.kinds
        if link_kinds is None:
            # the pumps are known here, the valves are not: a control on another link needs
            # its kind (a valve is not a pipe)
            others = [c.link for c in self.controls if c.link not in self.pump_of]
            others += [n for n in rule_links if n not in self.pump_of]
            if others:
                raise ValueError(
                    f"TankLevels: controls and rules name link(s) {sorted(set(others))}, which are "
                    f"not pumps; give LinkData.kinds (\"PIPE\", \"PUMP\" or \"VALVE\" per link)")
            link_kinds = ["PUMP" if n in self.pump_of else "PIPE" for n in self.link_names]
        self.link_kinds = list(link_kinds)
        # a control's numeric setting is in SI; the link's setting in the layer's units
        self.setting_scale = ([1.0] * len(self.link_names) if links.setting_scale is None
                              else [float(x) for x in links.setting_scale])
        # EPANET's tolerance on a setting a rule sets (`takeactions`), in the carried unit
        self.setting_tolerance = ([SETTING_TOLERANCE] * len(self.link_names)
                                  if links.setting_tolerance is None
                                  else [float(x) for x in links.setting_tolerance])
        # the links whose status the closure writes: the pumps and every link a control names
        named = [self.link_names.index(c.link) for c in self.controls]
        named += [self.link_names.index(n) for n in rule_links]
        self._status_links = torch.tensor(sorted({*self.pump_links.tolist(), *named}),
                                          dtype=torch.long)
        # the links whose setting a control may change
        set_by = {self.link_names.index(c.link) for c in self.controls
                  if c.setting is not None
                  or (c.status == OPEN
                      and self.link_kinds[self.link_names.index(c.link)] == "PUMP")}
        # a rule's SETTING on a pump or a valve (on a pipe it is a status), its OPEN on a pump
        for rule in self.rules:
            for a in (*rule.then_actions, *rule.else_actions):
                j = self.link_names.index(a.link)
                kind = self.link_kinds[j]
                if kind != "PIPE" and (a.attribute == "SETTING" or (
                        kind == "PUMP" and a.value == OPEN)):
                    set_by.add(j)
        # a pump with a speed pattern: `water_run` sets its carried speed every step
        set_by.update(self.link_names.index(n) for n in speed_patterns)
        self._setting_links = torch.tensor(sorted(set_by), dtype=torch.long)
        self.pressure_controls = [c for c in self.controls
                                  if c.trigger == "node" and c.node in self.junctions]
        self._has_clock = any(c.trigger != "node" for c in self.controls)
        self._held = False
        self.bottom = torch.tensor([t.elevation for t in self.tanks], dtype=F64)
        self.y_min = torch.tensor([t.min_level for t in self.tanks], dtype=F64)
        self.y_max = torch.tensor([t.max_level for t in self.tanks], dtype=F64)
        self.overflow = torch.tensor([bool(getattr(t, "overflow", False))
                                      for t in self.tanks])
        # the tanks' storage, read for their volumes when it is not a cylinder's
        self.storage = storage
        # EPANET's `tankstatus` rules, per link that touches a tank (see the class docstring)
        rules = list(tank_links)
        self.one_way = (torch.zeros(len(self.link_names), dtype=F64) if links.one_way is None
                        else torch.as_tensor(links.one_way, dtype=F64))
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
        setting = state.get("water.link_setting")
        if status is None or setting is None or "water.phi" not in state:
            raise KeyError(
                f"{type(self).__name__}: states 'water.phi', 'water.link_status' and "
                f"'water.link_setting' are all required; build them with initial_state(model)"
            )
        # the same inputs, unchanged in place, as the last call: its answer (the solve hands
        # the closure the same objects several times a step, and so does the run's report).
        # Every driver `_evaluate` reads is among them; a tensor is known by weak reference
        # and version counter (the answer keeps no graph of the inputs alive), anything else
        # by value.
        raw = (state.get("water.phi"), status, setting, drivers.get(STATUS_KEY),
               drivers.get(SETTING_KEY), drivers.get(ONE_WAY_KEY),
               drivers.get(TANK_INFLOW_KEY), drivers.get(TIME_KEY),
               drivers.get(RESERVOIR_HEAD_KEY))
        key = _identity(raw)
        fast = getattr(self, "_fast", None)
        if key is not None and fast is not None and _same(key, fast[0]):
            return self._completed(fast[1], state, drivers)
        out = self._evaluate(state, drivers, status, setting)
        self._fast = None if key is None else (key, self._graph_free(out))
        return out

    #: The answer's entries built from drivers a calibration may make learnable: a remembered
    #: answer is kept without them where they carry a graph, and `_completed` builds them
    #: again from the drivers of the call (the graph of the first call's drivers is freed
    #: by its `backward()`, and an equal-valued new leaf must receive the gradient).
    _GRAPH_KEYS = ("water.phi_boundary", SETTING_KEY)

    @classmethod
    def _graph_free(cls, out: dict[str, Tensor]) -> dict[str, Tensor]:
        """`out` without the `_GRAPH_KEYS` entries that carry a graph."""
        return {k: v for k, v in out.items()
                if not (k in cls._GRAPH_KEYS and v.requires_grad)}

    def _completed(self, remembered: dict[str, Tensor], state, drivers) -> dict[str, Tensor]:
        """A remembered answer with its `_GRAPH_KEYS` entries built from this call's
        `drivers` wherever they are missing or the driver they are built from requires grad
        (no graph is ever reused across calls)."""
        out = dict(remembered)
        heads = drivers.get(RESERVOIR_HEAD_KEY)
        if "water.phi_boundary" not in out or (heads is not None and heads.requires_grad):
            out["water.phi_boundary"] = self._boundary(drivers, self.levels(state))
        if self._setting_links.numel():
            given = drivers.get(SETTING_KEY)
            if SETTING_KEY not in out or (given is not None and given.requires_grad):
                out[SETTING_KEY] = self._link_setting(drivers, out["water.link_setting"])
        return out

    def _evaluate(self, state, drivers, status, setting) -> dict[str, Tensor]:
        level = self.levels(state)
        # every tank tests its level controls by volume, with a margin of one second of the
        # previous solve's inflow (`_apply_controls`): an input of the step (the driver, which
        # `water_steady` sets from the state it starts from), not the state's
        # `"water.storage_flow"`, which the solve itself overwrites
        inflow = drivers.get(TANK_INFLOW_KEY) if self.storage is not None else None
        time = self._time(drivers)
        read = (level.detach(), status.detach(), setting.detach(), drivers.get(STATUS_KEY),
                drivers.get(SETTING_KEY), drivers.get(ONE_WAY_KEY),
                None if inflow is None else inflow.detach(), time,
                drivers.get(RESERVOIR_HEAD_KEY))
        memo = getattr(self, "_memo", None)
        if memo is not None and all(
                (a is None and b is None) or (a is not None and b is not None
                                              and a.shape == b.shape and torch.equal(a, b))
                for a, b in zip(read, memo[0], strict=True)):
            return self._completed(memo[1], state, drivers)
        with torch.no_grad():
            if not self._held:
                status, setting = self._apply_controls(level, status, setting, inflow, time)
        self._refuse_outside(level)
        out = {"water.phi_boundary": self._boundary(drivers, level),
               STATUS_KEY: self._link_status(drivers, status),
               "water.link_status": status, "water.link_setting": setting}
        if self._setting_links.numel():
            out[SETTING_KEY] = self._link_setting(drivers, setting)
        limited = self._full_and_empty(level, drivers, out[STATUS_KEY])
        if limited is not None:
            out[STATUS_KEY], out[ONE_WAY_KEY] = limited
        self._memo = (tuple(None if t is None else t.clone() for t in read),
                      self._graph_free(out))
        return out

    def _time(self, drivers) -> Tensor | None:
        """The driver `"water.time"` when a time control needs it, else `None`."""
        if not self._has_clock:
            return None
        time = drivers.get(TIME_KEY)
        if time is None:
            raise KeyError(
                f"{type(self).__name__}: driver {TIME_KEY!r} (s since the simulation start) "
                f"is required by the network's AT TIME / AT CLOCKTIME controls; take the "
                f"drivers from initial_drivers(model) and set it each step")
        return torch.as_tensor(time).detach().to(F64)

    def _link_status(self, drivers, link_status: Tensor) -> Tensor:
        """Every link's status: the drivers' `"water.status"` (the build's when they carry
        none), with the pumps' and the controlled links' entries from the carried
        `link_status`."""
        base = drivers.get(STATUS_KEY, self.link_status).to(link_status.dtype)
        base, link_status = torch.broadcast_tensors(base, link_status)
        out = base.clone()
        out[..., self._status_links] = link_status[..., self._status_links]
        return out

    def _link_setting(self, drivers, link_setting: Tensor) -> Tensor:
        """Every link's setting: the drivers' `"water.setting"` (the build's when they carry
        none), with the entries of the links a control may set taken from the carried
        `link_setting` where it differs from the built one.

        TRADE-OFF: a control that sets the built value (an OPEN on a pump built at speed 1)
        is then no change, so the drivers' own value stays: a calibration driver at 0.9
        keeps 0.9 after such a control, where EPANET gives 1.0. A pump CLOSED by a control or
        a rule is at speed 0, as in EPANET; a pump closed by `[STATUS]` or by the solve keeps
        its speed, so code that reads the setting alone (the energy) must read
        `"water.status"` too."""
        base = drivers.get(SETTING_KEY, self.link_setting).to(link_setting.dtype)
        base, link_setting = torch.broadcast_tensors(base, link_setting)
        idx = self._setting_links
        moved = link_setting[..., idx] != self.link_setting[idx]
        out = base.clone()
        out[..., idx] = torch.where(moved, link_setting[..., idx], base[..., idx])
        return out

    def effective_setting(self, drivers, link_setting: Tensor) -> Tensor:
        """Every link's setting as the solve takes it: `_link_setting` (the carried value
        where a control, a rule or a speed pattern has moved it) when any link's setting can
        move, else the drivers' `"water.setting"` (the build's when they carry none)."""
        if self._setting_links.numel():
            return self._link_setting(drivers, link_setting)
        return drivers.get(SETTING_KEY, self.link_setting)

    @contextmanager
    def held_controls(self):
        """Within it the closure applies no control, carrying the status and setting it is
        given: EPANET's `pswitch` re-solves the same step with the status it has switched,
        without testing the step's other controls again."""
        self._held, self._memo, self._fast = True, None, None
        try:
            yield self
        finally:
            self._held, self._memo, self._fast = False, None, None

    @property
    def has_pressure_controls(self) -> bool:
        """True when a junction-pressure control is among the controls."""
        return bool(self.pressure_controls)

    def pressure_switch(self, state) -> tuple[Tensor, Tensor, Tensor]:
        """`(link_status, link_setting, changed)` after the junction-pressure controls
        (`apply_controls`, EPANET's `pswitch`) are tested on the converged `state`: the
        pressure head of a junction is its head less its elevation, within `Htol` of the
        setting. Where `changed`, the step is solved again with the new status."""
        pressures = {name: state["water.phi"][..., node].detach() / self.head_scale - elev
                     for name, (node, elev) in self.junctions.items()}
        with torch.no_grad():
            return apply_controls(
                self.pressure_controls, pressures=pressures,
                status=state["water.link_status"].detach(),
                setting=state["water.link_setting"].detach(),
                link_names=self.link_names, link_kinds=self.link_kinds,
                setting_scale=self.setting_scale)

    def _reservoir_heads(self, drivers) -> Tensor:
        """The reservoir heads: the driver `RESERVOIR_HEAD_KEY` when given, else as built."""
        return drivers.get(RESERVOIR_HEAD_KEY, self.reservoir_heads)

    def _boundary(self, drivers, level: Tensor) -> Tensor:
        """`"water.phi_boundary"`: the reservoir heads. A tank held at a boundary node has no
        column there (the layer holds it at its head in the step-start state, which `Model`
        passes to the solve as `history`), and a tank stepped inside the solve is an interior
        node. A fresh tensor, not a view of the driver: the remembered answer must keep no
        input alive (`__call__`)."""
        heads = self._reservoir_heads(drivers)
        batch = torch.broadcast_shapes(level.shape[:-1], heads.shape[:-1])
        return heads.expand(batch + heads.shape[-1:]).clone()

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
        `water_run` divides the inflow by). Given `inflow` (m3/s per tank), a
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

    def _levels(self, level: Tensor):
        """The tank levels by name (m)."""
        return {t.name: level[..., i] for i, t in enumerate(self.tanks)}

    def _apply_controls(self, level: Tensor, status: Tensor, setting: Tensor,
                        inflow: Tensor | None = None, time: Tensor | None = None):
        """`(status, setting)` with every time and tank-level control whose
        condition holds applied, in file order (a later control on the same link wins)
        (`apply_controls`). Every tank is tested by volume, as EPANET 2.2's `controls`
        (`hydraul.c`) tests it: ``V(level) >= V(value) - |q|`` (ABOVE),
        ``V(level) <= V(value) + |q|`` (BELOW), ``q`` the previous solve's inflow over one
        second (`inflow`, m3/s: the driver `"water.tank_inflow"`, which `water_steady`
        takes from the state it starts from, `with_tank_inflow`; zero at `initial_state`,
        as EPANET's `NodeDemand` of a tank at t = 0). A control so fires
        at equality and up to one second's inflow early: the margin that lets a step rounded
        to whole seconds (`controltimestep`) end just short of a crossing and still fire.
        A cylinder's volume is its area times its level, so for it the test is
        ``level >= value - |q| / A``. On a curve whose volume is flat between two levels the
        level and volume tests differ: at the flat stretch's volume every control level on
        it is reached (a tank held at `tankgrade(Vmax)` below its maximum level fires an
        ABOVE control set higher)."""
        levels = self._levels(level)
        tested = None
        if self.storage is not None and self.storage.linear:
            # a cylinder: V = A level, so the volume test is `level >= value - |q| / A`
            area = self.storage.capacity(level, level) * self.head_scale
            slack = (torch.zeros_like(level) if inflow is None
                     else inflow.detach().abs().expand(level.shape) / area)

            def tested(control):
                i = self.tank_of.get(control.node)
                if i is None:
                    return None
                if control.above:
                    return level[..., i] >= control.value - slack[..., i]
                return level[..., i] <= control.value + slack[..., i]
        elif self.storage is not None:
            phi = (self.bottom + level) * self.head_scale
            volume = self.storage.amount(phi)
            margin = (torch.zeros_like(volume) if inflow is None
                      else inflow.detach().abs().expand(volume.shape))

            def tested(control):
                i = self.tank_of.get(control.node)
                if i is None:
                    return None
                at = level.new_full(level.shape, control.value)
                v2 = self.storage.amount((self.bottom + at) * self.head_scale)[..., i]
                v1, vplus = volume[..., i], margin[..., i]
                return v1 >= v2 - vplus if control.above else v1 <= v2 + vplus
        out, new, _ = apply_controls(
            self.controls, time=time, clocktime=self.start_clocktime, levels=levels,
            status=status, setting=setting, link_names=self.link_names,
            link_kinds=self.link_kinds, setting_scale=self.setting_scale, tested=tested)
        return out, new

    def control_time(self, level: Tensor, rate: Tensor, drivers, state,
                     inflow: Tensor | None = None) -> float:
        """Seconds to the next control that fires (`next_control_time`, EPANET's
        `controltimestep`), `inf` if none: the next time or clock-time control and every
        tank level, changing linearly at `rate` (dy/dt, m/s, per tank), that reaches a
        level control's trigger, of the controls that change the status or setting `state`
        carries. On a volume curve the crossing is by volume (`_volume_time`), at the
        tanks' inflow `inflow` (m3/s) when given."""
        level, rate = level.detach(), rate.detach()
        if inflow is not None:
            # `controltimestep` skips a tank whose inflow is at most QZERO
            rate = torch.where(inflow.detach().abs() <= QZERO, torch.zeros_like(rate), rate)
        time = self._time(drivers)
        crossing = None
        if self._curved:
            def crossing(control):
                i = self.tank_of.get(control.node)
                if i is None:
                    return None
                target = level.clone()
                target[..., i] = control.value
                return self._volume_time(target, level, rate, inflow)[..., i]
        levels = self._levels(level)
        rates = {t.name: rate[..., i] for i, t in enumerate(self.tanks)}
        ahead = next_control_time(
            self.controls, time=torch.zeros((), dtype=F64) if time is None else time,
            clocktime=self.start_clocktime, levels=levels, rates=rates, crossing=crossing,
            status=state["water.link_status"].detach(),
            setting=state["water.link_setting"].detach(),
            link_names=self.link_names, link_kinds=self.link_kinds,
            setting_scale=self.setting_scale)
        return float(ahead.detach().min())


class TankStorageControls(TankLevels):
    """`TankLevels` for tanks stepped inside the solve, as interior storage nodes
    (`build_model(tank_scheme="backward_euler")` or `"trapezoidal"`): it refuses a level
    outside `[min, max]` by name instead of applying the full and empty rules.
    """

    def _refuse_outside(self, level: Tensor) -> None:
        """Refuse, by name, a tank outside its `[min, max]` levels: stepped inside the solve,
        a tank has no full or empty rule."""
        self._refuse_level_outside(level)

    def _full_and_empty(self, level: Tensor, drivers, link_status: Tensor):
        """`None`: the full and empty rules act at the step start on a held tank; here the
        tank moves inside the solve, and a level beyond a limit is refused instead."""
        return None
