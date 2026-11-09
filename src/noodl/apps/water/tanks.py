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
into `"water.phi_boundary"` and the per-pump status into `"water.status"`. A tank held at a
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

A level outside `[y_min, y_max]` is REFUSED by name: EPANET closes links instead, and
silently doing the same would change the network the user asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor
F64 = torch.float64


@dataclass(frozen=True)
class Control:
    """One `LINK <id> OPEN|CLOSED IF NODE <tank> BELOW|ABOVE <level>` line."""

    link: str
    status: str
    node: str
    test: str
    level: float


class TankLevels:
    """Simple controls on tank levels, and the boundary heads of a water layer whose tanks
    are held at boundary nodes (`build_model(tank_scheme="forward_euler")`).

    A tank's level is its head in `"water.phi"` at `tank_nodes` less its bottom (`levels`).
    Per call it applies the controls to those levels, refuses a level outside `[min, max]`
    by name, and writes the reservoir heads into `"water.phi_boundary"`, then zeros in the
    tanks' columns, which the layer fills with their step-start heads
    (`KirchhoffLayer.hold_storage`). Only the link status is carried. `event_step` is
    EPANET's shortening of the step to the next control crossing.
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
            if control.test not in ("BELOW", "ABOVE"):
                raise ValueError(
                    f"TankLevels: control on link {control.link!r} has test "
                    f"{control.test!r}; only BELOW and ABOVE are read"
                )
        self.tank_of = {t.name: i for i, t in enumerate(self.tanks)}
        self.pump_of = {name: i for i, name in enumerate(self.pump_names)}
        self.bottom = torch.tensor([t.elevation for t in self.tanks], dtype=F64)
        self.y_min = torch.tensor([t.min_level for t in self.tanks], dtype=F64)
        self.y_max = torch.tensor([t.max_level for t in self.tanks], dtype=F64)

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
        with torch.no_grad():
            status = self._apply_controls(level, status)
        self._refuse_outside(level)
        return {"water.phi_boundary": self._boundary(state, level), "water.status": status,
                "water.link_status": status}

    def _boundary(self, state, level: Tensor) -> Tensor:
        """The reservoir heads, then zeros in the tanks' columns (boundary nodes here, whose
        heads the layer holds: `KirchhoffLayer.hold_storage`)."""
        batch = level.shape[:-1]
        return torch.cat(
            [self.reservoir_heads.expand(batch + self.reservoir_heads.shape),
             torch.zeros(batch + level.shape[-1:], dtype=level.dtype, device=level.device)],
            dim=-1,
        )

    def _refuse_outside(self, level: Tensor) -> None:
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

    def event_step(self, level: Tensor, rate: Tensor, dt: float) -> float:
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
        """
        step = dt
        for control in self.controls:
            r = float(rate[self.tank_of[control.node]])
            if r == 0.0:
                continue
            crossing = (control.level - float(level[self.tank_of[control.node]])) / r
            if 1e-9 < crossing < step:
                step = crossing + 1e-9
        return step

    def _apply_controls(self, level: Tensor, status: Tensor) -> Tensor:
        out = status.clone()
        for control in self.controls:
            i = self.tank_of[control.node]
            j = self.pump_of[control.link]
            value = level[..., i]
            fires = value < control.level if control.test == "BELOW" else value > control.level
            new = 1.0 if control.status == "OPEN" else 0.0
            out[..., j] = torch.where(fires, torch.full_like(out[..., j], new), out[..., j])
        return out


class TankStorageControls(TankLevels):
    """`TankLevels` for tanks stepped inside the solve, as interior storage nodes
    (`build_model(tank_scheme="backward_euler")` or `"trapezoidal"`): it writes only the
    reservoir heads into `"water.phi_boundary"`, and its `event_step` never shortens a step
    below `min_step`.
    """

    def _boundary(self, state, level: Tensor) -> Tensor:
        """The reservoir heads (the tanks are interior nodes here)."""
        return self.reservoir_heads.expand(level.shape[:-1] + self.reservoir_heads.shape)

    def event_step(self, level: Tensor, rate: Tensor, dt: float,
                   min_step: float = 1.0) -> float:
        """`TankLevels.event_step`, never below `min_step` (s). The implicit step lands a
        little short of the projected crossing (its rate is the step's end rate, not the
        start's), and the next projection is then a fraction of a second away: a step that
        short makes the storage equation's `A / dt` large enough to stall Newton, while
        `min_step` carries the level past the trigger, so the control acts at the next
        step."""
        return max(super().event_step(level, rate, dt), min(min_step, dt))
