"""Tank levels and simple `[CONTROLS]`, as one closure carrying its own state.

`state_keys = ("water.tank_level", "water.link_status")` (spec 4.6a): the level of every
tank (m above its bottom) and the open/closed status of every controlled link. Both persist
across steps and belong to no layer.

Per call the closure (1) applies every control's level test with EPANET's semantics, under
`no_grad` -- the switch itself is a combinatorial decision; and (2) writes the prescribed head
at every tank into `"water.phi_boundary"` and the per-pump status into `"water.status"`. The
closure itself only PASSES the level through (spec 4.6a: a closure returns driver updates and
whatever state it carries across steps, unchanged, unless something else advances it);
ADVANCING the level to the next time step is `advance()`, called explicitly by the
extended-period driver (Task 12) with the net inflow from the solve this closure's own boundary
values just fed into -- it is not called from `__call__`, which only ever reads the CURRENT
level.

`event_step` is EPANET's ADAPTIVE shortening, and it is not optional: measured on Net1 over
24 h, a fixed 1 h step diverges from EPANET by 2.07 m because the pump switches a whole
hour late, while shortening the step to the linearly-projected trigger crossing costs 2
extra sub-steps and brings the worst reported-step difference to 8.181e-5 m (spec
amendment A12, row D3). It implements only the CONTROL-CROSSING half of EPANET's rule
(Manual section 13.1 item 17, p.113: the next step is the minimum of the nominal step and
the time to the next control crossing) -- not the demand-PERIOD boundary half, which this
milestone's driver does not need: Net1's pattern step (2 h) is an exact multiple of its
hydraulic step (1 h), so no demand-period boundary ever falls strictly inside a step and the
omission is masked on every fixture this plan measures against.

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
    """Tank levels, simple controls and the boundary heads they prescribe."""

    state_keys = ("water.tank_level", "water.link_status")

    def __init__(
        self,
        net,
        tanks,
        controls,
        pump_names,
        *,
        dt: float,
        reservoir_heads: Tensor,
        tank_first_index: int,
        head_scale: float = 1.0,
    ) -> None:
        self.net = net
        self.tanks = list(tanks)
        self.controls = list(controls)
        self.pump_names = list(pump_names)
        self.dt = float(dt)
        self.reservoir_heads = torch.as_tensor(reservoir_heads, dtype=F64)
        self.tank_first_index = int(tank_first_index)
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
                    f"of this network ({self.pump_names}); pipe controls are a recorded "
                    f"follow-up"
                )
            if control.test not in ("BELOW", "ABOVE"):
                raise ValueError(
                    f"TankLevels: control on link {control.link!r} has test "
                    f"{control.test!r}; only BELOW and ABOVE are read"
                )
        self.tank_of = {t.name: i for i, t in enumerate(self.tanks)}
        self.pump_of = {name: i for i, name in enumerate(self.pump_names)}
        self.bottom = torch.tensor([t.elevation for t in self.tanks], dtype=F64)
        self.area = torch.tensor(
            [torch.pi * t.diameter**2 / 4.0 for t in self.tanks], dtype=F64
        )
        self.y_min = torch.tensor([t.min_level for t in self.tanks], dtype=F64)
        self.y_max = torch.tensor([t.max_level for t in self.tanks], dtype=F64)

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        level = state.get("water.tank_level")
        status = state.get("water.link_status")
        if level is None or status is None:
            raise KeyError(
                "TankLevels: states 'water.tank_level' and 'water.link_status' are both "
                "required; build them with initial_state(model)"
            )
        with torch.no_grad():
            status = self._apply_controls(level, status)
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
        heads = (self.bottom + level) * self.head_scale
        boundary = torch.cat(
            [self.reservoir_heads.expand(heads.shape[:-1] + self.reservoir_heads.shape),
             heads],
            dim=-1,
        )
        return {
            "water.phi_boundary": boundary,
            "water.status": status,
            "water.tank_level": level,
            "water.link_status": status,
        }

    def advance(self, level: Tensor, inflow: Tensor, dt: float) -> Tensor:
        """Explicit Euler on the level, from the net inflow (m3/s) at each tank node."""
        return level + inflow * dt / self.area

    def event_step(self, level: Tensor, rate: Tensor, dt: float) -> float:
        """EPANET's shortened hydraulic step: the time to the next control crossing.

        Manual section 13.1 item 17, p.113: the next step is the minimum of the nominal
        step and the time until a tank level "reaches a point that triggers a change in
        status for some link", computed on the assumption that the level changes LINEARLY
        at the current solution's rate.

        `level` and `rate` (dy/dt, m/s) are given for EVERY tank, not just the one the
        caller happens to be tracking: each control tests its OWN tank (`control.node`)
        against that tank's own level and rate (N13). A single scalar pair, tested against
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
