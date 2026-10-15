"""Feedback control in a Modelica model: the signal blocks a sensor's output reaches.

A signal block is closed-loop when one of its inputs is wired from a sensor (a
`TemperatureSensor`'s `T`, `ComponentGraph.signal_wires`) or driven by a closed-loop block;
every `LimPID` is closed-loop. The closed-loop blocks become a continuous `ControlNetwork`
(`feedback_network`), built inputs first:
- `LimPID` becomes `LimitedPID`, transcribed from MSL 4.1.0 `Blocks/Continuous.mo` as a P
  controller, `y = clamp(k (wp u_s - u_m), yMin, yMax)`; the integral and derivative parts
  (`controllerType` PI, PD, PID), feed-forward and the steady-state and initial-output
  initialisations are refused by name;
- `Gain` becomes `Affine`; `Add`, `Add3`, `Sum`, `MultiSum` and `Feedback` become `Sum`;
  `Product` becomes `Product` and `Division` becomes `Divide`, with `signals.combine`'s
  equations;
- a sensor becomes a `Sensor` on its zone's temperature;
- every open-loop signal that feeds them becomes a `DriverValue` on `"controls.input"`,
  which `assemble` evaluates on the driver grid as point values, like a boundary value.

A closed-loop signal may drive only a `PrescribedHeatFlow`'s `Q_flow` into an unpinned zone,
and only from that zone's own temperature. On each piece of its limiters (every `LimitedPID`
below, inside or above its limits) such a heat flow is affine in the zone temperature,
`Q = A + B T`, and `FeedbackHeat` hands it to the zone's energy balance in that form: the
source `A` (W) and the removal rate `-B / C` (1/s, `C` the zone's heat capacity), so that the
transport step treats it implicitly, exactly within the exact scheme's step (a P controller
with `k gain = 15000 W/K` on a 75 kJ/K room relaxes in 5 s, shorter than the output
interval). A product of two closed-loop signals, or a division by one, is not affine and is
refused by name. The limiter corners are switches (`ControlNetwork.switching`), which
`run.simulate(scheme="midpoint")` locates on step boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.controls import (
    Affine,
    Block,
    Constant,
    ControlNetwork,
    Divide,
    DriverValue,
    FunctionReading,
    LimitedPID,
    Product,
    Sensor,
    Sum,
)
from noodl.apps.building_physics.modelica import expand, schema, signals
from noodl.apps.building_physics.modelica.graph import ComponentGraph
from noodl.apps.building_physics.modelica.schema import ModelicaDoc, Signal

Tensor = torch.Tensor
F64 = torch.float64
INPUT_KEY = "controls.input"


@dataclass(frozen=True)
class Feedback:
    """The network; the inputs fed open-loop (`"<block>.<input>"`, in `"controls.input"`
    order); each closed-loop signal's block position; every heat flow it drives as
    `(block, zone)`; the heat-flow inputs it drives (`"<component>.Q_flow"`); each zone's
    sensor blocks."""

    network: ControlNetwork
    inputs: tuple[str, ...]
    blocks: dict[str, int]
    heat: tuple[tuple[int, str], ...]
    heat_targets: frozenset[str]
    sensors: dict[str, tuple[int, ...]]


def _short(value, default: str) -> str:
    return str(value if value is not None else default).rsplit(".", 1)[-1]


def closed_loop(doc: ModelicaDoc, graph: ComponentGraph) -> set[str]:
    """The names of the closed-loop signal blocks (module docstring)."""
    by_target = {t: s for s in doc.signals for t in s.drives}
    sensed = {target for _, target in graph.signal_wires}
    closed: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in doc.signals:
            if s.name in closed or (s.cls not in schema.CONTROLLERS
                                    and s.cls not in schema.MATH):
                continue
            ins = [f"{s.name}.{p}" for p in signals.block_inputs(s)]
            if s.cls in schema.CONTROLLERS or any(
                    t in sensed or (t in by_target and by_target[t].name in closed)
                    for t in ins):
                closed.add(s.name)
                changed = True
    return closed


def _controller(sig: Signal, u: list[int], errors: list[str]) -> Block:
    where, p = f"{sig.name} ({sig.cls})", sig.parameters
    kind = _short(p.get("controllerType"), "PID")         # Continuous.mo: default PID
    bad = []
    if kind != "P":
        bad.append(f"controllerType {kind} is not supported (P is)")
    if p.get("withFeedForward", False):
        bad.append("feed-forward is not supported")
    init = _short(p.get("initType"), "InitialState")      # Continuous.mo: InitialState
    if init not in ("InitialState", "NoInit"):
        bad.append(f"initType {init} is not supported (InitialState and NoInit are)")
    for key in ("k", "yMax"):
        if key not in p:
            bad.append(f"parameter {key!r} is required and was not exported")
    if bad:
        errors.extend(f"{where}: {b}" for b in bad)
        return Constant(0.0, name=where)
    y_max = float(p["yMax"])
    return LimitedPID(u[0], u[1], gain=float(p["k"]), y_max=y_max,
                      y_min=float(p.get("yMin", -y_max)),
                      setpoint_weight=float(p.get("wp", 1.0)), name=where)


def _math(sig: Signal, u: list[int]) -> Block:
    where, p = f"{sig.name} ({sig.cls})", sig.parameters
    short = sig.cls.rsplit(".", 1)[-1]
    if short == "Gain":
        return Affine(u[0], gain=float(p["k"]), name=where)
    if short == "Feedback":
        return Sum(u[0], u[1], weights=(1.0, -1.0), name=where)
    if short == "Product":
        return Product(*u, name=where)
    if short == "Division":
        return Divide(u[0], u[1], name=where)
    if short in ("Add", "Add3"):
        weights = [float(p.get(f"k{i}", 1.0)) for i in range(1, len(u) + 1)]
    else:                                                 # Sum, MultiSum: k vector
        k = p.get("k")
        k = k if k is None or isinstance(k, list) else [k]
        weights = [1.0] * len(u) if k is None else [float(v) for v in k]
    return Sum(*u, weights=weights, name=where)


def feedback_network(doc: ModelicaDoc, graph: ComponentGraph, *,
                     zone_index: Mapping[str, int],
                     zone_temperature: Callable, pinned: set[str],
                     errors: list[str]) -> Feedback | None:
    """The continuous network of the closed-loop blocks, or None when there are none.
    `zone_index` maps a zone to its full-node position, `zone_temperature(state, drivers)`
    gives every node's temperature (K), `pinned` are the zones a fixed temperature holds;
    refusals are appended to `errors`."""
    closed = closed_loop(doc, graph)
    if not closed:
        return None
    by_target = {t: s for s in doc.signals for t in s.drives}
    wire_of = {target: src for src, target in graph.signal_wires}
    sensor_zone = {c.name: zone for c, zone in graph.heat_sensors}
    path = expand.modelica_path
    blocks: list[Block] = []
    reads: list[frozenset[str]] = []      # per block: the zones whose temperature it reads
    position: dict = {}
    inputs: list[str] = []
    visiting: set[str] = set()

    def add(block: Block, zones: frozenset[str] = frozenset()) -> int:
        blocks.append(block)
        reads.append(zones)
        return len(blocks) - 1

    def source(target: str) -> int:
        if target in wire_of:
            sensor = wire_of[target].split(".", 1)[0]
            if ("sensor", sensor) not in position:
                zone = sensor_zone[sensor]
                node = zone_index[zone]
                position[("sensor", sensor)] = add(Sensor(FunctionReading(
                    lambda s, d, n=node: zone_temperature(s, d)[..., n]),
                    name=f"{sensor} (temperature sensor)"), frozenset({zone}))
            return position[("sensor", sensor)]
        sig = by_target.get(target)
        if sig is None:
            errors.append(f"{target}: input of a feedback block that no supported signal "
                          f"drives")
            if ("none", target) not in position:
                position[("none", target)] = add(Constant(0.0, name=target))
            return position[("none", target)]
        if sig.name in closed:
            return block(sig)
        if ("input", target) not in position:
            position[("input", target)] = add(
                DriverValue(INPUT_KEY, len(inputs), name=f"{sig.name} ({sig.cls})"))
            inputs.append(target)
        return position[("input", target)]

    def block(sig: Signal) -> int:
        where = f"{sig.name} ({sig.cls})"
        if sig.name in position:
            return position[sig.name]
        if sig.name in visiting:
            errors.append(f"{where}: an algebraic loop of feedback blocks is not supported")
            return position.setdefault(("loop", sig.name), add(Constant(0.0, name=where)))
        visiting.add(sig.name)
        u = [source(f"{sig.name}.{p}") for p in signals.block_inputs(sig)]
        visiting.discard(sig.name)
        fed = [reads[i] for i in u]
        short = sig.cls.rsplit(".", 1)[-1]
        if short == "Product" and sum(bool(z) for z in fed) > 1:
            errors.append(f"{where}: multiplies two feedback signals; a heat flow must be "
                          f"affine in the zone temperature")
        if short == "Division" and fed and fed[-1]:
            errors.append(f"{where}: divides by a feedback signal; a heat flow must be "
                          f"affine in the zone temperature")
        made = (_controller(sig, u, errors) if sig.cls in schema.CONTROLLERS
                else _math(sig, u))
        position[sig.name] = add(made, frozenset().union(*fed))
        return position[sig.name]

    heat_zone = {f"{c.name}.Q_flow": zone for c, zone in graph.heat_sources}
    heat: list[tuple[int, str]] = []
    used: set[str] = set()
    for sig in doc.signals:                               # document order: deterministic
        if sig.name not in closed:
            continue
        b = block(sig)
        where = f"{sig.name} ({sig.cls})"
        for target in sig.drives:
            if target.split(".", 1)[0] in closed:
                continue
            zone = heat_zone.get(target)
            if zone is None:
                errors.append(f"{where}: drives {target}; a feedback signal may drive only a "
                              f"PrescribedHeatFlow's Q_flow")
            elif zone in pinned:
                errors.append(f"{where}: drives {target}, a heat flow into zone {path(zone)}, "
                              f"which a fixed temperature holds")
            elif reads[b] - {zone}:
                other = ", ".join(sorted(path(z) for z in reads[b] - {zone}))
                errors.append(f"{where}: drives {target}, a heat flow into zone {path(zone)}, "
                              f"from the temperature of zone {other}; a feedback heat flow "
                              f"may read only its own zone's temperature")
            else:
                heat.append((b, zone))
                used.add(target)
    sensors: dict[str, list[int]] = {}
    for key, i in position.items():
        if isinstance(key, tuple) and key[0] == "sensor":
            sensors.setdefault(sensor_zone[key[1]], []).append(i)
    network = ControlNetwork(blocks, (), sampled=False)
    return Feedback(network, tuple(inputs),
                    {s.name: position[s.name] for s in doc.signals if s.name in closed
                     and s.name in position},
                    tuple(heat), frozenset(used),
                    {z: tuple(sorted(v)) for z, v in sensors.items()})


class FeedbackHeat:
    """The closure that puts a feedback network's heat flows into the zone energy balance
    (module docstring): on the limiters' pieces at the state, it adds each heat flow's
    `A` to `"thermal.sources"` (full node, W) and adds `-B / C` to `"thermal.removal"`
    (thermal interior order, 1/s), with `C` the zone heat capacity (`"thermal.capacity"`
    from the drivers or the state, else the thermal layer's, `layer`). `heat` lists
    `(block, node)` per heat flow (none: the closure writes nothing, and `run.simulate` still
    records the network's outputs); `run._Midpoint` evaluates `affine` itself, on the piece of
    the whole step (`pieces_between`)."""

    def __init__(self, fb: Feedback, *, zone_index: Mapping[str, int],
                 zone_temperature: Callable, layer) -> None:
        self.network = fb.network
        self.heat = tuple((b, int(zone_index[z])) for b, z in fb.heat)
        self.zone_temperature = zone_temperature
        self.layer = layer
        self._sensors = {int(zone_index[z]): frozenset(v) for z, v in fb.sensors.items()}
        self._pids = [b for b in self.network.blocks if isinstance(b, LimitedPID)]

    def _walk(self, state, drivers, pieces: Tensor | None = None,
              bump: frozenset[int] = frozenset()) -> tuple[Tensor, Tensor]:
        """Every block's output (`(..., n_blocks)`) and every limiter's input `v`
        (`(..., n_limiters)`); on `pieces` (+1 at the upper limit, -1 at the lower, 0
        between) when given, else on the pieces `v` lies on. The sensor blocks in `bump`
        read 1 K more."""
        v: list[Tensor] = []

        def output(i, block, u, memory, y):
            if isinstance(block, LimitedPID):
                vi = block.unlimited(u, memory)
                if pieces is not None:
                    p = pieces[..., len(v)]
                    y = torch.where(p > 0, torch.as_tensor(block.y_max, dtype=F64),
                                    torch.where(p < 0, torch.as_tensor(block.y_min, dtype=F64),
                                                vi))
                v.append(vi)
            elif i in bump:
                y = y + 1.0
            return y

        out = self.network.evaluate(state, drivers, output=output)
        return out, (torch.stack(torch.broadcast_tensors(*v), dim=-1) if v
                     else torch.zeros(0, dtype=F64))

    def _limits(self, side: str) -> Tensor:
        """Every limiter's `y_max` or `y_min` on the last axis (floats or tensors, as the
        blocks hold them now)."""
        values = [torch.as_tensor(getattr(b, side), dtype=F64) for b in self._pids]
        if not values:
            return torch.zeros(0, dtype=F64)
        return torch.stack(torch.broadcast_tensors(*values), dim=-1)

    def _piece(self, v: Tensor) -> Tensor:
        one = torch.ones_like(v)
        return torch.where(v > self._limits("y_max"), one,
                           torch.where(v < self._limits("y_min"), -one, 0 * one))

    def pieces(self, state, drivers) -> Tensor:
        """Each limiter's piece at `state` (+1 at its upper limit, -1 at its lower, 0
        between)."""
        return self._piece(self._walk(state, drivers)[1])

    def pieces_between(self, s0, d0, s1, d1) -> Tensor:
        """The limiters' pieces of a step from `s0` to `s1`: those of the mean of their
        inputs at the two ends. A step lies on one side of every corner (`run._Switches`), so
        an end within the location tolerance of a corner takes the other end's piece."""
        return self._piece(0.5 * (self._walk(s0, d0)[1] + self._walk(s1, d1)[1]))

    def affine(self, state, drivers, pieces: Tensor) -> tuple[Tensor, Tensor]:
        """Full-node `(A, B)`: the heat flows into each zone on `pieces`, `A + B T` W with `T`
        the zone temperature, at `state`. `B` is the change of the heat flows when the zone's
        sensors read 1 K more, exact on a piece (the flows are affine there)."""
        T = torch.as_tensor(self.zone_temperature(state, drivers), dtype=F64)
        out, _ = self._walk(state, drivers, pieces)
        bumped: dict[int, Tensor] = {}
        parts: list[tuple[int, Tensor, Tensor]] = []
        for b, node in self.heat:
            if node not in bumped:
                bumped[node] = self._walk(state, drivers, pieces,
                                          self._sensors.get(node, frozenset()))[0]
            q = out[..., b]
            slope = bumped[node][..., b] - q
            parts.append((node, q - slope * T[..., node], slope))
        batch = broadcast_shapes(T.shape[:-1], *(a.shape for _, a, _ in parts))
        A = torch.zeros(batch + T.shape[-1:], dtype=F64)
        B = torch.zeros_like(A)
        for node, a, slope in parts:
            A[..., node] += a
            B[..., node] += slope
        return A, B

    def heat_flows(self, state, drivers) -> Tensor:
        """Full-node heat flows (W) at `state`."""
        A, B = self.affine(state, drivers, self.pieces(state, drivers))
        return A + B * torch.as_tensor(self.zone_temperature(state, drivers), dtype=F64)

    def removal(self, B: Tensor, capacity: Tensor | None = None) -> Tensor:
        """The removal rate (1/s, thermal interior order) of the full-node slopes `B` at the
        zones' heat `capacity` (J/K, interior order; default the layer's)."""
        cap = self.layer.capacity if capacity is None else capacity
        return -B[..., self.layer.interior_idx] / cap

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        if not self.heat:
            return {}  # a network that drives no heat flow (its outputs are still recorded)
        A, B = self.affine(state, drivers, self.pieces(state, drivers))
        cap = drivers.get("thermal.capacity", state.get("thermal.capacity"))
        base = drivers.get("thermal.sources")
        removal = self.removal(B, cap)
        base_removal = drivers.get("thermal.removal")
        return {"thermal.sources": A if base is None else base + A,
                "thermal.removal": removal if base_removal is None else base_removal + removal}
