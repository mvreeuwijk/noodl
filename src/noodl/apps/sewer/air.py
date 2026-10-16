"""Headspace air: the `Headspace` branch element, the `Drag` drive and the air-density
closure that feeds the existing `Stack` buoyancy drive.

Per `headspace` edge the branch law is

    dp = R(h) |Q| Q - D(h, V_w) - B(T_head, T_amb)

with the wall-friction resistance ``R(h) = f_air L rho / (2 D_h A_air^2)`` (Darcy-Weisbach
on the air segment's hydraulic diameter), the interfacial drag ``D`` from the moving water
surface, and the stack term ``B`` from the existing `drives.Stack`.

``A_air`` and ``D_h`` arrive as per-edge DRIVERS written by `SewerHydraulics` in the same
pass, exactly as `UpstreamDensityPowerLaw` reads its density driver: an element may read
`drivers` inside `flow`/`dflow`/`linear_init`, a Drive may read `drivers` and nothing else.

Coefficient status: ``f_air`` default 0.02 is UNVERIFIED (the reported
range for sewer crowns is 0.015-0.045, Edwini-Bonsu and Steffler 2006, paywalled);
``F_I_DEFAULT`` is CALIBRATED here against Pescod and Price's Test 8 as tabulated by
Edwini-Bonsu and Steffler 2004, Table 1 p.337.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from noodl.drives import Stack
from noodl.elements.base import ConstitutiveLaw
from noodl.elements.fixed import FixedFlow
from noodl.elements.powerlaw import PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64

#: Air density at 20 C and 101325 Pa (kg/m^3), the reference the elements are written at.
RHO_AIR_REF = 1.2041

#: Molar mass of dry air (kg/mol) and the universal gas constant (J/mol/K).
M_AIR = 0.0289647
R_GAS = 8.314462618

#: Standard atmospheric pressure (Pa).
P_ATM = 101325.0

#: Wall friction factor for the headspace (Darcy). UNVERIFIED; inside the 0.015-0.045 range
#: reported for sewer crowns.
F_AIR_DEFAULT = 0.02

#: Interfacial drag factor, CALIBRATED to Pescod and Price Test 8 (300 mm UPVC, 15 m, both
#: ends open, h/D = 0.6, V_w = 0.8 m/s, measured U_air = 0.20 m/s, i.e. U_air/V_w = 25 %) at
#: ``f_air = 0.02``. For a pipe open at both ends the balance ``R Q^2 = D`` has the closed
#: form ``U_air / V_w = sqrt((f_i / f_air) T D_h / A_air)``, which at that geometry gives
#: ``f_i = 0.02 * 0.25^2 / 1.668281 = 7.492741e-4``, rounded to 7.49e-4. The same value
#: reproduces Tests 7 and 9 at 24.14 % and 25.15 % against measured 35 % and 27.5 % (both
#: inside the 20-40 % acceptance band of verification check A1). MEASURED.
F_I_DEFAULT = 7.49e-4


class Headspace(ConstitutiveLaw):
    """``dp = R |Q| Q`` on a headspace edge, with ``R`` read from per-edge drivers.

    ``R(h) = f_air L rho / (2 D_h A_air^2)``; the inversion is the closed form
    ``Q = sign(dp) sqrt(|dp| / R)``, laminar-blended below ``dp_transition`` in the
    `PowerLaw` manner, with the safe input substituted on BOTH branches of the ``where``
    (the exponent 1/2 has an infinite derivative at ``dp = 0``).

    ``area_key`` and ``dh_key`` name full per-`pipe`-edge driver tensors; the element holds
    the positions of its own edges within that per-pipe order as a registered buffer, so the
    pairing between a `headspace` edge and the `pipe` edge it parallels is decided ONCE, at
    construction, and never re-derived.
    """

    def __init__(
        self,
        length,
        pipe_positions,
        *,
        f_air=F_AIR_DEFAULT,
        rho_air: float = RHO_AIR_REF,
        area_key: str = "sewer.A_air",
        dh_key: str = "sewer.D_h",
        dp_transition: float = 1e-8,
        kind: str = "headspace",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        # `ConstitutiveLaw._param` casts a plain Python float with `torch.get_default_dtype()`
        # (float32 in this repo, per `elements/duct.py`'s documented gotcha); casting
        # explicitly to `self.length`'s dtype first keeps a float64-constructed element
        # float64 throughout instead of silently losing precision on the default `f_air`.
        self.f_air = self._param(torch.as_tensor(f_air, dtype=self.length.dtype), learnable)
        self.rho_air = float(rho_air)
        self.area_key = area_key
        self.dh_key = dh_key
        self.dp_transition = float(dp_transition)
        self.register_buffer(
            "pipe_positions", torch.as_tensor(pipe_positions, dtype=torch.long)
        )

    def resistance(self, drivers: Mapping | None) -> Tensor:
        # Name only the key(s) that are actually missing, not always both -- a
        # caller who supplied `area_key` but forgot `dh_key` (or the reverse) otherwise
        # gets an error naming a driver it DID give, which reads as if the code were wrong
        # about its own requirement.
        missing = [
            key for key in (self.area_key, self.dh_key)
            if drivers is None or key not in drivers
        ]
        if missing:
            raise KeyError(
                f"Headspace (kind {self.kind!r}): driver(s) {missing} are required and "
                f"were not given; they are written by the SewerHydraulics closure in the "
                f"same pass"
            )
        a_air = drivers[self.area_key].index_select(-1, self.pipe_positions)
        d_h = drivers[self.dh_key].index_select(-1, self.pipe_positions)
        if bool(torch.any(a_air <= 0)):
            bad = (a_air <= 0).reshape(-1, a_air.shape[-1]).any(0).nonzero()
            raise ValueError(
                f"Headspace (kind {self.kind!r}): the headspace area is not positive on "
                f"edge positions {bad.flatten().tolist()}; a full pipe has no headspace "
                f"and no air path"
            )
        return self.f_air * self.length * self.rho_air / (2.0 * d_h * a_air**2)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = torch.sign(dp) * torch.sqrt(dp_safe / r)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / r) / dpt
        return torch.where(mask, slope * dp, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = 0.5 / torch.sqrt(r * dp_safe)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / r) / dpt
        return torch.where(mask, slope * torch.ones_like(dp), sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=r.dtype) / r) / dpt
        return torch.zeros_like(slope), slope


class Drag:
    """Interfacial drag as a `Drive`: the shear a moving water surface exerts on the air.

    ``D = (f_i / 2) rho U_s |U_s| T L / A_air`` with ``U_s = c_s V_w`` the surface velocity
    (Edwini-Bonsu and Steffler 2004's rigid-moving-lid framing, p.332), spread over the air
    area.

    SIGN. The layer evaluates ``dp = (phi_src - phi_tgt) + sum(drives)`` and the element's
    law is ``dp = R |Q| Q``, so the momentum balance ``phi_src - phi_tgt = R|Q|Q - D - B``
    means the
    drive returns ``+D``: with both ends at ambient the drag alone then produces
    ``R|Q|Q = D``, i.e. flow from the upstream manhole towards the downstream one, which is
    the direction the wastewater drags the air. (Returning ``-D`` reverses the airflow and
    was caught by the A1 measurement.)

    A `Drive` reads DRIVERS ONLY -- never `phi`, never another layer's state -- which is
    what keeps the Newton Jacobian exactly ``A_I diag(g') A_I^T``. The relative-velocity
    form ``(U_s - V_air)`` is therefore out of scope and not implemented.

    ``zero_positions`` names edge positions within this kind's block whose drive is zeroed:
    the outfall edge, whose "pipe" is the open end and carries no water surface of its own.
    """

    kind = "headspace"

    def __init__(
        self,
        length,
        pipe_positions,
        *,
        f_i=F_I_DEFAULT,
        c_s: float = 1.0,
        rho_air: float = RHO_AIR_REF,
        v_key: str = "sewer.v",
        width_key: str = "sewer.T",
        area_key: str = "sewer.A_air",
        zero_positions=(),
        kind: str = "headspace",
    ) -> None:
        self.kind = kind
        self.length = torch.as_tensor(length, dtype=F64)
        self.pipe_positions = torch.as_tensor(pipe_positions, dtype=torch.long)
        self.f_i = torch.as_tensor(f_i, dtype=F64)
        self.c_s = float(c_s)
        self.rho_air = float(rho_air)
        self.v_key = v_key
        self.width_key = width_key
        self.area_key = area_key
        mask = torch.ones(self.pipe_positions.shape[-1], dtype=F64)
        for p in zero_positions:
            mask[int(p)] = 0.0
        self.mask = mask

    def __call__(self, drivers: Mapping) -> Tensor:
        for key in (self.v_key, self.width_key, self.area_key):
            if key not in drivers:
                raise KeyError(
                    f"Drag (kind {self.kind!r}): driver {key!r} is required and was not "
                    f"given; it is written by the SewerHydraulics closure in the same pass"
                )
        v_w = drivers[self.v_key].index_select(-1, self.pipe_positions) * self.c_s
        width = drivers[self.width_key].index_select(-1, self.pipe_positions)
        a_air = drivers[self.area_key].index_select(-1, self.pipe_positions)
        drag = (
            0.5 * self.f_i * self.rho_air * v_w * v_w.abs() * width * self.length / a_air
        )
        return self.mask * drag


def air_density(temperature: Tensor, pressure: float = P_ATM) -> Tensor:
    """Ideal-gas air density ``rho = p M / (R T)`` (kg/m^3) at absolute temperature T (K)."""
    return pressure * M_AIR / (R_GAS * temperature)


@dataclass(frozen=True)
class AirPlan:
    """The headspace edges `add_air_edges` added: `positions` (each headspace edge's
    position in the per-pipe driver order: every pipe, then each outfall pipe again for its
    outfall-to-ambient edge), `outfall_pipe_idx` (the pipes draining to an outfall) and
    `n_pipes`."""

    positions: Tensor
    outfall_pipe_idx: Tensor
    n_pipes: int


def check_air_topology(net) -> None:
    """Refuse, by name, a network the headspace air cannot be built on: every junction must
    have exactly one outgoing conduit (the air side's assumption: its capacity and leak are
    the outgoing conduit's), no outfall an outgoing one. A loop, a junction with several
    outgoing conduits or none (an adverse conduit, or one drained only by a regulator) is
    refused, and so is one that reaches no outfall (a cycle or a dead end, which dynamic-wave
    routing itself allows)."""
    outgoing: dict[str, list[str]] = {m.name: [] for m in net.manholes}
    outfalls = {o.name for o in net.outfalls}
    for pipe in net.pipes:
        if pipe.u in outfalls:
            raise ValueError(
                f"build_model: air=True needs every outfall to be a conduit's downstream end, "
                f"but outfall {pipe.u!r} is the upstream end of conduit {pipe.name!r}")
        if pipe.u in outgoing:
            outgoing[pipe.u].append(pipe.name)
    bad = {name: pipes for name, pipes in outgoing.items() if len(pipes) != 1}
    if bad:
        raise ValueError(
            f"build_model: air=True assumes every junction has exactly one outgoing conduit "
            f"(a dendritic sewer), but junction(s) {sorted(bad)} have "
            f"{[len(v) for v in bad.values()]} (conduits {list(bad.values())}): a loop, a "
            f"branch, an adverse conduit or a junction drained only by a regulator; build "
            f"with air=False")
    drains = {pipe.u: pipe.v for pipe in net.pipes}
    for name in outgoing:                  # one outgoing conduit each: follow it downstream
        seen, node = set(), name
        while node in drains and node not in seen and node not in outfalls:
            seen.add(node)
            node = drains[node]
        if node not in outfalls:
            raise ValueError(
                f"build_model: air=True needs every junction to drain to an outfall, but "
                f"junction {name!r} reaches none (a cycle, or a dead end: dynamic-wave "
                f"routing allows these, the headspace air does not); build with air=False")


def add_air_edges(graph: Network, net, fans: tuple = ()) -> AirPlan:
    """Add the headspace, leak and fan edges of `net` to `graph` (which already holds the
    node ``"ambient"`` and the pipe edges) and return their `AirPlan`.

    Kind order: `headspace` (one per pipe, then one outfall edge PER outfall pipe), `leak`,
    `fan`. Each outfall pipe is the one that DRAINS TO an outfall (`p.v in outfall_names`),
    found by where it drains rather than assumed to be `net.pipes[-1]` (otherwise a network
    whose outfall conduit is listed first would silently attach the outfall-flagged headspace
    edge, and its geometry, to the wrong manhole). The builder wires exactly ONE
    outfall-to-ambient headspace edge per OUTFALL NODE, so two manholes draining directly into
    the same outfall are refused here by name. A forest has one outfall pipe per component --
    each gets its own appended headspace edge, borrowing its own pipe's length and diameter.
    """
    n_pipes = len(net.pipes)
    outfall_names = {o.name for o in net.outfalls}
    candidates = [i for i, p in enumerate(net.pipes) if p.v in outfall_names]
    if not candidates:
        raise ValueError(
            f"build_model: no pipe drains to an outfall; outfalls are {sorted(outfall_names)}")
    drained = [net.pipes[i].v for i in candidates]
    if len(drained) != len(set(drained)):
        dupes = sorted({v for v in drained if drained.count(v) > 1})
        raise ValueError(
            f"build_model: outfall(s) {dupes} each have more than one pipe "
            f"draining into them; a dendritic sewer (or forest) gives each outfall "
            f"exactly one")
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="headspace", name=f"{pipe.name}.air")
    for i in candidates:
        graph.add_edge(net.pipes[i].u, "ambient", kind="headspace", outfall=True)
    for manhole in net.manholes:
        graph.add_edge(manhole.name, "ambient", kind="leak", name=f"{manhole.name}.leak")
    for name in fans:
        graph.add_edge(name, "ambient", kind="fan", name=f"{name}.fan")
    return AirPlan(torch.tensor(list(range(n_pipes)) + candidates, dtype=torch.long),
                   torch.tensor(candidates, dtype=torch.long), n_pipes)


def air_layer(graph: Network, net, plan: AirPlan, *, f_air=F_AIR_DEFAULT, f_i=F_I_DEFAULT,
              c_s: float = 1.0, leak_area=8e-4, leak_cd=0.6,
              fans: tuple = ()) -> KirchhoffLayer:
    """The ``"air"`` `KirchhoffLayer` (Pa; m3/s) over the edges `add_air_edges` added:
    `Headspace` and the leak `PowerLaw` (and `FixedFlow` fans), the `Drag` and `Stack`
    drives. It reads the per-pipe drivers ``sewer.A_air`` (total over barrels, m2),
    ``sewer.D_h`` (one barrel's, m), ``sewer.T`` (total top width, m), ``sewer.v`` (m/s) and
    ``rho_air_nodes``, which the sewer's hydraulics closure writes."""
    length = torch.tensor([p.length for p in net.pipes], dtype=F64)
    diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
    n_pipes = plan.n_pipes
    n_outfall_edges = plan.outfall_pipe_idx.numel()
    air_lengths = torch.cat([length, length.index_select(-1, plan.outfall_pipe_idx)])
    positions = plan.positions
    # Leak law: `PowerLaw(C = Cd A sqrt(2/rho), n = 0.5, kind="leak")` built DIRECTLY, never
    # through `Orifice`: `Orifice` casts with `torch.get_default_dtype()` (float32 in this
    # repository) regardless of its inputs' own dtype, which would downcast the leak's `C` and
    # floor check C1's residual on the float32 arithmetic of the leak branch alone (measured
    # 1.352e-12 / 1.855e-11 in float32; 4.518e-13 / 6.141e-12 in float64). `leak_cd` and
    # `leak_area` are ordinary floats by default but may also be tensors carrying a leading
    # batch (instance) dimension (`benchmarks/sewer_diurnal.py` varies them per instance
    # without any core or element change, since `PowerLaw._param`'s pass-through rule and
    # ordinary broadcasting do the rest).
    leak_c = (
        torch.as_tensor(leak_cd, dtype=F64)
        * torch.as_tensor(leak_area, dtype=F64)
        * math.sqrt(2.0 / RHO_AIR_REF)
        * torch.ones(len(net.manholes), dtype=F64)
    )
    leak_n = torch.full((len(net.manholes),), 0.5, dtype=F64)
    laws = [
        Headspace(air_lengths, positions, f_air=f_air, kind="headspace"),
        PowerLaw(C=leak_c, n=leak_n, kind="leak", dp_transition=1e-8),
    ]
    if fans:
        laws.append(FixedFlow(torch.zeros(len(fans), dtype=F64), kind="fan"))
    crown = torch.cat([diameter, diameter.index_select(-1, plan.outfall_pipe_idx)])
    drives = [
        Drag(
            air_lengths, positions, f_i=f_i, c_s=c_s,
            zero_positions=tuple(range(n_pipes, n_pipes + n_outfall_edges)),
            kind="headspace",
        ),
        _stack_for("headspace", graph, net, crown=crown),
        _stack_for("leak", graph, net),
    ]
    return KirchhoffLayer(
        graph, "air", laws, drives=drives, boundary=["ambient"],
        quantity="pressure", unit="Pa", flow_unit="m3/s",
    )


def _ground_of(node_attrs) -> float:
    """A node's ground level: its own `ground`, else its `invert`, else `0.0` (`ambient`'s
    case, carrying neither). `is None` throughout, not an `or`-chain -- `or` treats an
    explicitly given `0.0` ground (or invert) as falsy and falls through to the next
    fallback, silently dropping a real ground-level-zero manhole to whatever its invert (or
    the hard-coded `0.0`) happens to be instead."""
    ground = node_attrs.get("ground")
    if ground is not None:
        return float(ground)
    invert = node_attrs.get("invert")
    return 0.0 if invert is None else float(invert)


def _stack_for(
    kind: str, graph: Network, net, *, crown: Tensor | None = None
) -> Stack:
    """The existing buoyancy drive on one air kind.

    On `headspace` edges `z_path` is the pipe CROWN elevation: the mean of the two end
    inverts plus the pipe's own diameter (the mean invert alone is
    the pipe's INVERT at its midpoint, not its crown; `crown` is the per-headspace-edge
    diameter, in the same order as `graph.endpoints("headspace")`, i.e. `net.pipes` order
    then the outfall pipe's diameter again for the extra outfall-to-ambient edge, exactly
    like `air_lengths` at the call site). `z_ref` is the node's ground level, so a warm
    headspace column in a manhole shaft is lighter than the outside column of the same
    height -- the building application's stack effect, unchanged.

    DATUM CONVENTION (a leak DOES carry a stack term). `ground` falls back to the node's own
    `invert` when no `ground` was given, and to `0.0` when neither was given -- which is exactly
    `ambient`'s case, since `ambient` is added as a bare node with no `invert` or `ground` attribute
    at all. On a `leak` edge `z_path` is the manhole's own ground level (`ground[src]`), so the
    MANHOLE term of the Stack law cancels (`z_ref[src] - z_path = 0`), but the AMBIENT term does
    not: `z_ref` at `ambient` is `0.0`, not the manhole's ground level, so every leak in fact
    carries a `g * rho_ambient * ground[manhole]` term, not zero. This is self-consistent rather
    than a bug: `ambient`'s `0.0` datum is used identically at every leak and at the outfall's
    open-air `headspace` edge, and `ambient` is itself a fixed boundary node
    (`air.phi_boundary = 0`), so the datum choice only ever shifts every air pressure by the
    same constant -- `air.phi` is DATUM-REFERENCED to this convention, not to a physical
    zero, and differences between manholes (what every drive and refusal actually reads)
    are unaffected. `crown` is unused (and must be `None`) for `kind='leak'`.
    """
    if kind != "headspace" and crown is not None:
        raise ValueError(
            f"_stack_for: crown is only meaningful for kind='headspace', got kind={kind!r}"
        )
    src, tgt = graph.endpoints(kind)
    invert = torch.tensor(
        [
            0.0 if (v := graph.graph.nodes[n].get("invert")) is None else v
            for n in graph.nodes
        ],
        dtype=F64,
    )
    ground = torch.tensor(
        [_ground_of(graph.graph.nodes[n]) for n in graph.nodes],
        dtype=F64,
    )
    if kind == "headspace":
        if crown is None:
            raise ValueError("_stack_for: kind='headspace' requires the crown diameters")
        z_path = 0.5 * (invert[src] + invert[tgt]) + crown
    else:
        z_path = ground[src]
    return Stack(kind, src=src, tgt=tgt, z_path=z_path, z_ref=ground,
                 rho_key="rho_air_nodes", g=9.80665)
