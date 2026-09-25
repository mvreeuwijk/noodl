"""Streets, junctions, and the builder that turns them into a `Model`.

The network is the ELIMINATED form: every street is a storage
node, every intersection has been eliminated into directed `route` edges between the
streets meeting there, and the atmosphere is the single boundary node that the `vent` and
`exchange` edges reach. There is no potential layer and no junction node; the flows are
written by `StreetFlows` into the driver `"street.q"`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from noodl.apps.street_aq.canyon import Z0_B_DEFAULT, Z0_S_DEFAULT
from noodl.apps.street_aq.routing import StreetFlows, StreetGeometry
from noodl.layers.reaction import Reaction
from noodl.layers.transport import TransportLayer, active_interior
from noodl.model import Drivers, Model, State
from noodl.topology import Network

_DTYPE = torch.float64
"""EVERY tensor this application builds is float64: `torch.get_default_dtype()` is float32
in this repository, so no tensor here may be built without an explicit dtype."""


@dataclass(frozen=True)
class Street:
    """One street segment. `u` and `v` are junction names; the azimuth points `u -> v`."""

    name: str
    u: str
    v: str
    length: float
    width: float
    height: float
    z0_b: float = Z0_B_DEFAULT
    emission_scale: float = 1.0


@dataclass(frozen=True)
class StreetNetwork:
    """Streets plus the x/y coordinates (metres) of every junction they name."""

    streets: list[Street]
    x: dict[str, float]
    y: dict[str, float]
    _azimuth: list[float] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for street in self.streets:
            if street.name in seen:
                raise ValueError(
                    f"StreetNetwork: duplicate street name {street.name!r}"
                )
            seen.add(street.name)
            if street.u == street.v:
                raise ValueError(
                    f"StreetNetwork: street {street.name!r} has u and v are the same "
                    f"junction ({street.u!r}); a street with both ends at one junction has "
                    f"no direction there and cannot be routed"
                )
            missing = [n for n in (street.u, street.v) if n not in self.x or n not in self.y]
            if missing:
                raise KeyError(
                    f"StreetNetwork: street {street.name!r} names junction(s) {missing} "
                    f"with no x/y coordinate"
                )
            for attribute in ("length", "width", "height", "z0_b"):
                value = float(getattr(street, attribute))
                if not value > 0:
                    raise ValueError(
                        f"StreetNetwork: street {street.name!r} has {attribute} "
                        f"{value!r}; it must be strictly positive"
                    )
        object.__setattr__(self, "_azimuth", [
            math.atan2(self.y[s.v] - self.y[s.u], self.x[s.v] - self.x[s.u])
            for s in self.streets
        ])

    @property
    def azimuth(self) -> list[float]:
        """Radians counter-clockwise from EAST, pointing `u -> v`, one per street."""
        return list(self._azimuth)

    @property
    def junctions(self) -> list[str]:
        """Junction names in first-appearance order -- the order `StreetFlows` indexes."""
        out: list[str] = []
        for street in self.streets:
            for node in (street.u, street.v):
                if node not in out:
                    out.append(node)
        return out

    def degree(self, node: str) -> int:
        """How many street ends meet at `node`. Degree one is a dead end."""
        return sum((s.u == node) + (s.v == node) for s in self.streets)


def street_geometry(net: StreetNetwork) -> StreetGeometry:
    """The plain-tensor view of `net` that `StreetFlows` consumes."""
    return StreetGeometry(
        names=[s.name for s in net.streets],
        u=[s.u for s in net.streets],
        v=[s.v for s in net.streets],
        length=torch.tensor([s.length for s in net.streets], dtype=_DTYPE),
        width=torch.tensor([s.width for s in net.streets], dtype=_DTYPE),
        height=torch.tensor([s.height for s in net.streets], dtype=_DTYPE),
        z0_b=torch.tensor([s.z0_b for s in net.streets], dtype=_DTYPE),
        azimuth=torch.tensor(net.azimuth, dtype=_DTYPE),
    )


def from_test_network() -> StreetNetwork:
    """IMPAQ's `build_test_network`: four junctions, three roads (`impaq.py:61-80`).

    Coordinates, widths, heights and the 0.15 m roughness are the prototype's own; the
    lengths are recomputed from the coordinates exactly as `compute_road_geometry` does,
    so `r1` and `r2` come out at 316.227766 m and `r3` at 300 m.
    """
    x = {"n0": 0.0, "n1": 300.0, "n2": 600.0, "n3": 300.0}
    y = {"n0": 400.0, "n1": 300.0, "n2": 400.0, "n3": 0.0}
    spec = [("r1", "n0", "n1", 20.0, 20.0), ("r2", "n1", "n2", 10.0, 20.0),
            ("r3", "n3", "n1", 20.0, 30.0)]
    streets = [
        Street(name, a, b, math.hypot(x[b] - x[a], y[b] - y[a]), width, height)
        for name, a, b, width, height in spec
    ]
    return StreetNetwork(streets=streets, x=x, y=y)


def munich_idealised(
    *, L: float = 100.0, W: float = 20.0, H: float = 20.0
) -> tuple[StreetNetwork, list[str]]:
    """The 12-street network of Kim et al. 2022 Fig. 1, p. 7374.

    Four real junctions on a square of side `L` (`A` NW, `B` NE, `C` SW, `D` SE) and eight
    dead ends one spacing out along each stub, so all twelve segments have the same length.
    `L`, `W` and `H` are ARGUMENTS because the paper never published them: the absolute
    concentrations of Fig. 1 cannot be reproduced, only the ratios and the pattern.
    Returns the network and the street names `"1"` ... `"12"`, with `"11"` the emitter.
    """
    coordinates = {
        "A": (0.0, L), "B": (L, L), "C": (0.0, 0.0), "D": (L, 0.0),
        "N1": (0.0, 2.0 * L), "N2": (L, 2.0 * L), "E3": (-L, L), "E5": (2.0 * L, L),
        "E8": (-L, 0.0), "E10": (2.0 * L, 0.0), "E11": (0.0, -L), "E12": (L, -L),
    }
    ends = [("N1", "A"), ("N2", "B"), ("E3", "A"), ("A", "B"), ("B", "E5"), ("C", "A"),
            ("D", "B"), ("E8", "C"), ("C", "D"), ("D", "E10"), ("E11", "C"), ("D", "E12")]
    streets = [Street(str(i + 1), a, b, L, W, H) for i, (a, b) in enumerate(ends)]
    net = StreetNetwork(
        streets=streets,
        x={k: v[0] for k, v in coordinates.items()},
        y={k: v[1] for k, v in coordinates.items()},
    )
    return net, [s.name for s in streets]


def build_model(
    net: StreetNetwork,
    *,
    canyon_wind: str = "soulhac",
    exchange: str = "sirane",
    routing: str = "sirane",
    direction_averaging: str = "none",
    n_theta: int | None = None,
    sigma_theta: float | None = None,
    species: Sequence[str] = ("nox",),
    chemistry: Reaction | None = None,
    scheme: str = "implicit",
    kappa: float | None = None,
    canyon_wind_min: float = 0.0,
    u_d_min: float = 0.0,
    stability: str = "impaq",
    roof_wind_form: str = "sirane",
    z0_s: float = Z0_S_DEFAULT,
    z_ref: float = 30.0,
    pblh_floor: bool = True,
    atmosphere: str = "atmosphere",
    layer_name: str = "street",
) -> tuple[Model, State, Drivers]:
    """`(Model, initial state, driver template)` for one street network.

    The graph: the boundary node `atmosphere` FIRST, then one node per street in the
    network's own order -- which makes the layer's active interior exactly the street order
    (checked below rather than assumed, because every index in this application depends on
    it). Then, for every junction, a directed `route` edge for each ordered pair of
    DISTINCT street ends meeting there; two `vent` edges per (street, end), one each way;
    two `exchange` edges per street, one each way. Every edge carries the attributes
    `StreetFlows` reads its indices from.

    `pblh_floor=True` (the default) applies MUNICH's `pblh := max(H, PBLH)` guard with the
    network's tallest street, which keeps `sigma_w` positive. Pass `False` for the
    prototype's unguarded behaviour -- and expect `exchange_velocity` to refuse the
    step if a street is taller than 1.25 times the boundary-layer height.

    `kappa=None` (the default) is passed straight through to `StreetFlows`,
    which resolves it to MUNICH's 0.41 whenever any MUNICH-style form is selected
    (`canyon_wind="exponential"`, `exchange="schulte"` or `roof_wind_form="macdonald"`) and
    to IMPAQ's 0.4 otherwise; an explicit float always wins over that resolution.
    """
    names = [s.name for s in net.streets]
    if atmosphere in names:
        raise ValueError(
            f"build_model: a street is called {atmosphere!r}, which is the name of "
            f"the boundary node; rename the street or pass another `atmosphere`"
        )
    graph = Network(dtype=_DTYPE)
    graph.add_node(atmosphere)
    for street in net.streets:
        graph.add_node(street.name, volume=street.length * street.width * street.height)
    junctions = net.junctions
    index = {name: j for j, name in enumerate(junctions)}
    # THE junction and slot enumeration. It is written onto the edges below and read back
    # from them by `StreetFlows`; nothing anywhere else re-derives it.
    slots: list[list[tuple[int, str]]] = [[] for _ in junctions]
    for i, street in enumerate(net.streets):
        slots[index[street.u]].append((i, "u"))
        slots[index[street.v]].append((i, "v"))
    for j, members in enumerate(slots):
        for a, (ia, _) in enumerate(members):
            for b, (ib, _) in enumerate(members):
                if a == b:
                    continue
                graph.add_edge(names[ia], names[ib], kind="route", junction=j,
                               slot_a=a, slot_b=b)
    for j, members in enumerate(slots):
        for s, (i, end) in enumerate(members):
            graph.add_edge(names[i], atmosphere, kind="vent", junction=j, slot=s,
                           street=i, end=end, direction="out")
            graph.add_edge(atmosphere, names[i], kind="vent", junction=j, slot=s,
                           street=i, end=end, direction="in")
    for i, street in enumerate(net.streets):
        graph.add_edge(street.name, atmosphere, kind="exchange", street=i,
                       direction="out")
        graph.add_edge(atmosphere, street.name, kind="exchange", street=i, direction="in")
    kinds = ("route", "vent", "exchange")
    interior, _inactive = active_interior(graph, kinds, [atmosphere])
    ordered = [graph.nodes[i] for i in interior.tolist()]
    if ordered != names:
        raise ValueError(
            f"build_model: the layer's active interior came out as {ordered}, not "
            f"the street order {names}; every index in this application assumes they agree"
        )
    capacity = torch.tensor(
        [s.length * s.width * s.height for s in net.streets], dtype=_DTYPE
    )
    n_species = len(tuple(species))
    layer = TransportLayer(
        graph, layer_name, capacity=capacity, flow_kind=kinds, boundary=[atmosphere],
        n_species=n_species, scheme=scheme, quantity="concentration", unit="kg/m3",
    )
    closure = StreetFlows(
        graph, layer, street_geometry(net), canyon_wind=canyon_wind, exchange=exchange,
        routing=routing, direction_averaging=direction_averaging, n_theta=n_theta,
        sigma_theta=sigma_theta, kappa=kappa, canyon_wind_min=canyon_wind_min,
        u_d_min=u_d_min, stability=stability, roof_wind_form=roof_wind_form, z0_s=z0_s,
        z_ref=z_ref, pblh_floor=pblh_floor, layer_name=layer_name,
    )
    reactions = [(layer_name, chemistry)] if chemistry is not None else []
    model = Model(graph, {layer_name: layer}, closures=[closure], reactions=reactions)
    n_streets = len(net.streets)
    state: State = {
        f"{layer_name}.x": torch.zeros(
            (n_streets,) if n_species == 1 else (n_streets, n_species), dtype=_DTYPE
        )
    }
    drivers: Drivers = {
        f"{layer_name}.x_boundary": torch.zeros(
            (1,) if n_species == 1 else (1, n_species), dtype=_DTYPE
        ),
        f"{layer_name}.sources": torch.zeros(
            (graph.n,) if n_species == 1 else (graph.n, n_species), dtype=_DTYPE
        ),
    }
    return model, state, drivers


def initial_state(model: Model) -> State:
    """Zero concentration everywhere, in each transport layer's own shape."""
    state: State = {}
    for name, layer in model.transport.items():
        if layer.quantity != "concentration":
            raise ValueError(
                f"initial_state: transport layer {name!r} has quantity "
                f"{layer.quantity!r}, which the street application has no initial value "
                f"for (it knows 'concentration'); build it with build_model, or "
                f"set that layer's own state key {name + '.x'!r} yourself"
            )
        shape = (layer.n_i,) if layer.n_species == 1 else (layer.n_i, layer.n_species)
        state[f"{name}.x"] = torch.zeros(shape, dtype=_DTYPE)
    return state


def street_index(model: Model, *, layer_name: str = "street") -> dict[str, int]:
    """Street name -> its row of `"<layer>.x"`."""
    layer = model.transport[layer_name]
    return {model.net.nodes[i]: r for r, i in enumerate(layer.interior_idx.tolist())}
