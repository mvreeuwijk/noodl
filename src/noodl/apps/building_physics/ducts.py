"""Duct networks: junctions, segments and terminals, built the way CONTAM builds them.

A junction is a node holding half the volume of every segment that meets it (length times
the segment's area); a terminal's own zero-length segment adds nothing (measured against
ContamX 3.4.1.7). A segment is an `AirDuct` edge between two junctions, a terminal a
`DuctTerminal` edge from its junction to a zone. A fan, damper or any other element in a
duct is an ordinary edge between two junctions, added by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from noodl.apps.building_physics.thermal import T_REF
from noodl.components import OuterRef
from noodl.drives import DuctStack, Stack, Wind
from noodl.elements import AirDuct, DuctTerminal

DEFAULT_ROUGHNESS = 9e-5       # m, CONTAM's default duct roughness
DEFAULT_TERMINAL_LOSS = 0.125  # CONTAM's default terminal loss coefficient


@dataclass
class _Segment:
    a: str
    b: str
    length: float
    diameter: float
    roughness: float
    loss: float
    area: float
    name: str | None


@dataclass
class _Terminal:
    junction: str
    zone: object
    area: float
    loss: float
    name: str | None
    wind: dict = field(default_factory=dict)


class DuctNetwork:
    """A duct system, described first and then added to a `Network` or a `Component`:

        ducts = DuctNetwork()
        ducts.junction("j1", z_ref=2.0)
        ducts.junction("j2", z_ref=2.0)
        ducts.segment("j1", "j2", length=4.0, diameter=0.2)
        ducts.terminal("j2", "room", area=0.0314)
        ducts.add_to(net)

    Junction volumes come from the segments, which is why the system is described before it
    is added. Edge kinds: `kind` for the segments, `terminal_kind` for the terminals.
    `ambient` names the ambient node (or, inside a component, its `outer` reference), the
    one zone whose terminals take the wind.
    """

    def __init__(self, *, kind: str = "duct", terminal_kind: str = "duct_terminal",
                 ambient: str = "ambient") -> None:
        self.kind, self.terminal_kind, self.ambient = kind, terminal_kind, ambient
        self._junctions: dict[str, dict] = {}
        self._segments: list[_Segment] = []
        self._terminals: list[_Terminal] = []

    def junction(self, name: str, *, z_ref: float, T0: float = T_REF) -> None:
        """A junction at height `z_ref` (m) holding air at `T0` (K)."""
        if name in self._junctions:
            raise ValueError(f"DuctNetwork: junction {name!r} is declared twice")
        self._junctions[name] = dict(z_ref=float(z_ref), T0=float(T0))

    def _known(self, name, what: str) -> None:
        if name not in self._junctions:
            raise KeyError(f"DuctNetwork: {what} names junction {name!r}, which was not "
                           f"declared with junction()")

    def segment(self, a: str, b: str, *, length: float, diameter: float,
                roughness: float = DEFAULT_ROUGHNESS, loss: float = 0.0,
                area: float | None = None, name: str | None = None) -> None:
        """A duct segment from junction `a` to `b` (positive flow a -> b): `length`,
        hydraulic `diameter` and `roughness` in m, `loss` the sum of its dynamic loss
        coefficients, `area` (m2) the circle of the diameter when not given."""
        where = f"DuctNetwork: segment {name or f'{a!r} -> {b!r}'}"
        self._known(a, "a segment")
        self._known(b, "a segment")
        if a == b:
            raise ValueError(f"{where}: a segment cannot join junction {a!r} to itself")
        for label, value in (("length", length), ("diameter", diameter),
                             ("roughness", roughness)) + ((("area", area),) if area is not None
                                                          else ()):
            if not value > 0:
                raise ValueError(f"{where}: {label} must be positive, got {value!r}")
        if not loss >= 0:
            raise ValueError(f"{where}: loss must be non-negative, got {loss!r}")
        if not roughness < 3.7 * diameter:
            raise ValueError(f"{where}: roughness must be below 3.7 diameters (the Colebrook "
                             f"equation has no solution beyond), got roughness {roughness!r} "
                             f"and diameter {diameter!r}")
        A = math.pi * diameter * diameter / 4.0 if area is None else float(area)
        self._segments.append(_Segment(a, b, float(length), float(diameter),
                                       float(roughness), float(loss), A, name))

    def terminal(self, junction: str, zone, *, area: float,
                 loss: float = DEFAULT_TERMINAL_LOSS, name: str | None = None,
                 Ch: float | None = None, azimuth: float | None = None,
                 profile: int | None = None, Cp: float | None = None,
                 facade: str | None = None) -> None:
        """A terminal from `junction` into `zone` (positive flow out of the duct): a zone's
        node name, or `outer("ambient")` inside a component. `area` (m2) is the duct's area
        at the terminal, `loss` (positive) its loss coefficient. A terminal in the ambient
        takes the wind as an envelope path does: from the `WindProfile` numbered `profile`
        at the wall's `azimuth` (or, inside a component, its `facade`), or from the constant
        pressure coefficient `Cp`, scaled by the speed modifier `Ch` (1 by default). Wind data
        that would act on nothing is refused: `Ch`, `azimuth` or `facade` without a `profile`
        or with `Cp = 0`, a `profile` and a `Cp` together, and wind on a terminal into a zone
        other than the ambient."""
        where = f"DuctNetwork: terminal {name or f'{junction!r} -> {zone!r}'}"
        self._known(junction, "a terminal")
        if not area > 0:
            raise ValueError(f"{where}: area must be positive, got {area!r}")
        if not loss > 0:
            raise ValueError(f"{where}: loss must be positive (with no loss nothing limits "
                             f"the flow), got {loss!r}")
        if profile is not None and not (float(profile) == int(profile) and profile >= 1):
            raise ValueError(f"{where}: profile must be a wind profile number from 1, got "
                             f"{profile!r}; for no profile leave it out")
        stray = [k for k, v in dict(Ch=Ch, azimuth=azimuth, facade=facade).items()
                 if v is not None]
        if profile is None and Cp is None and stray:
            raise ValueError(f"{where}: {', '.join(stray)} given without a profile or a "
                             f"Cp, so no wind would act; give the terminal's profile or "
                             f"its constant pressure coefficient Cp")
        if profile is None and Cp is not None and float(Cp) == 0.0 and stray:
            raise ValueError(f"{where}: {', '.join(stray)} given with Cp = 0, so no wind "
                             f"would act; give a nonzero Cp or the terminal's profile")
        if profile is not None and Cp is not None:
            raise ValueError(f"{where}: both a profile and a Cp given; the profile would "
                             f"replace the constant Cp, so give one of them")
        into = zone.name if isinstance(zone, OuterRef) else zone
        if (profile is not None or Cp is not None) and into != self.ambient:
            raise ValueError(f"{where}: wind on a terminal into {into!r}, which is not the "
                             f"ambient {self.ambient!r}; the wind acts only on a terminal "
                             f"into the ambient")
        wind: dict = {k: float(v) for k, v in
                      dict(Ch=Ch, azimuth=azimuth, profile=profile, Cp=Cp).items()
                      if v is not None}
        if facade is not None:
            wind["facade"] = facade
        self._terminals.append(_Terminal(junction, zone, float(area), float(loss), name,
                                         wind))

    def volumes(self) -> dict[str, float]:
        """Each junction's volume (m3): half of every attached segment's length x area."""
        v = dict.fromkeys(self._junctions, 0.0)
        for s in self._segments:
            half = 0.5 * s.length * s.area
            v[s.a] += half
            v[s.b] += half
        return v

    def add_to(self, target) -> None:
        """Add the junction nodes and the segment and terminal edges to `target`, a
        `Network` or a `Component`."""
        volumes = self.volumes()
        for name, attrs in self._junctions.items():
            target.add_node(name, volume=volumes[name], heat_capacity=0.0, **attrs)
        for s in self._segments:
            named = {"name": s.name} if s.name else {}
            target.add_edge(s.a, s.b, kind=self.kind, length=s.length, diameter=s.diameter,
                            roughness=s.roughness, loss=s.loss, area=s.area, **named)
        for t in self._terminals:
            named = {"name": t.name} if t.name else {}
            target.add_edge(t.junction, t.zone, kind=self.terminal_kind, area=t.area,
                            loss=t.loss, z_path=self._junctions[t.junction]["z_ref"],
                            **t.wind, **named)


def duct_elements_from_edges(net, *, kind: str = "duct", terminal_kind: str = "duct_terminal",
                             temperature_key: str | None = None,
                             opening_key: str | None = None, g: float = 9.80665,
                             ambient="ambient", profiles: Sequence = ()):
    """The elements and drives of the duct edges `DuctNetwork.add_to` made: an `AirDuct`
    and a `DuctStack` for the segments, a `DuctTerminal`, a `Stack` (at the junction's
    height) and, when a terminal has a wind profile or a nonzero `Cp`, a `Wind` for the
    terminals (`profiles` numbers the `WindProfile`s from 1, as for `Wind.from_network`;
    an edge with neither feels no wind). Returns
    `(elements, drives)`; a kind the network lacks contributes nothing.
    `temperature_key` names the full-node temperature driver for the viscosity (the
    viscosity at 293.15 K when None); `opening_key` the terminals' opening driver."""
    elements, drives = [], []
    kinds = set(net.edge_kinds())
    if kind in kinds:
        src, tgt = net.endpoints(kind)
        elements.append(AirDuct(net.edge_attr("length", kind), net.edge_attr("diameter", kind),
                                net.edge_attr("roughness", kind), net.edge_attr("loss", kind),
                                area=net.edge_attr("area", kind), src=src, tgt=tgt,
                                temperature_key=temperature_key, kind=kind))
        drives.append(DuctStack.from_network(net, kind, g=g))
    if terminal_kind in kinds:
        src, tgt = net.endpoints(terminal_kind)
        elements.append(DuctTerminal(net.edge_attr("area", terminal_kind),
                                     net.edge_attr("loss", terminal_kind), src=src, tgt=tgt,
                                     temperature_key=temperature_key, opening_key=opening_key,
                                     kind=terminal_kind))
        drives.append(Stack.from_network(net, terminal_kind, g=g))
        windy = ((net.edge_attr("profile", terminal_kind, default=0.0) > 0)
                 | (net.edge_attr("Cp", terminal_kind, default=0.0) != 0))
        if bool(windy.any()):
            drives.append(Wind.from_network(net, terminal_kind, ambient=ambient,
                                            profiles=profiles))
    return elements, drives
