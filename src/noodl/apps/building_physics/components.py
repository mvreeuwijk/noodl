"""Building components: room, door, crack, window, shaft, fan, duct and air_handler factories, each
returning a `noodl.components.Component`. Heights are relative to the component's placement,
as a CONTAM zone's or path's height is relative to its level. Importing this module registers
`z_ref` and `z_path` as heights and `facade` as a lookup in the `facades` table."""

from __future__ import annotations

import math
from collections.abc import Sequence

from noodl.apps.building_physics.air_handling import add_air_handler
from noodl.apps.building_physics.ducts import (
    DEFAULT_ROUGHNESS,
    DEFAULT_TERMINAL_LOSS,
    DuctNetwork,
)
from noodl.apps.building_physics.elements import add_large_opening
from noodl.apps.building_physics.thermal import T_REF, WallMass, Zone, add_zone
from noodl.components import (
    Component,
    ComponentError,
    register_elevations,
    register_table_lookup,
)

register_elevations(nodes=("z_ref",), edges=("z_path",))
register_table_lookup("facade", table="facades", target="azimuth")


def room(name: str, *, volume: float, T0: float = T_REF, z_ref: float = 0.0,
         wall: WallMass | None = None) -> Component:
    """A zone: air node `air` (m3, K, m) and, with `wall`, a lumped wall node conducting to
    the shared ambient. Port: `air`."""
    c = Component(name, template="room")
    add_zone(c, Zone("air", volume=volume, T0=T0, z_ref=z_ref, wall=wall))
    c.expose("air")
    return c


def door(name: str, *, H: float, W: float, z_mid: float, Cd: float = 0.78) -> Component:
    """A doorway between two zones (CONTAM's two-opening DR_PL2): terminals `a`, `b` and
    edges `low`, `high`. `z_mid` (m) is relative to the door's placement. Ports: `a`, `b`."""
    c = Component(name, template="door")
    c.add_terminal("a")
    c.add_terminal("b")
    add_large_opening(c, "a", "b", H=H, W=W, z_mid=z_mid, Cd=Cd, names=("low", "high"))
    c.expose("a", "b")
    return c


def _orifice(template: str, name: str, *, area: float, z_path: float, Cd: float,
             exterior: bool, azimuth: float | None = None, facade: str | None = None,
             Cp: float | None = None, Ch: float | None = None,
             profile: int | None = None) -> Component:
    wind = dict(azimuth=azimuth, facade=facade, Cp=Cp, Ch=Ch, profile=profile)
    given = {k: v for k, v in wind.items() if v is not None}
    c = Component(name, template=template)
    c.add_terminal("a")
    attrs = dict(z_path=z_path, Cd=Cd, area=area)
    if exterior:
        c.add_edge(c.outer("ambient"), "a", kind="airpath", name="path", **attrs, **given)
        c.expose("a")
    else:
        if given:
            raise ComponentError(
                f"{name}: wind attributes {sorted(given)} apply only to an exterior opening "
                f"(exterior=True)"
            )
        c.add_terminal("b")
        c.add_edge("a", "b", kind="airpath", name="path", **attrs)
        c.expose("a", "b")
    return c


def crack(name: str, *, area: float, z_path: float, Cd: float = 0.6, exterior: bool = False,
          azimuth: float | None = None, facade: str | None = None, Cp: float | None = None,
          Ch: float | None = None, profile: int | None = None) -> Component:
    """A small orifice (m2, m). Interior: between terminals `a` and `b`. Exterior: from the
    shared ambient to `a`, carrying the wind attributes `Wind.from_network` reads; give the
    facade direction as `azimuth` (degrees from north) or as `facade`, a key of the
    building's `inner_table("facades", ...)`. Edge: `path`."""
    return _orifice("crack", name, area=area, z_path=z_path, Cd=Cd, exterior=exterior,
                    azimuth=azimuth, facade=facade, Cp=Cp, Ch=Ch, profile=profile)


def window(name: str, *, area: float, z_path: float, Cd: float = 0.6, exterior: bool = True,
           azimuth: float | None = None, facade: str | None = None, Cp: float | None = None,
           Ch: float | None = None, profile: int | None = None) -> Component:
    """An opening in the envelope: a `crack` that is exterior unless told otherwise."""
    return _orifice("window", name, area=area, z_path=z_path, Cd=Cd, exterior=exterior,
                    azimuth=azimuth, facade=facade, Cp=Cp, Ch=Ch, profile=profile)


def shaft(name: str, *, levels: int, level_height: float, volume: float, area: float,
          T0: float = T_REF, Cd: float = 0.6) -> Component:
    """A tall space (stairwell, atrium, lift shaft) as a stack of zones, one per level,
    joined by openings (how stairwells are usually represented in multizone models): node
    `levels[i]` at z_ref = i * level_height, joined by an opening `slab[i]` (area m2) at the
    slab between levels i and i+1. Ports: `levels[0]`, `levels[1]`, ..."""
    if levels < 2:
        raise ComponentError(f"{name}: a shaft needs at least 2 levels, got {levels}")
    c = Component(name, template="shaft")
    for i in range(levels):
        c.add_node(f"levels[{i}]", volume=volume, T0=T0, z_ref=i * level_height,
                   heat_capacity=0.0)
    for i in range(levels - 1):
        c.add_edge(f"levels[{i}]", f"levels[{i + 1}]", kind="airpath", name=f"slab[{i}]",
                   z_path=(i + 1) * level_height, Cd=Cd, area=area)
    c.expose(*[f"levels[{i}]" for i in range(levels)])
    return c


def _check(name: str, argument: str, value, rule: str, ok) -> float:
    try:
        good = bool(ok(float(value)))
    except (TypeError, ValueError):
        good = False
    if not good:
        raise ComponentError(f"{name}: {argument} must be {rule}, got {value!r}")
    return float(value)


def fan(name: str, *, flow: float, exterior: str | None = None) -> Component:
    """A fan delivering the mass flow `flow` (kg/s, finite, not negative) whatever the
    pressures and the height (CONTAM's constant-mass-flow fan): from terminal `a` to `b`, or
    with `exterior="supply"` from the shared ambient into `a`, or with `exterior="exhaust"`
    from `a` to the ambient. Edge: `fan`, kind "fan"; build it with
    `fan_law_from_edges`. Ports: `a`, `b` (only `a` when exterior)."""
    if exterior not in (None, "supply", "exhaust"):
        raise ComponentError(f"{name}: exterior must be None, 'supply' or 'exhaust', got "
                             f"{exterior!r}")
    attrs = dict(flow=_check(name, "flow", flow, "finite and not negative (swap the ports to "
                             "reverse a fan)", lambda v: math.isfinite(v) and v >= 0.0))
    c = Component(name, template="fan")
    c.add_terminal("a")
    if exterior is None:
        c.add_terminal("b")
        c.add_edge("a", "b", kind="fan", name="fan", **attrs)
        c.expose("a", "b")
    elif exterior == "supply":
        c.add_edge(c.outer("ambient"), "a", kind="fan", name="fan", **attrs)
        c.expose("a")
    else:
        c.add_edge("a", c.outer("ambient"), kind="fan", name="fan", **attrs)
        c.expose("a")
    return c


def duct(name: str, *, length: float, diameter: float, roughness: float = DEFAULT_ROUGHNESS,
         loss: float = 0.0, terminal_area: float | None = None,
         terminal_loss: float = DEFAULT_TERMINAL_LOSS, z_a: float = 0.0, z_b: float = 0.0,
         T0: float = T_REF, exterior: bool = False, azimuth: float | None = None,
         facade: str | None = None, Ch: float | None = None,
         profile: int | None = None) -> Component:
    """A duct run with a terminal at each end: junctions `ends[0]` at height `z_a` and
    `ends[1]` at `z_b` (m, relative to the placement), each holding half the duct's air,
    joined by the segment `segment` (`length`, hydraulic `diameter` and `roughness` in m,
    `loss` its dynamic loss coefficients), with terminals `terminal_a` and `terminal_b`
    (area `terminal_area` in m2, the duct's own by default; loss coefficient
    `terminal_loss`). Interior: from port `a` to port `b`. Exterior: `terminal_a` opens to
    the shared ambient and takes the wind from the `WindProfile` numbered `profile` (with
    `azimuth` or `facade`, and `Ch`); `azimuth`, `facade` or `Ch` without a `profile` are
    refused (a constant `Cp` is offered by `DuctNetwork.terminal`). Invalid lengths,
    diameters or losses are refused naming the duct. Build it with
    `duct_laws_from_edges`. Ports: `a`, `b` (only `b` when exterior)."""
    given = {k: v for k, v in dict(azimuth=azimuth, facade=facade, Ch=Ch,
                                   profile=profile).items() if v is not None}
    if given and not exterior:
        raise ComponentError(f"{name}: wind attributes {sorted(given)} apply only to an "
                             f"exterior duct (exterior=True)")
    try:
        return _duct(name, length, diameter, roughness, loss, terminal_area, terminal_loss,
                     z_a, z_b, T0, exterior, given)
    except ComponentError:
        raise
    except ValueError as err:
        raise ComponentError(f"{name}: {err}") from err


def _duct(name, length, diameter, roughness, loss, terminal_area, terminal_loss, z_a, z_b,
          T0, exterior, given) -> Component:
    c = Component(name, template="duct")
    area = math.pi * diameter * diameter / 4.0 if terminal_area is None else terminal_area
    ducts = DuctNetwork()
    ducts.junction("ends[0]", z_ref=z_a, T0=T0)
    ducts.junction("ends[1]", z_ref=z_b, T0=T0)
    ducts.segment("ends[0]", "ends[1]", length=length, diameter=diameter,
                  roughness=roughness, loss=loss, name="segment")
    if exterior:
        start = c.outer("ambient")
    else:
        c.add_terminal("a")
        start = "a"
    c.add_terminal("b")
    ducts.terminal("ends[0]", start, area=area, loss=terminal_loss, name="terminal_a",
                   **given)
    ducts.terminal("ends[1]", "b", area=area, loss=terminal_loss, name="terminal_b")
    ducts.add_to(c)
    if exterior:
        c.expose("b")
    else:
        c.expose("a", "b")
    return c


def air_handler(name: str, *, supplies: Sequence[float], returns: Sequence[float],
                outdoor_air_fraction: float = 1.0, min_outdoor_air: float = 0.0,
                supply_volume: float = 1.0, return_volume: float = 1.0, T0: float = T_REF,
                kind: str = "air_handler") -> Component:
    """CONTAM's simple air-handling system: plenums `supply` and `return` (volumes in m3,
    temperature `T0` in K), supply points to the ports `supplies[i]` and return points from
    `returns[j]` (design flows in kg/s), the recirculation, and the outdoor air and exhaust
    to the shared ambient; `outdoor_air_fraction` and `min_outdoor_air` (kg/s) as in
    `add_air_handler`. After `flatten`, `air_handler_from_edges(net, kind)` gives its
    law and closure, and `air_handler_plenums(net, kind)` the nodes for
    `build_model(flow_balanced=...)`. Zero plenum volumes (CONTAM's default) also need
    `storage_free=...` and an implicit heat layer; the 1 m3 default here needs neither. Give
    each air handler of a building its own `kind`. Ports: `supplies[i]`, `returns[j]`."""
    for label, noun, flows in (("supplies", "supply", supplies), ("returns", "return", returns)):
        if len(flows) < 1:
            raise ComponentError(f"{name}: an air handler needs at least one {noun} "
                                 f"point, got {label}={list(flows)!r}")
        for i, f in enumerate(flows):
            _check(name, f"{label}[{i}]", f, "a finite design flow, not negative (kg/s)",
                   lambda v: math.isfinite(v) and v >= 0.0)
    _check(name, "outdoor_air_fraction", outdoor_air_fraction, "in [0, 1]",
           lambda v: 0.0 <= v <= 1.0)
    _check(name, "min_outdoor_air", min_outdoor_air, "finite and not negative (kg/s)",
           lambda v: math.isfinite(v) and v >= 0.0)
    _check(name, "supply_volume", supply_volume, "finite and not negative (m3)",
           lambda v: math.isfinite(v) and v >= 0.0)
    _check(name, "return_volume", return_volume, "finite and not negative (m3)",
           lambda v: math.isfinite(v) and v >= 0.0)
    c = Component(name, template="air_handler")
    supply_ports = [f"supplies[{i}]" for i in range(len(supplies))]
    return_ports = [f"returns[{j}]" for j in range(len(returns))]
    for port in (*supply_ports, *return_ports):
        c.add_terminal(port)
    add_air_handler(c, supplies=dict(zip(supply_ports, supplies, strict=True)),
                    returns=dict(zip(return_ports, returns, strict=True)), kind=kind,
                    supply_plenum="supply", return_plenum="return",
                    outdoor_air_fraction=outdoor_air_fraction,
                    min_outdoor_air=min_outdoor_air, supply_volume=supply_volume,
                    return_volume=return_volume, T0=T0)
    c.expose(*supply_ports, *return_ports)
    return c
