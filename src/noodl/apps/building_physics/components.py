"""Building components: room, door, crack, window and shaft factories, each returning a
`noodl.components.Component`. Heights are relative to the component's placement, as a
CONTAM zone's or path's height is relative to its level. Importing this module registers
`z_ref` and `z_path` as heights and `facade` as a lookup in the `facades` table."""

from __future__ import annotations

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
