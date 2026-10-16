"""Composite MBL classes the reader expands into classes it knows, before `graph.build`.

`MediumColumnDynamic` (MBL v13.0.0 `Airflow/Multizone/MediumColumnDynamic.mo`) is a mixing
volume `vol` between two half-height columns:

- `colTop` (`h/2`, `densitySelection = fromBottom`) from the model's `port_a` to
  `vol.ports[2]`;
- `colBot` (`h/2`, `fromTop`) from `vol.ports[1]` to the model's `port_b`.

Both half columns therefore take the volume's density. `vol` has the model's `V`,
`energyDynamics` (and `massDynamics = energyDynamics`, declared `final`), its `p_start`,
`T_start`, `X_start` and `C_start`, `m_flow_nominal = V/360` and `nPorts = 2`; everything
else (`C_nominal`, `mSenFac`, `initialize_p`) keeps the `MixingVolume` default, since the
model does not pass it on. The parts are named `<name>|vol`, `<name>|colTop` and
`<name>|colBot`. `|` cannot occur in a Modelica name, so the parts never clash with the
model's own instances. The reader reports the volume under its Modelica path `<name>.vol`
(`ModelicaNames.nodes`). With `use_HeatTransfer = true` the column exchanges heat through
`heatPort`, which is refused (raises `ModelicaImportError`, naming every such instance).
"""

from __future__ import annotations

import dataclasses

from noodl.apps.building_physics.modelica import graph, schema
from noodl.apps.building_physics.modelica.schema import Component, ModelicaDoc

SEPARATOR = "|"
_VOLUME = "Buildings.Fluid.MixingVolumes.MixingVolume"
_COLUMN = "Buildings.Airflow.Multizone.MediumColumn"
_SELECTION = "Buildings.Airflow.Multizone.Types.densitySelection."
# MediumColumnDynamic.mo: `vol(final nPorts=2, final m_flow_nominal=V/360, final V=V, final
# energyDynamics=energyDynamics, final massDynamics=massDynamics, final p_start=p_start,
# final T_start=T_start, final X_start=X_start, final C_start=C_start)`; `massDynamics =
# energyDynamics` (the `extends` clause); `h = 3` (its declaration); `energyDynamics`
# defaults to `DynamicFreeInitial` (`Fluid/Interfaces/LumpedVolumeDeclarations.mo`).
_PASSED = ("p_start", "T_start", "X_start", "C_start")
_FREE = "Modelica.Fluid.Types.Dynamics.DynamicFreeInitial"


def modelica_path(name: str) -> str:
    """The Modelica path of an expanded part: `"sha|vol"` -> `"sha.vol"`."""
    return name.replace(SEPARATOR, ".")


def expand(doc: ModelicaDoc) -> ModelicaDoc:
    """`doc` with every `MediumColumnDynamic` replaced by its parts (module docstring)."""
    errors: list[str] = []
    components: list[Component] = []
    renamed: dict[str, str] = {}
    internal: list[tuple[str, str]] = []
    for c in doc.components:
        if c.cls not in schema.DYNAMIC_COLUMNS:
            components.append(c)
            continue
        where = f"{c.name} ({c.cls})"
        p = c.parameters
        if p.get("use_HeatTransfer", False):
            errors.append(f"{where}: use_HeatTransfer = true is not supported (heat exchange "
                          f"through the column's heatPort)")
            continue
        if "V" not in p:
            errors.append(f"{where}: parameter 'V' is required and was not exported")
            continue
        V, h = float(p["V"]), float(p.get("h", 3.0))
        energy = p.get("energyDynamics", _FREE)
        vol = {k: p[k] for k in _PASSED if k in p}
        vol.update(V=V, energyDynamics=energy, massDynamics=energy, nPorts=2,
                   m_flow_nominal=V / 360.0)
        name_vol = f"{c.name}{SEPARATOR}vol"
        top, bot = f"{c.name}{SEPARATOR}colTop", f"{c.name}{SEPARATOR}colBot"
        components += [
            Component(name=name_vol, cls=_VOLUME, parameters=vol),
            Component(name=top, cls=_COLUMN,
                      parameters={"h": h / 2.0, "densitySelection": _SELECTION + "fromBottom"}),
            Component(name=bot, cls=_COLUMN,
                      parameters={"h": h / 2.0, "densitySelection": _SELECTION + "fromTop"}),
        ]
        renamed[f"{c.name}.port_a"] = f"{top}.port_a"
        renamed[f"{c.name}.port_b"] = f"{bot}.port_b"
        internal += [(f"{top}.port_b", f"{name_vol}.ports[2]"),
                     (f"{name_vol}.ports[1]", f"{bot}.port_a")]
    graph._raise(errors)  # the same "refused N items" report as `graph.build`
    connections = tuple((renamed.get(a, a), renamed.get(b, b)) for a, b in doc.connections)
    return dataclasses.replace(doc, components=tuple(components),
                               connections=connections + tuple(internal))
