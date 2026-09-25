"""Modelica Buildings Library (MBL) multizone import.

Design: `docs/superpowers/specs/2026-09-24-modelica-import-design.md`. This package reads the
intermediate JSON an OpenModelica export script writes (spec section 4,
`scripts/modelica_export.py`) and builds a noodl `Model` from it. It never parses Modelica
source and never evaluates a Modelica expression: every parameter it reads was already
evaluated by OpenModelica.

`schema.load` parses and structurally validates the JSON; `graph.build` resolves it into a
`ComponentGraph` (nodes, fused hydrostatic-column paths, two-way edges, heat pins, sources),
refusing unsupported content; `assemble.build` turns that graph into the model, its initial
state and its drivers (`signals` evaluates the signal blocks on the experiment grid);
`run.simulate` steps the result over the grid. `read_modelica` chains them and returns the
same `(Model, State, Drivers)` triple as the CONTAM route
(`noodl.apps.building_physics.read_prj` + `project_to_model`).
"""

from __future__ import annotations

from pathlib import Path

from noodl.apps.building_physics.modelica import assemble, graph, schema
from noodl.apps.building_physics.modelica.assemble import ModelicaNames
from noodl.apps.building_physics.modelica.run import simulate, step_drivers
from noodl.apps.building_physics.modelica.schema import ModelicaImportError


def read_modelica(path: str | Path, *, return_names: bool = False):
    """Read one `noodl-modelica/1` JSON file into `(model, state, drivers)`.

    With `return_names=True` a fourth item, the `ModelicaNames` mapping MBL instance names to
    the model's edge columns and node positions (and carrying the experiment grid), is
    returned too. Raises `ModelicaImportError` naming every unsupported instance.
    """
    doc = schema.load(path)
    model, state, drivers, names = assemble.build(graph.build(doc), doc)
    if return_names:
        return model, state, drivers, names
    return model, state, drivers


__all__ = ["ModelicaImportError", "ModelicaNames", "read_modelica", "simulate", "step_drivers"]
