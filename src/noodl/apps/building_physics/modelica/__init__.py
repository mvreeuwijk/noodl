"""Modelica Buildings Library (MBL) multizone import.

This package reads the intermediate JSON an OpenModelica export script writes
(`scripts/modelica_export.py`) and builds a noodl `Model` from it. It never parses
Modelica source and never evaluates a Modelica expression: every parameter it reads was
already evaluated by OpenModelica.

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
from noodl.apps.building_physics.modelica.run import extrapolate, simulate, step_drivers
from noodl.apps.building_physics.modelica.schema import ModelicaImportError


def read_modelica(path: str | Path, *, return_names: bool = False, substeps: int = 1,
                  mass_storage: bool | None = None):
    """Read one `noodl-modelica/1` JSON file into `(model, state, drivers)`.

    The drivers are evaluated on `assemble.driver_grid`: the experiment's output grid with
    each interval split into `substeps` equal steps, plus the event times of the signals
    whose output jumps; `run.simulate` steps over all of it and reports the times asked for.

    `mass_storage` (`assemble.build`, `storage` module): `None` (the default) models the
    compressible mass storage of every volume whose `massDynamics` is not `SteadyState`, as
    MBL does; `False` reads the quasi-steady airflow; `True` also refuses a model with no
    storing volume.

    With `return_names=True` a fourth item, the `ModelicaNames` mapping MBL instance names to
    the model's edge columns and node positions (and carrying the experiment grid), is
    returned too. Raises `ModelicaImportError` naming every unsupported instance.
    """
    doc = schema.load(path)
    model, state, drivers, names = assemble.build(graph.build(doc), doc, substeps=substeps,
                                                  mass_storage=mass_storage)
    if return_names:
        return model, state, drivers, names
    return model, state, drivers


__all__ = ["ModelicaImportError", "ModelicaNames", "extrapolate", "read_modelica", "simulate",
           "step_drivers"]
