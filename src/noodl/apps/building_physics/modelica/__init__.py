"""Modelica Buildings Library (MBL) multizone import.

Design: `docs/superpowers/specs/2026-09-24-modelica-import-design.md`. This package reads the
intermediate JSON an OpenModelica export script writes (spec section 4,
`scripts/modelica_export.py`, Task 8) and builds a component graph noodl can turn into a
`Model` (Task 6). It never parses Modelica source and never evaluates a Modelica expression:
every parameter it reads was already evaluated by OpenModelica.

`schema.load` parses and structurally validates the JSON; `graph.build` resolves it into a
`ComponentGraph` (nodes, fused hydrostatic-column paths, two-way edges, heat pins, sources),
refusing unsupported content. `read_modelica`, the public `(Model, State, Drivers)` entry point
mirroring the CONTAM route (`noodl.apps.building_physics.read_prj`), is Task 6's addition; this
package only exports the shared error type until then.
"""

from __future__ import annotations

from noodl.apps.building_physics.modelica.schema import ModelicaImportError

__all__ = ["ModelicaImportError"]
