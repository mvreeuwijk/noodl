"""Modelica electrical import: OpenModelica exports of `Modelica.Electrical.Analog` (MSL) and
`Buildings.Electrical` (MBL) example models, in the `noodl-modelica-electrical/1` format
written by `scripts/modelica_export_electrical.py`.

`read_modelica_circuit(path)` reads one file into `(model, state, drivers, names)`:
`schema.load` parses and structurally validates the JSON; `netlist.build` resolves it into
devices on numbered nodes, refusing unsupported content with one error naming every problem;
`assemble.build` turns the netlist into a model with its laws, its drivers on the step grid,
its start state and a `CircuitNames` that maps the reference's columns to noodl quantities.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from noodl.apps.electrical.modelica import assemble, netlist, schema
from noodl.apps.electrical.modelica.assemble import CircuitNames, Column
from noodl.apps.electrical.modelica.schema import ModelicaImportError

__all__ = ["CircuitNames", "Column", "ModelicaImportError", "read_modelica_circuit"]


def read_modelica_circuit(path: str | Path, *, substeps: int = 1, omit: Iterable[str] = ()):
    """Read a `noodl-modelica-electrical/1` file into `(model, state, drivers, names)`.

    `substeps` splits every interval of the step grid (the output times and the signal
    events) into that many equal steps; `omit` names components to drop, with their
    connections, before the netlist is built. Raises `ModelicaImportError` naming every
    problem of a model the reader cannot convert in full.
    """
    doc = schema.load(path)
    return assemble.build(netlist.build(doc, omit=omit), doc, substeps=substeps)
