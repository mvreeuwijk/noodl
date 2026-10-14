"""Modelica electrical import: OpenModelica exports of `Modelica.Electrical.Analog` (MSL) and
`Buildings.Electrical` (MBL) example models, in the `noodl-modelica-electrical/1` format
written by `scripts/modelica_export_electrical.py`.

`read_modelica_circuit(path)` reads one file into `(model, state, drivers, names)`:
`schema.load` parses and structurally validates the JSON; `netlist.build` resolves it into
devices on numbered nodes, refusing unsupported content with one error naming every problem;
`assemble.build` turns the netlist into a model with its laws, its drivers on the step grid,
its start state and a `CircuitNames` that maps the reference's columns to noodl quantities.
`simulate_circuit` runs it over its step grid (`run` module); `extrapolate` combines two runs
at `substeps` `r` and `2r` by Richardson extrapolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from noodl.apps.building_physics.modelica.run import extrapolate
from noodl.apps.electrical.modelica import assemble, netlist, schema
from noodl.apps.electrical.modelica.assemble import CircuitNames, Column
from noodl.apps.electrical.modelica.run import simulate_circuit
from noodl.apps.electrical.modelica.schema import ModelicaImportError

__all__ = ["CircuitNames", "Column", "ModelicaImportError", "extrapolate",
           "read_modelica_circuit", "simulate_circuit"]


def read_modelica_circuit(path: str | Path, *, substeps: int = 1, omit: Iterable[str] = ()):
    """Read a `noodl-modelica-electrical/1` file into `(model, state, drivers, names)`.

    `substeps` splits every interval of the step grid (the output times and the signal
    events) into that many equal steps; `omit` names components to drop, with their
    connections, before the netlist is built. Raises `ModelicaImportError` naming every
    problem of a model the reader cannot convert in full.

    Comparing with the reference at round-off needs tight Newton tolerances: with the
    layer's defaults an algebraic model such as MBL's `ACACTransformer` example stays
    farther from OpenModelica than `1e-9` relative. `simulate_circuit` solves algebraic
    models at `atol=1e-12, rtol=1e-14` by default (`run.STEADY_ATOL`, `run.STEADY_RTOL`;
    `atol=1e-11`, `run.THERMAL_STEADY_ATOL`, with a heat-port thermal network); pass the
    same to `model.steady` when solving by hand.
    """
    doc = schema.load(path)
    return assemble.build(netlist.build(doc, omit=omit), doc, substeps=substeps)
