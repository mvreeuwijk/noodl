"""Export one Modelica electrical model to `noodl-modelica-electrical/1` JSON and a CSV.

Run inside WSL, where OpenModelica is installed::

    python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.Resistor \
        [--out tests/data/modelica_electrical] [--mbl ~/modelica/modelica-buildings] \
        [--tolerance 1e-12] [--intervals N] [--no-simulate] [--keep DIR]

The model is a `Modelica.Electrical.Analog` example of the Modelica Standard Library (MSL
4.1.0) or a `Buildings.Electrical` example of the Modelica Buildings Library (MBL). It writes
`<stem>.json` and `<stem>.csv` into `--out`, where `<stem>` is the model name without its
leading `Modelica.Electrical.` or `Buildings.Electrical.` (`Analog.Examples.Resistor`,
`DC.Loads.Examples.Resistor`). Standard library only; it never imports `noodl` and is never
run by the test suite. The pure-Python writer needs no OpenModelica, so the contract test
(`tests/apps/electrical/modelica/test_export_contract.py`) imports this file by path.

OpenModelica is driven exactly as by the building export script `scripts/modelica_export.py`,
which this script imports by path and reuses (instance tree, experiment, component and
connection extraction, signals and observers, full-precision Real parameters, JSON and CSV
writers, omc helpers); see its docstring. The differences:

Libraries and simulation
------------------------
`loadModel(Modelica, {"4.1.0"})` and the MBL `package.mo` are loaded for every model. There is
no medium. The model is simulated once with `simulate(<model>, startTime, stopTime,
numberOfIntervals=<--intervals, or the experiment's>, tolerance=<--tolerance, default 1e-12>,
method="dassl", outputFormat="mat", fileNamePrefix="noodl_export", simflags="-noEventEmit
-emit_protected")`; `readSimulationResultVars` must list every compared, initial and derived
name below, else `ExportError` names the missing ones. Every model must start at
`StartTime = 0` (`ExportError` otherwise): initial and derived values are read with
`val(<name>, 0)`. A failed simulation raises `ExportError`; `--no-simulate` writes the JSON
only, with the instance API's six-digit Reals flagged `"approximate"` and no initial or
derived values.

Document
--------
The building document with `"format": "noodl-modelica-electrical/1"`, no `"medium"`, and:
`"msl"` (the MSL version loaded), `"experiment"` (StartTime, StopTime, Interval, Tolerance as
SIMULATED: the interval of `--intervals`, the tolerance of `--tolerance`),
`"experiment_declared"` (the model's own experiment annotation, only the four keys above that
the annotation declares), `"compared"` (the CSV columns after `time`), and per component
`"initial"` and `"derived"` where they apply. Every public parameter is exported, so the
converter flags `ground_AC`/`ground_DC` are in the file for the reader to check. A component
whose `useHeatPort`, `useTemperatureDependency` or `use_C` OpenModelica left unevaluated
raises `ExportError` (these flags decide columns, states and ports; never assumed false).

Class groups (the reader's schema must hold equal sets; the contract tests check it):

- `GROUNDS`: `Modelica.Electrical.Analog.Basic.Ground`.
- `WIRES` (ideal current sensors; their two ends are the same node): MSL `CurrentSensor`, MBL
  DC and AC one-phase `GeneralizedSensor`.
- `STATES` (class -> state variables): `Capacitor` `v`, `Inductor` `i`, `Transformer` `i1`,
  `i2`, `HeatCapacitor` `T`, MBL `TwoPortRCLine` `Vc` (only when its parameter `use_C` is
  true). `val(<inst>.<var>, 0)` of each goes into the component's `"initial"`: the
  reference's own initial state.
- `LINES` (the one-phase and the three-phase balanced MBL `Line`): their inner `line`'s
  simulated `line.R`, `line.L`, `line.C` (`val(<inst>.line.R, 0)`, ...) go into the
  component's `"derived"`, because the one-phase `Line` divides `R` and `L` by 3 while the
  balanced one's redeclaration does not, and only the simulation shows which applies.

Compared columns, per component in document order, skipping observers, wires and grounds:

- class `Modelica.Electrical.Analog.*`: `i` when the class has that variable, else `i1`,
  `i2` (those it has); plus `v` for `Capacitor`; plus `T_heatPort` when the parameter
  `useHeatPort` is true;
- `Modelica.Thermal.HeatTransfer.Components.HeatCapacitor`: `T`; other thermal components
  (conductors, fixed temperatures): none;
- class `Buildings.Electrical.*`: for each connector named `terminal`, `terminal_n`,
  `terminal_p` (in that order): a type name containing `.DC.Interfaces.` gives
  `<c>.v[1]`, `<c>.v[2]`, `<c>.i[1]`; one containing `.AC.` gives `<c>.v[1]`, `<c>.v[2]`,
  `<c>.i[1]`, `<c>.i[2]`; plus `Vc` for a `TwoPortRCLine` with `use_C`.

CSV
---
`# key: value` header lines (model, mbl_commit, openmodelica, msl, tolerance, start_time,
stop_time, interval, solver), then `time` and the `compared` columns, every number in
Python's shortest round-trip form of OpenModelica's double.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_BASE_PATH = Path(__file__).resolve().parent / "modelica_export.py"
_spec = importlib.util.spec_from_file_location("noodl_modelica_export_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

ExportError = base.ExportError
MBL_COMMIT_EXPECTED = base.MBL_COMMIT_EXPECTED

FORMAT = "noodl-modelica-electrical/1"

GROUNDS = frozenset({"Modelica.Electrical.Analog.Basic.Ground"})
WIRES = frozenset({
    "Modelica.Electrical.Analog.Sensors.CurrentSensor",
    "Buildings.Electrical.DC.Sensors.GeneralizedSensor",
    "Buildings.Electrical.AC.OnePhase.Sensors.GeneralizedSensor",
})
RC_LINE = "Buildings.Electrical.DC.Lines.TwoPortRCLine"
CAPACITOR = "Modelica.Electrical.Analog.Basic.Capacitor"
HEAT_CAPACITOR = "Modelica.Thermal.HeatTransfer.Components.HeatCapacitor"
STATES = {
    CAPACITOR: ("v",),
    "Modelica.Electrical.Analog.Basic.Inductor": ("i",),
    "Modelica.Electrical.Analog.Basic.Transformer": ("i1", "i2"),
    HEAT_CAPACITOR: ("T",),
    RC_LINE: ("Vc",),
}
LINES = frozenset({
    "Buildings.Electrical.AC.OnePhase.Lines.Line",
    "Buildings.Electrical.AC.ThreePhasesBalanced.Lines.Line",
})
LINE_DERIVED = ("line.R", "line.L", "line.C")
TERMINALS = ("terminal", "terminal_n", "terminal_p")
_PREFIXES = ("Modelica.Electrical.", "Buildings.Electrical.")
_EXPERIMENT_KEYS = ("StartTime", "StopTime", "Interval", "Tolerance")
# Boolean parameters that decide columns, states or ports; each must have a value.
FLAGS = ("useHeatPort", "useTemperatureDependency", "use_C")


# =====================================================================================
# Pure-Python writer (no OpenModelica needed)
# =====================================================================================
def stem(model: str) -> str:
    """The output file stem: `model` without its leading library package."""
    for prefix in _PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def check_start_time(experiment: dict) -> None:
    """Refuse a start time other than 0 (initial and derived values are read at t = 0)."""
    t0 = float(experiment.get("StartTime", 0.0))
    if t0 != 0.0:
        raise ExportError(f"StartTime = {t0!r}: only models starting at 0 are exported "
                          f"(initial and derived values are read at t = 0)")


def simulated_experiment(experiment: dict, tolerance: float,
                         intervals: int | None) -> tuple[dict, int, dict]:
    """`(experiment as simulated, numberOfIntervals, declared experiment)`.

    `experiment` is what the building script's `introspect` returns: `values` (the evaluated
    settings), `numberOfIntervals` and `declared` (the annotation's modifier names).
    """
    v = experiment["values"]
    n = int(intervals) if intervals is not None else int(experiment["numberOfIntervals"])
    interval = (v["Interval"] if intervals is None
                else (float(v["StopTime"]) - float(v["StartTime"])) / n)
    simulated = {"StartTime": float(v["StartTime"]), "StopTime": float(v["StopTime"]),
                 "Interval": float(interval), "Tolerance": float(tolerance)}
    declared = {k: v[k] for k in _EXPERIMENT_KEYS if k in experiment["declared"]}
    return simulated, n, declared


def build_circuit_document(instance: dict, *, model: str, experiment: dict,
                           experiment_declared: dict, mbl_commit: str, openmodelica: str,
                           msl: str) -> dict:
    """The `noodl-modelica-electrical/1` document for one `getModelInstance` tree."""
    doc = base.build_document(instance, model=model, medium={}, experiment=experiment,
                              mbl_commit=mbl_commit, openmodelica=openmodelica)
    unresolved = [f"{c['name']}.{f}" for c in doc["components"]
                  for f in FLAGS if f in c.get("unevaluated", [])]
    if unresolved:
        raise ExportError(f"OpenModelica gave no value for {unresolved}; these flags decide "
                          f"columns and ports and are never assumed false")
    return {
        "format": FORMAT,
        "model": doc["model"],
        "mbl_commit": doc["mbl_commit"],
        "openmodelica": doc["openmodelica"],
        "msl": msl,
        "experiment": doc["experiment"],
        "experiment_declared": dict(experiment_declared),
        "components": doc["components"],
        "connections": doc["connections"],
        "signals": doc["signals"],
    }


def _active(comp: dict) -> bool:
    return (comp.get("role") != "observer" and comp["class"] not in WIRES
            and comp["class"] not in GROUNDS)


def _connector_types(type_: dict) -> dict[str, str]:
    return {el["name"]: base._type_name(el.get("type")) for el, _ in base._walk_elements(type_)
            if el.get("$kind") == "component"}


def compared_variables(instance: dict, doc: dict) -> list[str]:
    """CSV columns after `time` (rule in the module docstring)."""
    by_name = {c["name"]: c for c in base._model_components(instance)}
    columns: list[str] = []
    for comp in doc["components"]:
        if not _active(comp):
            continue
        name, cls, params = comp["name"], comp["class"], comp["parameters"]
        type_ = by_name[name]["type"]
        if cls.startswith("Modelica.Electrical.Analog."):
            variables = base._variables_of(type_)
            if "i" in variables:
                columns.append(f"{name}.i")
            else:
                columns += [f"{name}.{v}" for v in ("i1", "i2") if v in variables]
            if cls == CAPACITOR:
                columns.append(f"{name}.v")
            if params.get("useHeatPort") is True:
                columns.append(f"{name}.T_heatPort")
        elif cls == HEAT_CAPACITOR:
            columns.append(f"{name}.T")
        elif cls.startswith("Buildings.Electrical."):
            connectors = _connector_types(type_)
            for terminal in TERMINALS:
                type_name = connectors.get(terminal)
                if type_name is None:
                    continue
                c = f"{name}.{terminal}"
                if ".DC.Interfaces." in type_name:
                    columns += [f"{c}.v[1]", f"{c}.v[2]", f"{c}.i[1]"]
                elif ".AC." in type_name:
                    columns += [f"{c}.v[1]", f"{c}.v[2]", f"{c}.i[1]", f"{c}.i[2]"]
                else:
                    raise ExportError(f"{c}: connector type {type_name!r} is neither an MBL DC "
                                      f"nor an AC terminal")
            if cls == RC_LINE and params.get("use_C") is True:
                columns.append(f"{name}.Vc")
    return columns


def initial_variables(instance: dict, doc: dict) -> dict[str, list[str]]:
    """State variables whose start value is recorded, per component (`STATES`)."""
    del instance  # the class and parameters in `doc` decide
    out: dict[str, list[str]] = {}
    for comp in doc["components"]:
        cls = comp["class"]
        if not _active(comp) or cls not in STATES:
            continue
        if cls == RC_LINE and comp["parameters"].get("use_C") is not True:
            continue
        out[comp["name"]] = list(STATES[cls])
    return out


def derived_variables(doc: dict) -> dict[str, list[str]]:
    """The simulated inner-line values recorded for every MBL `Line` (`LINES`)."""
    return {c["name"]: list(LINE_DERIVED) for c in doc["components"] if c["class"] in LINES}


def _result_names(group: dict[str, list[str]]) -> list[str]:
    return [f"{name}.{var}" for name, variables in group.items() for var in variables]


def record_values(doc: dict, initial: dict[str, list[str]], derived: dict[str, list[str]],
                  values: dict[str, float]) -> None:
    """Write the read-back values into each component's `"initial"` and `"derived"`."""
    entries = {c["name"]: c for c in doc["components"]}
    for key, group in (("initial", initial), ("derived", derived)):
        for name, variables in group.items():
            entries[name][key] = {v: values[f"{name}.{v}"] for v in variables}


def csv_header(*, model: str, mbl_commit: str, openmodelica: str, msl: str,
               experiment: dict) -> dict[str, str]:
    """The CSV's `# key: value` header lines, from the experiment as simulated."""
    return {
        "model": model,
        "mbl_commit": mbl_commit,
        "openmodelica": openmodelica,
        "msl": msl,
        "tolerance": repr(float(experiment["Tolerance"])),
        "start_time": repr(float(experiment["StartTime"])),
        "stop_time": repr(float(experiment["StopTime"])),
        "interval": repr(float(experiment["Interval"])),
        "solver": "dassl",
    }


def check_result_variables(wanted: list[str], available) -> None:
    """Raise `ExportError` naming every name in `wanted` that the result does not hold."""
    have = set(available)
    missing = [w for w in wanted if w not in have]
    if missing:
        raise ExportError(f"variables missing from the simulation result: {missing}")


# =====================================================================================
# OpenModelica driver
# =====================================================================================
def simulate_circuit_model(model: str, mbl: Path, workdir: Path, experiment: dict,
                           tolerance: float, intervals: int) -> list[str]:
    """Simulate `model` once (settings in the module docstring); the result's variable names.

    `experiment` is the simulated experiment (`StartTime`, `StopTime`).
    """
    script = base._load_libs(mbl) + base._echo_block("model_sim", f"""simulate({model},
  startTime={experiment["StartTime"]!r}, stopTime={experiment["StopTime"]!r},
  numberOfIntervals={int(intervals)}, tolerance={float(tolerance)!r}, method="dassl",
  outputFormat="mat", fileNamePrefix="noodl_export",
  simflags="-noEventEmit -emit_protected")""") + """
getErrorString();
names := readSimulationResultVars("noodl_export_res.mat", readParameters=true,
  openmodelicaStyle=true);
print("@@VARS=\\n");
for nm in names loop print(nm + "\\n"); end for;
print("@@ENDVARS\\n");
"""
    out = base._run_omc(script, workdir, "simulate.mos")
    if not (workdir / "noodl_export_res.mat").is_file():
        raise ExportError(f"simulation of {model} failed:\n"
                          f"{base._echoed(out, 'model_sim')[-3000:]}")
    start = out.find("@@VARS=\n")
    end = out.find("@@ENDVARS", start)
    if start < 0 or end < 0:
        raise ExportError(f"omc listed no result variables for {model}:\n{out[-2000:]}")
    return out[start + len("@@VARS=\n"):end].split()


def export(model: str, *, out: Path, mbl: Path, run_simulation: bool = True,
           keep: Path | None = None, tolerance: float = 1e-12,
           intervals: int | None = None) -> dict:
    """Export `model`; returns a summary (paths, settings, timings, CSV rows and size)."""
    t_start = time.perf_counter()
    commit = base._mbl_commit(mbl)
    if commit != MBL_COMMIT_EXPECTED:
        print(f"warning: MBL commit {commit} is not {MBL_COMMIT_EXPECTED}", file=sys.stderr)
    omc = base._omc_version()
    name = stem(model)
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"noodl_export_{name}_"))
    summary: dict = {"model": model}
    try:
        instance, experiment, msl = base.introspect(model, mbl, workdir)
        check_start_time(experiment["values"])
        sim_experiment, n, declared = simulated_experiment(experiment, tolerance, intervals)
        t_intro = time.perf_counter()
        doc = build_circuit_document(instance, model=model, experiment=sim_experiment,
                                     experiment_declared=declared, mbl_commit=commit,
                                     openmodelica=omc, msl=msl)
        columns = compared_variables(instance, doc)
        doc["compared"] = columns
        initial = initial_variables(instance, doc)
        derived = derived_variables(doc)
        refs = base.exact_value_refs(instance, doc)
        matrix = None
        if run_simulation:
            available = simulate_circuit_model(model, mbl, workdir, sim_experiment,
                                               tolerance, n)
            recorded = _result_names(initial) + _result_names(derived)
            check_result_variables(columns + recorded, available)
            have = set(available)
            params = [r for r in refs if r in have]
            _, matrix, values = base.read_results(workdir, [], columns, params + recorded)
            missing = base.apply_exact_values(doc, refs, values)
            record_values(doc, initial, derived, values)
        else:
            missing = base.apply_exact_values(doc, refs, {})
        t_sim = time.perf_counter()
        doc["parameter_precision"] = (
            "Reals from the simulation result (full double precision)" if not missing else
            f"Reals from the simulation result except {len(missing)} value(s) absent from it; "
            f"those keep the instance API's six significant digits (see 'approximate')")
        for root in {mbl.as_posix(), mbl.resolve().as_posix()}:
            doc = base.relative_library_paths(doc, root)
        json_path = out / f"{name}.json"
        base.write_json(doc, json_path)
        summary.update(json=str(json_path), approximate=missing,
                       stop_time=sim_experiment["StopTime"], intervals=n,
                       interval=sim_experiment["Interval"], tolerance=tolerance,
                       introspect_s=round(t_intro - t_start, 1),
                       simulate_s=round(t_sim - t_intro, 1))
        if matrix is not None:
            header = csv_header(model=model, mbl_commit=commit, openmodelica=omc, msl=msl,
                                experiment=sim_experiment)
            csv_path = out / f"{name}.csv"
            rows = base.write_csv(csv_path, header, columns, matrix)
            summary.update(csv=str(csv_path), rows=rows, columns=len(columns),
                           csv_kB=round(csv_path.stat().st_size / 1024, 1))
        summary["total_s"] = round(time.perf_counter() - t_start, 1)
        return summary
    finally:
        if keep is not None:
            shutil.copytree(workdir, keep / workdir.name, dirs_exist_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="fully qualified model name")
    ap.add_argument("--out", default="tests/data/modelica_electrical", type=Path)
    ap.add_argument("--mbl", default=os.path.expanduser("~/modelica/modelica-buildings"),
                    type=Path)
    ap.add_argument("--tolerance", type=float, default=1e-12,
                    help="DASSL tolerance of the reference simulation (default 1e-12)")
    ap.add_argument("--intervals", type=int, default=None,
                    help="number of output intervals (default: the model's experiment)")
    ap.add_argument("--no-simulate", action="store_true",
                    help="write the JSON only (Reals keep six significant digits)")
    ap.add_argument("--keep", type=Path, default=None,
                    help="copy the omc working directory here (debugging)")
    args = ap.parse_args(argv)
    summary = export(args.model, out=args.out, mbl=args.mbl.expanduser(),
                     run_simulation=not args.no_simulate, keep=args.keep,
                     tolerance=args.tolerance, intervals=args.intervals)
    print(json.dumps(summary))
    if not args.no_simulate and "csv" not in summary:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
