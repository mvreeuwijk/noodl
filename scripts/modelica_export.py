"""Export one Modelica Buildings Library (MBL) model to `noodl-modelica/1` JSON and a CSV.

Run inside WSL, where OpenModelica is installed::

    python3 scripts/modelica_export.py Buildings.Airflow.Multizone.Validation.ThreeRoomsContam \
        [--out tests/data/modelica] [--mbl ~/modelica/modelica-buildings] [--no-simulate]

It writes `<Short>.json` and `<Short>.csv` into `--out`, where `<Short>` is
the last component of the model name. Standard library only; it never imports `noodl` and is
never run by the test suite. The pure-Python writer (`build_document`, `compared_variables`,
`write_json`, `write_csv`) needs no OpenModelica, so the contract test
(`tests/apps/building_physics/modelica/test_export_contract.py`) imports this file by path.

`main` exits non-zero if a requested simulation fails (the JSON is still written, with the
instance API's Reals instead of the simulated values, for inspection); `--no-simulate` always
exits 0, since it never asks for a simulation in the first place.

How OpenModelica is driven
--------------------------
OMPython is not installed in the WSL environment, so the script writes `.mos` scripts and
runs `omc` on them. Four `omc` runs per model, all in a fresh
temporary directory:

1. *Introspection.* `loadModel(Modelica, {"4.1.0"})` (MBL v13 needs MSL 4.1.0; install it
   once with `installPackage(Modelica, "4.1.0")`, which needs no sudo), then
   `loadFile(<mbl>/Buildings/package.mo)`, then `getModelInstance(<model>)` written to a file,
   plus `getSimulationOptions` (the evaluated experiment settings) and
   `getAnnotationNamedModifiers(<model>, "experiment")` (which of them the model declares).
   `getModelInstance` is OpenModelica's instance API (the one OMEdit uses): it returns the
   model's elements AFTER inheritance and redeclaration are applied (so
   `ThreeRoomsContamDiscretizedDoor`'s `dooOpeClo` is a `DoorDiscretizedOperable`, and the
   components and connections of `ThreeRoomsContam` appear under its `extends` node), with
   each component's class, each parameter's binding and, where the front end evaluates it,
   its `value`, and every `connect()`. This one call replaces the older `getComponents`/
   `getNthConnection`/`getNthInheritedClass`/`getComponentModifierValue` walk, which returns
   unevaluated modifier TEXT and does not apply a redeclaration made in an `extends` clause.
2. *Medium.* The medium is the package every fluid component's `redeclare package Medium =`
   modifier names (the model's local `Medium`, possibly declared in a base model). Its
   library class comes from `isShortDefinition`/`getInheritedClasses` on the declaring class;
   its constants from a generated probe model whose parameters are bound to
   `<model>.Medium.p_default`, `T_default`, `X_default`, `nXi`, `extraPropertiesNames`,
   `dStp`, `pStp`, read back with `getModelInstance` (a constant the medium does not declare,
   such as `dStp` in `PerfectGas`, has no value and is left out).
3. *Simulation.* A second probe holding only the medium's Real constants, and the model
   itself: `simulate(<model>, startTime, stopTime, numberOfIntervals, tolerance,
   method="dassl", outputFormat="mat", simflags="-noEventEmit -emit_protected")` with the
   experiment's own settings; then `readSimulationResultVars` lists what the result holds.
   `-emit_protected` is needed because the MBL validation models declare their components
   `protected` (`OneWayFlow`); `-noEventEmit` keeps output rows on the grid.
4. *Read-back* (no libraries loaded): `readSimulationResult` for the compared trajectories
   and `val(<parameter>, 0)` for every Real parameter, echoed by omc in full precision.

Parameter values
----------------
The instance API prints every Real with SIX significant digits, bindings and evaluated
values alike (`parameter Real a = 0.123456789012345` comes back as `0.123457`). So the
instance API supplies the structure, the Integer/Boolean/String/enumeration parameters and
the list of parameters, and every Real parameter value is then re-read from the .mat
simulation result, where OpenModelica stores the values it simulated with in full double
precision. A Real parameter absent from the result (none in the models tried so far) keeps the
six-digit value and is named in its component's `"approximate"` list; the top-level
`"parameter_precision"` field states which case applies. A parameter whose binding the front
end did not evaluate (typically a medium function call such as `rho_default =
Medium.density(...)`) is left out and named in the component's `"unevaluated"` list; the
reader never needs these (it recomputes derived medium quantities itself). Only PUBLIC
parameters are exported, inherited ones included. Nothing is parsed out of `.mo` text.
Enumerations are written as their fully qualified literal
(`"Modelica.Fluid.Types.Dynamics.FixedInitial"`); the reader keeps only the last component.

Components, signals and observers
---------------------------------
Every top-level element of the model (and of its base models) whose class is a `model` or
`block` is exported; plain variables of the model (`Real dP = ...`) are not. A block is a
SIGNAL when one of its outputs reaches, directly or through other blocks, an input of a
non-block component; its `drives` lists every input its outputs are connected to (a string
for one, a list for several). Every other block, and every
`Buildings.Fluid.Sensors.*` instance, is an OBSERVER: listed in `components` with
`"role": "observer"`. `connections` holds every `connect()` whose two ends are components
(signal wiring is carried by `drives` alone, since the reader resolves connection ends
against `components`).

CSV
---
Written by this script from the .mat result (so that one simulation yields both the exact
parameters and the trajectories; OpenModelica's own CSV writer would need a second
compilation and cannot hold parameters): `#` header lines (model, MBL commit, OpenModelica
and MSL versions, tolerance, start, stop, interval, solver), then `time` and one column per
compared variable, every number in Python's shortest round-trip form of OpenModelica's double.
Columns: each one-way flow element's `m_flow`, each four-port element's `m1_flow`/`m2_flow`
(and `mAB_flow`/`mBA_flow` when the class declares them as variables), each volume's `T`,
`p`, `Xi[1]` (when the medium has moisture) and `C[k]` (one per trace substance). If the
simulation fails (a refused model may not simulate), the JSON is still written, with Reals
flagged approximate, and no CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FORMAT = "noodl-modelica/1"

# Class groups this script needs to choose compared variables. They must equal the reader's
# (`noodl.apps.building_physics.modelica.schema`); the contract test checks that.
VOLUMES = frozenset({
    "Buildings.Fluid.MixingVolumes.MixingVolume",
    "Buildings.Fluid.Delays.DelayFirstOrder",
})
ONE_WAY = frozenset({
    "Buildings.Airflow.Multizone.Orifice",
    "Buildings.Airflow.Multizone.EffectiveAirLeakageArea",
    "Buildings.Airflow.Multizone.Point_m_flow",
    "Buildings.Airflow.Multizone.Points_m_flow",
    "Buildings.Airflow.Multizone.Coefficient_V_flow",
    "Buildings.Airflow.Multizone.Coefficient_m_flow",
    "Buildings.Airflow.Multizone.Table_V_flow",
    "Buildings.Airflow.Multizone.Table_m_flow",
})
FOUR_PORT = frozenset({
    "Buildings.Airflow.Multizone.DoorOpen",
    "Buildings.Airflow.Multizone.DoorOperable",
    "Buildings.Airflow.Multizone.DoorDiscretizedOpen",
    "Buildings.Airflow.Multizone.DoorDiscretizedOperable",
    "Buildings.Airflow.Multizone.ZonalFlow_ACS",
    "Buildings.Airflow.Multizone.ZonalFlow_m_flow",
})
OBSERVER_PREFIXES = ("Buildings.Fluid.Sensors.",)

MBL_COMMIT_EXPECTED = "55abf579598ca81cae0a82f337350375958e6722"
MSL_VERSION = "4.1.0"


class ExportError(RuntimeError):
    """Raised when OpenModelica fails or the model cannot be exported faithfully."""


# =====================================================================================
# Pure-Python writer (no OpenModelica needed)
# =====================================================================================
def _type_name(t) -> str:
    return t.get("name", "") if isinstance(t, dict) else str(t or "")


def _restriction(t) -> str:
    return t.get("restriction", "") if isinstance(t, dict) else ""


def _walk_elements(cls: dict):
    """Yield `(element, owner)` for every element of `cls` and, recursively, of its bases.

    Base-class elements come first, in declaration order, like the flattened model.
    """
    for el in cls.get("elements", []) or []:
        if el.get("$kind") == "extends":
            base = el.get("baseClass")
            if isinstance(base, dict):
                yield from _walk_elements(base)
        else:
            yield el, cls


def _walk_connections(cls: dict):
    for el in cls.get("elements", []) or []:
        if el.get("$kind") == "extends" and isinstance(el.get("baseClass"), dict):
            yield from _walk_connections(el["baseClass"])
    yield from cls.get("connections", []) or []


_UNEVALUATED = object()


def _literal(value):
    """A JSON value for an instance-API literal, or `_UNEVALUATED` for an expression."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        out = [_literal(v) for v in value]
        return _UNEVALUATED if any(v is _UNEVALUATED for v in out) else out
    if isinstance(value, dict) and value.get("$kind") == "enum":
        return value["name"]
    return _UNEVALUATED


def parameters_of(type_: dict) -> tuple[dict, list[str]]:
    """The public parameters of one component class: `(values, unevaluated_names)`."""
    values: dict = {}
    unevaluated: list[str] = []
    for el, _owner in _walk_elements(type_):
        if el.get("$kind") != "component":
            continue
        prefixes = el.get("prefixes") or {}
        if prefixes.get("variability") != "parameter" or prefixes.get("public") is False:
            continue
        v = el.get("value") or {}
        result = _UNEVALUATED
        if "value" in v:
            result = _literal(v["value"])
        if result is _UNEVALUATED and "binding" in v:
            result = _literal(v["binding"])
        if result is _UNEVALUATED:
            if "binding" in v:
                unevaluated.append(el["name"])
            continue
        values[el["name"]] = result
    return values, unevaluated


def _variables_of(type_: dict) -> set[str]:
    return {
        el["name"] for el, _ in _walk_elements(type_)
        if el.get("$kind") == "component"
        and (el.get("prefixes") or {}).get("variability") not in ("parameter", "constant")
    }


def _outputs_of(type_: dict) -> set[str]:
    out = set()
    for el, _ in _walk_elements(type_):
        if el.get("$kind") != "component":
            continue
        tn = _type_name(el.get("type"))
        direction = (el.get("prefixes") or {}).get("direction")
        if direction == "output" or (tn.startswith("Modelica.Blocks.Interfaces.")
                                     and tn.endswith("Output")):
            out.add(el["name"])
    return out


def _cref_text(cref: dict) -> str:
    parts = []
    for p in cref.get("parts", []):
        text = p["name"]
        subs = p.get("subscripts")
        if subs:
            if not all(isinstance(s, int) for s in subs):
                raise ExportError(f"connection subscript {subs!r} in {cref!r} is not a literal "
                                  f"integer (for-loop connections are not supported)")
            text += "[" + ",".join(str(s) for s in subs) + "]"
        parts.append(text)
    return ".".join(parts)


def _model_components(instance: dict) -> list[dict]:
    """Top-level components of the model (inherited ones included), model/block classes only."""
    comps = []
    seen = set()
    for el, _owner in _walk_elements(instance):
        if el.get("$kind") != "component":
            continue
        if _restriction(el.get("type")) not in ("model", "block"):
            continue
        if el["name"] in seen:
            raise ExportError(f"component {el['name']!r} appears twice in the instance tree")
        seen.add(el["name"])
        comps.append(el)
    return comps


def build_document(instance: dict, *, model: str, medium: dict, experiment: dict,
                   mbl_commit: str, openmodelica: str, extra: dict | None = None) -> dict:
    """The `noodl-modelica/1` document for one `getModelInstance` tree."""
    comps = _model_components(instance)
    by_name = {c["name"]: c for c in comps}
    cls_of = {n: _type_name(c["type"]) for n, c in by_name.items()}
    is_block = {n: _restriction(c["type"]) == "block" for n, c in by_name.items()}
    outputs = {n: _outputs_of(c["type"]) for n, c in by_name.items() if is_block[n]}

    connections = []
    for conn in _walk_connections(instance):
        a, b = _cref_text(conn["lhs"]), _cref_text(conn["rhs"])
        connections.append((a, b))

    def port_base(ref: str) -> tuple[str, str]:
        inst, _, port = ref.partition(".")
        return inst, re.sub(r"\[.*\]$", "", port.split(".")[0])

    # Direct targets of each block's outputs, in connection order.
    targets: dict[str, list[str]] = {n: [] for n in outputs}
    for a, b in connections:
        for src, dst in ((a, b), (b, a)):
            inst, port = port_base(src)
            if inst in outputs and port in outputs[inst]:
                targets[inst].append(dst)

    observer = {n: cls_of[n].startswith(OBSERVER_PREFIXES) for n in by_name}
    # A block is a signal when it reaches a non-block, non-observer component input.
    signal: dict[str, bool] = {n: False for n in outputs}
    changed = True
    while changed:
        changed = False
        for n in outputs:
            if signal[n]:
                continue
            for t in targets[n]:
                inst = port_base(t)[0]
                if inst not in by_name:
                    continue
                if (not is_block[inst] and not observer[inst]) or signal.get(inst, False):
                    signal[n] = True
                    changed = True
                    break
    for n in outputs:
        if not signal[n]:
            observer[n] = True

    components, signals = [], []
    for n, c in by_name.items():
        params, unevaluated = parameters_of(c["type"])
        entry = {"name": n, "class": cls_of[n], "parameters": params}
        if unevaluated:
            entry["unevaluated"] = unevaluated
        if signal.get(n):
            ts = targets[n]
            entry["drives"] = ts[0] if len(ts) == 1 else list(ts)
            signals.append(entry)
            continue
        if observer[n]:
            entry["role"] = "observer"
        components.append(entry)

    signal_names = {s["name"] for s in signals}
    kept = [[a, b] for a, b in connections
            if port_base(a)[0] not in signal_names and port_base(b)[0] not in signal_names]
    unknown = sorted({port_base(r)[0] for pair in kept for r in pair} - set(by_name))
    if unknown:
        raise ExportError(f"connections reference instances that are not components: {unknown}")

    doc = {
        "format": FORMAT,
        "model": model,
        "mbl_commit": mbl_commit,
        "openmodelica": openmodelica,
        "experiment": dict(experiment),
        "medium": dict(medium),
        "components": components,
        "connections": kept,
        "signals": signals,
    }
    if extra:
        doc.update(extra)
    return doc


def compared_variables(instance: dict, doc: dict) -> list[str]:
    """CSV columns (after `time`): flow-element flows first, then volume states."""
    by_name = {c["name"]: c for c in _model_components(instance)}
    n_c = len(doc["medium"].get("extraPropertiesNames", []))
    n_xi = int(doc["medium"].get("nXi", 0))
    flows, states = [], []
    for comp in doc["components"]:
        name, cls = comp["name"], comp["class"]
        if comp.get("role") == "observer":
            continue
        variables = _variables_of(by_name[name]["type"])
        if cls in ONE_WAY:
            flows.append(f"{name}.m_flow")
        elif cls in FOUR_PORT:
            for v in ("m1_flow", "m2_flow", "mAB_flow", "mBA_flow"):
                if v in variables:
                    flows.append(f"{name}.{v}")
        elif cls in VOLUMES:
            states += [f"{name}.T", f"{name}.p"]
            states += [f"{name}.Xi[{k + 1}]" for k in range(min(n_xi, 1))]
            states += [f"{name}.C[{k + 1}]" for k in range(n_c)]
    return flows + states


_NOT_REAL = ("Integer", "Boolean", "String")


def _real_parameter_names(type_: dict) -> set[str]:
    """Public parameters of a class whose declared type is Real (or a Real-derived type)."""
    return {
        el["name"] for el, _ in _walk_elements(type_)
        if el.get("$kind") == "component"
        and (el.get("prefixes") or {}).get("variability") == "parameter"
        and _type_name(el.get("type")) not in _NOT_REAL
    }


def _flatten(value, index=()):
    if isinstance(value, list):
        for i, v in enumerate(value, start=1):
            yield from _flatten(v, (*index, i))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield index, value


def exact_value_refs(instance: dict, doc: dict) -> dict[str, tuple[str, str, tuple]]:
    """Result-file names of every Real parameter value in `doc`.

    Maps `"<instance>.<param>[i,j]"` to `(instance, param, (i, j))`. The instance API prints
    Reals with six significant digits (`0.123456789012345` comes back as `0.123457`, both as a
    binding and as a value), so every Real is re-read from the simulation result, where
    OpenModelica stores the parameter values it simulated with in full double precision.
    """
    by_name = {c["name"]: c for c in _model_components(instance)}
    refs = {}
    for entry in [*doc["components"], *doc["signals"]]:
        reals = _real_parameter_names(by_name[entry["name"]]["type"])
        for pname, value in entry["parameters"].items():
            if pname not in reals:
                continue
            for index, _ in _flatten(value):
                sub = "[" + ",".join(map(str, index)) + "]" if index else ""
                refs[f"{entry['name']}.{pname}{sub}"] = (entry["name"], pname, index)
    return refs


def apply_exact_values(doc: dict, refs: dict, values: dict[str, float]) -> list[str]:
    """Replace each referenced value by `values[ref]`; returns the refs `values` lacks.

    A parameter with any element missing from `values` keeps the instance-API value and is
    named in its component's `"approximate"` list (six significant digits, see above).
    """
    entries = {e["name"]: e for e in [*doc["components"], *doc["signals"]]}
    missing = []
    for ref, (name, pname, index) in refs.items():
        entry = entries[name]
        if ref not in values:
            missing.append(ref)
            approx = entry.setdefault("approximate", [])
            if pname not in approx:
                approx.append(pname)
            continue
        if not index:
            entry["parameters"][pname] = values[ref]
            continue
        target = entry["parameters"][pname]
        for i in index[:-1]:
            target = target[i - 1]
        target[index[-1] - 1] = values[ref]
    return missing


def relative_library_paths(obj, mbl: str | Path):
    """`obj` with every string naming a file under the MBL checkout `mbl` rewritten as a
    `modelica://` URI relative to the library root.

    OpenModelica evaluates `loadResource("modelica://Buildings/...")` bindings (a weather
    file's `filNam`, say) to absolute paths on the exporting machine; a committed fixture
    must not carry them. The reader never opens these files, so the URI is informative only.
    """
    root = str(mbl).rstrip("/") + "/"
    if isinstance(obj, str):
        return "modelica://" + obj[len(root):] if obj.startswith(root) else obj
    if isinstance(obj, list):
        return [relative_library_paths(v, mbl) for v in obj]
    if isinstance(obj, dict):
        return {k: relative_library_paths(v, mbl) for k, v in obj.items()}
    return obj


def write_json(doc: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def write_csv(out: str | Path, header: dict, columns: list[str],
              series: list[list[float]]) -> int:
    """Write the reference CSV: `# key: value` header lines, then `time` and `columns`.

    `series[0]` is time and `series[k]` the values of `columns[k - 1]`, one entry per output
    instant, as `readSimulationResult` returns them. Numbers are written with Python's
    shortest round-trip representation, so every double is reproduced exactly. Output
    instants must strictly increase (the simulation runs with `-noEventEmit`); anything else
    raises `ExportError`. Returns the number of rows.
    """
    if len(series) != len(columns) + 1 or len({len(s) for s in series}) != 1:
        raise ExportError(f"CSV series do not match the {len(columns)} columns plus time")
    times = series[0]
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ExportError("output instants are not strictly increasing (event rows?)")
    lines = [f"# {k}: {v}" for k, v in header.items()]
    lines.append(",".join(f'"{c}"' for c in ["time", *columns]))
    lines += [",".join(repr(float(s[i])) for s in series) for i in range(len(times))]
    Path(out).write_text("\n".join(lines) + "\n")
    return len(times)


def parse_omc_matrix(text: str) -> list:
    """Parse an OpenModelica array literal such as `{{0.0, 1.0}, {2.5, NaN}}`."""
    body = text.strip().replace("{", "[").replace("}", "]")
    try:
        return json.loads(body)  # json accepts NaN and Infinity
    except json.JSONDecodeError as exc:
        raise ExportError(f"cannot parse OpenModelica array: {text[:200]!r}") from exc


# =====================================================================================
# OpenModelica driver
# =====================================================================================
def _mos_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_omc(script: str, workdir: Path, name: str, timeout: float = 3600) -> str:
    path = workdir / name
    path.write_text(script)
    proc = subprocess.run(["omc", name], cwd=workdir, capture_output=True, text=True,
                          timeout=timeout)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "@@FAIL" in out:
        raise ExportError(f"omc {name} failed (exit {proc.returncode}):\n{out[-4000:]}")
    return out


def _load_libs(mbl: Path) -> str:
    return f"""echo(false);
if not loadModel(Modelica, {{{_mos_str(MSL_VERSION)}}}) then
  print("@@FAIL loadModel(Modelica): " + getErrorString() + "\\n"); exit(1);
end if;
if not loadFile({_mos_str(str(mbl / "Buildings" / "package.mo"))}) then
  print("@@FAIL loadFile(Buildings): " + getErrorString() + "\\n"); exit(1);
end if;
getErrorString();
"""


def _marker(out: str, key: str) -> str:
    m = re.search(rf"^@@{key}=(.*)$", out, flags=re.M)
    if m is None:
        raise ExportError(f"omc output has no {key}:\n{out[-2000:]}")
    return m.group(1).strip()


def _echoed(out: str, key: str) -> str:
    """The value omc echoed between `@@<key>=` and `@@END<key>` (full double precision).

    `String(r, ...)` in a .mos script limits the digits, so values are printed by omc's own
    echo of a statement's result instead; `echo(true)` itself echoes `true` first.
    """
    m = re.search(rf"@@{key}=(?:true)?\s*(.*?)\s*@@END{key}", out, flags=re.S)
    if m is None:
        raise ExportError(f"omc output has no {key}:\n{out[-2000:]}")
    return m.group(1)


def _echo_block(key: str, statement: str) -> str:
    return (f'print("@@{key}=");\necho(true);\n{statement};\necho(false);\n'
            f'print("@@END{key}\\n");\n')


def introspect(model: str, mbl: Path, workdir: Path) -> tuple[dict, dict, str]:
    """`(instance tree, experiment, MSL version)` for `model`."""
    script = _load_libs(mbl) + f"""
s := getModelInstance({model});
if s == "" then print("@@FAIL getModelInstance: " + getErrorString() + "\\n"); exit(1); end if;
writeFile("instance.json", s);
""" + _echo_block("simopts", f"getSimulationOptions({model})") + f"""
names := getAnnotationNamedModifiers({model}, "experiment");
print("@@annotation=");
for nm in names loop print(nm + " "); end for;
print("\\n");
print("@@msl=" + getVersion(Modelica) + "\\n");
"""
    out = _run_omc(script, workdir, "introspect.mos")
    instance = json.loads((workdir / "instance.json").read_text())
    # getSimulationOptions -> (startTime, stopTime, tolerance, numberOfIntervals, interval).
    fields = _echoed(out, "simopts").strip("()").split(",")
    if len(fields) != 5:
        raise ExportError(f"unexpected getSimulationOptions output:\n{out[-2000:]}")
    t0, t1, tol, n_intervals, dt = fields
    experiment = {"StartTime": float(t0), "StopTime": float(t1), "Interval": float(dt),
                  "Tolerance": float(tol)}
    declared = _marker(out, "annotation").split()
    return instance, {"values": experiment, "numberOfIntervals": int(n_intervals),
                      "declared": declared}, _marker(out, "msl")


def medium_reference(instance: dict) -> str:
    """The package name every fluid component redeclares `Medium` to (must be one)."""
    names = set()
    for comp in _model_components(instance):
        mods = comp.get("modifiers") or {}
        med = mods.get("Medium") if isinstance(mods, dict) else None
        if isinstance(med, dict):
            value = med.get("$value", med)
            if isinstance(value, dict) and value.get("baseClass"):
                names.add(value["baseClass"])
    if len(names) != 1:
        raise ExportError(f"expected exactly one medium package, found {sorted(names)}")
    return names.pop()


_PROBE = """model NoodlMediumProbe
  package M = {ref};
  parameter Real p_default = M.p_default;
  parameter Real T_default = M.T_default;
  parameter Real X_default[M.nX] = M.X_default;
  parameter Integer nXi = M.nXi;
  parameter String extraPropertiesNames[M.nC] = M.extraPropertiesNames;
  parameter Real dStp = M.dStp;
  parameter Real pStp = M.pStp;
end NoodlMediumProbe;"""
_PROBE_REALS = ("p_default", "T_default", "dStp", "pStp")


def model_classes(instance: dict) -> list[str]:
    """The model's class name followed by every base class, depth first (declaration order)."""
    names = [instance["name"]]
    for el in instance.get("elements", []) or []:
        if el.get("$kind") == "extends" and isinstance(el.get("baseClass"), dict):
            names += model_classes(el["baseClass"])
    return names


def probe_medium(instance: dict, mbl: Path, workdir: Path, medium_ref: str) -> dict:
    """The medium's library class and its constants as the instance API gives them.

    Returns `{"class", "ref", "instance"}`; `instance` is the probe model's instance tree, in
    which a constant the medium does not declare (`dStp` in `PerfectGas`) has no value. A
    model-local medium (`package Medium = Buildings.Media.Air(...)`, possibly declared in a
    base model, as `ThreeRoomsContamDiscretizedDoor` inherits it) is resolved to its library
    class with `isShortDefinition`/`getInheritedClasses` on the class that declares it.
    """
    model = instance["name"]
    full_ref = medium_ref if "." in medium_ref else f"{model}.{medium_ref}"
    probe = _PROBE.format(ref=full_ref)
    script = _load_libs(mbl)
    if "." in medium_ref:
        script += f'print("@@medium_base={medium_ref}\\n");\n'
    else:
        for i, cls in enumerate(model_classes(instance)):
            script += f"""{"if" if i == 0 else "elseif"} isShortDefinition({cls}.{medium_ref}) then
  bases := getInheritedClasses({cls}.{medium_ref});
  print("@@medium_base=" + typeNameString(bases[1]) + "\\n");
  print("@@medium_base_short=" + String(isShortDefinition(bases[1])) + "\\n");
"""
        script += 'end if;\n'
    script += f"""
if not loadString({_mos_str(probe)}) then
  print("@@FAIL probe: " + getErrorString() + "\\n"); exit(1);
end if;
writeFile("medium.json", getModelInstance(NoodlMediumProbe));
getErrorString();
"""
    out = _run_omc(script, workdir, "medium.mos")
    if "@@medium_base=" not in out:
        raise ExportError(f"medium {medium_ref!r} is not a short class definition in "
                          f"{model_classes(instance)}")
    base = _marker(out, "medium_base")
    if "@@medium_base_short=true" in out:
        raise ExportError(f"medium {full_ref} is an alias of an alias ({base}); not supported")
    return {"class": base, "ref": full_ref,
            "instance": json.loads((workdir / "medium.json").read_text())}


def _probe_sim_model(probe_info: dict) -> str:
    """A probe model holding only the Real constants the medium declares, for simulation."""
    lines = ["model NoodlMediumReals", f"  package M = {probe_info['ref']};"]
    for el in probe_info["instance"].get("elements", []):
        v = (el.get("value") or {}).get("value", _UNEVALUATED)
        if v is _UNEVALUATED:
            continue
        if el["name"] in _PROBE_REALS:
            lines.append(f"  parameter Real {el['name']} = M.{el['name']};")
        elif el["name"] == "X_default":
            lines.append("  parameter Real X_default[M.nX] = M.X_default;")
    lines.append("end NoodlMediumReals;")
    return "\n".join(lines)


def simulate(model: str, mbl: Path, workdir: Path, experiment: dict, probe_info: dict,
             run_model: bool) -> tuple[list[str] | None, str]:
    """Simulate the medium's Real-constants probe and (if `run_model`) the model.

    Returns `(result variable names, or None if the model was not simulated, omc output)`.
    The model is simulated once with the experiment's settings, `outputFormat="mat"` (the
    .mat result holds every parameter in full precision besides the trajectories),
    `-emit_protected` (the MBL validation models declare their components `protected`, e.g.
    `OneWayFlow`) and `-noEventEmit` (output rows on the grid only).
    """
    script = _load_libs(mbl) + f"""
if not loadString({_mos_str(_probe_sim_model(probe_info))}) then
  print("@@FAIL probe: " + getErrorString() + "\\n"); exit(1);
end if;
""" + _echo_block("probe_sim", 'simulate(NoodlMediumReals, stopTime=1, numberOfIntervals=1, '
                  'outputFormat="mat", fileNamePrefix="noodl_probe")')
    if run_model:
        v = experiment["values"]
        script += _echo_block("model_sim", f"""simulate({model}, startTime={v["StartTime"]!r},
  stopTime={v["StopTime"]!r}, numberOfIntervals={experiment["numberOfIntervals"]},
  tolerance={v["Tolerance"]!r}, method="dassl", outputFormat="mat",
  fileNamePrefix="noodl_export", simflags="-noEventEmit -emit_protected")""") + """
getErrorString();
names := readSimulationResultVars("noodl_export_res.mat", readParameters=true,
  openmodelicaStyle=true);
print("@@VARS=\\n");
for nm in names loop print(nm + "\\n"); end for;
print("@@ENDVARS\\n");
"""
    out = _run_omc(script, workdir, "simulate.mos")
    if not (workdir / "noodl_probe_res.mat").is_file():
        raise ExportError(f"medium probe simulation failed:\n{_echoed(out, 'probe_sim')}")
    names = None
    if run_model and (workdir / "noodl_export_res.mat").is_file():
        m = re.search(r"@@VARS=\n(.*?)@@ENDVARS", out, flags=re.S)
        names = m.group(1).split() if m else None
    return names, out


def read_results(workdir: Path, probe_names: list[str], series: list[str] | None,
                 params: list[str]) -> tuple[dict, list | None, dict]:
    """Read full-precision numbers back from the .mat results (one omc run, no libraries).

    Returns `(probe values, series matrix or None, parameter values)`; `val(p, t0)` reads
    each parameter, `readSimulationResult` the compared trajectories.
    """
    script = "echo(false);\n"
    for i, name in enumerate(probe_names):
        script += _echo_block(f"probe{i}", f'val({name}, 0.0, "noodl_probe_res.mat")')
    if series is not None:
        script += 'n := readSimulationResultSize("noodl_export_res.mat");\n'
        script += _echo_block("series", 'readSimulationResult("noodl_export_res.mat", {'
                              + ", ".join(["time", *series]) + "}, n)")
        for i, name in enumerate(params):
            script += _echo_block(f"param{i}", f'val({name}, 0.0, "noodl_export_res.mat")')
    script += 'print("@@errors=" + getErrorString() + "\\n");\n'
    out = _run_omc(script, workdir, "read.mos")
    probe = {n: float(_echoed(out, f"probe{i}")) for i, n in enumerate(probe_names)}
    matrix = parse_omc_matrix(_echoed(out, "series")) if series is not None else None
    values = {n: float(_echoed(out, f"param{i}")) for i, n in enumerate(params)}
    for name, v in [*probe.items(), *values.items()]:
        if v != v:  # NaN: val() found no such variable
            raise ExportError(f"OpenModelica returned NaN for {name}")
    return probe, matrix, values


def medium_from_probe(probe_info: dict, reals: dict) -> dict:
    """The JSON `medium` object: strings/integers from the probe tree, Reals exact."""
    medium: dict = {"class": probe_info["class"]}
    for el in probe_info["instance"].get("elements", []):
        v = (el.get("value") or {}).get("value", _UNEVALUATED)
        if v is _UNEVALUATED or _literal(v) is _UNEVALUATED:
            continue  # the medium does not declare this constant (e.g. dStp in PerfectGas)
        value = _literal(v)
        name = el["name"]
        if name in _PROBE_REALS:
            value = reals[name]
        elif name == "X_default":
            value = [reals[f"X_default[{i + 1}]"] for i in range(len(value))]
        medium[name] = value
    for key in ("p_default", "T_default", "X_default", "nXi", "extraPropertiesNames"):
        if key not in medium:
            raise ExportError(f"medium {probe_info['ref']}: OpenModelica gave no value for {key}")
    return medium


def _probe_real_names(probe_info: dict) -> list[str]:
    names = []
    for el in probe_info["instance"].get("elements", []):
        v = (el.get("value") or {}).get("value", _UNEVALUATED)
        if v is _UNEVALUATED:
            continue
        if el["name"] in _PROBE_REALS:
            names.append(el["name"])
        elif el["name"] == "X_default":
            names += [f"X_default[{i + 1}]" for i in range(len(v))]
    return names


def _omc_version() -> str:
    return subprocess.run(["omc", "--version"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _mbl_commit(mbl: Path) -> str:
    return subprocess.run(["git", "-C", str(mbl), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def export(model: str, out: Path, mbl: Path, run_simulation: bool = True,
           keep: Path | None = None) -> dict:
    """Export `model`; returns a summary (paths, timings, CSV rows, precision notes)."""
    t_start = time.perf_counter()
    commit = _mbl_commit(mbl)
    if commit != MBL_COMMIT_EXPECTED:
        print(f"warning: MBL commit {commit} is not {MBL_COMMIT_EXPECTED}", file=sys.stderr)
    omc = _omc_version()
    short = model.rsplit(".", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"noodl_export_{short}_"))
    summary: dict = {"model": model}
    try:
        instance, experiment, msl = introspect(model, mbl, workdir)
        t_intro = time.perf_counter()
        probe_info = probe_medium(instance, mbl, workdir, medium_reference(instance))
        t_probe = time.perf_counter()
        result_names, sim_out = simulate(model, mbl, workdir, experiment, probe_info,
                                         run_simulation)
        t_sim = time.perf_counter()
        # A provisional document (instance-API values) fixes the compared variables and the
        # Real parameters to re-read; nXi/nC come from the probe tree.
        provisional = medium_from_probe(probe_info, _ProvisionalReals(probe_info))
        extra = {"modelica_standard_library": msl,
                 "experiment_declared": experiment["declared"]}
        kwargs = dict(model=model, experiment=experiment["values"], mbl_commit=commit,
                      openmodelica=omc, extra=extra)
        doc = build_document(instance, medium=provisional, **kwargs)
        variables = compared_variables(instance, doc)
        refs = exact_value_refs(instance, doc)
        simulated = result_names is not None
        if run_simulation and not simulated:
            print(f"warning: simulation of {model} failed; JSON only, Reals approximate\n"
                  f"{_echoed(sim_out, 'model_sim')[-3000:]}", file=sys.stderr)
        available = set(result_names or [])
        if simulated:
            absent = [v for v in variables if v not in available]
            if absent:
                raise ExportError(f"compared variables missing from the result: {absent}")
        params = [r for r in refs if r in available]
        probe_vals, matrix, values = read_results(
            workdir, _probe_real_names(probe_info), variables if simulated else None, params)
        doc["medium"] = medium_from_probe(probe_info, probe_vals)
        missing = apply_exact_values(doc, refs, values)
        doc["parameter_precision"] = (
            "Reals from the simulation result (full double precision)" if not missing else
            f"Reals from the simulation result except {len(missing)} value(s) absent from it; "
            f"those keep the instance API's six significant digits (see 'approximate')")
        for root in {mbl.as_posix(), mbl.resolve().as_posix()}:
            doc = relative_library_paths(doc, root)
        write_json(doc, out / f"{short}.json")
        summary.update(json=str(out / f"{short}.json"), approximate=missing,
                       introspect_s=round(t_intro - t_start, 1),
                       medium_s=round(t_probe - t_intro, 1),
                       simulate_s=round(t_sim - t_probe, 1))
        if simulated:
            v = experiment["values"]
            header = {
                "model": model,
                "mbl_commit": commit,
                "openmodelica": omc,
                "modelica_standard_library": msl,
                "tolerance": repr(v["Tolerance"]),
                "start_time": repr(v["StartTime"]),
                "stop_time": repr(v["StopTime"]),
                "interval": repr(v["Interval"]),
                "solver": "dassl",
                "result": "OpenModelica .mat result (simflags -noEventEmit -emit_protected), "
                          "read with readSimulationResult",
            }
            rows = write_csv(out / f"{short}.csv", header, variables, matrix)
            summary.update(csv=str(out / f"{short}.csv"), rows=rows, columns=len(variables))
        summary["total_s"] = round(time.perf_counter() - t_start, 1)
        return summary
    finally:
        if keep is not None:
            shutil.copytree(workdir, keep / workdir.name, dirs_exist_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)


class _ProvisionalReals(dict):
    """Probe Reals as the instance API printed them, before the exact values are read."""

    def __init__(self, probe_info: dict) -> None:
        super().__init__()
        for el in probe_info["instance"].get("elements", []):
            v = (el.get("value") or {}).get("value", _UNEVALUATED)
            if isinstance(v, list):
                for i, x in enumerate(v):
                    self[f"{el['name']}[{i + 1}]"] = x
            elif v is not _UNEVALUATED:
                self[el["name"]] = v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="fully qualified model name")
    ap.add_argument("--out", default="tests/data/modelica", type=Path)
    ap.add_argument("--mbl", default=os.path.expanduser("~/modelica/modelica-buildings"),
                    type=Path)
    ap.add_argument("--no-simulate", action="store_true",
                    help="write the JSON only (Reals keep six significant digits)")
    ap.add_argument("--keep", type=Path, default=None,
                    help="copy the omc working directory here (debugging)")
    args = ap.parse_args(argv)
    summary = export(args.model, args.out, args.mbl.expanduser(),
                     run_simulation=not args.no_simulate, keep=args.keep)
    print(json.dumps(summary))
    # A requested simulation that failed still gets a JSON-only export ("warning: simulation
    # of ... failed" above, on stderr) but no "csv" key in the summary; a batch script over
    # many models should not treat that as success.
    if not args.no_simulate and "csv" not in summary:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
