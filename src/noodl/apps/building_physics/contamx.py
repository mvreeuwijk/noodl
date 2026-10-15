"""Drive NIST's ContamX engine through `contamxpy` and return results in noodl's order.

Optional dependency: `pip install .[contam]` (Windows x86-64 only; the wheel bundles the
engine, no CONTAM installation is needed). Everything here is a thin driver; no physics.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

import torch

# The result files `run_project` reads back (`_result_files`). ContamX writes them only when
# the project's output flags ask for them, so `_isolated` never copies one into the scratch
# directory.
_RESULT_SUFFIXES = (".ebw", ".age", ".ach")
_HELP = "contamxpy is required for ContamX parity: pip install .[contam] (Windows x86-64 only)"


def _cxlib():
    try:
        from contamxpy import cxLib
    except ImportError as exc:  # also raised when sys.modules["contamxpy"] is None
        raise ImportError(_HELP) from exc
    return cxLib


def _set_airflow_convergence(prj: Path, rcnvg: float, acnvg: float) -> None:
    """Rewrite `afrcnvg`/`afacnvg` in the COPY's `!sim_af` run-control record.

    ContamX stops its airflow Newton iteration at these tolerances (TN 1887r1 Appendix A:
    `sim_af afcalc afmaxi afrcnvg afacnvg afrelax uac Pbldg uPb`; relative and absolute,
    the latter in kg/s). NIST's sample projects carry 1e-5 / 1e-6, which leaves ContamX's own
    path flows unbalanced at the ~1e-6 relative level; a comparison meant to resolve
    anything finer has to ask the reference engine for it. contamxpy exposes no setter, so
    the record is edited in the scratch copy -- never in the fixture.
    """
    lines = prj.read_bytes().decode().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("!sim_af"):
            tok = lines[i + 1].split()
            if len(tok) < 9:
                break
            tok[3], tok[4] = f"{float(rcnvg):.6e}", f"{float(acnvg):.6e}"
            ending = lines[i + 1][len(lines[i + 1].rstrip("\r\n")):]
            lines[i + 1] = " ".join(tok) + ending
            prj.write_bytes("".join(lines).encode())
            return
    raise ValueError(f"contamx: {prj.name} has no '!sim_af' record with 9 fields to edit")


def _named_files(prj: Path) -> list[str]:
    """The weather and contaminant file names the project's run control gives (not `null`).

    Only bare file names are accepted: a name with a directory part would be opened by
    ContamX relative to the scratch directory (or outside it), not beside the project."""
    names = []
    for line in prj.read_text().splitlines():
        body, _, comment = line.partition("!")
        if ("weather file" in comment or "contaminant file" in comment) and body.strip():
            name = body.strip()
            if name.lower() == "null":
                continue
            if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError(
                    f"contamx: {prj.name} names the {comment.strip()} {name!r}; only a bare "
                    f"file name beside the project is supported"
                )
            names.append(name)
    return names


@contextlib.contextmanager
def _isolated(prj_path, airflow_convergence=None):
    """Yield a path to a COPY of the project inside a scratch directory.

    ContamX writes its `.sim`, `.log`, `.ach` and `.xlog` output beside the `.prj` it is
    handed, not into the working directory. Pointing it straight at `tests/data/contam/`
    would therefore drop four untracked files into a tracked fixture directory on every run,
    and two runs of the same project could not proceed concurrently. The copy carries every
    sibling that shares the project's stem, except an earlier run's `.ebw`, `.age` and `.ach`
    (`_RESULT_SUFFIXES`); the `-UseApi` projects need none, because they
    name `null` for the weather, contaminant and values files and take ambient conditions
    from the API instead. A project whose auxiliary files are named differently will not be
    found by the engine, which then refuses the setup and is reported by `_open` -- it is
    never silently simulated without them. The weather and contaminant files the project's
    run control names are copied too, whatever their stem. ContamX opens them relative to
    the process WORKING DIRECTORY, not the project's directory, so `run_project` runs
    inside the scratch directory.
    """
    src = Path(prj_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"contamx: no such project file: {src}")
    with tempfile.TemporaryDirectory(prefix="noodl-contamx-") as tmp:
        for sibling in sorted(src.parent.glob(f"{src.stem}.*")):
            if sibling.suffix.lower() in _RESULT_SUFFIXES:
                continue                       # an earlier run's: never read back as this one's
            shutil.copy2(sibling, Path(tmp) / sibling.name)
        for name in _named_files(src):
            if (src.parent / name).is_file() and not (Path(tmp) / name).exists():
                shutil.copy2(src.parent / name, Path(tmp) / name)
        if airflow_convergence is not None:
            _set_airflow_convergence(Path(tmp) / src.name, *airflow_convergence)
        yield Path(tmp) / src.name


def _open(prj_path, ambient: dict, *, reported_as=None):
    """Set up a ContamX simulation on `prj_path`.

    `reported_as` is the path the CALLER named. `prj_path` is always the scratch COPY
    `_isolated` made (in a temporary directory that is deleted before the caller ever sees
    the exception), so naming it in the refusal message points at a path that no longer
    exists and that the caller never asked for. Report the original.
    """
    cxLib = _cxlib()

    def init(cx):
        cx.setAmbtPressure(float(ambient["Pb"]))
        cx.setAmbtWindSpeed(float(ambient["Ws"]))
        cx.setAmbtWindDirection(float(ambient["Wd"]))
        cx.setAmbtTemperature(float(ambient["Ta"]))
        for i, mf in ambient.get("mf", {}).items():
            cx.setAmbtMassFraction(int(i), float(mf))

    cx = cxLib(str(Path(prj_path)), 0, True, init)
    cx.setVerbosity(0)
    if cx.setupSimulation(1):
        cx.endSimulation()
        raise RuntimeError(
            f"contamx: ContamX refused "
            f"{prj_path if reported_as is None else reported_as} (setupSimulation != 0)"
        )
    return cx


def _snapshot(cx) -> tuple[list[float], list[list[float]], list[float]]:
    """Net path flows [kg/s], zone mass fractions [-], in contamxpy's path and zone order, and
    duct terminal flows [kg/s], out of each terminal into its zone, in junction-number order
    (`getDuctTerminalFlow`; an empty list without terminals).

    `getPathFlow` returns the path's two directional flows; their sum is the net flow, and
    that net is POSITIVE IN THE from_zone -> to_zone DIRECTION -- the same orientation
    `prj.py` gives the edge it builds for the path, so nothing is negated here. The
    convention is measured, not assumed and not tuned to a test outcome; two
    cases whose direction is known before the engine is consulted fix it, both run against
    ContamX 3.4.1.7 via contamxpy 0.0.9:

    * `tests/data/contam/doorway_damper_fan.prj`, path 5, carries flow element 1: an
      `fan_cmf` constant-MASS-flow fan rated 0.200683 kg/s, on a path declared `n# 1` to
      `m# -1`, i.e. zone -> ambient. A fixed-flow fan has no freedom; it must deliver its
      rated flow in the from->to direction. The engine reports `[+0.20068299770355225, 0.0]`
      -- positive, and equal to the rating to eight figures.
    * `tests/data/contam/test_OneZoneWthCtmStack-UseApi.prj` with the ambient at 273.15 K
      around a zone at 293.15 K and no wind. Its two paths both run ambient -> zone, at
      relative heights 0.0 and 1.5 m. Buoyancy must admit cold air at the LOW opening and
      expel warm air at the high one, so path 1 must be positive and path 2 negative. The
      engine reports +0.127148 and -0.127148 kg/s, and reverses both when the ambient is
      warmed to 313.15 K instead.

    `contamxpy`'s `Path` docstring is consistent with this ("from_zone: Number of *From* zone
    used to indicate positive flow direction: from_zone -> to_zone"), but it documents the
    field rather than `getPathFlow`'s sign, so the two measurements above are the evidence.
    """
    flows = [sum(cx.getPathFlow(p.nr)) for p in cx.paths]
    mf = [[cx.getZoneMassFraction(z.nr, c) for c in range(cx.nContaminants)] for z in cx.zones]
    terminals = [cx.getDuctTerminalFlow(i + 1) for i in range(cx.nDuctTerminals)]
    return flows, mf, terminals


def _result(cx, flows, mf, terminals, dt=None) -> dict:
    # `from_zone`/`to_zone` are contamxpy's own numbering, in which AMBIENT IS 0 -- the `.prj`
    # file writes -1 for the same thing, so these do not compare directly with `PrjPath`.
    out = {
        "path_nr": [p.nr for p in cx.paths],
        "from_zone": [p.from_zone for p in cx.paths],
        "to_zone": [p.to_zone for p in cx.paths],
        "zone_nr": [z.nr for z in cx.zones],
        "zone_name": [z.name for z in cx.zones],
        "flow": torch.tensor(flows, dtype=torch.float64),
        "mf": torch.tensor(mf, dtype=torch.float64),
        "terminal_flow": torch.tensor(terminals, dtype=torch.float64),
    }
    if dt is not None:
        out["dt"] = dt
    return out


def _data_rows(path: Path) -> list[list[str]]:
    """The time-step rows of a ContamX result text file: tab-separated `date time ...`."""
    rows = []
    for line in path.read_text().splitlines():
        tok = line.split("\t")
        if len(tok) > 2 and "/" in tok[0] and ":" in tok[1]:
            rows.append(tok)
    return rows


def _result_files(prj) -> dict:
    """The occupant exposure (`.ebw`), age-of-air (`.age`) and building air change rate
    (`.ach`) files ContamX writes next to a project it ran, when the project's output flags
    ask for them (`contamxpy` has no API for these). One row per step, none at t = 0:
    `"exposure"` (N - 1, n_exposures, K) kg/kg, each exposure's step mean in exposure
    order; `"age"` (N - 1, n_zones) h, the local mean age of air in zone-number order;
    `"air_change_rate"` (N - 1,) 1/h, the total column (envelope plus ducts and air
    handlers); `"building_volume"` m3, the conditioned volume the `.ach` header gives.
    A file that is not there gives no key. ContamX writes the exposure file one row per
    step and species, a step's species together (TN 1887), so a step's rows form its
    (n_exposures, K) block. The values are printed to 7 significant digits.

    Refused by name (ValueError): a file without its header line or without step rows, and
    a header whose counts (exposures and species, zones) disagree with the rows."""
    prj = Path(prj)
    out: dict = {}
    ebw, age, ach = (prj.with_suffix(s) for s in _RESULT_SUFFIXES)
    if ebw.exists():
        head, rows = _header_and_rows(ebw, 2)
        n_exp, n_species = int(head[0]), int(head[1])
        if any(len(r) != 3 + n_exp for r in rows) or len(rows) % n_species:
            raise ValueError(f"contamx: {ebw.name} declares {n_exp} exposures and "
                             f"{n_species} species, which its rows do not match")
        values = [[float(v) for v in r[3:]] for r in rows]
        table = torch.tensor(values, dtype=torch.float64).reshape(-1, n_species, n_exp)
        out["exposure"] = table.transpose(-1, -2)
    if age.exists():
        head, rows = _header_and_rows(age, 1)
        n_zones = int(head[0])
        if any(len(r) != 2 + n_zones for r in rows):
            raise ValueError(f"contamx: {age.name} declares {n_zones} zones, which its rows "
                             f"do not match")
        out["age"] = torch.tensor([[float(v) for v in r[2:]] for r in rows],
                                  dtype=torch.float64)
    if ach.exists():
        head, rows = _header_and_rows(ach, 6)
        if any(len(r) != 5 for r in rows):
            raise ValueError(f"contamx: {ach.name} rows are not date, time, path, duct, "
                             f"total")
        out["air_change_rate"] = torch.tensor([float(r[4]) for r in rows],
                                              dtype=torch.float64)
        out["building_volume"] = float(head[5])
    return out


def _header_and_rows(path: Path, n_head: int) -> tuple[list[str], list[list[str]]]:
    """A result file's header fields (at least `n_head`) and its step rows; refused by name
    when either is missing (an empty or truncated file)."""
    lines = path.read_text().splitlines()
    head = lines[0].split("\t") if lines else []
    rows = _data_rows(path)
    if len(head) < n_head or not rows:
        missing = "header" if len(head) < n_head else "step rows"
        raise ValueError(f"contamx: {path.name} has no {missing} (an empty or truncated "
                         f"result file)")
    return head, rows


def run_steady(prj_path, *, ambient: dict, airflow_convergence=None) -> dict:
    """Path net flows [kg/s], zone mass fractions and duct terminal flows (`terminal_flow`,
    kg/s, out of each terminal into its zone, in junction-number order) after the initial
    steady-state solve.

    `airflow_convergence=(rcnvg, acnvg)` overrides the project's airflow tolerances in the
    scratch copy (see `_set_airflow_convergence`); `None` runs the project as written.
    """
    with _isolated(prj_path, airflow_convergence) as prj:
        cx = _open(prj, ambient, reported_as=prj_path)
        try:
            return _result(cx, *_snapshot(cx))
        finally:
            cx.endSimulation()


def run_transient(prj_path, *, steps: int, ambient: dict, airflow_convergence=None) -> dict:
    """`steps` steps of the project's own time step; results stacked with the initial state
    first: flow (steps+1, n_paths), mf (steps+1, n_zones, K), terminal_flow
    (steps+1, n_terminals). `airflow_convergence` as in `run_steady`."""
    with _isolated(prj_path, airflow_convergence) as prj:
        cx = _open(prj, ambient, reported_as=prj_path)
        try:
            dt = float(cx.getSimTimeStep())
            f0, m0, j0 = _snapshot(cx)
            flows, mfs, terms = [f0], [m0], [j0]
            for _ in range(int(steps)):
                cx.doSimStep(1)
                f, m, j = _snapshot(cx)
                flows.append(f)
                mfs.append(m)
                terms.append(j)
            return _result(cx, flows, mfs, terms, dt)
        finally:
            cx.endSimulation()


def run_project(prj_path, *, airflow_convergence=None) -> dict:
    """Run the project AS WRITTEN: its own run control (start, end, time step), weather and
    ambient-concentration files and schedules; no ambient value is set through the API.
    Returns the initial state and every step: `time` (N,) s since the start, `flow`
    (N, n_paths) kg/s, `mf` (N, n_zones, K), `terminal_flow` (N, n_terminals): flow out of
    each duct terminal into its zone, kg/s, in junction-number order, `dt`, and the path and
    zone numbering of `run_transient`, and `control` (N, n_reports): the project's named pass
    (split) nodes, in node order, as `getOutputControlValue` reports them: the value the
    network computed for that step ((N, 0) when the project has none). `airflow_convergence`
    as in `run_steady`. When the project's output flags ask for them, the occupants'
    exposure, the age of air and the building air change rate ContamX writes are returned
    too, one row per step (`_result_files`: `exposure`, `age`, `air_change_rate`,
    `building_volume`); a key is absent when its file is.

    ContamX (3.4.1.7) opens the project's weather and contaminant files relative to the
    process's WORKING DIRECTORY, so the run changes into the scratch copy's directory for its
    duration (not thread-safe; nothing in this package runs ContamX concurrently). The
    project-data callback is registered (with nothing to do) because `contamxpy` fills
    `cx.paths`/`cx.zones` only through it.
    """
    cxLib = _cxlib()
    with _isolated(prj_path, airflow_convergence) as prj, contextlib.chdir(prj.parent):
        cx = cxLib(str(prj), 0, True, lambda _cx: None)
        cx.setVerbosity(0)
        if cx.setupSimulation(1):
            cx.endSimulation()
            raise RuntimeError(f"contamx: ContamX refused {prj_path} (setupSimulation != 0)")
        try:
            dt = float(cx.getSimTimeStep())
            d0, t0 = cx.getSimStartDate(), cx.getSimStartTime()
            d1, t1 = cx.getSimEndDate(), cx.getSimEndTime()
            duration = (d1 - d0) * 86400.0 + t1 - t0
            n = round(duration / dt)
            if n < 1 or n * dt != duration:
                raise ValueError(
                    f"contamx: {prj_path} runs {duration} s, not a whole number (>= 1) of "
                    f"{dt} s time steps"
                )
            def control():
                return [cx.getOutputControlValue(i + 1) for i in range(cx.nOutputControls)]

            f, m, j = _snapshot(cx)
            flows, mfs, terms, times, controls = [f], [m], [j], [0.0], [control()]
            for k in range(1, n + 1):
                cx.doSimStep(1)
                t = (cx.getCurrentDayOfYear() - d0) * 86400.0 + cx.getCurrentTimeInSec() - t0
                if abs(t - k * dt) > 1e-6:
                    raise RuntimeError(f"contamx: step {k} reports t = {t} s, not {k * dt} s")
                f, m, j = _snapshot(cx)
                flows.append(f)
                mfs.append(m)
                terms.append(j)
                times.append(t)
                controls.append(control())
            out = _result(cx, flows, mfs, terms, dt)
            out["time"] = torch.tensor(times, dtype=torch.float64)
            out["control"] = torch.tensor(controls, dtype=torch.float64).reshape(n + 1, -1)
        finally:
            cx.endSimulation()
        out.update(_result_files(prj))         # complete only once the simulation ended
    return out
