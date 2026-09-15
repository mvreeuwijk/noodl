"""Drive NIST's ContamX engine through `contamxpy` and return results in tellegen's order.

Optional dependency: `pip install .[contam]` (Windows x86-64 only; the wheel bundles the
engine, no CONTAM installation is needed). Everything here is a thin driver; no physics.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

import torch

_HELP = "contamxpy is required for ContamX parity: pip install .[contam] (Windows x86-64 only)"


def _cxlib():
    try:
        from contamxpy import cxLib
    except ImportError as exc:  # also raised when sys.modules["contamxpy"] is None
        raise ImportError(_HELP) from exc
    return cxLib


@contextlib.contextmanager
def _isolated(prj_path):
    """Yield a path to a COPY of the project inside a scratch directory.

    ContamX writes its `.sim`, `.log`, `.ach` and `.xlog` output beside the `.prj` it is
    handed, not into the working directory. Pointing it straight at `tests/data/contam/`
    would therefore drop four untracked files into a tracked fixture directory on every run,
    and two runs of the same project could not proceed concurrently. The copy carries every
    sibling that shares the project's stem; the `-UseApi` projects need none, because they
    name `null` for the weather, contaminant and values files and take ambient conditions
    from the API instead. A project whose auxiliary files are named differently will not be
    found by the engine, which then refuses the setup and is reported by `_open` -- it is
    never silently simulated without them.
    """
    src = Path(prj_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"contamx: no such project file: {src}")
    with tempfile.TemporaryDirectory(prefix="tellegen-contamx-") as tmp:
        for sibling in sorted(src.parent.glob(f"{src.stem}.*")):
            shutil.copy2(sibling, Path(tmp) / sibling.name)
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


def _snapshot(cx) -> tuple[list[float], list[list[float]]]:
    """Net path flows [kg/s] and zone mass fractions [-], in contamxpy's path and zone order.

    `getPathFlow` returns the path's two directional flows; their sum is the net flow, and
    that net is POSITIVE IN THE from_zone -> to_zone DIRECTION -- the same orientation
    `prj.py` gives the edge it builds for the path, so nothing is negated here. The
    convention is measured, not assumed and not tuned to a test outcome (Ruling R12); two
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
    return flows, mf


def _result(cx, flows, mf, dt=None) -> dict:
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
    }
    if dt is not None:
        out["dt"] = dt
    return out


def run_steady(prj_path, *, ambient: dict) -> dict:
    """Path net flows [kg/s] and zone mass fractions after the initial steady-state solve."""
    with _isolated(prj_path) as prj:
        cx = _open(prj, ambient, reported_as=prj_path)
        try:
            flows, mf = _snapshot(cx)
            return _result(cx, flows, mf)
        finally:
            cx.endSimulation()


def run_transient(prj_path, *, steps: int, ambient: dict) -> dict:
    """`steps` steps of the project's own time step; results stacked with the initial state
    first: flow (steps+1, n_paths), mf (steps+1, n_zones, K)."""
    with _isolated(prj_path) as prj:
        cx = _open(prj, ambient, reported_as=prj_path)
        try:
            dt = float(cx.getSimTimeStep())
            f0, m0 = _snapshot(cx)
            flows, mfs = [f0], [m0]
            for _ in range(int(steps)):
                cx.doSimStep(1)
                f, m = _snapshot(cx)
                flows.append(f)
                mfs.append(m)
            return _result(cx, flows, mfs, dt)
        finally:
            cx.endSimulation()
