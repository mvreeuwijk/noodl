"""SWMM corpus harness: how much of a folder of SWMM `.inp` files does noodl run, and how
closely does it agree with SWMM?

    python benchmarks/swmm_corpus.py CORPUS_DIR [--window 600] [--timeout 300] [--jobs 4]
                                     [--json out.json] [--limit N] [--pattern GLOB] [--tight]

For every `.inp` under CORPUS_DIR (the corpus is an argument, never part of the repository):

1. ``read_swmm_inp`` then ``build_model`` (air and quality off) -- any ValueError there is a
   refusal BY NAME and is counted as ``refused:<normalised reason>`` (paths, line numbers,
   names and numbers removed so the same reason groups across files); any other exception is
   ``build_error``.
2. simulate `--window` seconds: ``simulate_dynamic_wave`` under DYNWAVE (the file's time
   series, patterns and dry-weather flow reach it as ``model.inflow_schedule``, which the run
   uses by default); under KINWAVE and STEADY the model's resolved state (the existing
   quasi-steady path) at the inflows the schedule gives at the END of the window --
   ``sim_error`` on an exception.
3. run pyswmm on a temporary COPY of the file with its dates cut to the same window (the
   ``FILE`` paths of its time series made absolute, as the copy lives elsewhere), and
   compare node depths and link flows at the end of the window (both in SI): ``ok`` carries
   the maximum depth and flow differences and both wall times; ``swmm_error`` when SWMM
   itself fails.

Both engines report how many steps did not converge within their trials: SWMM's "% of Steps
Not Converging" from its report, noodl's steps that used up the file's MAX_TRIALS
(``out["converged"]``). Where SWMM does not converge it returns its last trial, which at its
default HEAD_TOLERANCE (0.005 ft) and MAX_TRIALS (8) is off its own fixed point by up to
decimetres on some corpus files; ``--tight`` runs BOTH engines on a copy with HEAD_TOLERANCE
1e-12 and MAX_TRIALS 500, so that the comparison measures the model rather than SWMM's
iteration error (``swmm_nonconverging_pct`` then says where SWMM has no converged answer at
all).

Each stage of each file runs in its own subprocess (``python -I``) with a timeout (``timeout``
status), so a pathological or crashing file cannot stall or kill the run. The corpus is
UNTRUSTED DATA: it is only ever read as text and handed to the parsers; nothing from it is
executed, and SWMM runs on a copy in a temporary directory. Results go to a JSON file and a
printed summary of the status categories.

Under KINWAVE the noodl state is quasi-steady (no wave travel), so its agreement with SWMM
after a short window measures how far SWMM still is from steady state as much as the model.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
MARK = "@@RESULT@@"


# ----------------------------------------------------------------------------------------
# workers (run inside a `python -I` subprocess)
# ----------------------------------------------------------------------------------------


def normalise_reason(message: str) -> str:
    """A refusal message with its file path, line number, quoted names and numbers removed,
    so the same reason groups across files."""
    text = " ".join(str(message).split())
    text = re.sub(r"^.*?\.inp:\s*", "", text, flags=re.I)
    text = re.sub(r"^(line \d+( of \[[A-Z_]+\])?:\s*)", "", text)
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'<x>'", text)
    text = re.sub(r"\bline \d+\b", "line <n>", text)
    text = re.sub(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", "<n>", text)
    return text.replace(" (line <n>)", "").split(";")[0][:160]


def _noodl_worker(path: str, window: float) -> dict:
    """Read, build and simulate `path` in noodl; the result dict is printed as JSON."""
    sys.path.insert(0, str(SRC))
    import torch

    torch.set_num_threads(1)
    from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

    out: dict = {}
    t0 = time.perf_counter()
    try:
        net, _, _ = read_swmm_inp(path)
        out["routing"] = net.routing
        model, state, drivers = build_model(net, air=False, quality=False)
    except ValueError as err:
        out.update(status="refused", reason=normalise_reason(err))
        return out
    except Exception as err:
        out.update(status="build_error", reason=normalise_reason(f"{type(err).__name__}: {err}"))
        return out
    out["build_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        if net.routing == "dynamic_wave":
            _, hist = simulate_dynamic_wave(model, state, drivers, duration=window,
                                            record=("water.phi", "water.q"))
            hyd = model.dynamic_wave
            junctions = list(hyd.junction_names)
            nodes = junctions + list(hyd.outfall_names)
            idx = torch.cat([model.junction_idx, hyd.outfall_idx])
            y = hist["water.phi"][-1, idx] - hyd.node_invert[idx]
            out["depths"] = {n: float(y[j]) for j, n in enumerate(nodes)}
            names = list(model.net.nodes)
            out["flows"] = {
                n: [float(hist["water.q"][-1, i]), names[int(hyd.edge_u[i])],
                    names[int(hyd.edge_v[i])]]
                for i, n in enumerate(model.link_names)}
            out["steps"] = int(hist["water.q"].shape[0])
            out["nonconverged_steps"] = int((~hist["converged"]).sum())
        else:
            if net.inflows is not None:        # the schedule at the end of the window
                drivers = dict(drivers, inflow=model.inflow_schedule.driver(
                    list(model.net.nodes))(window))
            resolved = model._apply_closures(state, drivers)
            out["flows"] = {p.name: [float(resolved["sewer.q"][i]), p.u, p.v]
                            for i, p in enumerate(net.pipes)}
            out["link_depths"] = {p.name: float(resolved["sewer.h"][i])
                                  for i, p in enumerate(net.pipes)}
    except Exception as err:
        out.update(status="sim_error", reason=normalise_reason(f"{type(err).__name__}: {err}"))
        return out
    out.update(status="ran", sim_s=time.perf_counter() - t1)
    return out


def _swmm_worker(path: str, qf: float, lf: float) -> dict:
    """Run pyswmm on `path` (already cut to the window); depths (m) and flows (m3/s)."""
    import pyswmm

    out: dict = {}
    t0 = time.perf_counter()
    try:
        with pyswmm.Simulation(path) as sim:
            conn = {k.linkid: list(k.connections) for k in pyswmm.Links(sim)}
            for _ in sim:
                pass
            out["depths"] = {n.nodeid: n.depth * lf for n in pyswmm.Nodes(sim)}
            out["flows"] = {k.linkid: k.flow * qf for k in pyswmm.Links(sim)}
            out["link_depths"] = {k.linkid: k.depth * lf for k in pyswmm.Links(sim)}
            out["conn"] = conn
            out["error"] = float(sim.flow_routing_error)
        rpt = Path(path).with_suffix(".rpt")
        found = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)",
                          rpt.read_text(errors="replace")) if rpt.exists() else None
        out["nonconverging_pct"] = float(found.group(1)) if found else None
    except Exception as err:
        out.update(status="swmm_error", reason=normalise_reason(f"{type(err).__name__}: {err}"))
        return out
    out.update(status="ran", swmm_s=time.perf_counter() - t0)
    return out


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------


def _run_worker(args: list[str], timeout: float) -> dict:
    """Run this script as a worker in an isolated interpreter; its result dict, or a
    ``timeout`` / ``crash`` status."""
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), *args]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": f"more than {timeout:g} s"}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
    return {"status": "crash", "reason": normalise_reason(f"exit {done.returncode}: {tail[0]}")}


_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%b-%d-%Y", "%d/%m/%Y")


def _parse_date(text: str) -> datetime.date:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unreadable date {text!r}")


def _parse_time(text: str) -> datetime.timedelta:
    parts = [float(p) for p in text.split(":")]
    parts += [0.0] * (3 - len(parts))
    return datetime.timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


_FILE_LINE = re.compile(r'^(\s*\S+\s+FILE\s+)(?:"([^"]*)"|(\S+))', re.I)


def absolutise_files(lines: list[str], base: Path | None) -> list[str]:
    """`lines` of an `.inp` with the path of each `[TIMESERIES]` ``name FILE path`` made
    absolute against `base`, the original file's folder (SWMM reads a copy that lives
    elsewhere)."""
    if base is None:
        return lines
    out, section = [], None
    for raw in lines:
        body = raw.split(";", 1)[0]
        if body.strip().startswith("["):
            section = body.strip().strip("[]").upper()
        match = _FILE_LINE.match(body) if section == "TIMESERIES" else None
        if match:
            path = Path(match.group(2) or match.group(3))
            raw = f'{match.group(1)}"{path if path.is_absolute() else base / path}"'
        out.append(raw)
    return out


#: The [OPTIONS] ``--tight`` sets in both engines' copies: each step converged to its fixed point.
TIGHT_OPTIONS = {"HEAD_TOLERANCE": "1e-12", "MAX_TRIALS": "500"}


def cut_to_window(text: str, window: float, base: Path | None = None,
                  options: dict | None = None) -> tuple[str, str]:
    """`(text, flow_units)` of an `.inp` whose START/END (and REPORT_START) are set so the
    run lasts `window` s from the file's own start; `base`, the original file's folder, makes
    the time series ``FILE`` paths absolute; `options` sets further [OPTIONS] (keyword ->
    value text, e.g. `TIGHT_OPTIONS`)."""
    lines = text.splitlines()
    section, found, options_at = None, {}, None
    for i, raw in enumerate(lines):
        body = raw.split(";", 1)[0].strip()
        if body.startswith("["):
            section = body.strip("[]").upper()
            if section == "OPTIONS":
                options_at = i
            continue
        if section == "OPTIONS" and body:
            f = body.split()
            if len(f) >= 2:
                found[f[0].upper()] = (i, f[1])
    if options_at is None:
        raise ValueError("no [OPTIONS] section")
    start = (_parse_date(found["START_DATE"][1]) if "START_DATE" in found
             else datetime.date(2004, 1, 1))
    clock = _parse_time(found["START_TIME"][1]) if "START_TIME" in found else datetime.timedelta()
    begin = datetime.datetime.combine(start, datetime.time()) + clock
    end = begin + datetime.timedelta(seconds=window)
    values = {"START_DATE": begin.strftime("%m/%d/%Y"), "START_TIME": begin.strftime("%H:%M:%S"),
              "REPORT_START_DATE": begin.strftime("%m/%d/%Y"),
              "REPORT_START_TIME": begin.strftime("%H:%M:%S"),
              "END_DATE": end.strftime("%m/%d/%Y"), "END_TIME": end.strftime("%H:%M:%S"),
              **(options or {})}
    for key, value in values.items():
        if key in found:
            lines[found[key][0]] = f"{key} {value}"
        else:
            lines.insert(options_at + 1, f"{key} {value}")
            found = {k: (i + (i > options_at), v) for k, (i, v) in found.items()}
    lines = absolutise_files(lines, base)
    return "\n".join(lines) + "\n", found.get("FLOW_UNITS", (0, "CFS"))[1].upper()


def run_file(path: Path, root: Path, window: float, timeout: float,
             tight: bool = False) -> dict:
    """One corpus file: the noodl worker, then (when noodl ran) the SWMM worker and the
    comparison. With `tight` both run on a copy with `TIGHT_OPTIONS`."""
    with tempfile.TemporaryDirectory(prefix="swmm_corpus_") as tmp:
        return _run_file(path, root, window, timeout, tight, Path(tmp))


def _run_file(path: Path, root: Path, window: float, timeout: float, tight: bool,
              tmp: Path) -> dict:
    """`run_file` in the temporary folder `tmp`."""
    row: dict = {"file": str(path.relative_to(root)), "tight": tight}
    path = path.resolve()
    source = path
    if tight:
        try:
            text, _ = cut_to_window(path.read_text(errors="replace"), window, path.parent,
                                    TIGHT_OPTIONS)
            source = tmp / "noodl.inp"
            source.write_text(text)
        except Exception:
            pass        # noodl reads the file as it is; SWMM's stage reports the cut failing
    noodl = _run_worker(["--noodl", str(source), str(window)], timeout)
    row.update({k: v for k, v in noodl.items() if k not in ("depths", "flows", "link_depths")})
    if noodl.get("status") != "ran":
        row["category"] = (f"refused:{noodl['reason']}" if noodl["status"] == "refused"
                           else noodl["status"] if noodl["status"] != "timeout"
                           else "timeout:noodl")
        return row
    sys.path.insert(0, str(SRC))
    from noodl.apps.sewer.inp import unit_factors
    try:
        text, units = cut_to_window(path.read_text(errors="replace"), window, path.parent,
                                    TIGHT_OPTIONS if tight else None)
        lf, qf = unit_factors(units)
    except Exception as err:
        row.update(category="swmm_error", reason=normalise_reason(err))
        return row
    copy = tmp / "model.inp"
    copy.write_text(text)
    swmm = _run_worker(["--swmm", str(copy), repr(qf), repr(lf)], timeout)
    if swmm.get("status") != "ran":
        row.update(category=("timeout:swmm" if swmm["status"] == "timeout"
                             else "swmm_error"), reason=swmm.get("reason", ""))
        return row
    row["swmm_s"] = swmm["swmm_s"]
    row["swmm_continuity_error"] = swmm["error"]
    row["swmm_nonconverging_pct"] = swmm.get("nonconverging_pct")
    depth_diff, flow_diff, flow_ref = 0.0, 0.0, 0.0
    for name, y in noodl.get("depths", {}).items():
        if name in swmm["depths"]:
            depth_diff = max(depth_diff, abs(y - swmm["depths"][name]))
    for name, y in noodl.get("link_depths", {}).items():
        if name in swmm["link_depths"]:
            depth_diff = max(depth_diff, abs(y - swmm["link_depths"][name]))
    for name, (q, u, v) in noodl["flows"].items():
        if name not in swmm["flows"]:
            continue
        # SWMM reports in the file's orientation; noodl's pipes may be reversed
        sign = 1.0 if swmm["conn"].get(name, [u, v]) == [u, v] else -1.0
        flow_diff = max(flow_diff, abs(q - sign * swmm["flows"][name]))
        flow_ref = max(flow_ref, abs(swmm["flows"][name]))
    row.update(category="ok", max_depth_diff_m=depth_diff, max_flow_diff_m3s=flow_diff,
               max_swmm_flow_m3s=flow_ref)
    return row


#: The bins (m) of `depth_histogram`.
DEPTH_BINS = ((1e-3, "<1 mm"), (1e-2, "1-10 mm"), (5e-2, "1-5 cm"), (0.2, "5-20 cm"),
              (1.0, "0.2-1 m"), (float("inf"), ">1 m"))


def depth_histogram(rows: list[dict]) -> dict[str, int]:
    """How many of the compared `rows` have their maximum depth difference in each of
    `DEPTH_BINS`."""
    counts = {label: 0 for _, label in DEPTH_BINS}
    for r in rows:
        label = next(lb for edge, lb in DEPTH_BINS if r["max_depth_diff_m"] < edge)
        counts[label] += 1
    return counts


def summarise(rows: list[dict]) -> str:
    """The printed summary: status counts, the top refusal reasons and the agreement and
    timing of the runs that completed."""
    counts = collections.Counter(
        "refused" if r["category"].startswith("refused:") else r["category"] for r in rows)
    reasons = collections.Counter(r["category"][len("refused:"):] for r in rows
                                  if r["category"].startswith("refused:"))
    lines = [f"{len(rows)} files"]
    lines += [f"  {k:14s} {v:6d}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    lines.append("top refusal reasons:")
    lines += [f"  {n:6d}  {reason}" for reason, n in reasons.most_common(15)]
    ok = [r for r in rows if r["category"] == "ok"]
    if ok:
        lines.append("max depth difference (files): " + ", ".join(
            f"{label} {n}" for label, n in depth_histogram(ok).items()))
        lines.append(f"ok files ({len(ok)}):")
        for r in sorted(ok, key=lambda r: r["file"]):
            lines.append(
                f"  {r['file']}: {r['routing']} depth diff {r['max_depth_diff_m']:.3g} m, "
                f"flow diff {r['max_flow_diff_m3s']:.3g} m3/s (SWMM max "
                f"{r['max_swmm_flow_m3s']:.3g}), noodl {r['sim_s']:.2f} s, "
                f"SWMM {r['swmm_s']:.3f} s, steps not converging: SWMM "
                f"{r.get('swmm_nonconverging_pct')} %, noodl {r.get('nonconverged_steps')}"
                + (" [quasi-steady vs SWMM's transient]" if r["routing"] == "kinematic_wave"
                   else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry; see the module docstring."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--noodl":
        print(MARK + json.dumps(_noodl_worker(argv[1], float(argv[2]))))
        return 0
    if argv and argv[0] == "--swmm":
        print(MARK + json.dumps(_swmm_worker(argv[1], float(argv[2]), float(argv[3]))))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", type=Path, help="folder searched recursively for .inp files")
    ap.add_argument("--window", type=float, default=600.0, help="simulated seconds")
    ap.add_argument("--timeout", type=float, default=300.0, help="per stage and file, s")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--json", type=Path, default=Path("swmm_corpus.json"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N files")
    ap.add_argument("--pattern", default="*.inp", help="file glob")
    ap.add_argument("--tight", action="store_true",
                    help="both engines at HEAD_TOLERANCE 1e-12, MAX_TRIALS 500")
    args = ap.parse_args(argv)
    files = sorted(p for p in args.corpus.rglob(args.pattern) if p.is_file()
                   and p.suffix.lower() == ".inp")
    if args.limit:
        files = files[:args.limit]
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_file, p, args.corpus, args.window, args.timeout,
                               args.tight)
                   for p in files]
        for i, fut in enumerate(futures, 1):
            rows.append(fut.result())
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", file=sys.stderr, flush=True)
    args.json.write_text(json.dumps({"window_s": args.window, "tight": args.tight,
                                     "files": rows}, indent=1))
    print(summarise(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
