"""Reference AC and DC power-flow results for the MATPOWER cases in tests/data/power_flow.

Run ONCE, in a scratch virtual environment that is not noodl's (pandapower and PYPOWER are
never noodl dependencies):

    python -m venv pfvenv
    pfvenv/Scripts/python -m pip install pandapower==3.5.6 PYPOWER==5.1.21 \
        matpowercaseframes==2.1.1 numpy==2.4.6 scipy==1.18.1 pandas==2.3.3
    pfvenv/Scripts/python -I scripts/power_flow_reference.py tests/data/power_flow

For each `<case>.m` it writes `<case>.json` next to it (format
`noodl-power-flow-reference/1`). The stored bus and generator values are pandapower's
(`runpp(init="flat", enforce_q_lims=False, trafo_model="pi")` and `rundcpp`), on the case
converted by `from_ppc`; the stored DC branch flows are PYPOWER's `rundcpf` (pandapower splits
branches into lines and transformers and loses the file order). PYPOWER, a line-by-line port
of MATPOWER, is run on the same parsed case as an independent check: the script refuses to
write a fixture unless the two agree to 1e-10 on every stored bus and generator quantity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

CASES = ["case9", "case14", "case30", "case118", "case9_features"]
AGREE = 1e-10
# Column indices (0-based) of MATPOWER's idx_bus / idx_gen / idx_brch.
BUS_I, BUS_TYPE, VM, VA, BASE_KV = 0, 1, 7, 8, 9
GEN_BUS, PG, QG, GEN_STATUS = 0, 1, 2, 7
BR_STATUS, PF = 10, 13


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _parse(path: Path) -> dict:
    from pandapower.converter.matpower.from_mpc import _m2ppc

    raw = _m2ppc(str(path), None)
    ppc = {k: np.array(raw[k], dtype=float) for k in ("bus", "gen", "branch")}
    ppc["baseMVA"] = float(raw["baseMVA"])
    ppc["version"] = "2"
    # `_m2ppc` shifts every bus number down by one (0-based, as PYPOWER's internal order);
    # the fixtures name buses by their MATPOWER numbers, which these cases keep as 1..n.
    if not np.array_equal(ppc["bus"][:, BUS_I], np.arange(len(ppc["bus"]))):
        raise SystemExit(f"{path}: buses are not numbered 1..n in table order")
    return ppc


def _gen_bus_totals(bus_ids, gen_bus, p, q, on):
    """Per bus carrying at least one in-service generator, in bus-table order."""
    ids, P, Q = [], [], []
    for b in bus_ids:
        sel = (gen_bus == b) & on
        if sel.any():
            ids.append(int(b))
            P.append(float(p[sel].sum()))
            Q.append(float(q[sel].sum()))
    return ids, np.array(P), np.array(Q)


def _pandapower(ppc):
    import pandapower as pp
    from pandapower.converter.pypower.from_ppc import from_ppc

    p = copy.deepcopy(ppc)
    # from_ppc derives impedance bases from baseKV; case14 has baseKV 0 throughout, which
    # makes the conversion divide by zero. Per-unit results do not depend on baseKV.
    p["bus"][p["bus"][:, BASE_KV] == 0, BASE_KV] = 1.0
    net = from_ppc(p, f_hz=60)
    pp.runpp(net, init="flat", enforce_q_lims=False, calculate_voltage_angles=True,
             trafo_model="pi", tolerance_mva=1e-11, max_iteration=50, numba=False)
    vm = net.res_bus.vm_pu.to_numpy()
    va = np.deg2rad(net.res_bus.va_degree.to_numpy())
    bus_ids = ppc["bus"][:, BUS_I]

    def gens():
        gb, gp, gq, on = [], [], [], []
        for table in ("ext_grid", "gen", "sgen"):
            res = net[f"res_{table}"]
            gb += net[table].bus.tolist()
            gp += res.p_mw.tolist()
            gq += res.q_mvar.tolist() if "q_mvar" in res else [0.0] * len(res)
            on += net[table].in_service.tolist()
        # from_ppc numbers pandapower buses by bus-table row
        return (bus_ids[np.array(gb, dtype=int)], np.array(gp), np.array(gq),
                np.array(on, dtype=bool))

    ac = (vm, va, *_gen_bus_totals(bus_ids, *gens()))
    pp.rundcpp(net, trafo_model="pi", calculate_voltage_angles=True, numba=False)
    dc_va = np.deg2rad(net.res_bus.va_degree.to_numpy())
    gb, gp, _, on = gens()
    dc = (dc_va, *_gen_bus_totals(bus_ids, gb, gp, np.zeros_like(gp), on)[:2])
    return ac, dc


def _pypower(ppc):
    from pypower.api import ppoption, rundcpf, runpf

    opt = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-13, PF_MAX_IT=30, ENFORCE_Q_LIMS=0)
    p = copy.deepcopy(ppc)
    slack = p["bus"][:, BUS_TYPE] == 3
    p["bus"][~slack, VM] = 1.0   # flat start; the slack keeps its file angle
    p["bus"][~slack, VA] = 0.0
    r, ok = runpf(p, opt)
    if not ok:
        raise RuntimeError("PYPOWER runpf did not converge")
    bus_ids = r["bus"][:, BUS_I]
    on = r["gen"][:, GEN_STATUS] > 0
    ac = (r["bus"][:, VM], np.deg2rad(r["bus"][:, VA]),
          *_gen_bus_totals(bus_ids, r["gen"][:, GEN_BUS], r["gen"][:, PG], r["gen"][:, QG], on))
    d, ok = rundcpf(copy.deepcopy(ppc), opt)
    if not ok:
        raise RuntimeError("PYPOWER rundcpf did not converge")
    on = d["gen"][:, GEN_STATUS] > 0
    branch_on = d["branch"][:, BR_STATUS] > 0
    dc = (np.deg2rad(d["bus"][:, VA]),
          *_gen_bus_totals(bus_ids, d["gen"][:, GEN_BUS], d["gen"][:, PG],
                           np.zeros(len(on)), on)[:2],
          d["branch"][branch_on, PF])
    return ac, dc


def _agree(name, a, b):
    worst = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)), initial=0.0))
    if worst > AGREE:
        raise SystemExit(f"{name}: pandapower and PYPOWER differ by {worst:.3e} > {AGREE}")
    return worst


def main(folder: str) -> None:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    for case in CASES:
        path = Path(folder) / f"{case}.m"
        ppc = _parse(path)
        base = ppc["baseMVA"]
        (vm, va, gids, pg, qg), (dva, dgids, dpg) = _pandapower(ppc)
        (vm2, va2, gids2, pg2, qg2), (dva2, dgids2, dpg2, dpf) = _pypower(ppc)
        if gids != gids2 or dgids != dgids2:
            raise SystemExit(f"{case}: generator buses differ: {gids} / {gids2}")
        check = {
            "vm": _agree(f"{case} vm", vm, vm2), "va": _agree(f"{case} va", va, va2),
            "pg": _agree(f"{case} pg", pg / base, pg2 / base),
            "qg": _agree(f"{case} qg", qg / base, qg2 / base),
            "dc_va": _agree(f"{case} dc va", dva, dva2),
            "dc_pg": _agree(f"{case} dc pg", dpg / base, dpg2 / base),
        }
        out = {
            "format": "noodl-power-flow-reference/1",
            "case": case,
            "case_sha256_lf": _sha256_lf(path),
            "base_mva": base,
            "engines": {k: version(k) for k in
                        ("pandapower", "PYPOWER", "matpowercaseframes", "numpy", "scipy")},
            "options": {"runpp": {"init": "flat", "enforce_q_lims": False,
                                  "trafo_model": "pi", "tolerance_mva": 1e-11},
                        "pypower": {"PF_TOL": 1e-13, "ENFORCE_Q_LIMS": 0}},
            "pandapower_vs_pypower_max_abs": check,
            "ac": {"bus": [int(b) + 1 for b in ppc["bus"][:, BUS_I]], "vm": vm.tolist(),
                   "va_rad": va.tolist(), "gen_bus": [b + 1 for b in gids],
                   "pg_pu": (pg / base).tolist(), "qg_pu": (qg / base).tolist()},
            "dc": {"va_rad": dva.tolist(), "gen_bus": [b + 1 for b in dgids],
                   "pg_pu": (dpg / base).tolist(), "branch_pf_pu": (dpf / base).tolist()},
        }
        target = path.with_suffix(".json")
        target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"{case}: wrote {target.name}; pandapower vs PYPOWER max abs "
              + ", ".join(f"{k} {v:.1e}" for k, v in check.items()))


if __name__ == "__main__":
    main(sys.argv[1])
