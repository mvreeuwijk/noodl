"""Generates the figures of the examples gallery, docs/examples/*.md, in a light and a dark variant.

Run manually after changing an example: `.venv/Scripts/python scripts/make_example_figures.py`
(needs matplotlib, which is not a runtime dependency). Pass names to draw only those figures.

Each figure repeats the computation of its page and plots it, using the public API only. The
output is `docs/assets/examples/<name>.svg` and `<name>-dark.svg`. The pages show the pair
with Material's `#only-light` and `#only-dark` image suffixes. The colours are those of the
hand-written figures (`scripts/make_dark_figures.py`), light value and dark twin.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
OUT = ROOT / "docs" / "assets" / "examples"
F64 = torch.float64

# role -> (light, dark), the palette of scripts/make_dark_figures.py
THEMES = {
    "light": {"bg": "#F7F9FA", "ink": "#1E3A52", "muted": "#7C8A99", "line": "#DCE3E8",
              "a": "#17A398", "b": "#C2703D", "tint": "#D8F0ED"},
    "dark": {"bg": "#262A33", "ink": "#E3EAF0", "muted": "#A3AFBC", "line": "#4A525E",
             "a": "#2EC4B6", "b": "#E68F57", "tint": "#1E4541"},
}


def style(theme: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["bg"], "axes.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"], "axes.edgecolor": theme["line"],
        "axes.labelcolor": theme["ink"], "text.color": theme["ink"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "grid.color": theme["line"], "grid.linewidth": 0.8, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "normal",
        "legend.frameon": False, "lines.linewidth": 2.0,
        "svg.fonttype": "none", "svg.hashsalt": "noodl",
        "font.family": "sans-serif",
    })


def save(fig, name: str, theme_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "-dark" if theme_name == "dark" else ""
    fig.savefig(OUT / f"{name}{suffix}.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


# --- the computations (each repeats its page) -------------------------------------------


def natural_ventilation() -> dict:
    from noodl.apps.building_physics import (
        Zone,
        add_large_opening,
        add_zone,
        build_model,
        initial_state,
        orifice_law_from_edges,
    )
    from noodl.heads import Stack
    from noodl.topology import Network

    t_mean = 283.15
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=t_mean + 5.0))
    add_zone(net, Zone("B", volume=60.0, T0=t_mean + 2.0))
    add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
    net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
    net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)
    model = build_model(
        net, air_laws=[orifice_law_from_edges(net, "airpath")],
        heads=[Stack.from_network(net, "airpath")],
        coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50,
    )
    state = initial_state(model)
    heater = torch.zeros(net.n, dtype=F64)
    heater[net.node_index("A")] = 300.0
    dt = 600.0
    hours, a, b, out, door = [], [], [], [], []
    for k in range(144):
        t = (k + 1) * dt
        outdoor = t_mean + 6.0 * math.sin(2.0 * math.pi * (t - 9.0 * 3600.0) / 86400.0)
        drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64),
                   "thermal.x_boundary": torch.tensor([outdoor], dtype=F64),
                   "thermal.sources": heater}
        state = model.step(state, drivers, dt)
        hours.append(t / 3600.0)
        a.append(float(state["thermal.x"][0]) - 273.15)
        b.append(float(state["thermal.x"][1]) - 273.15)
        out.append(outdoor - 273.15)
        door.append(abs(float(state["air.q"][0])))
    return {"hours": hours, "A": a, "B": b, "outdoor": out, "door": door}


def draw_natural_ventilation(data: dict, theme: dict):
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.6),
                                      height_ratios=(3, 1.4), layout="constrained")
    top.plot(data["hours"], data["outdoor"], color=theme["muted"], lw=1.5, ls="--",
             label="outdoors")
    top.plot(data["hours"], data["A"], color=theme["b"], label="room A (heated)")
    top.plot(data["hours"], data["B"], color=theme["a"], label="room B")
    top.set_ylabel("temperature (°C)")
    top.legend(loc="upper left", ncols=3)
    bottom.plot(data["hours"], data["door"], color=theme["ink"])
    bottom.set_ylabel("doorway (kg/s)")
    bottom.set_xlabel("time (h)")
    bottom.set_xlim(0, 24)
    bottom.set_xticks(range(0, 25, 6))
    return fig


def calibrate_demands() -> dict:
    from noodl.apps.water import build_model, initial_drivers, read_epanet_inp, water_steady

    work = Path(tempfile.mkdtemp())
    shutil.copy(DATA / "water" / "Net1.inp", work / "Net1.inp")
    net = read_epanet_inp(work / "Net1.inp")
    model, state, _ = build_model(net)
    water = model.refs.water

    def heads(demand):
        drivers = initial_drivers(model, demand=demand)
        return water.head.named(water_steady(model, state, drivers)[water.head])

    unknown = ["12", "21", "31"]
    sensors = ["11", "12", "13", "22", "23", "32"]
    truth = {"12": 0.012, "21": 0.020, "31": 0.016}
    measured = heads(truth)
    target = torch.stack([measured[s] for s in sensors])
    guess = torch.full((3,), 0.008, dtype=F64, requires_grad=True)

    def misfit(demands):
        h = heads(dict(zip(unknown, demands, strict=True)))
        return (((torch.stack([h[s] for s in sensors]) - target) * 1000.0) ** 2).sum()

    optimiser = torch.optim.LBFGS([guess], max_iter=50, tolerance_grad=1e-14,
                                  tolerance_change=1e-16, line_search_fn="strong_wolfe")
    trace = [([1000 * float(g) for g in guess], float(misfit(guess)))]

    def closure():
        optimiser.zero_grad()
        loss = misfit(guess)
        loss.backward()
        trace.append(([1000 * float(g) for g in guess], float(loss)))
        return loss

    for _ in range(2):
        optimiser.step(closure)
    # one point per distinct evaluation, in order
    return {"unknown": unknown, "truth": [1000 * truth[j] for j in unknown], "trace": trace}


def draw_calibrate_demands(data: dict, theme: dict):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    trace = data["trace"]
    n = range(len(trace))
    colours = [theme["a"], theme["b"], theme["ink"]]
    for i, name in enumerate(data["unknown"]):
        left.plot(n, [t[0][i] for t in trace], color=colours[i], label=f"junction {name}")
        left.axhline(data["truth"][i], color=colours[i], lw=1.0, ls="--")
    left.set_ylim(0, 40)
    left.set_xlabel("misfit evaluation")
    left.set_ylabel("estimated demand (L/s)")
    left.set_title("dashed: true demand", loc="left", color=theme["muted"])
    left.legend(loc="center right")
    right.semilogy(n, [max(t[1], 1e-12) for t in trace], color=theme["ink"])
    right.set_xlabel("misfit evaluation")
    right.set_ylabel("misfit (mm²)")
    return fig


def water_day() -> dict:
    from noodl.apps.water import build_model, read_epanet_inp, water_run

    work = Path(tempfile.mkdtemp())
    shutil.copy(DATA / "water" / "Net1.inp", work / "Net1.inp")
    net = read_epanet_inp(work / "Net1.inp")
    model, _, drivers = build_model(net)
    series = {}
    for scale in (0.8, 1.0, 1.2):
        scaled = {**drivers, "water.sources": drivers["water.sources"] * scale}
        run = water_run(model, drivers=scaled, quality=False)
        series[scale] = ([float(t) / 3600.0 for t in run.time],
                         [float(x) for x in run.tank_level[:, 0]])
    return series


def draw_water_day(data: dict, theme: dict):
    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    colours = {0.8: theme["a"], 1.0: theme["ink"], 1.2: theme["b"]}
    for scale, (hours, level) in data.items():
        ax.plot(hours, level, color=colours[scale], label=f"demand × {scale}")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("tank 2 level (m)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.legend(loc="upper left", ncols=3)
    return fig


def sewer_sulfide() -> dict:
    from noodl.apps.sewer import build_model, to_ppm, tree_steady

    model, state, drivers = build_model(tree_steady())
    base = drivers["inflow"].clone()
    labels = list(model.refs.air_quality.concentration.labels)
    dt = 300.0
    hours, history = [], []
    for step in range(60):
        factor = 1.0 + 0.25 * math.sin(2.0 * math.pi * (step * dt / 3600.0) / 8.0)
        step_drivers = {**drivers, "inflow": base * factor,
                        "bod_in": torch.where(base > 0, 0.30 * factor, 0.0 * base)}
        state = model.step(state, step_drivers, dt)
        history.append([float(v) for v in to_ppm(state["air_quality.x"], drivers["T_head"])])
        hours.append((step + 1) * dt / 3600.0)
    return {"labels": labels, "hours": hours, "ppm": history}


def draw_sewer_sulfide(data: dict, theme: dict):
    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    colours = [theme["a"], theme["b"], theme["ink"], theme["muted"], "#7E57C2"]
    if theme is THEMES["dark"]:
        colours[4] = "#B39DDB"
    for i, label in enumerate(data["labels"]):
        ax.plot(data["hours"], [row[i] for row in data["ppm"]], color=colours[i],
                label=f"manhole {label}")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("headspace H₂S (ppm)")
    ax.set_xlim(0, 5)
    ax.legend(loc="upper left", ncols=3)
    return fig


def street_release() -> dict:
    from noodl.apps.street_aq import build_model, initial_drivers, twelve_street_grid

    net, _ = twelve_street_grid()
    model, state, _ = build_model(net, pblh_floor=False)
    street = model.refs.street
    n = 24
    angle = torch.arange(n, dtype=F64) * (2 * math.pi / n)
    drivers = initial_drivers(model, values={
        street.boundary_concentration: {"atmosphere": 0.0},
        street.sources: {"11": 1.0e-3},
        "U_ref": torch.full((n,), 3.0, dtype=F64), "theta_w": angle,
        "h_abl": torch.full((n,), 1000.0, dtype=F64),
    })
    state = {key: value.expand(n, *value.shape).clone() for key, value in state.items()}
    solved = model.steady(state, drivers)
    ug = (solved[street.concentration] * 1e9)
    segments = [((net.x[s.u], net.y[s.u]), (net.x[s.v], net.y[s.v])) for s in net.streets]
    return {"segments": segments, "ug": ug.tolist()}


def draw_street_release(data: dict, theme: dict):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("noodl", [theme["tint"], theme["a"], theme["b"]])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), layout="constrained")
    norm = LogNorm(vmin=1.0, vmax=4000.0)
    for ax, degrees in zip(axes, (0, 45, 90), strict=True):
        values = data["ug"][degrees // 15]
        lc = LineCollection(data["segments"], linewidths=6, capstyle="round",
                            colors=[cmap(norm(max(v, 1.0))) for v in values])
        ax.add_collection(lc)
        ax.set_xlim(-130, 230)
        ax.set_ylim(-130, 260)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"wind toward {degrees}°", color=theme["ink"])
        rad = math.radians(degrees)
        ax.annotate("", xy=(-90 + 35 * math.cos(rad), 190 + 35 * math.sin(rad)),
                    xytext=(-90, 190), arrowprops={"arrowstyle": "-|>", "color": theme["muted"]})
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label("concentration (µg/m³)")
    bar.outline.set_edgecolor(theme["line"])
    return fig


def street_building() -> dict:
    from noodl.apps.building_physics import project_to_model, read_prj
    from noodl.apps.street_aq import Street, StreetNetwork, street_index
    from noodl.apps.street_aq import build_model as build_street
    from noodl.couple import (
        CONCENTRATION_TO_MASS_FRACTION,
        MATH_RAD_TO_COMPASS_DEG,
        DriverAlias,
        ValueLink,
        union,
    )

    work = Path(tempfile.mkdtemp())
    for f in (DATA / "contam").glob("valThreeZonesWthCtm*"):
        shutil.copy(f, work / f.name)
    x = {"n0": 0.0, "n1": 30.0, "n2": 60.0, "n3": 30.0}
    y = {"n0": 40.0, "n1": 30.0, "n2": 40.0, "n3": 0.0}
    spec = [("r1", "n0", "n1"), ("r2", "n1", "n2"), ("r3", "n3", "n1")]
    net = StreetNetwork(
        streets=[Street(n, a, b, math.hypot(x[b] - x[a], y[b] - y[a]), 2.0, 3.0)
                 for n, a, b in spec], x=x, y=y)
    street_model, street_state, street_drivers = build_street(net, species=("nox",), z_ref=10.0)
    graph = street_model.net
    emissions = torch.zeros(graph.n, dtype=F64)
    for s in net.streets:
        emissions[graph.node_index(s.name)] = 1.0e-7
    street_drivers = {
        **street_drivers, "street.x_boundary": torch.full((1,), 2.0e-8, dtype=F64),
        "street.sources": emissions, "U_ref": torch.tensor(2.5, dtype=F64),
        "theta_w": torch.tensor(0.0, dtype=F64), "h_abl": torch.tensor(1200.0, dtype=F64)}
    street_state = {"street.x": torch.zeros(3, dtype=F64)}
    building = project_to_model(read_prj(work / "valThreeZonesWthCtm-UseApi.prj"))
    link = ValueLink(
        from_model="street", from_key="street.x", from_index=street_index(street_model)["r2"],
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert=CONCENTRATION_TO_MASS_FRACTION, two_way=True)
    wind = [DriverAlias(source=("street", "U_ref"), targets=(("building", "V_met", None),)),
            DriverAlias(source=("street", "theta_w"),
                        targets=(("building", "theta_w", MATH_RAD_TO_COMPASS_DEG),))]
    city, state, drivers = union(
        {"street": (street_model, street_state, street_drivers), "building": building},
        shared=[link, *wind], substeps={"building": 6}, iterate_rtol=1e-8, iterate_max=50)
    ratios = []
    for _ in range(6):
        state = city.step(state, drivers, dt=3600.0)
        street_c = float(state["street"]["street.x"][1])
        ratios.append([float(z) * 1.2 / street_c for z in state["building"]["species.x"][:, 0]])
    return {"hours": list(range(1, 7)), "ratios": ratios}


def draw_street_building(data: dict, theme: dict):
    fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
    colours = [theme["a"], theme["b"], theme["ink"]]
    for i in range(3):
        ax.plot(data["hours"], [r[i] for r in data["ratios"]], color=colours[i], marker="o",
                label=f"zone {i + 1}")
    ax.axhline(1.0, color=theme["muted"], lw=1.0, ls="--")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("indoor / street concentration")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left", ncols=3)
    return fig


FIGURES = {
    "natural-ventilation": (natural_ventilation, draw_natural_ventilation),
    "calibrate-demands": (calibrate_demands, draw_calibrate_demands),
    "water-day": (water_day, draw_water_day),
    "sewer-sulfide": (sewer_sulfide, draw_sewer_sulfide),
    "street-release": (street_release, draw_street_release),
    "street-building": (street_building, draw_street_building),
}


def main(names: list[str]) -> int:
    unknown = sorted(set(names) - set(FIGURES))
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}; known: {', '.join(FIGURES)}",
              file=sys.stderr)
        return 1
    here = os.getcwd()
    for name in names or FIGURES:
        compute, draw = FIGURES[name]
        data = compute()
        os.chdir(here)
        for theme_name, theme in THEMES.items():
            style(theme)
            save(draw(data, theme), name, theme_name)
        print(f"wrote {name}.svg and {name}-dark.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
