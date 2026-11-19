"""The numbers behind each gallery figure are the numbers its page prints.

`scripts/make_example_figures.py` repeats each example's computation to draw it. This test runs
those computations and compares them with the `text` output printed on the example's page, so a
page that changes without its figure (or the reverse) fails here. Only the data are checked,
not the drawing.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "docs" / "examples"


@pytest.fixture(scope="module")
def figures():
    spec = importlib.util.spec_from_file_location(
        "make_example_figures", ROOT / "scripts" / "make_example_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def printed(name: str) -> str:
    text = (EXAMPLES / f"{name}.md").read_text(encoding="utf-8")
    return re.findall(r"```text\n(.*?)```", text, re.S)[-1]


def numbers(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", line)]


def match(values, shown, decimals):
    """The figure's values rounded to the page's precision equal the printed ones."""
    tol = 0.5 * 10 ** -decimals + 1e-9
    assert len(values) == len(shown)
    for v, s in zip(values, shown, strict=True):
        assert abs(v - s) <= tol, (values, shown)


def test_natural_ventilation(figures):
    data = figures.natural_ventilation()
    for line in printed("natural-ventilation").strip().splitlines():
        hour, a, b, door = numbers(line.replace("A ", " ").replace("B ", " "))
        k = int(hour) * 6 - 1
        assert data["hours"][k] == pytest.approx(hour)
        match([data["A"][k], data["B"][k]], [a, b], 1)
        match([data["door"][k]], [door], 2)


def test_calibrate_demands(figures):
    data = figures.calibrate_demands()
    lines = printed("calibrate-demands").strip().splitlines()
    assert data["trace"][0][1] == pytest.approx(numbers(lines[0])[-1], abs=0.06)
    match(data["trace"][-1][0], numbers(lines[2]), 1)
    match(data["truth"], numbers(lines[3]), 1)


def test_water_day(figures):
    data = figures.water_day()
    for scale, line in zip((0.8, 1.0, 1.2), printed("water-day").strip().splitlines(), strict=True):
        match([data[scale][1][-1]], [numbers(line)[1]], 3)


def test_sewer_sulfide(figures):
    data = figures.sewer_sulfide()
    lines = printed("sewer-sulfide").strip().splitlines()
    columns = list(zip(*data["ppm"], strict=True))
    match([max(c) for c in columns], numbers(lines[1].split("[")[1]), 2)
    match(list(data["ppm"][-1]), numbers(lines[2].split("[")[1]), 2)


def test_street_release(figures):
    data = figures.street_release()
    for line in printed("street-release").strip().splitlines():
        degrees, peak, above = re.fullmatch(
            r"wind toward\s+(\d+) deg: street 11\s+([\d.]+) ug/m3, (\d+) streets.*", line).groups()
        degrees, peak, above = int(degrees), float(peak), int(above)
        row = data["ug"][degrees // 15]
        assert sum(v > 1.0 for v in row) == above
        if degrees in (0, 180):
            assert max(row) == pytest.approx(peak, abs=0.06)


def test_street_building(figures):
    data = figures.street_building()
    for hour, line in enumerate(printed("street-building").strip().splitlines()):
        shown = numbers(line.split("[")[1])
        match(data["ratios"][hour], shown, 3)
