"""Shared fixtures: small typed networks in float64 for exact linear-algebra tests."""

from __future__ import annotations

import os

import pytest
import torch

from benchmarks.composed_model import build_composed
from noodl.elements import PowerLaw
from noodl.topology import Network


@pytest.fixture
def triangle() -> Network:
    """Three nodes, one directed loop: a->b->c->a, all edges kind "airpath", float64."""
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    net.add_edge("c", "a", kind="airpath")
    return net


@pytest.fixture
def two_zone() -> Network:
    """Ambient plus two zones in series: ambient->z1->z2->ambient, kind "airpath", float64."""
    net = Network(dtype=torch.float64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    return net


@pytest.fixture
def two_zone_layer():
    """ambient (boundary) -> z1 -> z2 -> ambient, one PowerLaw airpath element.

    Edges (kind="airpath", insertion order): ambient->z1, z1->z2, z2->ambient.
    C = [0.01, 0.02, 0.01], n = 0.65. boundary = ["ambient"], interior = ["z1", "z2"].
    A layer built from it (named "zones") takes the wind as its head driver
    `"zones.edge_head"`, shaped (..., 3) with the wind magnitude in column 0 (ambient->z1)
    and zero elsewhere.

    Returns (net, elements, boundary).
    """
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    elements = [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=torch.float64), 0.65)]
    boundary = ["ambient"]
    return net, elements, boundary


@pytest.fixture(scope="session")
def composed_model():
    """The reference composed model: 8 buildings, a 40-node street and a 30-node sewer
    network, about 1030 nodes and 2200 edges, with one reference physics configuration
    (`benchmarks.composed_model.build_composed`, defaults, ensemble=1). Used by the
    tests that need a realistic joined topology, and by the composed-model scaling gate.
    Tests must not mutate the returned model's elements.
    """
    return build_composed()


@pytest.fixture
def contamx():
    """NIST's contamxpy (bundles ContamX 3.4.1.7; Windows x86-64). Skips when absent."""
    return pytest.importorskip(
        "contamxpy", reason="contamxpy not installed: pip install .[contam] (Windows only)"
    )


def _builtin(value):
    """A recorded property as a builtin type: numpy and torch scalars (and arrays, as lists)
    cannot cross pytest-xdist's execnet channel, which crashes the worker that recorded them."""
    if isinstance(value, (list, tuple)):
        return [_builtin(v) for v in value]
    if isinstance(value, torch.Tensor) or hasattr(value, "tolist"):
        return value.tolist()
    return value


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    report.user_properties = [(k, _builtin(v)) for k, v in report.user_properties]
    return report


@pytest.fixture(scope="session", autouse=True)
def _private_scratch_directory(tmp_path_factory):
    """Give every test process its own TMP/TEMP/TMPDIR for EPANET's scratch files.

    EPANET 2.2 (the `epanet22.dll` inside `wntr`, driven by `wntr.sim.EpanetSimulator`) writes
    its hydraulics and output scratch files `enN` into the directory named by `TMP`, numbering
    them from a counter that restarts in each process. Under `pytest -n 4` two workers then
    open the same file, and a run reads another network's results: heads off by ~100 m in
    `test_two_loop_heads_and_flows`, a different test failing in about every second run, each
    one passing alone. A per-worker directory removes the collision (the tests' own
    `file_prefix` already lives in their `tmp_path`).
    """
    names = ("TMP", "TEMP", "TMPDIR")
    saved = {name: os.environ.get(name) for name in names}
    scratch = str(tmp_path_factory.mktemp("epanet-tmp"))
    for name in names:
        os.environ[name] = scratch
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
