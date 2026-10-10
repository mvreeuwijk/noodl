"""Helpers for the core component tests: tiny factories with plain attributes, and a
registry that starts empty in every test (the building app registers on import)."""

import pytest

from noodl.components import Component, _registry


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    monkeypatch.setattr(_registry, "_NODE_ELEVATIONS", set())
    monkeypatch.setattr(_registry, "_EDGE_ELEVATIONS", set())
    monkeypatch.setattr(_registry, "_TABLE_LOOKUPS", {})


def zone(name, **attrs):
    """One real node `air`, exposed."""
    c = Component(name, template="zone")
    c.add_node("air", **attrs)
    c.expose("air")
    return c


def link(name, kind="airpath", **attrs):
    """Two terminals `a`, `b` and one edge `path` between them, both exposed (a door)."""
    c = Component(name, template="link")
    c.add_terminal("a")
    c.add_terminal("b")
    c.add_edge("a", "b", kind=kind, name="path", **attrs)
    c.expose("a", "b")
    return c


def leak(name, kind="airpath", **attrs):
    """One terminal `a` and an edge from the shared `ambient` to it."""
    c = Component(name, template="leak")
    c.add_terminal("a")
    c.add_edge(c.outer("ambient"), "a", kind=kind, name="path", **attrs)
    c.expose("a")
    return c
