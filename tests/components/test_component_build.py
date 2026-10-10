"""Building a component tree: what each call accepts, and the errors it raises at the call."""

import pytest

from noodl.components import Component, ComponentError, OuterRef, PortRef, _registry
from tests.components.conftest import link, zone


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("bad", ["", "a.b", 3, None])
def test_component_name_must_be_a_plain_string(bad):
    with pytest.raises(ComponentError, match="name"):
        Component(bad)


@pytest.mark.parametrize("bad", ["", "x.y"])
def test_node_terminal_edge_and_port_names_are_checked(bad):
    c = Component("c")
    c.add_node("n")
    with pytest.raises(ComponentError):
        c.add_node(bad)
    with pytest.raises(ComponentError):
        c.add_terminal(bad)
    with pytest.raises(ComponentError):
        c.add_edge("n", "n", kind="k", name=bad)
    with pytest.raises(ComponentError):
        c.expose(**{bad: "n"})


def test_template_and_name_are_kept():
    c = Component("r1", template="office")
    assert (c.name, c.template) == ("r1", "office")
    assert Component("x").template is None


# ---------------------------------------------------------------- nodes and edges
def test_nodes_lists_real_nodes_and_terminals_in_order():
    c = Component("c")
    c.add_node("air", volume=1.0)
    c.add_terminal("door")
    assert c.nodes == ["air", "door"]


def test_duplicate_node_is_an_error():
    c = Component("c")
    c.add_node("air")
    with pytest.raises(ComponentError, match="already exists"):
        c.add_terminal("air")


def test_inner_name_may_not_clash_with_a_node():
    c = Component("c")
    c.add_node("ambient")
    with pytest.raises(ComponentError, match="already exists"):
        c.inner("ambient")
    d = Component("d")
    d.inner("ambient")
    with pytest.raises(ComponentError, match="already exists"):
        d.add_node("ambient")


def test_add_edge_returns_its_name_and_auto_names_skip_taken_ones():
    c = Component("c")
    c.add_node("a")
    c.add_node("b")
    assert c.add_edge("a", "b", kind="k", name="e1") == "e1"
    assert c.add_edge("a", "b", kind="k") == "e0"   # the smallest free e<k>
    assert c.add_edge("a", "b", kind="k") == "e2"   # e1 is taken
    assert c.edges == ["e1", "e0", "e2"]


def test_duplicate_edge_name_is_an_error():
    c = Component("c")
    c.add_node("a")
    c.add_edge("a", "a", kind="k", name="x")
    with pytest.raises(ComponentError, match="already exists"):
        c.add_edge("a", "a", kind="k", name="x")


def test_add_edge_to_unknown_node_names_the_known_ones():
    c = Component("c")
    c.add_node("air")
    with pytest.raises(ComponentError, match=r"unknown node 'aier'.*'air'"):
        c.add_edge("aier", "air", kind="k")


def test_add_edge_needs_a_kind():
    c = Component("c")
    c.add_node("a")
    with pytest.raises(ComponentError, match="kind"):
        c.add_edge("a", "a", kind="")


def test_add_edge_accepts_child_port_and_outer():
    f = Component("f")
    r = f.add(zone("r"))
    f.add_edge(r.ports.air, f.outer("ambient"), kind="k")
    assert f.edges == ["e0"]


def test_add_edge_refuses_a_grandchild_port():
    b = Component("b")
    f = b.add(Component("f"))
    r = f.add(zone("r"))
    b.add_node("n")
    with pytest.raises(ComponentError, match="re-expose"):
        b.add_edge(r.ports.air, "n", kind="k")


def test_add_edge_refuses_something_that_is_not_an_endpoint():
    c = Component("c")
    c.add_node("a")
    with pytest.raises(ComponentError, match="not a node name"):
        c.add_edge("a", 42, kind="k")


# ---------------------------------------------------------------- ports
def test_expose_own_nodes_and_renamed_child_port():
    f = Component("f")
    r = f.add(zone("r"))
    f.add_terminal("t")
    f.expose("t", corridor=r.ports.air)
    assert list(f.ports) == ["t", "corridor"]


def test_expose_unknown_node_is_an_error():
    with pytest.raises(ComponentError, match="unknown node"):
        Component("c").expose("air")


def test_expose_positional_portref_is_an_error():
    f = Component("f")
    r = f.add(zone("r"))
    with pytest.raises(ComponentError, match="new_name="):
        f.expose(r.ports.air)


def test_duplicate_port_is_an_error():
    c = zone("z")
    with pytest.raises(ComponentError, match="already exists"):
        c.expose("air")


def test_ports_namespace_attribute_item_iter_len_contains():
    c = Component("s")
    for name in ("levels[0]", "levels[1]", "add"):
        c.add_terminal(name)
        c.expose(name)
    assert isinstance(c.ports["levels[0]"], PortRef)
    assert c.ports.add.name == "add"          # no clash with Component.add
    assert callable(c.add)
    assert len(c.ports) == 3 and "levels[1]" in c.ports and "x" not in c.ports
    assert list(c.ports) == ["levels[0]", "levels[1]", "add"]


def test_unknown_port_lists_the_ports():
    c = zone("z")
    with pytest.raises(AttributeError, match=r"no port 'aier'.*'air'"):
        _ = c.ports.aier
    with pytest.raises(KeyError, match="no port"):
        c.ports["aier"]


def test_portrefs_compare_by_component_identity_and_name():
    a, b = zone("z"), zone("z")
    assert a.ports.air == a.ports.air
    assert a.ports.air != b.ports.air


# ---------------------------------------------------------------- children
def test_add_returns_the_child_and_lists_it():
    f = Component("f")
    r = zone("r")
    assert f.add(r, at=(1, 2, 3)) is r
    assert f.children == {"r": r}
    assert f._children["r"][1] == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("at", [
    (1.0, 2.0), (1, 2, 3, 4), "abc", ("1", "2", "3"), (float("nan"), 0, 0),
    (0, float("inf"), 0), (True, 0, 0), 5,
])
def test_at_must_be_three_numbers(at):
    with pytest.raises(ComponentError, match=r"\(x, y, z\)"):
        Component("f").add(zone("r"), at=at)


def test_a_child_can_be_added_once():
    r = zone("r")
    Component("f").add(r)
    with pytest.raises(ComponentError, match="already part of 'f'"):
        Component("g").add(r)


def test_component_cannot_contain_itself_or_an_ancestor():
    b = Component("b")
    f = b.add(Component("f"))
    with pytest.raises(ComponentError, match="itself"):
        b.add(b)
    with pytest.raises(ComponentError, match="ancestor"):
        f.add(b)


def test_child_names_are_unique_and_distinct_from_nodes():
    f = Component("f")
    f.add(zone("r"))
    with pytest.raises(ComponentError, match="already exists"):
        f.add(zone("r"))
    f.add_node("n")
    with pytest.raises(ComponentError, match="already exists"):
        f.add(zone("n"))


def test_add_refuses_a_non_component():
    with pytest.raises(ComponentError, match="Component"):
        Component("f").add("r")


# ---------------------------------------------------------------- connect
def test_connect_child_ports_and_own_nodes():
    f = Component("f")
    r, d = f.add(zone("r")), f.add(link("d"))
    f.add_terminal("t")
    f.connect(r.ports.air, d.ports.a)
    f.connect("t", d.ports.b)
    assert len(f._connects) == 2


def test_connect_refuses_outer_grandchild_self_and_unadded():
    b = Component("b")
    f = b.add(Component("f"))
    r = f.add(zone("r"))
    d = b.add(link("d"))
    with pytest.raises(ComponentError, match="outer"):
        b.connect(b.outer("ambient"), d.ports.a)
    with pytest.raises(ComponentError, match="re-expose"):
        b.connect(r.ports.air, d.ports.a)
    with pytest.raises(ComponentError, match="itself"):
        b.connect(d.ports.a, d.ports.a)
    stray = link("stray")
    with pytest.raises(ComponentError, match="direct child"):
        b.connect(stray.ports.a, d.ports.a)


# ---------------------------------------------------------------- inner / outer / tables
def test_inner_and_table_duplicates_are_errors():
    c = Component("c")
    c.inner("ambient", z_ref=0.0)
    c.inner_table("facades", south=180.0)
    with pytest.raises(ComponentError, match="already exists"):
        c.inner("ambient")
    with pytest.raises(ComponentError, match="already exists"):
        c.inner_table("facades")


def test_outer_is_a_reference_by_name():
    assert Component("c").outer("ambient") == OuterRef("ambient")


# ---------------------------------------------------------------- registry
def test_registry_records_elevations_and_lookups():
    _registry.register_elevations(nodes=("z_ref",), edges=("z_path",))
    _registry.register_table_lookup("facade", table="facades", target="azimuth")
    _registry.register_table_lookup("facade", table="facades", target="azimuth")  # same: fine
    assert _registry.node_elevations() == {"z_ref"}
    assert _registry.edge_elevations() == {"z_path"}
    assert _registry.table_lookups() == {"facade": ("facades", "azimuth")}


def test_registry_refuses_a_conflicting_lookup():
    _registry.register_table_lookup("facade", table="facades", target="azimuth")
    with pytest.raises(ValueError, match="already"):
        _registry.register_table_lookup("facade", table="other", target="azimuth")


def test_unknown_node_that_is_an_inner_hints_at_outer():
    c = Component("c")
    c.inner("ambient")
    c.add_terminal("a")
    with pytest.raises(ComponentError, match=r"reached with outer\('ambient'\)"):
        c.add_edge("a", "ambient", kind="airpath")


def test_name_template_and_ports_are_read_only():
    c = Component("c", template="t")
    c.add_node("a")
    for attr in ("name", "template", "ports"):
        with pytest.raises(AttributeError, match="fixed once created"):
            setattr(c, attr, "x")
    assert (c.name, c.template) == ("c", "t")


def test_a_failing_expose_adds_no_port():
    c = Component("c")
    c.add_node("a")
    with pytest.raises(ComponentError):
        c.expose("a", "missing")
    assert list(c.ports) == []
    with pytest.raises(ComponentError, match="already exists"):
        c.expose("a", "a")
    assert list(c.ports) == []
    c.expose("a")
    c.add_node("b")
    with pytest.raises(ComponentError, match="already exists"):
        c.expose("b", a="b")
    assert list(c.ports) == ["a"]


def test_at_accepts_zero_d_real_tensors_and_rejects_the_rest():
    import torch

    f = Component("f")
    r = f.add(zone("r"), at=(torch.tensor(3.0), torch.tensor(1), 0.0))
    assert f._children["r"][1] == (3.0, 1.0, 0.0)
    assert all(type(v) is float for v in f._children["r"][1])
    for bad in (torch.tensor(float("nan")), torch.tensor(True), torch.tensor(1 + 1j),
                torch.tensor([1.0])):
        with pytest.raises(ComponentError, match=r"\(x, y, z\)"):
            Component("g").add(zone("s"), at=(bad, 0, 0))
    assert r.name == "r"
