"""The `PowerNetwork` data model and its validation."""

import pytest

from noodl.apps.electrical.power import (
    BUS_KINDS,
    Branch,
    Bus,
    FixedShunt,
    Generator,
    Load,
    PowerNetwork,
)


def _net(**kw):
    args = dict(
        buses=(Bus(1, "slack", vm=1.02), Bus(2, "pq")),
        branches=(Branch(1, 2, 0.01, 0.1, 0.02),),
        generators=(Generator(1, 50.0),),
        loads=(Load(2, 40.0, 10.0),),
        shunts=(FixedShunt(2, 0.0, 5.0),),
    )
    args.update(kw)
    return PowerNetwork(**args)


def test_a_valid_network_keeps_its_file_units_and_defaults():
    net = _net()
    assert BUS_KINDS == ("pq", "pv", "slack")
    assert net.bus_ids() == (1, 2)
    assert net.base_mva == 100.0 and net.name == "" and dict(net.notes) == {}
    g = net.generators[0]
    assert (g.q_mvar, g.vm_set, g.in_service, g.q_max_mvar, g.q_min_mvar) == (
        0.0, 1.0, True, float("inf"), float("-inf"))
    br = net.branches[0]
    assert (br.ratio, br.shift_deg, br.in_service) == (0.0, 0.0, True)


def test_duplicate_bus_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate bus"):
        _net(buses=(Bus(1, "slack"), Bus(2, "pq"), Bus(2, "pq")))


def test_an_unknown_bus_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        _net(buses=(Bus(1, "slack"), Bus(2, "isolated")))


def test_a_nonpositive_base_is_refused():
    with pytest.raises(ValueError, match="base_mva"):
        _net(base_mva=0)


@pytest.mark.parametrize("field, value, match", [
    ("branches", (Branch(1, 2, 0.0, 0.1, 0.0), Branch(2, 7, 0.0, 0.1, 0.0)), r"branch 1.*bus 7"),
    ("branches", (Branch(7, 2, 0.0, 0.1, 0.0),), r"branch 0.*bus 7"),
    ("generators", (Generator(5, 10.0),), r"generator 0.*bus 5"),
    ("loads", (Load(3, 1.0, 0.0),), r"load 0.*bus 3"),
    ("shunts", (FixedShunt(4, 0.0, 1.0),), r"shunt 0.*bus 4"),
])
def test_elements_on_unknown_buses_are_refused(field, value, match):
    with pytest.raises(ValueError, match=match):
        _net(**{field: value})


def test_a_network_is_hashable_and_its_notes_do_not_take_part_in_equality():
    net = _net(notes={"ignored_fields": ["gencost"]})
    assert hash(net) == hash(_net())
    assert net == _net() and {net: 1}[_net()] == 1
