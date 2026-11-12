"""Reading WSIMOD's model description: the class schemas, the demos, the overrides, the refusals.

Tests that compare against WSIMOD itself skip when `wsimod` is not installed.
"""

from __future__ import annotations

import copy
import inspect
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from noodl.apps.wsimod import build_model
from noodl.apps.wsimod.classes import (
    ARC_SCHEMAS,
    NODE_SCHEMAS,
    OUT_OF_SCOPE,
    REQUIRED,
    SURFACE_SCHEMAS,
)
from noodl.apps.wsimod.config import DEFAULT_ORCHESTRATION, parse_date, read_wsimod_config

DATA = Path(__file__).resolve().parents[2] / "data" / "wsimod"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _signature_chain(cls) -> dict[str, object]:
    """The keywords and defaults of ``cls``'s constructor chain: its own ``__init__`` and, while
    one takes ``**kwargs``, the next class of the MRO's (a subclass's default wins)."""
    parameters: dict[str, object] = {}
    for klass in cls.__mro__:
        if klass is object:
            break
        forwards = False
        for name, p in inspect.signature(klass.__init__).parameters.items():
            if name == "self" or p.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                forwards = True
                continue
            default = REQUIRED if p.default is inspect.Parameter.empty else p.default
            parameters.setdefault(name, copy.deepcopy(default))
        if not forwards:
            break
    return parameters


def test_the_class_schemas_match_wsimods_constructors():
    pytest.importorskip("wsimod")
    from wsimod.arcs import arcs
    from wsimod.nodes import land
    from wsimod.nodes.nodes import NODES_REGISTRY

    assert len(NODE_SCHEMAS) == 19
    assert set(NODE_SCHEMAS) == set(NODES_REGISTRY) - {"EnfieldFoulSewer"}
    for table, module in (
        (NODE_SCHEMAS, NODES_REGISTRY),
        (SURFACE_SCHEMAS, vars(land)),
        (ARC_SCHEMAS, vars(arcs)),
    ):
        for name, schema in table.items():
            assert schema.name == name
            assert dict(schema.parameters) == _signature_chain(module[name]), name
    for name in OUT_OF_SCOPE:
        assert name not in NODE_SCHEMAS | SURFACE_SCHEMAS | ARC_SCHEMAS


def test_the_renamed_types_are_the_names_wsimod_gives_its_instances():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    model = wsimod_model(
        {
            "nodes": [
                {"name": "g", "type_": "Groundwater", "node_type_override": "QueueGroundwater"},
                {"name": "r", "type_": "Reservoir", "node_type_override": "RiverReservoir"},
                {"name": "d", "type_": "Demand", "node_type_override": "ResidentialDemand"},
                {"name": "w", "type_": "Waste"},
            ]
        }
    )
    renamed = {n: type(o).__name__ for n, o in model.nodes.items()}
    assert renamed == {"g": "Groundwater", "r": "Reservoir", "d": "Demand", "w": "Waste"}
    assert NODE_SCHEMAS["QueueGroundwater"].renamed_type == "Groundwater"
    assert NODE_SCHEMAS["RiverReservoir"].renamed_type == "Reservoir"
    assert NODE_SCHEMAS["ResidentialDemand"].renamed_type == "Demand"
    assert all(
        s.renamed_type == s.name for k, s in NODE_SCHEMAS.items() if k not in {
            "QueueGroundwater", "RiverReservoir", "ResidentialDemand"
        }
    )  # fmt: skip


@pytest.mark.parametrize("case", ["quickstart", "oxford"])
def test_the_demos_read_as_wsimod_loads_them(case):
    pytest.importorskip("wsimod")
    from wsimod.core import constants

    from tests.verification._wsimod_reference import wsimod_model

    model = wsimod_model(DATA / case)
    net = read_wsimod_config(DATA / case)

    assert [n.name for n in net.nodes] == list(model.nodes)
    assert [a.name for a in net.arcs] == list(model.arcs)
    for arc, spec in zip(model.arcs.values(), net.arcs, strict=True):
        assert (spec.in_port, spec.out_port) == (arc.in_port.name, arc.out_port.name)
        assert (spec.capacity, spec.preference) == (arc.capacity, arc.preference)
    assert net.pollutants.names == tuple(constants.POLLUTANTS)
    assert net.pollutants.additive == tuple(constants.ADDITIVE_POLLUTANTS)
    assert net.pollutants.non_additive == tuple(constants.NON_ADDITIVE_POLLUTANTS)
    assert net.float_accuracy == constants.FLOAT_ACCURACY
    if case == "oxford":
        assert net.float_accuracy == 1e-8
    assert net.dates == [d._date for d in model.dates]
    written = yaml.safe_load((DATA / case / "config.yml").read_text(encoding="utf-8"))
    assert net.date_labels == written["dates"]
    assert net.orchestration == [
        (next(iter(e)), next(iter(e.values()))) for e in model.orchestration
    ]

    compared = 0

    def compare(spec_parameters, obj, where):
        nonlocal compared
        for key, value in spec_parameters.items():
            if not _is_number(value) or not hasattr(obj, key):
                continue
            actual = getattr(obj, key)
            if type(obj).__name__ == "PerviousSurface" and key == "depth":
                value = value * obj.total_porosity  # the simulated depth (land.py:759)
            assert actual == pytest.approx(value, rel=1e-15, abs=0), f"{where}.{key}"
            compared += 1

    def same_data(spec_data, obj, where):
        expected = {(v, d._date): x for (v, d), x in obj.data_input_dict.items()}
        assert spec_data == expected, where

    for spec in net.nodes:
        obj = model.nodes[spec.name]
        assert NODE_SCHEMAS[spec.cls].renamed_type == type(obj).__name__
        assert model.nodes_type[spec.slot_key][spec.name] is obj
        compare(spec.parameters, obj, spec.name)
        if spec.data:
            same_data(spec.data, obj, spec.name)
        if hasattr(obj, "surfaces"):
            assert [s.name for s in spec.surfaces] == [s.surface for s in obj.surfaces]
            for surface_spec, surface in zip(spec.surfaces, obj.surfaces, strict=True):
                assert surface_spec.cls == type(surface).__name__
                compare(surface_spec.parameters, surface, f"{spec.name}.{surface_spec.name}")
                same_data(surface_spec.data, surface, f"{spec.name}.{surface_spec.name}")
    assert compared >= (45 if case == "oxford" else 25)  # not vacuous


def test_the_small_config_reads():
    net = read_wsimod_config(DATA / "small")
    assert [(n.name, n.cls) for n in net.nodes] == [
        ("inflow", "Catchment"),
        ("river", "River"),
        ("outlet", "Waste"),
    ]
    assert [(a.in_port, a.out_port) for a in net.arcs] == [
        ("inflow", "river"),
        ("river", "outlet"),
    ]
    assert len(net.dates) == 10 and net.dates[0] == datetime(2000, 1, 1)
    assert net.pollutants.K == 0
    assert net.orchestration == list(DEFAULT_ORCHESTRATION)
    assert net.nodes[0].data[("flow", datetime(2000, 1, 3))] == 95.0
    assert net.nodes[1].parameters["length"] == 200
    replaced = read_wsimod_config(DATA / "small", overrides={"float_accuracy": 1e-8})
    assert replaced.float_accuracy == 1e-8


def test_the_small_config_loads_in_wsimod_too():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    model = wsimod_model(DATA / "small")
    assert list(model.nodes) == ["inflow", "river", "outlet"]
    assert len(model.dates) == 10


def test_foul_sewers_and_type_overrides_take_their_slot_keys():
    net = read_wsimod_config(
        {
            "nodes": [
                {"name": "foul_1", "type_": "Sewer"},
                {"name": "storm_1", "type_": "Sewer"},
                {"name": "h", "type_": "Demand", "node_type_override": "ResidentialDemand"},
            ]
        }
    )
    foul, storm, houses = net.nodes
    assert (foul.cls, foul.slot_key) == ("Sewer", "Foul")
    assert (storm.cls, storm.slot_key) == ("Sewer", "Sewer")
    assert (houses.cls, houses.slot_key) == ("ResidentialDemand", "Demand")


def _everything() -> dict:
    """A mapping with a node of each class the override test touches (WSIMOD's post-read form)."""
    return {
        "nodes": [
            {
                "name": "land",
                "type_": "Land",
                "surfaces": [
                    {
                        "surface": "urban",
                        "type_": "ImperviousSurface",
                        "area": 10,
                        "pore_depth": 0.002,
                    },
                    {
                        "surface": "rural",
                        "type_": "PerviousSurface",
                        "area": 100,
                        "depth": 0.5,
                        "total_porosity": 0.4,
                        "field_capacity": 0.3,
                        "wilting_point": 0.12,
                    },
                ],
            },
            {"name": "river", "type_": "River", "length": 200, "width": 20},
            {"name": "wwtw", "type_": "WWTW"},
            {"name": "fwtw", "type_": "FWTW"},
            {"name": "sewer", "type_": "Sewer", "capacity": 0.04},
        ],
        "overrides": {
            "nodes": {
                "land": {
                    "type_": "Land",
                    "name": "land",
                    "surface_residence_time": 2,
                    "percolation_residence_time": 40,
                    "surfaces": {
                        "urban": {"area": 12, "pore_depth": 0.003, "et0_to_e": 0.9},
                        "rural": {
                            "depth": 0.6,
                            "total_porosity": 0.45,
                            "field_capacity": 0.28,
                            "area": 90,
                            "pollutant_load": {"phosphate": 1e-7},
                        },
                    },
                },
                "river": {"type_": "River", "name": "river", "length": 300, "width": 25},
                "wwtw": {
                    "type_": "WWTW",
                    "name": "wwtw",
                    "treatment_throughput_capacity": 20,
                    "percent_solids": 0.001,
                    "liquor_multiplier": {"volume": 0.05},
                    "stormwater_storage_capacity": 15,
                },
                "fwtw": {
                    "type_": "FWTW",
                    "name": "fwtw",
                    "treatment_throughput_capacity": 8,
                    "service_reservoir_storage_capacity": 30,
                },
                "sewer": {
                    "type_": "Sewer",
                    "name": "sewer",
                    "capacity": 0.05,
                    "pipe_time": 1,
                    "pipe_timearea": {0: 0.5, 1: 0.5},
                    "chamber_area": 2,
                },
            }
        },
    }


def test_overrides_merge_into_the_parameters():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    mapping = _everything()
    model = wsimod_model(mapping)
    net = read_wsimod_config(mapping)
    close = {"rel": 1e-15, "abs": 0}

    nodes = {n.name: n for n in net.nodes}
    for name, keys in {
        "land": ["surface_residence_time", "percolation_residence_time"],
        "river": ["length", "width"],
        "wwtw": [
            "treatment_throughput_capacity",
            "percent_solids",
            "stormwater_storage_capacity",
        ],
        "fwtw": ["treatment_throughput_capacity", "service_reservoir_storage_capacity"],
        "sewer": ["capacity", "pipe_time", "chamber_area"],
    }.items():
        for key in keys:
            assert getattr(model.nodes[name], key) == pytest.approx(
                nodes[name].parameters[key], **close
            ), (name, key)
    assert nodes["sewer"].parameters["pipe_timearea"] == model.nodes["sewer"].pipe_timearea
    # liquor_multiplier: the one key overridden, the others kept
    for name in ("wwtw", "fwtw"):
        assert nodes[name].parameters["liquor_multiplier"] == model.nodes[name].liquor_multiplier
    assert nodes["wwtw"].parameters["liquor_multiplier"]["volume"] == 0.05
    assert nodes["wwtw"].parameters["liquor_multiplier"]["phosphate"] == 0.7
    assert nodes["fwtw"].parameters["liquor_multiplier"]["volume"] == 0.03

    land = model.nodes["land"]
    for surface_spec in nodes["land"].surfaces:
        surface = land.get_surface(surface_spec.name)
        for key, value in surface_spec.parameters.items():
            if key in ("area", "pore_depth", "et0_to_e", "field_capacity", "total_porosity"):
                assert getattr(surface, key) == pytest.approx(value, **close), (
                    surface_spec.name,
                    key,
                )
        if surface_spec.cls == "PerviousSurface":
            # WSIMOD's own round trip: the simulated depth over the porosity (land.py:858)
            assert surface.depth / surface.total_porosity == pytest.approx(
                surface_spec.parameters["depth"], **close
            )
    rural = next(s for s in nodes["land"].surfaces if s.name == "rural")
    assert rural.parameters["pollutant_load"] == land.get_surface("rural").pollutant_load
    assert rural.parameters["depth"] == 0.6 and rural.parameters["wilting_point"] == 0.12


def test_an_override_of_pollutant_load_reaches_only_its_node():
    net = read_wsimod_config(
        {
            "nodes": [
                {"name": "a", "type_": "Demand"},
                {"name": "b", "type_": "Demand"},
            ],
            "overrides": {
                "nodes": [{"type_": "Demand", "name": "a", "pollutant_load": {"phosphate": 2.0}}]
            },
        }
    )
    a, b = net.nodes
    assert a.parameters["pollutant_load"] == {"phosphate": 2.0}
    assert b.parameters["pollutant_load"] == {}
    assert NODE_SCHEMAS["Demand"].parameters["pollutant_load"] == {}


def test_monthly_daily_and_timestamp_dates_name_the_same_day():
    net = read_wsimod_config(
        {
            "nodes": [
                {"name": "c", "type_": "Catchment", "data_input_dict": {("flow", "2000-01"): 5.0}}
            ],
            "dates": ["2000-01-01", "2000-01-02 00:00:00", "2000-01-05"],
        }
    )
    assert net.dates == [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 5)]
    assert net.date_labels == ["2000-01-01", "2000-01-02 00:00:00", "2000-01-05"]
    assert net.nodes[0].data[("flow", datetime(2000, 1, 1))] == 5.0
    assert parse_date("2000-01") == parse_date("2000-01-01 00:00:00")


def _base() -> dict:
    return {
        "nodes": [
            {"name": "a", "type_": "Catchment"},
            {"name": "b", "type_": "Waste"},
            {
                "name": "land",
                "type_": "Land",
                "surfaces": [{"surface": "g", "type_": "PerviousSurface"}],
            },
        ],
        "arcs": [{"name": "ab", "type_": "Arc", "in_port": "a", "out_port": "b"}],
    }


def _with(**changes) -> dict:
    mapping = _base()
    mapping.update(changes)
    return mapping


def _node(**node) -> dict:
    mapping = _base()
    mapping["nodes"].append({"name": "x", **node})
    return mapping


def _arc(**arc) -> dict:
    mapping = _base()
    mapping["arcs"].append({"name": "x", "in_port": "a", "out_port": "b", **arc})
    return mapping


def _override(node=None, arc=None) -> dict:
    mapping = _base()
    mapping["overrides"] = {"nodes": [node] if node else [], "arcs": [arc] if arc else []}
    return mapping


REFUSALS = [
    (_with(extensions=["x.py"]), "extensions"),
    (_with(unified_data_file="all.parquet"), "unified_data_file"),
    (_with(foo=1), "foo"),
    (_with(overrides={"surfaces": {}}), "surfaces"),
    (_with(dates=["01/02/2000"]), "01/02/2000"),
    (_with(dates=["2000-01-01 12:00"]), "12:00"),
    (_with(pollutants=["a"], additive_pollutants=["b"], non_additive_pollutants=[]), "'b'"),
    (_node(type_="EnfieldFoulSewer"), "EnfieldFoulSewer"),
    (_node(type_="Land", surfaces=[{"type_": "GrowingSurface"}]), "3c"),
    (_node(type_="Land", surfaces=[{"type_": "PerviousSurface", "capacity": 1}]), "capacity"),
    (_node(type_="Land", surfaces=[{"type_": "ImperviousSurface", "depth": 1}]), "depth"),
    (_node(type_="Widget"), "Widget"),
    (_node(type_="Waste", capacityy=1), "capacityy"),
    (_node(type_="Waste", filename="x.csv"), "data_input_dict"),  # a constructor key
    (_node(type_="Catchment", filename="missing.csv"), "missing.csv"),
    (_node(type_="Waste", name="a"), "'a'"),
    (_arc(type_="Arc", decays={}), "decays"),
    (_arc(type_="QueueArc", decays={}), "read only by DecayArc and DecayArcAlt"),
    (_arc(type_="DecayArc", parent="x"), "parent"),
    (_arc(type_="DecayArcAlt", parent="x"), "parent"),
    (_arc(type_="Arc", parent="x"), "parent"),
    (_arc(type_="DecayArc", decays=None), "decays must be a mapping"),
    (_arc(type_="DecayArc", decays={"phosphate": {"constant": 1}}), "exponent"),
    (_arc(type_="DecayArcAlt", decays={"unobtainium": {"constant": 1, "exponent": 1}}), "unob"),
    (_arc(type_="Arc", diameter=1), "diameter"),
    (_arc(type_="Arc", number_of_timesteps=2), "number_of_timesteps"),
    (_arc(type_="QueueArc", number_of_timesteps=1.5), "number_of_timesteps"),
    (_arc(type_="Arc", capacity="big"), "capacity"),
    (_arc(type_="Arc", in_port="nowhere"), "nowhere"),
    (_override(node={"type_": "Catchment", "name": "nowhere"}), "nowhere"),
    (_override(node={"type_": "Reservoir", "name": "a"}), "Reservoir"),
    (_override(node={"type_": "Land", "name": "a"}), "'a'"),  # a is filed under Catchment
    (_override(node={"type_": "Catchment", "name": "a", "capacity": 1}), "capacity"),
    (_override(node={"type_": "Catchment", "name": "a", "filename": "x.csv"}), "x.csv"),
    (_override(node={"type_": "Catchment", "name": "a", "filename": 3}), "filename"),
    (
        _override(node={"type_": "Land", "name": "land", "surfaces": {"g": {"filename": "x.csv"}}}),
        "filename",
    ),
    (_override(node={"type_": "Catchment", "name": "a", "data_input_dict": {}}), "data_input_dict"),
    (
        _override(
            node={"type_": "Land", "name": "land", "surfaces": {"g": {"data_input_dict": "x.csv"}}}
        ),
        "data_input_dict",
    ),
    (
        _override(node={"type_": "Land", "name": "land", "surfaces": {"g": {"capacity": 1}}}),
        "capacity",
    ),
    (_override(node={"type_": "Land", "name": "land", "surfaces": {"nowhere": {}}}), "nowhere"),
    (
        _with(
            arcs=[],
            nodes=[
                {
                    "name": "l",
                    "type_": "Land",
                    "surfaces": [{"surface": "u", "type_": "ImperviousSurface"}],
                }
            ],
            overrides={
                "nodes": [
                    {"type_": "Land", "name": "l", "surfaces": {"u": {"depth": 1}}},
                ]
            },
        ),
        "depth",
    ),
    (_node(type_="Groundwater", decays={"phosphate": {"constant": 1}}), "exponent"),
    # a DecayQueueTank's number_of_timesteps is an override only (tanks.py:682-685, X28)
    (_node(type_="QueueGroundwater", decays={}, number_of_timesteps=0), "number_of_timesteps"),
    # read_decays at node level (Task 5's refusals)
    (_node(type_="Storage", decays=[1]), "decays must be a mapping"),
    (_node(type_="Storage", decays={"unobtainium": {"constant": 1, "exponent": 1}}), "unobtainium"),
    (_node(type_="Storage", decays={"volume": {"constant": 1, "exponent": 1}}), "volume"),
    (_node(type_="Reservoir", decays={"phosphate": {"constant": 1}}), "exponent"),
    (_node(type_="River", decays={"phosphate": {"constant": 1, "exponent": 0}}), "exponent"),
    (_node(type_="Storage", decays={"phosphate": {"constant": "a", "exponent": 1}}), "constant"),
    (_node(type_="Storage", decays={"phosphate": {"constant": 1, "exponent": None}}), "exponent"),
    (_node(type_="Land", surfaces=[{"type_": "PerviousSurface", "decays": {"a": 1}}]), "'a'"),
    (_node(type_="Waste", data_input_dict={}), "data_input_dict"),
    (_override(node={"type_": "Catchment", "name": "a", "decays": {}}), "decays"),
    (_override(arc={"name": "nowhere", "type_": "Arc"}), "nowhere"),
    (_override(arc={"name": "ab", "type_": "Arc", "diameter": 1}), "diameter"),
]


def _decay_arc_mapping(type_: str, **arc) -> dict:
    """``_base()`` with a ``type_`` arc ``x`` out of the catchment ``a``."""
    mapping = _arc(type_=type_, number_of_timesteps=1, **arc)
    mapping["overrides"] = {"nodes": [], "arcs": []}
    return mapping


@pytest.mark.parametrize("type_", ["DecayArc", "DecayArcAlt"])
def test_the_decay_arcs_are_read_with_their_decays(type_):
    """A ``Decay*`` arc is a configured class now (it was refused as 3b's); its ``decays``
    defaults to ``{}`` (``arcs.py:671``, ``:734``), and the reader's checks of a node's
    ``decays`` apply. A ``decays`` override is refused (an arc's ``apply_overrides`` takes
    ``capacity`` and ``preference`` only, ``arcs.py:87-96``), and so is a 3b message."""
    assert "DecayArc" not in OUT_OF_SCOPE and "DecayArcAlt" not in OUT_OF_SCOPE
    plain = read_wsimod_config(_decay_arc_mapping(type_)).arcs[-1]
    assert (plain.cls, plain.number_of_timesteps, plain.decays.keys) == (type_, 1, ())
    decays = {"phosphate": {"constant": 0.1, "exponent": 1.05}}
    spec = read_wsimod_config(_decay_arc_mapping(type_, decays=decays)).arcs[-1]
    assert spec.decays.keys == ("phosphate",) and spec.decays.constant == {"phosphate": 0.1}
    assert read_wsimod_config(_base()).arcs[0].decays is None  # a plain arc has none
    mapping = _decay_arc_mapping(type_, decays=decays)
    mapping["overrides"]["arcs"] = [{"name": "x", "decays": {"phosphate": decays["phosphate"]}}]
    with pytest.raises(ValueError, match="decays") as refused:
        read_wsimod_config(mapping)
    assert "3b" not in str(refused.value)
    with pytest.raises(ValueError, match="decays") as refused:
        read_wsimod_config(_arc(type_="Arc", decays={}))
    assert "3b" not in str(refused.value)


@pytest.mark.parametrize("type_", ["DecayArc", "DecayArcAlt"])
def test_a_decay_arc_needs_its_in_ports_temperature(type_):
    """``core.py:484-487``: a ``Decay*`` arc reads its in-port's ``("temperature", t)`` at every
    decay, in both modes; a missing date is refused at build time, naming the arc and the
    node. A ``filename`` override of that node is read, so it is not listed unread."""
    mapping = {
        "dates": ["2000-01-01", "2000-01-02"],
        "nodes": [
            {
                "name": "a",
                "type_": "Node",
                "data_input_dict": {("temperature", "2000-01-01"): 12.0},
            },
            {"name": "b", "type_": "Waste"},
        ],
        "arcs": [{"name": "x", "type_": type_, "in_port": "a", "out_port": "b"}],
    }
    for pollutants in (True, False):
        with pytest.raises(ValueError, match=r"arc 'x'.*node 'a'.*temperature.*2000-01-02"):
            build_model(read_wsimod_config(mapping), pollutants=pollutants)
    mapping["nodes"][0]["data_input_dict"][("temperature", "2000-01-02")] = 13.0
    build_model(read_wsimod_config(mapping), pollutants=False)


def test_a_decay_arcs_in_port_filename_override_is_read(tmp_path):
    """D10: the ``filename`` of a node that reads no forcing is listed unread, unless a
    ``Decay*`` arc leaves it, whose decay reads it (``core.py:464-470``)."""
    path = tmp_path / "a.csv"
    path.write_text("variable,time,value\ntemperature,2000-01-01,11.0\n", encoding="utf-8")
    mapping = {
        "dates": ["2000-01-01"],
        "nodes": [{"name": "a", "type_": "Node"}, {"name": "b", "type_": "Waste"}],
        "arcs": [{"name": "x", "type_": "Arc", "in_port": "a", "out_port": "b"}],
        "overrides": {"nodes": [{"type_": "Node", "name": "a", "filename": str(path)}]},
    }
    assert read_wsimod_config(copy.deepcopy(mapping)).unread == ["a.filename"]
    mapping["arcs"][0]["type_"] = "DecayArc"
    assert read_wsimod_config(mapping).unread == []


def _forcing_csv(path: Path, rows: dict[str, float]) -> str:
    path.write_text(
        "variable,time,value\n" + "".join(f"flow,{d},{v!r}\n" for d, v in rows.items()),
        encoding="utf-8",
    )
    return str(path)


def _as_dates(data) -> dict:
    """A forcing dict with its dates as ``datetime`` (WSIMOD's ``to_datetime`` keys compare by
    ``__eq__``, not by hash)."""
    return {(k[0], datetime.fromisoformat(str(k[1]))): v for k, v in data.items()}


def _filename_override(value) -> dict:
    """``_base()`` with a ``filename`` override of the catchment ``a`` (and its inline data), in
    the dict form WSIMOD's ``add_overrides`` takes."""
    mapping = _base()
    mapping["nodes"][0]["data_input_dict"] = {("flow", "2000-01-01"): 1.0}
    mapping["overrides"] = {
        "nodes": {"a": {"type_": "Catchment", "name": "a", "filename": value}},
        "arcs": {},
    }
    return mapping


def test_a_filename_override_replaces_the_forcing(tmp_path):
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    path = _forcing_csv(tmp_path / "flow.csv", {"2000-01-01": 7.5, "2000-01-02": 8.5})
    mapping = _filename_override(path)
    ours = read_wsimod_config(mapping).nodes[0].data
    theirs = wsimod_model(mapping).nodes["a"].data_input_dict
    assert ours == {
        ("flow", datetime(2000, 1, 1)): 7.5,
        ("flow", datetime(2000, 1, 2)): 8.5,
    }
    assert _as_dates(theirs) == ours


def test_a_relative_filename_override_opens_from_the_working_directory(tmp_path, monkeypatch):
    """Not from the configuration's directory (nodes.py:93-99, X27)."""
    _forcing_csv(tmp_path / "flow.csv", {"2000-01-01": 3.0})
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    network = read_wsimod_config(_filename_override("flow.csv"))
    assert network.nodes[0].data == {("flow", datetime(2000, 1, 1)): 3.0}
    monkeypatch.chdir(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="flow.csv"):
        read_wsimod_config(_filename_override("flow.csv"))


def test_filename_override_edge_cases(tmp_path):
    """D23 as the source reads (nodes.py:93-99): a falsy non-string is ignored, the empty string
    is a string that WSIMOD opens (``FileNotFoundError``), another non-string is refused (WSIMOD
    raises ``RuntimeError``), and a class that reads no forcing accepts the file and lists it."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    inline = {("flow", datetime(2000, 1, 1)): 1.0}
    for ignored in (None, 0, []):
        mapping = _filename_override(ignored)
        assert read_wsimod_config(mapping).nodes[0].data == inline
        assert _as_dates(wsimod_model(mapping).nodes["a"].data_input_dict) == inline
    with pytest.raises(ValueError, match="does not exist"):
        read_wsimod_config(_filename_override(""))
    with pytest.raises(FileNotFoundError):
        wsimod_model(_filename_override(""))

    with pytest.raises(ValueError, match="filename"):
        read_wsimod_config(_filename_override(3))
    with pytest.raises(RuntimeError, match="Not recognised format"):
        wsimod_model(_filename_override(3))

    path = _forcing_csv(tmp_path / "waste.csv", {"2000-01-01": 2.0})
    mapping = _filename_override(path)
    mapping["overrides"]["nodes"] = {"b": {"type_": "Waste", "name": "b", "filename": path}}
    assert read_wsimod_config(mapping).unread == ["b.filename"]
    wsimod_model(mapping)  # WSIMOD reads the file and accepts it
    assert read_wsimod_config(_filename_override(path)).unread == []


def test_groundwater_forcing_is_lost_and_listed_unread():
    """storage.py:202-203: the configured forcing is reset by ``Node.__init__`` (X11, D10)."""
    mapping = {
        "nodes": [
            {
                "name": "g",
                "type_": "Groundwater",
                "data_input_dict": {("temperature", "2000-01-01"): 10.0},
            },
            {"name": "b", "type_": "Waste"},
        ],
        "arcs": [],
    }
    network = read_wsimod_config(mapping)
    assert network.unread == ["g.data_input_dict"]
    assert network.nodes[0].data == {}
    mapping["dates"] = ["2000-01-01"]
    model, _, _ = build_model(read_wsimod_config(mapping), pollutants=False)
    assert model.unread == ["g.data_input_dict"]


@pytest.mark.parametrize(("mapping", "named"), REFUSALS)
def test_refusals_name_the_key(mapping, named):
    with pytest.raises(ValueError, match=named):
        read_wsimod_config(mapping)


def test_a_config_without_nodes_is_refused():
    with pytest.raises(ValueError, match="nodes"):
        read_wsimod_config({"arcs": []})


def test_an_orchestration_is_read_as_slot_method_pairs():
    net = read_wsimod_config(
        {"nodes": [], "orchestration": [{"Land": "run"}, {"Catchment": "route"}]}
    )
    assert net.orchestration == [("Land", "run"), ("Catchment", "route")]
    with pytest.raises(ValueError, match="orchestration"):
        read_wsimod_config({"nodes": [], "orchestration": [{"Land": "run", "River": "x"}]})


def test_overrides_replace_top_level_keys_and_the_overrides_section():
    net = read_wsimod_config(
        _base(), overrides={"nodes": [{"name": "w", "type_": "Waste"}], "arcs": []}
    )
    assert [n.name for n in net.nodes] == ["w"] and net.arcs == []
    section = {"nodes": [{"type_": "Catchment", "name": "a", "bogus": 1}]}
    with pytest.raises(ValueError, match="bogus"):
        read_wsimod_config(_base(), overrides={"overrides": section})
    base = _base()
    net = read_wsimod_config(
        base,
        overrides={
            "overrides": {"arcs": [{"name": "ab", "type_": "Arc", "capacity": 7.0}]},
        },
    )
    assert net.arcs[0].capacity == 7.0
    assert "overrides" not in base  # the caller's mapping is untouched


def test_a_foul_sewer_is_overridden_by_its_slot_key():
    net = read_wsimod_config(
        {
            "nodes": [
                {"name": "foul_1", "type_": "Sewer", "capacity": 0.1},
                {"name": "storm", "type_": "Sewer"},
            ],
            "overrides": {"nodes": [{"type_": "Foul", "name": "foul_1", "capacity": 0.2}]},
        }
    )
    assert net.nodes[0].parameters["capacity"] == 0.2
    assert net.nodes[1].parameters["capacity"] == 0
    with pytest.raises(ValueError, match="Sewer"):
        read_wsimod_config(
            {
                "nodes": [{"name": "foul_1", "type_": "Sewer"}],
                "overrides": {"nodes": [{"type_": "Sewer", "name": "foul_1", "capacity": 1}]},
            }
        )


def test_nodes_and_arcs_keep_the_configuration_order_in_both_forms():
    names = ["z", "m", "a", "k"]
    nodes = [{"name": n, "type_": "Waste"} for n in names]
    arcs = [
        {"name": f"{x}{y}", "type_": "Arc", "in_port": x, "out_port": y}
        for x, y in [("k", "z"), ("a", "z"), ("m", "z"), ("z", "a")]
    ]
    as_lists = read_wsimod_config({"nodes": nodes, "arcs": arcs})
    as_mappings = read_wsimod_config(
        {"nodes": {n["name"]: n for n in nodes}, "arcs": {a["name"]: a for a in arcs}}
    )
    for net in (as_lists, as_mappings):
        assert [n.name for n in net.nodes] == names
        assert [a.name for a in net.arcs] == ["kz", "az", "mz", "za"]


def test_surfaces_given_as_a_dict_on_the_node_read_like_a_list():
    rural = {"surface": "rural", "type_": "PerviousSurface", "area": 3}
    urban = {"surface": "urban", "type_": "ImperviousSurface"}
    as_dict = read_wsimod_config(
        {"nodes": [{"name": "l", "type_": "Land", "surfaces": {"rural": rural, "urban": urban}}]}
    )
    as_list = read_wsimod_config(
        {"nodes": [{"name": "l", "type_": "Land", "surfaces": [rural, urban]}]}
    )
    for net in (as_dict, as_list):
        assert [(s.name, s.cls) for s in net.nodes[0].surfaces] == [
            ("rural", "PerviousSurface"),
            ("urban", "ImperviousSurface"),
        ]
        assert net.nodes[0].surfaces[0].parameters["area"] == 3


def test_two_treatment_works_do_not_share_their_dicts():
    net = read_wsimod_config(
        {
            "nodes": [{"name": "w1", "type_": "WWTW"}, {"name": "w2", "type_": "WWTW"}],
            "overrides": {
                "nodes": [
                    {
                        "type_": "WWTW",
                        "name": "w1",
                        "liquor_multiplier": {"volume": 0.1},
                        "process_parameters": {"phosphate": {"constant": 0.5}},
                    }
                ]
            },
        }
    )
    w1, w2 = (n.parameters for n in net.nodes)
    assert w1["liquor_multiplier"] is not w2["liquor_multiplier"]
    assert w1["process_parameters"] is not w2["process_parameters"]
    assert w1["liquor_multiplier"]["volume"] == 0.1 and w2["liquor_multiplier"]["volume"] == 0.03
    assert w1["process_parameters"]["phosphate"]["constant"] == 0.5
    assert w2["process_parameters"]["phosphate"]["constant"] == 0.01
    assert NODE_SCHEMAS["WWTW"].parameters["liquor_multiplier"] == {}


def test_a_demand_with_a_residential_override_runs_in_the_demand_slot():
    """``Model.save`` writes ``type_: Demand`` with ``node_type_override: ResidentialDemand``:
    the slot key follows ``type_``, the class the override."""
    network = read_wsimod_config(
        {
            "pollutants": [],
            "additive_pollutants": [],
            "non_additive_pollutants": [],
            "dates": ["2000-01-01"],
            "nodes": [
                {
                    "name": "homes",
                    "type_": "Demand",
                    "node_type_override": "ResidentialDemand",
                    "data_input_dict": {("temperature", "2000-01-01"): 5.0},
                }
            ],
            "arcs": [],
        }
    )
    assert (network.nodes[0].slot_key, network.nodes[0].cls) == ("Demand", "ResidentialDemand")


def test_a_rivers_filename_override_is_read_with_nitrate_among_the_pollutants(tmp_path):
    """``storage.py:750``: a ``River`` with ``nitrate`` reads its temperature at every
    ``calculate_discharge``, so its ``filename`` override is not listed unread; without
    ``nitrate`` it is."""
    path = tmp_path / "r.csv"
    path.write_text("variable,time,value\ntemperature,2000-01-01,11.0\n", encoding="utf-8")

    def mapping(pollutants):
        return {
            "dates": ["2000-01-01"],
            "pollutants": pollutants,
            "additive_pollutants": [x for x in pollutants if x != "temperature"],
            "non_additive_pollutants": ["temperature"] if "temperature" in pollutants else [],
            "nodes": [{"name": "r", "type_": "River"}],
            "arcs": [],
            "overrides": {"nodes": [{"type_": "River", "name": "r", "filename": str(path)}]},
        }

    assert read_wsimod_config(mapping(["phosphate", "temperature"])).unread == ["r.filename"]
    assert read_wsimod_config(mapping(["nitrate", "phosphate", "temperature"])).unread == []


def test_an_arc_override_of_decays_is_refused_with_the_reason():
    """M3: WSIMOD's ``Arc.apply_overrides`` consumes only ``capacity`` and ``preference``
    (``arcs.py:87-96``): the refusal says so instead of "must be a number"."""
    decays = {"phosphate": {"constant": 0.1, "exponent": 1.05}}
    mapping = _decay_arc_mapping("DecayArc", decays=decays)
    mapping["overrides"]["arcs"] = [{"name": "x", "decays": decays}]
    with pytest.raises(ValueError, match=r"arc 'x'.*decays cannot be overridden.*arcs\.py:87-96"):
        read_wsimod_config(mapping)
