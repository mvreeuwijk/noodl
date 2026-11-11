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
    (_node(type_="Waste", filename="x.csv"), "data_input_dict"),
    (_node(type_="Catchment", filename="missing.csv"), "missing.csv"),
    (_node(type_="Waste", name="a"), "'a'"),
    (_arc(type_="DecayArc"), "3b"),
    (_arc(type_="Arc", decays={}), "decays"),
    (_arc(type_="Arc", diameter=1), "diameter"),
    (_arc(type_="Arc", number_of_timesteps=2), "number_of_timesteps"),
    (_arc(type_="QueueArc", number_of_timesteps=1.5), "number_of_timesteps"),
    (_arc(type_="Arc", capacity="big"), "capacity"),
    (_arc(type_="Arc", in_port="nowhere"), "nowhere"),
    (_override(node={"type_": "Catchment", "name": "nowhere"}), "nowhere"),
    (_override(node={"type_": "Reservoir", "name": "a"}), "Reservoir"),
    (_override(node={"type_": "Land", "name": "a"}), "'a'"),  # a is filed under Catchment
    (_override(node={"type_": "Catchment", "name": "a", "capacity": 1}), "capacity"),
    (_override(node={"type_": "Catchment", "name": "a", "filename": "x.csv"}), "filename"),
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
    (_node(type_="Storage", decays={}), "3b"),
    (_node(type_="Groundwater", decays={"phosphate": {"constant": 1}}), "decays"),
    (_node(type_="Land", surfaces=[{"type_": "PerviousSurface", "decays": {"a": 1}}]), "3b"),
    (_node(type_="Waste", data_input_dict={}), "data_input_dict"),
    (_override(node={"type_": "Catchment", "name": "a", "decays": {}}), "decays"),
    (_override(arc={"name": "nowhere", "type_": "Arc"}), "nowhere"),
    (_override(arc={"name": "ab", "type_": "Arc", "diameter": 1}), "diameter"),
]


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
