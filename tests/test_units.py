"""Units are part of the checked closure contract: `Model.check` reports `unit-mismatch`
when a closure's declared unit differs from the unit of the field or input it refers to, and
`unit-undeclared` for a field with no unit. Strings are compared after trivial respelling;
nothing is converted."""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import pytest
import torch

from noodl.contracts import closure_spec
from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.refs import normalize_unit, units_agree
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parent / "data"


def _model(closures=(), *, source_unit="kg/s"):
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    for a, b in (("ambient", "z1"), ("z1", "z2"), ("z2", "ambient")):
        net.add_edge(a, b, kind="airpath")
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"],
        quantity="pressure", unit="Pa")
    species = TransportLayer(
        net, "species", capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", quantity="mass_fraction", unit="kg/kg",
        source_unit=source_unit, capacity_unit="kg", flow_unit="kg/s")
    return Model(net, {"air": air, "species": species}, closures=closures)


class _Closure:
    def __init__(self, *, outputs=(), output_units=None, input_specs=None):
        self.outputs = tuple(outputs)
        if output_units is not None:
            self.output_units = output_units
        self._specs = input_specs or {}

    @property
    def input_specs(self):
        return self._specs

    def __call__(self, state, drivers):
        return {}


def _issues(model, code):
    return [i for i in model.check().issues if i.code == code]


@pytest.mark.parametrize("a, b", [
    ("m3", "m^3"), ("m3", "m³"), ("kg / m3", "kg/m^3"), ("m**3/s", "m3/s"),
    ("1", "-"), ("-", ""), ("[-]", "dimensionless"), ("°C", "degC"),
    ("µg/m3", "ug/m^3"),
])
def test_trivially_equivalent_spellings_normalise_alike(a, b):
    assert normalize_unit(a) == normalize_unit(b) and units_agree(a, b)


@pytest.mark.parametrize("a, b", [("kg/s", "kg"), ("mPa", "MPa"), ("m3", "m2"), ("1", "1/s")])
def test_different_units_never_compare_equal(a, b):
    assert not units_agree(a, b)


def test_an_output_unit_that_differs_from_the_layer_field_is_a_mismatch_warning():
    model = _model([_Closure(outputs=("species.sources",),
                             output_units={"species.sources": "kg/h"})])
    (issue,) = _issues(model, "unit-mismatch")
    assert issue.level == "warning" and issue.key == "species.sources"
    assert "[kg/h]" in issue.message and "[kg/s]" in issue.message
    assert not model.check().errors


@pytest.mark.parametrize("layer_unit, declared", [
    ("kg/s", "kg/s"), ("kg/s", " kg / s "), ("m3/s", "m^3/s"), ("m3/s", "m³/s"),
    ("-", "1"), ("1", ""),
])
def test_an_output_unit_that_agrees_up_to_spelling_is_not_reported(layer_unit, declared):
    model = _model([_Closure(outputs=("species.sources",),
                             output_units={"species.sources": declared})],
                   source_unit=layer_unit)
    assert not _issues(model, "unit-mismatch")


def test_a_declared_input_unit_that_differs_from_a_layer_field_is_a_mismatch():
    spec = {"species.sources": dict(description="x", unit="g/s", over="nodes")}
    (issue,) = _issues(_model([_Closure(input_specs=spec)]), "unit-mismatch")
    assert issue.key == "species.sources" and "input" in issue.message


def test_two_closures_declaring_one_input_in_different_units_are_a_mismatch():
    def reads(unit):
        return _Closure(input_specs={"T": dict(description="t", unit=unit, over="nodes")})

    assert [i.key for i in _issues(_model([reads("K"), reads("degC")]), "unit-mismatch")] == ["T"]
    assert not _issues(_model([reads("degC"), reads("°C")]), "unit-mismatch")


def test_an_output_is_compared_with_a_declared_input_of_another_closure():
    reader = _Closure(input_specs={"rho": dict(description="d", unit="kg/m3", over="nodes")})
    ok = _Closure(outputs=("rho",), output_units={"rho": "kg/m^3"})
    bad = _Closure(outputs=("rho",), output_units={"rho": "g/m3"})
    assert not _issues(_model([ok, reader]), "unit-mismatch")
    assert len(_issues(_model([bad, reader]), "unit-mismatch")) == 1


def test_a_unit_that_is_not_declared_is_never_a_mismatch():
    model = _model([_Closure(outputs=("species.sources",), output_units={"species.sources": ""})])
    assert not _issues(model, "unit-mismatch")


def test_a_field_without_a_unit_is_reported_as_info():
    issues = _issues(_model(source_unit=""), "unit-undeclared")
    assert {i.level for i in issues} == {"info"}
    assert "species.sources" in {i.key for i in issues}
    assert "species.sources" not in {i.key for i in _issues(_model(), "unit-undeclared")}


def test_the_closure_spec_reads_output_units_and_refuses_a_malformed_one():
    spec = closure_spec(_Closure(outputs=("a",), output_units={"a": "K"}), index=0)
    assert dict(spec.output_units) == {"a": "K"}
    assert dict(closure_spec(_Closure(), index=0).output_units) == {}
    for bad in ("K", {"a": 1}, {1: "K"}):
        with pytest.raises(TypeError, match="output_units"):
            closure_spec(_Closure(output_units=bad), index=0)


# ---- every application's standard build -----------------------------------------------


def _contam(path):
    from noodl.apps.building_physics.prj import project_to_model, read_prj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return project_to_model(read_prj(path))[0]


def _water():
    from noodl.apps.water.network import build_model, twoloop

    return build_model(twoloop(), quality=0.5)[0]


def _sewer_kinematic():
    from noodl.apps.sewer import build_model, tree_steady

    return build_model(tree_steady(), air=False, quality=True, species=("bod", "tracer"))[0]


def _sewer_dynamic():
    from noodl.apps.sewer import build_model, read_swmm_inp

    net = read_swmm_inp(DATA / "sewer" / "storage_dynwave.inp")[0]
    return build_model(net, quality=True, species=("bod",))[0]


def _street():
    from noodl.apps.street_aq.chemistry import photostationary_for_streets
    from noodl.apps.street_aq.network import build_model, from_test_network

    species = ("no", "no2", "o3")
    return build_model(from_test_network(), species=species, pblh_floor=False,
                       chemistry=photostationary_for_streets(species))[0]


def _wsimod():
    from noodl.apps.wsimod.network import build_model

    return build_model(DATA / "wsimod" / "quickstart_topology.json")[0]


def _electrical():
    from noodl.apps.electrical import Circuit, Resistor, VoltageSource, build_model

    return build_model(Circuit([VoltageSource("V1", "a", "0", 1.0),
                                Resistor("R1", "a", "0", 100.0)]))[0]


_CASES = (
    [pytest.param(lambda p=p: _contam(p), id=Path(p).stem)
     for p in sorted(glob.glob(str(DATA / "contam" / "*.prj")))]
    + [pytest.param(f, id=f.__name__.strip("_"))
       for f in (_water, _sewer_kinematic, _sewer_dynamic, _street, _wsimod, _electrical)]
)


@pytest.mark.parametrize("build", _CASES)
def test_every_applications_standard_build_has_no_unit_mismatch(build):
    try:
        model = build()
    except ValueError as exc:       # a committed project the reader refuses (duct leakage)
        pytest.skip(str(exc))
    assert [str(i) for i in _issues(model, "unit-mismatch")] == []


def test_the_dynamic_wave_closure_carried_state_has_units():
    model = _sewer_dynamic()
    assert model.refs.fields()["sewer.pipe_flow"].unit == "m3/s"
    assert not _issues(model, "unit-undeclared")
