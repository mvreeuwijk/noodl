"""`noodl._closures.ClosureSchedule`: dispatch, `written`, probe records, the state-key
refusal and its carve-out, the live closure list, and copies."""

import copy
import pickle
from types import SimpleNamespace

import pytest

from noodl._closures import ClosureSchedule
from noodl.contracts import StepContext

# "heat" is a Kirchhoff layer; "street" a transport layer no Kirchhoff layer owns (its ".q"
# is a driver); "species" a transport layer whose flows "heat" provides.
LAYERS = {"heat": None, "street": None, "species": None}
TRANSPORT = {"street": None, "species": None}
FLOW_LAYER_OF = {"street": None, "species": "heat"}


def _schedule(closures):
    return ClosureSchedule(closures, layers=LAYERS, transport=TRANSPORT,
                           flow_layer_of=FLOW_LAYER_OF)


class Writes:
    def __init__(self, **out):
        self.out = out

    def __call__(self, state, drivers):
        return dict(self.out)


class Clock:
    """Integrating closure carrying "clock.t"."""

    integrates = True
    state_keys = ("clock.t",)

    def __call__(self, state, drivers, ctx):
        dt = 0.0 if ctx is None else ctx.dt
        return {"clock.t": state.get("clock.t", 0.0) + dt}


def _probe():
    return SimpleNamespace(driver_reads=set(), state_reads=set(), closure_outputs={},
                           closure_values={}, closure_writers={}, closure_returned={})


def test_apply_runs_in_order_and_later_closures_see_earlier_writes():
    def first(state, drivers):
        return {"a": 1.0}

    def second(state, drivers):
        return {"b": drivers["a"] + 1.0}

    closures = [first, second]
    out = _schedule(closures).apply(closures, {}, {"given": 0.0})
    assert out == {"given": 0.0, "a": 1.0, "b": 2.0}


def test_dispatch_passes_ctx_to_integrating_closures_only():
    seen = []

    def algebraic(state, drivers):
        seen.append("two")
        return {}

    closures = [algebraic, Clock()]
    sched = _schedule(closures)
    out = sched.apply(closures, {"clock.t": 1.0}, {}, ctx=StepContext(dt=2.0))
    assert out["clock.t"] == 3.0 and seen == ["two"]
    assert sched.apply(closures, {"clock.t": 1.0}, {})["clock.t"] == 1.0   # a query
    with pytest.raises(ValueError, match="cannot be evaluated by steady"):
        sched.apply(closures, {}, {}, ctx=StepContext(dt=None))
    assert sched.integrating(closures) == {id(closures[1])}


def test_written_names_only_the_closure_state_keys_a_closure_returned():
    closures = [Clock(), Writes(other=1.0)]
    sched = _schedule(closures)
    assert sched.closure_state_keys == {"clock.t": closures[0]}
    written: set = set()
    sched.apply(closures, {}, {"clock.t": 9.0}, written=written, ctx=StepContext(dt=1.0))
    assert written == {"clock.t"}
    written = set()
    sched.apply([closures[1]], {}, {"clock.t": 9.0}, written=written)
    assert written == set()          # a caller's same-named driver is not a closure's write


def test_probe_records_reads_and_returns():
    def reader(state, drivers):
        return {"out": state["s"] + drivers["d"]}

    closures = [reader]
    probe = _probe()
    out = _schedule(closures).apply(closures, {"s": 1.0, "t": 0.0}, {"d": 2.0}, probe=probe)
    assert out["out"] == 3.0
    assert probe.state_reads == {"s"} and "d" in probe.driver_reads
    assert probe.closure_outputs == {"out": 3.0} == probe.closure_values
    assert probe.closure_writers == {"out": "function"}
    assert probe.closure_returned == {id(reader): {"out"}}


@pytest.mark.parametrize("key", ["heat.phi", "heat.q", "species.x", "species.q", "street.x"])
def test_returning_a_layer_state_key_is_refused(key):
    closures = [Writes(**{key: 1.0})]
    with pytest.raises(ValueError, match=rf"returned '{key}', which is a state key"):
        _schedule(closures).apply(closures, {}, {})


def test_the_ownerless_transport_flow_is_a_driver():
    closures = [Writes(**{"street.q": 1.0})]
    sched = _schedule(closures)
    assert sched.apply(closures, {}, {})["street.q"] == 1.0
    assert not sched.is_layer_state_key("street.q")
    assert sched.is_layer_state_key("species.q") and sched.is_layer_state_key("heat.phi")
    assert not sched.is_layer_state_key("rho_amb")


def test_construction_refusals():
    class DeclaresState:
        outputs = ("heat.phi",)

        def __call__(self, state, drivers):
            return {}

    with pytest.raises(ValueError, match="declares output 'heat.phi', which is a state key"):
        _schedule([DeclaresState()])

    class CarriesLayer(Clock):
        state_keys = ("species.x",)

    with pytest.raises(ValueError, match="which is layer 'species''s own state key"):
        _schedule([CarriesLayer()])

    class CarriesFlow(Clock):
        state_keys = ("street.q",)

    assert "street.q" in _schedule([CarriesFlow()]).closure_state_keys
    with pytest.raises(ValueError, match="both declare the state key 'clock.t'"):
        _schedule([Clock(), Clock()])


def test_check_coupling():
    class Filter:
        requires_coupling = "iterate"
        reads_flows_of = "heat"

        def __call__(self, state, drivers):
            return {}

    sched = _schedule([Filter()])
    sched.check_coupling("iterate", {"heat": 1e-9})
    with pytest.raises(ValueError, match="Filter reads the flows of layer 'heat'"):
        sched.check_coupling("pingpong", {"heat": 1e-9})
    with pytest.raises(ValueError, match=r"iterate_flow_tol=\{\}"):
        sched.check_coupling("iterate", {})


def test_a_closure_added_to_the_live_list_is_read_on_first_use():
    closures = [Writes(a=1.0)]
    sched = _schedule(closures)
    closures.append(Clock())
    closures.insert(0, Writes(z=0.0))
    out = sched.apply(closures, {}, {}, ctx=StepContext(dt=2.0))
    assert out == {"z": 0.0, "a": 1.0, "clock.t": 2.0}
    assert sched.integrating(closures) == {id(closures[2])}
    closures[:] = [closures[0]]
    assert sched.apply(closures, {}, {}) == {"z": 0.0}


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))])
def test_a_copy_keeps_dispatching_integrating_closures(clone):
    closures = [Clock()]
    sched = _schedule(closures)
    sched.apply(closures, {}, {}, ctx=StepContext(dt=1.0))       # fills the identity lookup
    copied_sched, copied_closures = clone((sched, closures))
    assert copied_sched._ids is None
    out = copied_sched.apply(copied_closures, {"clock.t": 1.0}, {}, ctx=StepContext(dt=2.0))
    assert out["clock.t"] == 3.0
    assert copied_sched.integrating(copied_closures) == {id(copied_closures[0])}
