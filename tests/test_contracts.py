"""`noodl.contracts`: closure dispatch, the one reader of the closure attributes, and the
`species_columns` checks."""

from __future__ import annotations

import functools
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from noodl import contracts
from noodl.contracts import (
    CLOSURE_ATTRIBUTES,
    AlgebraicClosure,
    IntegratingClosure,
    StepContext,
    call_closure,
    check_species_columns,
    closure_spec,
    reaction_species_columns,
)

S, D = {"s": torch.tensor(1.0)}, {"d": torch.tensor(2.0)}


def _mark(**attrs):
    def deco(f):
        for k, v in attrs.items():
            setattr(f, k, v)
        return f
    return deco


# ---- dispatch ----------------------------------------------------------------------------


def test_two_arg_closure_is_called_without_ctx():
    spec = closure_spec(lambda s, d: {"x": s["s"] + d["d"]}, index=0)
    assert not spec.integrates and not spec.declares_integrates
    assert spec(S, D, StepContext(dt=1.0))["x"] == 3.0     # ctx is never passed on


def test_three_arg_closure_receives_the_ctx():
    @_mark(integrates=True)
    def clock(state, drivers, ctx):
        return {"dt": torch.tensor(-1.0 if ctx is None else ctx.dt)}

    spec = closure_spec(clock, index=0)
    assert spec.integrates and spec.declares_integrates
    assert spec(S, D, StepContext(dt=2.0))["dt"] == 2.0
    assert spec(S, D, None)["dt"] == -1.0                    # a query
    assert spec(S, D)["dt"] == -1.0


def test_star_args_integrating_closure_is_accepted():
    @_mark(integrates=True)
    def f(*args):
        return {"n": torch.tensor(float(len(args)))}

    assert closure_spec(f, index=0)(S, D, StepContext(dt=1.0))["n"] == 3.0


def test_bound_method_partial_and_lambda():
    class Obj:
        integrates = True

        def __call__(self, state, drivers, ctx):
            return {"k": torch.tensor(1.0)}

        def method(self, state, drivers):
            return {"m": torch.tensor(2.0)}

    assert closure_spec(Obj(), index=0)(S, D, StepContext(1.0))["k"] == 1.0
    assert closure_spec(Obj().method, index=0)(S, D)["m"] == 2.0

    def scaled(factor, state, drivers):
        return {"p": torch.tensor(factor)}

    assert closure_spec(functools.partial(scaled, 4.0), index=0)(S, D)["p"] == 4.0
    assert closure_spec(lambda s, d: {}, index=0)(S, D) == {}


def test_steady_refuses_an_integrating_closure():
    @_mark(integrates=True)
    def f(s, d, ctx):
        return {}

    spec = closure_spec(f, index=0)
    with pytest.raises(ValueError, match=r"declares integrates=True and cannot be evaluated "
                                         r"by steady\(\), which has no interval"):
        spec(S, D, StepContext(dt=None))
    # an algebraic closure is fine under steady()
    assert closure_spec(lambda s, d: {}, index=0)(S, D, StepContext(dt=None)) == {}


def test_call_closure_takes_a_bare_closure_or_a_spec():
    @_mark(integrates=True)
    def f(s, d, ctx):
        return {"dt": torch.tensor(ctx.dt)}

    assert call_closure(f, S, D, StepContext(3.0))["dt"] == 3.0
    assert call_closure(closure_spec(f, index=0), S, D, StepContext(4.0))["dt"] == 4.0
    assert call_closure(lambda s, d: {"a": s["s"]}, S, D)["a"] == 1.0


def test_spec_survives_deepcopy_and_pickle_with_its_closure():
    import copy

    class Clock:
        integrates = True
        t = 0.0

        def __call__(self, state, drivers, ctx):
            if ctx is not None:
                self.t += ctx.dt
            return {"t": torch.tensor(self.t)}

    spec = closure_spec(Clock(), index=0)
    clone = copy.deepcopy(spec)
    assert clone.closure is not spec.closure and clone.integrates
    assert clone(S, D, StepContext(2.0))["t"] == 2.0
    assert spec.closure.t == 0.0


def test_protocols():
    assert isinstance(lambda s, d: {}, AlgebraicClosure)
    assert not isinstance(3, AlgebraicClosure)

    class Integ:
        integrates = True

        def __call__(self, s, d, ctx):
            return {}

    assert isinstance(Integ(), IntegratingClosure)


# ---- the declarations ---------------------------------------------------------------------


def test_every_declared_attribute_is_read_into_the_spec():
    labels = {"a": ("x", "y")}
    specs = {"u": 1}

    @_mark(integrates=False, state_keys=["h"], key_labels=labels, inputs=["a", "b"],
           outputs=("c",), input_specs=specs, edge_kinds=["pipe"], requires_coupling="iterate",
           reads_flows_of="net", species_columns={"t": {"bod": 0}})
    def f(s, d):
        return {}

    spec = closure_spec(f, index=0)
    assert spec.state_keys == ("h",) and spec.declares_integrates and not spec.integrates
    assert spec.key_labels == labels
    assert spec.reads == {"a", "b"} and spec.writes == {"c"}
    assert spec.edge_kinds == ("pipe",)
    assert spec.requires_coupling == "iterate" and spec.reads_flows_of == "net"
    assert spec.species_columns == {"t": {"bod": 0}}
    assert spec.input_specs == specs
    assert spec.label == repr(f)


def test_undeclared_defaults():
    spec = closure_spec(lambda s, d: {}, index=0)
    assert spec.reads is None and spec.writes is None
    assert spec.state_keys == () and spec.edge_kinds == () and spec.key_labels == {}
    assert spec.species_columns == {} and spec.input_specs == {}
    assert spec.requires_coupling is None and spec.reads_flows_of is None


def test_declared_empty_is_not_undeclared():
    @_mark(inputs=(), outputs=())
    def f(s, d):
        return {}

    spec = closure_spec(f, index=0)
    assert spec.reads == frozenset() and spec.writes == frozenset()


def test_input_specs_is_read_lazily():
    class Lazy:
        calls = 0

        @property
        def input_specs(self):
            Lazy.calls += 1
            return {"k": 1}

        def __call__(self, s, d):
            return {}

    spec = closure_spec(Lazy(), index=0)
    n = Lazy.calls
    assert spec.input_specs == {"k": 1} and Lazy.calls == n + 1


def test_not_callable_message():
    with pytest.raises(TypeError, match=r"^Model: closure 2 is a int, which is not callable; "
                                        r"a closure is called as closure\(state, drivers\)"):
        closure_spec(3, index=2)
    with pytest.raises(TypeError, match="^Other: closure 0"):
        closure_spec(None, index=0, owner="Other")


def test_state_keys_without_integrates_message():
    @_mark(state_keys=("h",))
    def f(s, d):
        return {}

    with pytest.raises(ValueError, match=r"declares state_keys \('h',\) but not `integrates`; "
                                         r"set `integrates = True`"):
        closure_spec(f, index=0)


def test_integrates_true_with_two_positional_parameters_message():
    @_mark(integrates=True)
    def f(s, d):
        return {}

    with pytest.raises(ValueError, match=r"declares integrates=True but its call signature "
                                         r"takes 2 positional arguments; an integrating "
                                         r"closure is called as closure\(state, drivers, ctx\)"):
        closure_spec(f, index=0)


@pytest.mark.parametrize("attr", ["inputs", "outputs", "state_keys", "edge_kinds"])
def test_a_bare_string_declaration_is_refused(attr):
    def f(s, d, ctx=None):
        return {}

    f.integrates = False
    setattr(f, attr, "rho_amb")
    with pytest.raises(TypeError, match=rf"declares {attr} as the string 'rho_amb'"):
        closure_spec(f, index=0)


def test_a_non_iterable_declaration_is_refused():
    @_mark(outputs=3)
    def f(s, d):
        return {}

    with pytest.raises(TypeError, match="declares outputs as a int"):
        closure_spec(f, index=0)


@pytest.mark.parametrize("bad", [1, "yes", 0.0])
def test_a_non_bool_integrates_is_refused(bad):
    @_mark(integrates=bad)
    def f(s, d, ctx):
        return {}

    with pytest.raises(TypeError, match="it must be True or False"):
        closure_spec(f, index=0)


@pytest.mark.parametrize("bad", [["t"], {"t": ["bod"]}, {"t": {"bod": "0"}},
                                 {"t": {"bod": True}}, {1: {"bod": 0}}, {"t": {1: 0}}])
def test_malformed_species_columns_is_refused(bad):
    @_mark(species_columns=bad)
    def f(s, d):
        return {}

    with pytest.raises(TypeError, match="species_columns that is not"):
        closure_spec(f, index=0)


def test_reaction_species_columns():
    assert reaction_species_columns(SimpleNamespace()) == {}
    r = SimpleNamespace(species_columns={"t": {"a": 1}})
    assert reaction_species_columns(r) == {"t": {"a": 1}}
    with pytest.raises(TypeError, match="reaction .* declares species_columns"):
        reaction_species_columns(SimpleNamespace(species_columns={"t": ["a"]}))


# ---- the table and the single reader ------------------------------------------------------


def test_closure_attributes_table_is_complete():
    names = [a.name for a in CLOSURE_ATTRIBUTES]
    assert len(names) == len(set(names))
    assert set(names) == {
        "integrates", "state_keys", "key_labels", "inputs", "outputs", "input_specs",
        "edge_kinds", "requires_coupling", "reads_flows_of", "species_columns"}
    assert all(a.meaning and a.consumer and a.type for a in CLOSURE_ATTRIBUTES)


def test_closure_spec_reads_every_attribute_in_the_table():
    # every attribute of the table is read from the closure by closure_spec / ClosureSpec
    source = Path(contracts.__file__).read_text(encoding="utf-8")
    for attr in CLOSURE_ATTRIBUTES:
        assert re.search(rf"""["']{attr.name}["']|getattr\([^)]*["']{attr.name}["']""", source), \
            attr.name


@pytest.mark.xfail(
    reason="until the remaining S01 adopters (sewer network.py, building outputs.py) read "
           "declarations through closure_spec",
    strict=False)
def test_no_code_outside_contracts_reads_a_closure_attribute_directly():
    root = Path(contracts.__file__).parent
    names = "|".join(a.name for a in CLOSURE_ATTRIBUTES)
    pattern = re.compile(rf"""(?:getattr|hasattr)\(\s*[^,()]+,\s*["'](?:{names})["']""")
    # (relative path, matched text): `input_specs` of the model and of a reaction are read by
    # refs.py itself -- they are not closures, which closure_spec reads.
    allowed: set[tuple[str, str]] = {
        ("refs.py", 'getattr(model, "input_specs"'),
        ("refs.py", 'getattr(r, "input_specs"'),
    }
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "contracts.py" and path.parent == root:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = pattern.search(line)
            if m and (path.relative_to(root).as_posix(), m.group(0)) not in allowed:
                offenders.append(f"{path.relative_to(root).as_posix()}:{lineno}: {line.strip()}")
    assert offenders == []


# ---- check_species_columns ----------------------------------------------------------------


def _layer(n_species, species_names=None):
    return SimpleNamespace(n_species=n_species, species_names=species_names)


def test_check_species_columns_accepts_matching_declarations():
    transport = {"sew": _layer(3, ("bod", "sulfide", "o2")), "anon": _layer(2)}
    check_species_columns("c", {}, transport)
    check_species_columns("c", {"sew": {"bod": 0, "o2": 2}}, transport)
    check_species_columns("c", {"anon": {"a": 0, "b": 1}}, transport)


def test_check_species_columns_unknown_layer():
    with pytest.raises(ValueError, match=r"^c: species_columns names 'x', not a transport "
                                         r"layer of this model \(\['sew'\]\)"):
        check_species_columns("c", {"x": {"a": 0}}, {"sew": _layer(1)})


def test_check_species_columns_column_out_of_range():
    for col in (2, -1):
        with pytest.raises(ValueError, match="outside the layer's 2 species column"):
            check_species_columns("c", {"t": {"a": col}}, {"t": _layer(2)})


def test_check_species_columns_disagrees_with_species_names():
    # the R05 class: defaults (bod=0, sulfide=1) over species ("sulfide", "bod")
    transport = {"t": _layer(2, ("sulfide", "bod"))}
    with pytest.raises(ValueError, match=r"species_names\[0\] = 'sulfide'"):
        check_species_columns("c", {"t": {"bod": 0, "sulfide": 1}}, transport)
    check_species_columns("c", {"t": {"bod": 1, "sulfide": 0}}, transport)


def test_check_species_columns_layer_without_names_needs_distinct_columns():
    with pytest.raises(ValueError, match="no species_names to tell them apart"):
        check_species_columns("c", {"t": {"a": 0, "b": 0}}, {"t": _layer(2)})
