"""The component catalogue (`docs/catalogue/`) cannot fall behind the code.

Adding or changing a public class, function or option means updating its docstring and
re-running `python scripts/gen_catalogue.py`; these tests fail until both are done.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_catalogue", ROOT / "scripts" / "gen_catalogue.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_every_public_name_and_option_is_documented():
    assert gen.undocumented() == [], (
        "public names without a docstring (or options without a description) -- document "
        "them, then run `python scripts/gen_catalogue.py`"
    )


def test_the_committed_catalogue_matches_the_code():
    stale = [
        path.name for path, text in gen.generate().items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    assert stale == [], (
        f"docs/catalogue is out of date for {stale}; run `python scripts/gen_catalogue.py`"
    )


def test_every_catalogue_page_is_in_the_navigation():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    missing = [p.name for p in gen.generate() if f"catalogue/{p.name}" not in nav]
    assert missing == []


def test_components_elements_and_drives_state_their_units():
    """A user must learn the unit of every coefficient from help()/the catalogue."""
    import inspect

    from noodl.apps.street_aq import Street
    from noodl.drives import Stack, Wind
    from noodl.elements import Conductance, Damper, Duct, FanCurve, Orifice, PowerLaw, Quadratic

    first = lambda o: (inspect.getdoc(o) or "").split("\n\n")[0]  # noqa: E731
    assert "(m)" in first(Street) and "z0_b" in first(Street)
    assert "Pa" in first(PowerLaw) and "flow unit" in first(PowerLaw)
    assert "m2" in first(Orifice)
    assert "per Pa" in first(Conductance)
    assert "Pa" in first(FanCurve) and "q_max" in first(FanCurve)
    assert "Pa" in first(Quadratic)
    assert "Pa" in first(Damper)
    assert "(m)" in first(Duct)
    assert inspect.getdoc(Stack.from_network) and "z_path" in inspect.getdoc(Stack.from_network)
    assert "m/s" in first(Wind)


def test_every_documented_building_attribute_is_read_by_the_code():
    import re

    from noodl.apps.building_physics import EDGE_ATTRIBUTES, NODE_ATTRIBUTES

    src = ROOT / "src" / "noodl"
    code = "".join(p.read_text(encoding="utf-8") for p in [
        src / "drives.py", src / "refs.py", src / "apps" / "building_physics" / "thermal.py",
        src / "apps" / "building_physics" / "elements.py",
        src / "apps" / "building_physics" / "prj.py",
        src / "apps" / "building_physics" / "components.py",
        src / "components" / "_flatten.py",
    ])
    names = list(NODE_ATTRIBUTES) + [a for kind in EDGE_ATTRIBUTES.values() for a in kind]
    unread = [n for n in names if not re.search(rf"[\"']{re.escape(n)}[\"']", code)]
    assert unread == []
    page = gen.render(*[p for p in gen.PAGES if p[0] == "building_physics"][0])
    assert "## Network attributes" in page and "`z_path`" in page and "`heat_capacity`" in page
    assert "NODE_ATTRIBUTES" not in page


def test_generator_signatures_roles_and_summaries():
    from noodl.apps import building_physics as bp
    from noodl.drives import Drive
    from noodl.elements.base import Element
    from noodl.model import Closure, StepContext
    from noodl.refs import Field
    from noodl.validation import SetupReport

    assert "*args" not in gen._signature("Field", Field)
    assert gen._signature("Drive", Drive) == "Drive(...)"
    assert gen._role("SetupReport", SetupReport) == "class"
    assert gen._role("StepContext", StepContext) == "class"
    assert gen._role("Element", Element) == "base"
    assert gen._role("Closure", Closure) == "base"
    assert gen._summary(bp.initial_drivers).endswith(".")


def test_alias_constants_are_the_same_object_and_listed_once():
    import importlib

    for old, new in gen.CONSTANT_ALIASES.items():
        mod = importlib.import_module("noodl.apps.street_aq")
        assert getattr(mod, old) is getattr(mod, new), (old, new)
    page = gen.render(*[p for p in gen.PAGES if p[0] == "street_aq"][0])
    assert page.count("| `KAPPA_040`") == 1 and "| `KAPPA` |" not in page
    assert "also `KAPPA`" in page
