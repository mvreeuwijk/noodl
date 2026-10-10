"""Generate the component catalogue (`docs/catalogue/*.md`) from the code itself.

One page per application plus one for the core library. Every public name (the module's
`__all__`) is listed under its role -- set-up, network components, elements, drives,
closures, reactions, sources, files, functions, constants -- with the first line of its
docstring and its call signature. A name exported twice for one object (the old,
model-named spelling kept as an alias of the physical name) is listed once, with its
aliases.

Run `python scripts/gen_catalogue.py` after adding or changing anything public.
`tests/test_docs_catalogue.py` fails when the committed pages are stale or when a public
class or function has no docstring, so the catalogue cannot fall behind the code.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "catalogue"

PAGES = [
    # (file stem, title, page link for the application, modules, intro)
    ("core", "Core library", None,
     ["noodl.topology", "noodl.model", "noodl.refs", "noodl.validation", "noodl.layers",
      "noodl.elements", "noodl.drives", "noodl.nodesources", "noodl.couple"],
     "The framework every application builds on: the network, the layers, the model, the "
     "set-up helpers, and the general-purpose elements and drives."),
    ("building_physics", "Building physics", "../applications/building_physics.md",
     ["noodl.apps.building_physics", "noodl.elements", "noodl.drives"],
     "Multi-zone airflow, heat and contaminants (CONTAM's physics; CONTAM `.prj`/`.wth`, "
     "EPW and Modelica Buildings Library readers). The airflow elements and drives a "
     "building uses come from the core library and are listed here too."),
    ("street_aq", "Street air quality", "../applications/street_aq.md",
     ["noodl.apps.street_aq"],
     "Street-canyon networks: canyon exchange, junction routing, above-roof plumes, "
     "chemistry and exposure (SIRANE and MUNICH physics, case readers and writers)."),
    ("sewer", "Sewers", "../applications/sewer.md", ["noodl.apps.sewer"],
     "Gravity sewers: hydraulics, headspace air and sulfide (SWMM `.inp` reader)."),
    ("water", "Water distribution", "../applications/water.md", ["noodl.apps.water"],
     "Pressurised water networks: pipes, pumps, valves, tanks and demand (EPANET `.inp` "
     "reader)."),
    ("wsimod", "Flow allocation (WSIMOD)", "../applications/allocation.md",
     ["noodl.apps.wsimod"],
     "Rule-based allocation of water between stores on capacity-limited arcs."),
]

ROLES = [
    ("setup", "Set-up", "Build the model and its starting state and inputs."),
    ("component", "Network components",
     "Descriptions of the physical objects a network is built from."),
    ("layer", "Layers and model", "The conservation layers and the model that steps them."),
    ("element", "Elements (branch laws)",
     "Flow through an edge as a function of the potential difference across it."),
    ("drive", "Drives", "Terms added to an edge's potential difference (stack, wind, fans)."),
    ("closure", "Closures",
     "Functions of the state that compute drivers each step (densities, flows, levels)."),
    ("reaction", "Reactions", "Transformations applied to transported species each step."),
    ("source", "Sources", "Nodal sources and withdrawals."),
    ("file", "Files", "Readers and writers of other tools' file formats."),
    ("class", "Other classes", "Supporting types."),
    ("function", "Functions", "Calculations and helpers."),
    ("constant", "Constants and tables", "Named values and option tables."),
]
OPTION_MODULES = {"street_aq": "noodl.apps.street_aq.closures"}
ATTRIBUTE_MODULES = {"building_physics": "noodl.apps.building_physics.attributes"}
"""Pages whose application reads node and edge attributes off the network: the module
defining `NODE_ATTRIBUTES` and `EDGE_ATTRIBUTES`."""
ATTRIBUTE_TABLE_NAMES = {"NODE_ATTRIBUTES", "EDGE_ATTRIBUTES"}
"""Pages whose application has named model options: the module defining `OPTIONS` (option
-> allowed values), `PRESETS` (preset -> its choices) and, in the docstring under
`OPTIONS`, one `- `option`: meaning` bullet per option."""

SETUP_NAMES = {"build_model", "initial_state", "initial_drivers", "project_to_model",
               "drivers_from", "state_from", "check_setup"}


def _summary(obj) -> str:
    """The docstring's first paragraph, on one line; empty when there is none -- including a
    dataclass's auto-generated `Name(field, ...)` text, which is no description."""
    doc = inspect.getdoc(obj) or ""
    if inspect.isclass(obj) and doc.startswith(f"{obj.__name__}("):
        return ""
    return doc.strip().split("\n\n")[0].replace("\n", " ").strip()


def _signature(name: str, obj) -> str:
    try:
        target = obj
        if inspect.isclass(obj) and not dataclasses.is_dataclass(obj):
            target = obj.__init__
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return f"{name}(...)" if callable(obj) else name
    # Names and defaults only: the annotations are in the API reference, and quoted forward
    # references (`name: 'str'`) make a one-line call form hard to read.
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values() if p.name != "self"
    ]
    text = str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))
    return f"{name}{text}"


def _role(name: str, obj) -> str:
    from noodl.elements.base import Element
    from noodl.layers.reaction import Reaction
    from noodl.model import Model
    from noodl.nodesources import NodeSource

    if name in SETUP_NAMES:
        return "setup"
    if not (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj)):
        return "constant"
    if inspect.isclass(obj):
        if issubclass(obj, Element):
            return "element"
        if issubclass(obj, Reaction):
            return "reaction"
        if issubclass(obj, NodeSource) or name.endswith("Source"):
            return "source"
        if obj is Model or name.endswith("Layer") or name in {"Network", "Ports", "CoupledModel"}:
            return "layer"
        if dataclasses.is_dataclass(obj):
            return "component"
        # The instances' own `__call__`, defined by the class or a base (not `object`'s).
        call = next((k.__dict__["__call__"] for k in obj.__mro__[:-1]
                     if "__call__" in k.__dict__), None)
        if call is not None:
            try:
                params = [p for p in inspect.signature(call).parameters if p != "self"]
            except (TypeError, ValueError):
                params = []
            if params[:1] == ["drivers"] or name.endswith("Drive") or name in {
                "Stack", "Wind", "WindProfile",
            }:
                return "drive"
            if params[:2] == ["state", "drivers"]:
                return "closure"
        return "class"
    if name.startswith(("read_", "write_")):
        return "file"
    return "function"


def _entries(modules: list[str]) -> list[dict]:
    seen: dict[int, dict] = {}
    order: list[dict] = []
    for modname in modules:
        mod = importlib.import_module(modname)
        names = getattr(mod, "__all__", None)
        if names is None:
            names = [n for n, o in vars(mod).items() if not n.startswith("_") and (
                inspect.isclass(o) or inspect.isfunction(o)) and o.__module__ == modname]
        for name in names:
            obj = getattr(mod, name)
            key = id(obj) if (inspect.isclass(obj) or inspect.isfunction(obj)) else (modname, name)
            if key in seen:
                entry = seen[key]
                primary = getattr(obj, "__name__", None)
                if name == primary and entry["name"] != primary:
                    entry["aliases"].append(entry["name"])
                    entry["name"] = name
                elif name != entry["name"]:
                    entry["aliases"].append(name)
                continue
            entry = {"name": name, "obj": obj, "aliases": [], "module": modname}
            seen[key] = entry
            order.append(entry)
    for entry in order:
        primary = getattr(entry["obj"], "__name__", None)
        if primary in entry["aliases"]:
            entry["aliases"].remove(primary)
            entry["aliases"].append(entry["name"])
            entry["name"] = primary
    return order


def _constant_text(obj) -> str:
    if isinstance(obj, (int, float, str)):
        return f"`{obj!r}`"
    if isinstance(obj, dict):
        return f"table of {len(obj)} entries: " + ", ".join(f"`{k}`" for k in list(obj)[:8]) + (
            ", ..." if len(obj) > 8 else "")
    return f"`{type(obj).__name__}`"


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def option_help(modname: str) -> dict[str, str]:
    """`{option: meaning}` from the docstring written under `OPTIONS` in `modname`."""
    tree = ast.parse(Path(importlib.import_module(modname).__file__).read_text("utf-8"))
    body = tree.body
    for node, after in zip(body, body[1:], strict=False):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "OPTIONS" for t in targets):
            if isinstance(after, ast.Expr) and isinstance(after.value, ast.Constant):
                text = after.value.value
                found = re.findall(r"^- `(\w+)`: (.*?)(?=^- `|\Z)", text, re.M | re.S)
                return {k: " ".join(v.split()) for k, v in found}
    return {}


def _options_section(modname: str) -> list[str]:
    mod = importlib.import_module(modname)
    options, presets, helps = mod.OPTIONS, mod.PRESETS, option_help(modname)
    lines = [
        "## Model options", "",
        f"Keyword arguments of `build_model` choosing the physics; a preset "
        f"(`preset=`, one of {', '.join(f'`{p}`' for p in presets)}) sets them all, and any "
        f"given explicitly overrides it. A value marked with a preset's name is that "
        f"preset's choice. Defined in `{modname}`.", "",
        "| Option | Values | What it chooses |", "|---|---|---|",
    ]
    for opt, values in options.items():
        shown = []
        for v in values:
            marks = [p for p, chosen in presets.items() if chosen.get(opt) == v]
            shown.append(f"`{v}`" + (f" ({', '.join(marks)})" if marks else ""))
        lines.append(f"| `{opt}` | {'<br>'.join(shown)} | {_esc(helps.get(opt, ''))} |")
    numeric = [k for k in next(iter(presets.values()))
               if k not in options and not isinstance(presets[next(iter(presets))][k], dict)]
    if numeric:
        lines += ["", "Numeric settings each preset fixes (each also a `build_model` keyword):",
                  "", "| Setting | " + " | ".join(f"`{p}`" for p in presets) + " |",
                  "|---|" + "---|" * len(presets)]
        for k in numeric:
            lines.append(f"| `{k}` | " + " | ".join(
                f"`{presets[p].get(k)!r}`" for p in presets) + " |")
    return lines + [""]


def _attributes_section(modname: str) -> list[str]:
    mod = importlib.import_module(modname)
    lines = [
        "## Network attributes", "",
        "What the builders, elements and drives read off `net.add_node(...)` and "
        "`net.add_edge(...)`, by name. Defined in `" + modname + "`.", "",
        "**Nodes**", "", "| Attribute | Meaning | Read by |", "|---|---|---|",
    ]
    for attr, (meaning, reader) in mod.NODE_ATTRIBUTES.items():
        lines.append(f"| `{attr}` | {_esc(meaning)} | `{reader}` |")
    for kind, attrs in mod.EDGE_ATTRIBUTES.items():
        lines += ["", f"**Edges of kind `{kind}`**", "", "| Attribute | Meaning | Read by |",
                  "|---|---|---|"]
        for attr, (meaning, reader) in attrs.items():
            lines.append(f"| `{attr}` | {_esc(meaning)} | `{reader}` |")
    return lines + [""]


def render(stem: str, title: str, link: str | None, modules: list[str], intro: str) -> str:
    entries = _entries(modules)
    by_role: dict[str, list[dict]] = {r: [] for r, _, _ in ROLES}
    for e in entries:
        by_role[_role(e["name"], e["obj"])].append(e)
    lines = [
        f"# {title}: catalogue",
        "",
        "<!-- Generated by scripts/gen_catalogue.py from the code; do not edit by hand. -->",
        "",
        intro + (f" How the pieces fit together: [{title}]({link})." if link else ""),
        "Every application is set up the same way; see [Using noodl](../usage.md).",
        "",
        f"Import from `{modules[0]}`" + (
            "" if len(modules) == 1 else " and the modules named in each table") + ". "
        "The full reference, with every parameter, is in the [API reference](../api.md).",
        "",
    ]
    if stem in OPTION_MODULES:
        lines += _options_section(OPTION_MODULES[stem])
    if stem in ATTRIBUTE_MODULES:
        lines += _attributes_section(ATTRIBUTE_MODULES[stem])
    for role, heading, blurb in ROLES:
        group = by_role[role]
        if role == "constant":
            group = [e for e in group if e["name"] not in ATTRIBUTE_TABLE_NAMES]
        if not group:
            continue
        lines += [f"## {heading}", "", blurb, ""]
        if role == "constant":
            lines += ["| Name | Value |", "|---|---|"]
            for e in group:
                lines.append(f"| `{e['name']}` | {_esc(_constant_text(e['obj']))} |")
        else:
            lines += ["| Name | What it does | How to call it |", "|---|---|---|"]
            for e in group:
                name = e["name"]
                if e["aliases"]:
                    name += "<br><small>also " + ", ".join(
                        f"`{a}`" for a in sorted(e["aliases"])) + "</small>"
                else:
                    name = f"`{name}`"
                if e["aliases"]:
                    name = f"`{e['name']}`" + name[len(e["name"]):]
                where = "" if len(modules) == 1 else f"<br><small>`{e['module']}`</small>"
                lines.append(
                    f"| {name}{where} | {_esc(_summary(e['obj']))} | "
                    f"`{_esc(_signature(e['name'], e['obj']))}` |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def undocumented() -> list[str]:
    """Public classes and functions without a docstring, as `module.name`."""
    missing = []
    for _, _, _, modules, _ in PAGES:
        for e in _entries(modules):
            obj = e["obj"]
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and not _summary(obj):
                missing.append(f"{e['module']}.{e['name']}")
    for modname in OPTION_MODULES.values():
        helps = option_help(modname)
        for opt in importlib.import_module(modname).OPTIONS:
            if opt not in helps:
                missing.append(f"{modname}.OPTIONS[{opt!r}] (no `- `{opt}`: ...` bullet)")
    return missing


def generate() -> dict[Path, str]:
    return {OUT / f"{stem}.md": render(stem, title, link, mods, intro)
            for stem, title, link, mods, intro in PAGES}


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    missing = undocumented()
    if missing:
        print("public names without a docstring:\n  " + "\n  ".join(missing))
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in generate().items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
