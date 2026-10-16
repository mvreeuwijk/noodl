"""The parts of the two JSON formats that both readers share.

A signal record (`Signal`, read by `signal_from`), the structural checks the two `load`
functions build on (`require_dict`, `require_str`, `require_list`), and `flag`, the one way
a Boolean parameter is read. Each format's components, connections and top-level keys are
its reader's own (`noodl.apps.building_physics.modelica.schema`,
`noodl.apps.electrical.modelica.schema`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from noodl.modelica.errors import ModelicaImportError

__all__ = ["CONTROLLERS", "Signal", "fail", "flag", "require_dict", "require_list", "require_str",
           "signal_from"]

# The feedback blocks a signal may feed (MSL 4.1.0); the building reader evaluates them
# (`noodl.apps.building_physics.modelica.feedback`).
CONTROLLERS = frozenset({"Modelica.Blocks.Continuous.LimPID"})


@dataclass(frozen=True)
class Signal:
    """One instance from the JSON's `signals` array: a driver plus the input(s) it feeds.

    `drives` is written in the JSON as one `"<instance>.<input>"` string or a list of them
    (one block output may feed several inputs: `Examples/ZonalFlow.mo` connects one
    `Constant` to both `floExc.mAB_flow` and `floExc.mBA_flow`); it is always a tuple here.
    """

    name: str
    cls: str
    parameters: dict
    drives: tuple[str, ...]


def fail(message: str) -> ModelicaImportError:
    """A `ModelicaImportError` with the readers' `"modelica: "` prefix."""
    return ModelicaImportError(f"modelica: {message}")


def require_dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise fail(f"{where} must be an object, found {value!r}")
    return dict(value)


def require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise fail(f"{where} must be a string, found {value!r}")
    return value


def require_list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise fail(f"{where} must be an array, found {value!r}")
    return value


def signal_from(raw: Any, index: int) -> Signal:
    """The `index`-th entry of a document's `signals` array."""
    where = f"signals[{index}]"
    obj = require_dict(raw, where)
    name = require_str(obj.get("name"), f"{where}.name")
    where = f"{where} ({name})"
    cls = require_str(obj.get("class"), f"{where}.class")
    parameters = require_dict(obj.get("parameters", {}), f"{where}.parameters")
    drives = obj.get("drives")
    if isinstance(drives, str):
        drives = [drives]
    if not (isinstance(drives, list) and drives and all(isinstance(d, str) for d in drives)):
        raise fail(f"{where}.drives must be a string or a non-empty list of strings, "
                   f"found {obj.get('drives')!r}")
    return Signal(name=name, cls=cls, parameters=parameters, drives=tuple(drives))


def flag(parameters: Mapping, key: str, *, default: bool = False,
         problems: list[str] | None = None, where: str = "") -> bool:
    """The Boolean parameter `key`: its Modelica `default` when the export does not hold it.

    Any other value than true or false is a problem: appended to `problems` (and `default`
    returned) when a list is given, raised as a `ModelicaImportError` otherwise. `where`
    names the instance in the message.
    """
    value = parameters.get(key, default)
    if isinstance(value, bool):
        return value
    message = f"{where}: parameter {key} must be true or false, found {value!r}"
    if problems is None:
        raise fail(message)
    problems.append(message)
    return default
