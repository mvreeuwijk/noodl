"""CONTAM control nodes (TN 1887 Appendix A, section 12b) for the `.prj` reader.

`read_control_nodes` parses the section into `PrjControlNode` records. `plan_controls` keeps
the nodes an actuator or a named report node depends on. An actuator is a zone's, path's,
source's, segment's, terminal's or AHS path's control reference; a named report node is a
`pas` node with a name, which ContamX reports through its API. The nodes are put in ContamX's
computation order, the file's sequence numbers. A needed node of a type this reader does not
build is refused, naming what needs it; unused nodes of any known type are not built (NIST's
own sample projects carry integration nodes that feed only report nodes). `control_network`
builds the sampled `ControlNetwork` once the model's layers exist (the sensors' readings
depend on them). What each node computes is in `controls`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from noodl.apps.building_physics.controls import (
    AboveLimit,
    Absolute,
    Actuator,
    Affine,
    And,
    Average,
    BelowLimit,
    Block,
    Complement,
    Constant,
    ControlNetwork,
    Divide,
    DriverValue,
    Excess,
    ExclusiveOr,
    Hysteresis,
    Identity,
    IncrementalPI,
    LowerBandSwitch,
    Maximum,
    Minimum,
    Or,
    Positive,
    Product,
    ProportionalControl,
    Sensor,
    Shortfall,
    Sum,
    UpperBandSwitch,
)

# Node type -> the number of data values on the line after its description (None: no data
# line; -1: a count followed by that many node numbers), TN 1887 Appendix A section 12b.
_DATA: dict[str, int | None] = {
    "sns": 12, "sch": 1, "set": 1, "cvf": 1, "dvf": 1, "log": 5, "pas": None, "mod": 2,
    "hys": 3, "abs": None, "bin": None, "dls": 2, "dlx": 2, "int": None, "rav": 1,
    "inv": None, "and": None, "or": None, "xor": None, "add": None, "sub": None,
    "mul": None, "div": None, "sum": -1, "avg": -1, "max": -1, "min": -1, "lls": None,
    "uls": None, "lbs": 1, "ubs": 1, "llc": None, "ulc": None, "pc1": 1, "pi1": 4,
    "sup": 4, "sph": None,
}
_UNSUPPORTED = {
    "cvf": "a continuous values file", "dvf": "a discrete values file",
    "mod": "a modifier", "dls": "a scheduled delay", "dlx": "an exponential delay",
    "int": "integration over time", "rav": "a running average", "sup": "a super node",
    "sph": "a phantom node",
}
_ARITY = {"pas": 1, "log": 1, "hys": 1, "abs": 1, "bin": 1, "inv": 1, "pc1": 1, "pi1": 1,
          "and": 2, "or": 2, "xor": 2, "add": 2, "sub": 2, "mul": 2, "div": 2, "lls": 2,
          "uls": 2, "lbs": 2, "ubs": 2, "llc": 2, "ulc": 2}
NO_NAME = "<none>"
INPUT_KEY = "controls.input"


@dataclass(frozen=True)
class PrjControlNode:
    """One control node: `nr type seq flags inreq n1 n2 name`, its description line and its
    data values (`data`, strings as in the file)."""

    nr: int
    type: str
    seq: int
    flags: int
    n1: int
    n2: int
    name: str
    data: tuple[str, ...]

    @property
    def inputs(self) -> tuple[int, ...]:
        """The nodes it reads: `n1`, `n2`, or the listed nodes of sum/avg/max/min."""
        if _DATA.get(self.type) == -1:
            count = int(self.data[0])
            return tuple(int(v) for v in self.data[1:1 + count])
        return tuple(n for n in (self.n1, self.n2) if n)

    @property
    def label(self) -> str:
        return f"control node {self.nr} ({self.type})"


def read_control_nodes(lines, count: int) -> dict[int, PrjControlNode]:
    """`count` nodes from the section body (`lines` is the reader's line cursor); the
    description line may be blank, and a node without data may carry a blank line."""
    nodes: dict[int, PrjControlNode] = {}
    for _ in range(count):
        tok = lines.record()
        if len(tok) < 7:
            raise ValueError(f"prj: a control node line needs 7 fields, found {tok}")
        nr, typ = int(tok[0]), tok[1]
        if typ not in _DATA:
            raise ValueError(f"prj: control node {nr} has type {typ!r}, which this reader "
                             f"does not know (its data layout is unknown)")
        lines.raw()                                    # the description, possibly blank
        n_data, data = _DATA[typ], ()
        if n_data is not None:
            data = tuple(lines.record())
            want = n_data if n_data >= 0 else 1 + int(data[0])
            if len(data) < want:
                raise ValueError(f"prj: control node {nr} ({typ}) needs {want} data values, "
                                 f"found {list(data)}")
        nodes[nr] = PrjControlNode(nr=nr, type=typ, seq=int(tok[2]), flags=int(tok[3]),
                                   n1=int(tok[5]), n2=int(tok[6]),
                                   name=" ".join(tok[7:]) or NO_NAME, data=data)
    lines.expect_end("control nodes")
    return nodes


@dataclass(frozen=True)
class Actuation:
    """A record's control reference: node `node` writes driver `key` at positions `index`
    (the whole driver when `None`), as a schedule on the same record would."""

    node: int
    key: str
    index: tuple[int, ...] | None
    record: str                                        # "path 5", "zone 1", ...


@dataclass(frozen=True)
class ControlPlan:
    """The nodes to build, in evaluation order; the position of every schedule node on
    `"controls.input"`; the named report nodes and their positions in `order`; the
    actuations."""

    order: tuple[int, ...]
    schedule_slots: dict[int, int]
    outputs: tuple[tuple[str, int], ...]
    actuations: tuple[Actuation, ...]


def plan_controls(nodes: Mapping[int, PrjControlNode],
                  actuations: Sequence[Actuation]) -> ControlPlan | None:
    """The `ControlPlan`, or `None` when no actuator and no named report node exists."""
    reported = [n for n in sorted(nodes) if nodes[n].type == "pas" and nodes[n].name != NO_NAME]
    if not actuations and not reported:
        return None
    stack = [(a.node, a.record) for a in actuations]
    stack += [(n, f"report node {nodes[n].name!r}") for n in reported]
    needed: dict[int, str] = {}
    while stack:
        nr, why = stack.pop()
        if nr in needed:
            continue
        node = nodes.get(nr)
        if node is None:
            raise ValueError(f"prj: {why} references control node {nr}, which the 'control "
                             f"nodes' section does not define")
        if node.type in _UNSUPPORTED:
            raise ValueError(f"prj: {why} depends on {node.label}: {_UNSUPPORTED[node.type]} "
                             f"is not supported")
        if _DATA[node.type] == -1:
            if not node.inputs:
                raise ValueError(f"prj: {node.label} lists no input nodes")
        elif len(node.inputs) != _ARITY.get(node.type, 0):
            raise ValueError(f"prj: {node.label} needs {_ARITY.get(node.type, 0)} input(s), "
                             f"found {list(node.inputs)}")
        needed[nr] = why
        stack += [(i, f"{why} (through {node.label})") for i in node.inputs]
    order = tuple(sorted(needed, key=lambda n: (nodes[n].seq, n)))
    slots = {n: i for i, n in enumerate(n for n in order if nodes[n].type == "sch")}
    position = {n: i for i, n in enumerate(order)}
    outputs = tuple((nodes[n].name, position[n]) for n in reported)
    return ControlPlan(order, slots, outputs, tuple(actuations))


def _block(node: PrjControlNode, u: list[int], plan: ControlPlan, reading) -> Block:
    d, name, t = node.data, node.label, node.type
    if t == "sns":   # flags bit 0: offset and scale; bit 1: time constant (measured)
        scaled, lagged = bool(node.flags & 1), bool(node.flags & 2)
        return Sensor(reading(node), offset=float(d[0]) if scaled else 0.0,
                      scale=float(d[1]) if scaled else 1.0,
                      time_constant=float(d[2]) if lagged else 0.0, name=name)
    if t == "sch":
        return DriverValue(INPUT_KEY, plan.schedule_slots[node.nr], name=name)
    if t == "set":
        return Constant(float(d[0]), name=name)
    if t == "pas":
        return Identity(u[0], name=name)
    if t == "log":
        return Affine(u[0], offset=float(d[0]), gain=float(d[1]), name=name)
    if t == "hys":
        return Hysteresis(u[0], slack=float(d[0]), slope=float(d[1]), initial=float(d[2]),
                          name=name)
    if t == "sub":
        return Sum(u[0], u[1], weights=(1.0, -1.0), name=name)
    if t in ("lbs", "ubs"):
        cls = LowerBandSwitch if t == "lbs" else UpperBandSwitch
        return cls(u[0], u[1], band=float(d[0]), name=name)
    if t == "pc1":
        return ProportionalControl(u[0], gain=float(d[0]), name=name)
    if t == "pi1":
        return IncrementalPI(u[0], kp=float(d[0]), ki=float(d[1]), initial_output=float(d[2]),
                             initial_error=float(d[3]), name=name)
    unary = {"abs": Absolute, "bin": Positive, "inv": Complement}
    binary = {"and": And, "or": Or, "xor": ExclusiveOr, "lls": BelowLimit,
              "uls": AboveLimit, "llc": Shortfall, "ulc": Excess, "div": Divide}
    many = {"add": Sum, "sum": Sum, "mul": Product, "avg": Average, "max": Maximum,
            "min": Minimum}
    if t in unary:
        return unary[t](u[0], name=name)
    if t in binary:
        return binary[t](u[0], u[1], name=name)
    return many[t](*u, name=name)


def control_network(plan: ControlPlan, nodes: Mapping[int, PrjControlNode], *,
                    reading: Callable[[PrjControlNode], object],
                    actuations: Sequence[Actuation] | None = None) -> ControlNetwork:
    """The sampled network of `plan`; `reading(node)` is a sensor node's reading.
    `actuations` (default: the plan's) are the ones the model takes."""
    position = {n: i for i, n in enumerate(plan.order)}
    blocks = [_block(nodes[n], [position[i] for i in nodes[n].inputs], plan, reading)
              for n in plan.order]
    actuators = [Actuator(position[a.node], a.key, a.index, name=a.record)
                 for a in (plan.actuations if actuations is None else actuations)]
    return ControlNetwork(blocks, actuators)
