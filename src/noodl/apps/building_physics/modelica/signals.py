"""Signal sources of the Modelica Standard Library, evaluated on a time grid (spec section 6).

Transcribed from the Modelica Standard Library (MSL) v4.1.0 source, not from its
documentation: ``Modelica/Blocks/Sources.mo`` (``Constant`` :164-175, ``Step`` :204-216,
``Ramp`` :244-262, ``Sine`` :291-310, ``Pulse`` :773-800, ``TimeTable`` :1378-1484,
``CombiTimeTable`` :1588-1720), the shared base ``Modelica/Blocks/Interfaces.mo``
(``SignalSource`` :477-480: ``offset = 0``, ``startTime = 0``) and, because
``CombiTimeTable`` evaluates in C, ``Modelica/Resources/C-Sources/ModelicaStandardTables.c``
(``ModelicaStandardTables_CombiTimeTable_getValue``, :921-1190). Every default below is the
MSL declaration's own; a parameter MSL declares with only a ``start`` value (no default:
``Constant.k``, ``Ramp.duration``, ``Sine.f``, ``Pulse.period``) is required and its absence
is refused, since OpenModelica's export always writes the evaluated value.

``evaluate(signal, t)`` returns the block output ``y`` at every entry of ``t`` (float64, the
shape of ``t``). The time events MSL uses to locate table intervals and pulse periods are
replaced by the equivalent closed-form interval selection at each grid time: a grid time
that falls on a table knot, a start time or a period boundary takes the value AFTER the
event, which is what the solver reports at an output instant that coincides with a time
event. "On" is judged with MSL's own relative epsilon: every event comparison uses
``t + TimeEps |t|`` with ``TimeEps = 100 eps`` (``TimeTable``'s ``getInterpolationCoefficients``,
``Sources.mo:1411-1413,1473``), so a grid time computed as ``start + k interval`` that lands a
rounding error before an event is treated as at it; the signal VALUE is still evaluated at
``t`` itself.

Limits of the closed forms: ``Periodic`` ``CombiTimeTable`` maps ``t`` into the table range
with ``tOffset = floor((t - tMin)/T) T`` (``ModelicaStandardTables.c:1856``) and then
interpolates as inside the table. The C code's additional event-interval corrections
(``:953-1002``) only change the value returned DURING event iteration at an interval
boundary; off event instants the two agree, and at an instant the closed form returns the
post-event value, like the other blocks.

Supported ``CombiTimeTable`` settings: ``smoothness`` ``LinearSegments`` (the default) or
``ConstantSegments``; ``extrapolation`` ``LastTwoPoints`` (the default), ``HoldLastPoint``,
``Periodic`` or ``NoExtrapolation`` (a grid time outside the table raises, as the C code
does; the end points themselves are accepted); the table given in the model, one output
column. The spline smoothness options, ``tableOnFile = true`` and several output columns
are refused by name. Enumeration values may be written in full
(``Modelica.Blocks.Types.Smoothness.LinearSegments``) or by their last component.

Math blocks
-----------
A ``Modelica.Blocks.Math`` block in the JSON's ``signals`` combines other signals: MBL's
examples add an offset to a ramp (``Examples/Orifice.mo``: ``Math.Add``), sum two sources
(``Validation/OneWayFlow.mo``: ``Math.Sum``) or scale a sine into a heat flow
(``Examples/ClosedDoors.mo``: ``Math.Gain``). ``math_inputs`` names a block's input
connectors (``u``; ``u1``, ``u2``(, ``u3``); ``u[1]`` .. ``u[n]``), ``combine`` applies its
equation to the input values, transcribed from MSL v4.1.0 ``Modelica/Blocks/Math.mo``:
``Gain`` :552 ``y = k*u`` (``k`` required, declared with only a start value); ``MultiSum``
:624 ``y = k*u``, 0 when ``nu = 0`` (``nu = 0``, ``k = fill(1, nu)``,
``Interfaces.mo:395``); ``Sum`` :791 ``y = k*u`` (``nin = 1``, ``Interfaces.mo:376``,
``k = ones(nin)``); ``Feedback`` :832 ``y = u1 - u2``; ``Add`` :880 ``y = k1*u1 + k2*u2``
(``k1 = k2 = +1``); ``Add3`` :934 likewise with ``k3``; ``Product`` :976 ``y = u1*u2``;
``Division`` :1004 ``y = u1/u2``. The caller (``assemble``) finds each input's value by the
signal that ``drives`` ``"<block>.<input>"``, so a chain of blocks is evaluated from its
sources outwards.
"""

from __future__ import annotations

import math

import torch

from noodl.apps.building_physics.modelica.schema import ModelicaImportError, Signal

Tensor = torch.Tensor
F64 = torch.float64

_PREFIX = "Modelica.Blocks.Sources."


def _where(sig: Signal) -> str:
    return f"{sig.name} ({sig.cls})"


def _required(sig: Signal, key: str) -> float:
    if key not in sig.parameters:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: parameter {key!r} is required (MSL declares it with "
            f"no default) and was not exported"
        )
    return float(sig.parameters[key])


def _get(sig: Signal, key: str, default: float) -> float:
    return float(sig.parameters.get(key, default))


_TIME_EPS = 100 * torch.finfo(F64).eps  # Sources.mo:1473 (100*Modelica.Constants.eps)


def _ev(t: Tensor) -> Tensor:
    """``t`` nudged forward by MSL's relative event epsilon, for event comparisons only."""
    return t + _TIME_EPS * t.abs()


def _enum(value, default: str) -> str:
    return str(value if value is not None else default).rsplit(".", 1)[-1]


def _constant(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:164-168: y = k.
    return torch.full_like(t, _required(sig, "k"))


def _step(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:204-208: y = offset + (if time < startTime then 0 else height).
    height, offset = _get(sig, "height", 1.0), _get(sig, "offset", 0.0)
    start = _get(sig, "startTime", 0.0)
    return offset + torch.where(_ev(t) < start, torch.zeros_like(t),
                                torch.full_like(t, height))


def _ramp(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:244-252: y = offset + (if time < startTime then 0 else if time < startTime
    # + duration then (time - startTime)*height/duration else height).
    height, offset = _get(sig, "height", 1.0), _get(sig, "offset", 0.0)
    duration = _required(sig, "duration")
    start = _get(sig, "startTime", 0.0)
    safe = duration if duration > 0.0 else 1.0  # the branch is unreachable at duration 0
    rising = (t - start) * height / safe
    te = _ev(t)
    y = torch.where(te < start + duration, rising, torch.full_like(t, height))
    return offset + torch.where(te < start, torch.zeros_like(t), y)


def _sine(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:291-305: continuous -> offset + amplitude*(sin(phase) before startTime);
    # otherwise offset + (0 before startTime).
    amplitude, offset = _get(sig, "amplitude", 1.0), _get(sig, "offset", 0.0)
    f = _required(sig, "f")
    phase = _get(sig, "phase", 0.0)
    start = _get(sig, "startTime", 0.0)
    continuous = bool(sig.parameters.get("continuous", False))
    wave = amplitude * torch.sin(2 * math.pi * f * (t - start) + phase)
    before = amplitude * math.sin(phase) if continuous else 0.0
    return offset + torch.where(_ev(t) < start, torch.full_like(t, before), wave)


def _pulse(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:773-800: T_width = period*width/100; count = integer((time -
    # startTime)/period); T_start = startTime + count*period; y = offset + (if time <
    # startTime or nperiod == 0 or (nperiod > 0 and count >= nperiod) then 0 else if time <
    # T_start + T_width then amplitude else 0).
    amplitude, offset = _get(sig, "amplitude", 1.0), _get(sig, "offset", 0.0)
    width = _get(sig, "width", 50.0)
    period = _required(sig, "period")
    nperiod = int(sig.parameters.get("nperiod", -1))
    start = _get(sig, "startTime", 0.0)
    t_width = period * width / 100.0
    te = _ev(t)
    count = torch.floor((te - start) / period)
    t_start = start + count * period
    off = te < start
    if nperiod == 0:
        off = torch.ones_like(off)
    elif nperiod > 0:
        off = off | (count >= nperiod)
    high = torch.where(te < t_start + t_width, torch.full_like(t, amplitude),
                       torch.zeros_like(t))
    return offset + torch.where(off, torch.zeros_like(t), high)


def _table(sig: Signal) -> Tensor:
    if "table" not in sig.parameters:
        raise ModelicaImportError(f"modelica: {_where(sig)}: parameter 'table' is required")
    table = torch.as_tensor(sig.parameters["table"], dtype=F64)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 2:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: 'table' must be a non-empty matrix with a time column "
            f"and at least one value column, got shape {tuple(table.shape)}"
        )
    return table


def _linear(x: Tensor, y: Tensor, j: Tensor, t: Tensor) -> Tensor:
    """Linear through rows ``j`` and ``j + 1`` at ``t``; the right value where the two times
    coincide (a discontinuity: TimeTable :1440-1447, ``ModelicaStandardTables.c``
    ``isNearlyEqual(t0, t1) -> y1``)."""
    x0, x1, y0, y1 = x[j], x[j + 1], y[j], y[j + 1]
    dx = x1 - x0
    degenerate = dx <= 100 * torch.finfo(F64).eps * x1.abs()
    safe = torch.where(degenerate, torch.ones_like(dx), dx)
    return torch.where(degenerate, y1, y0 + (y1 - y0) * (t - x0) / safe)


def _timetable(sig: Signal, t: Tensor) -> Tensor:
    # Sources.mo:1378-1484. Before startTime: offset. One row: offset + table[1, 2].
    # Otherwise the interval is found by `while next < nrow and tp >= table[next, 1]` (the
    # first knot strictly after tp, capped to the last row, at least row 2), so beyond either
    # end the first/last two rows extrapolate linearly; y = a*(t - shiftTime) + b.
    table = _table(sig)
    if table.shape[0] > 1 and float(table[0, 0]) != 0.0:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: the first point in time has to be 0 (Sources.mo:1466), "
            f"got table[1,1] = {float(table[0, 0])!r}"
        )
    offset = _get(sig, "offset", 0.0)
    time_scale = _get(sig, "timeScale", 1.0)
    start = _get(sig, "startTime", 0.0)
    shift = _get(sig, "shiftTime", start)
    ts = t / time_scale
    ts_e = _ev(ts)  # getInterpolationCoefficients: tp = timeScaled + TimeEps*|timeScaled|
    before = ts_e < start / time_scale
    x, y = table[:, 0].contiguous(), table[:, 1].contiguous()
    if table.shape[0] == 1:
        value = torch.full_like(t, float(y[0]))
    else:
        tp = ts - shift / time_scale
        tp_e = ts_e - shift / time_scale
        nxt = torch.searchsorted(x, tp_e, right=True).clamp(1, x.numel() - 1)
        value = _linear(x, y, nxt - 1, tp)
    return offset + torch.where(before, torch.zeros_like(t), value)


def _combitimetable(sig: Signal, t: Tensor) -> Tensor:
    if bool(sig.parameters.get("tableOnFile", False)):
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: tableOnFile = true is not supported (the table must be "
            f"given in the model)"
        )
    table = _table(sig)
    columns = sig.parameters.get("columns", list(range(2, table.shape[1] + 1)))
    columns = [int(c) for c in (columns if isinstance(columns, list) else [columns])]
    if len(columns) != 1:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: a CombiTimeTable driving one input needs exactly one "
            f"output column, got columns {columns}"
        )
    smoothness = _enum(sig.parameters.get("smoothness"), "LinearSegments")
    extrapolation = _enum(sig.parameters.get("extrapolation"), "LastTwoPoints")
    if smoothness not in ("LinearSegments", "ConstantSegments"):
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: smoothness {smoothness!r} is not supported "
            f"(LinearSegments and ConstantSegments are)"
        )
    if extrapolation not in ("LastTwoPoints", "HoldLastPoint", "Periodic", "NoExtrapolation"):
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: extrapolation {extrapolation!r} is not supported"
        )
    offsets = sig.parameters.get("offset", [0.0])
    offsets = offsets if isinstance(offsets, list) else [offsets]
    offset = float(offsets[0])
    time_scale = _get(sig, "timeScale", 1.0)
    start = _get(sig, "startTime", 0.0)
    shift = _get(sig, "shiftTime", start)

    # ModelicaStandardTables.c:926-928: before startTime the table returns 0.
    ts = t / time_scale
    ts_e = _ev(ts)
    before = ts_e < start / time_scale
    x, y = table[:, 0].contiguous(), table[:, columns[0] - 1].contiguous()
    n = x.numel()
    if n == 1:  # :938-941
        return offset + torch.where(before, torch.zeros_like(t), torch.full_like(t, float(y[0])))
    tp = ts - shift / time_scale  # :948-950
    tp_e = ts_e - shift / time_scale
    t_min, t_max = x[0], x[-1]
    if extrapolation == "Periodic":  # :953-1002, tOffset = floor((t - tMin)/T)*T (:1856)
        period = t_max - t_min
        k_off = torch.floor((tp_e - t_min) / period) * period
        tp, tp_e = tp - k_off, tp_e - k_off
        left = torch.zeros_like(tp, dtype=torch.bool)
        right = torch.zeros_like(tp, dtype=torch.bool)
    else:
        left = tp_e < t_min  # :1003-1005
        right = tp_e >= t_max  # :1006-1014
        if extrapolation == "NoExtrapolation":  # :1172-1178 (end points accepted here)
            outside = (tp_e < t_min) | (tp > t_max)
            if bool(outside.any()):
                bad = t[outside].tolist()
                raise ModelicaImportError(
                    f"modelica: {_where(sig)}: extrapolation NoExtrapolation and times {bad} "
                    f"lie outside the table"
                )
            right = torch.zeros_like(right)

    # In the table (:1016-1100): `last` is the row with x[last] <= t (findRowIndex), capped
    # to the second-to-last row.
    last = (torch.searchsorted(x, tp_e, right=True) - 1).clamp(0, n - 2)
    if smoothness == "LinearSegments":  # :1080-1092
        inside = _linear(x, y, last, tp)
    else:  # CONSTANT_SEGMENTS :1094-1099
        last = torch.where(tp_e >= x[last + 1], last + 1, last)
        inside = y[last]

    if extrapolation == "HoldLastPoint":  # :1166-1169
        lo, hi = torch.full_like(t, float(y[0])), torch.full_like(t, float(y[-1]))
    else:  # LAST_TWO_POINTS :1127-1142, LINEAR and CONSTANT segments alike
        lo = _linear(x, y, torch.zeros_like(last), tp)
        hi = _linear(x, y, torch.full_like(last, n - 2), tp)
        # Coincident end rows: the outer value (y0 at the left, y1 at the right).
        if float(x[1] - x[0]) <= 0.0:
            lo = torch.full_like(t, float(y[0]))
    value = torch.where(left, lo, torch.where(right, hi, inside))
    return offset + torch.where(before, torch.zeros_like(t), value)


_BLOCKS = {
    "Constant": _constant,
    "Step": _step,
    "Ramp": _ramp,
    "Sine": _sine,
    "Pulse": _pulse,
    "TimeTable": _timetable,
    "CombiTimeTable": _combitimetable,
}


_MATH_PREFIX = "Modelica.Blocks.Math."
_MATH_TWO = ("u1", "u2")
_MATH_BLOCKS = ("Gain", "Add", "Add3", "Sum", "MultiSum", "Product", "Feedback", "Division")


def _math_short(sig: Signal) -> str | None:
    if not sig.cls.startswith(_MATH_PREFIX):
        return None
    short = sig.cls[len(_MATH_PREFIX):]
    return short if short in _MATH_BLOCKS else None


def is_math(sig: Signal) -> bool:
    """Whether ``sig`` is a ``Modelica.Blocks.Math`` block this module combines."""
    return _math_short(sig) is not None


def _vector_size(sig: Signal, short: str) -> int:
    key, default = ("nin", 1) if short == "Sum" else ("nu", 0)  # Interfaces.mo:376, :395
    return int(sig.parameters.get(key, default))


def math_inputs(sig: Signal) -> tuple[str, ...]:
    """The input connector names of Math block ``sig``, in MSL declaration order."""
    short = _math_short(sig)
    if short is None:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: signal block is not supported (supported Math blocks: "
            f"{', '.join(_MATH_PREFIX + k for k in _MATH_BLOCKS)})"
        )
    if short == "Gain":
        return ("u",)
    if short == "Add3":
        return ("u1", "u2", "u3")
    if short in ("Sum", "MultiSum"):
        return tuple(f"u[{i + 1}]" for i in range(_vector_size(sig, short)))
    return _MATH_TWO


def _gains(sig: Signal, n: int) -> list[float]:
    k = sig.parameters.get("k", [1.0] * n)
    k = k if isinstance(k, list) else [k]
    if len(k) != n:
        raise ModelicaImportError(
            f"modelica: {_where(sig)}: gain vector k has {len(k)} entries for {n} inputs"
        )
    return [float(v) for v in k]


def combine(sig: Signal, inputs: dict[str, Tensor]) -> Tensor:
    """The output ``y`` of Math block ``sig`` from its input values (by connector name)."""
    short = _math_short(sig)
    names = math_inputs(sig)
    u = [torch.as_tensor(inputs[n], dtype=F64) for n in names]
    if short == "Gain":
        return _required(sig, "k") * u[0]
    if short in ("Sum", "MultiSum"):
        k = _gains(sig, len(u))
        if not u:
            return torch.zeros((), dtype=F64)
        return sum((kk * uu for kk, uu in zip(k, u, strict=True)), torch.zeros((), dtype=F64))
    if short in ("Add", "Add3"):
        k = [_get(sig, f"k{i + 1}", 1.0) for i in range(len(u))]
        return sum((kk * uu for kk, uu in zip(k, u, strict=True)), torch.zeros((), dtype=F64))
    if short == "Product":
        return u[0] * u[1]
    if short == "Feedback":
        return u[0] - u[1]
    return u[0] / u[1]  # Division


def evaluate(signal: Signal, t: Tensor) -> Tensor:
    """The output ``y`` of ``signal`` at the times ``t`` (s), float64, shape of ``t``.

    Raises ``ModelicaImportError`` naming the signal for a block this module does not
    transcribe, a missing required parameter, or an unsupported table setting.
    """
    short = signal.cls[len(_PREFIX):] if signal.cls.startswith(_PREFIX) else None
    fn = _BLOCKS.get(short) if short is not None else None
    if fn is None:
        raise ModelicaImportError(
            f"modelica: {_where(signal)}: signal source is not supported (supported: "
            f"{', '.join(_PREFIX + k for k in _BLOCKS)})"
        )
    t = torch.as_tensor(t, dtype=F64)
    return fn(signal, t).to(F64)
