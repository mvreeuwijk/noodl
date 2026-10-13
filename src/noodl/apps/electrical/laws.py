"""Constitutive laws of electrical circuit devices, each carrying every device of its kind.

Sign convention (Modelica Standard Library): a device on the edge `p -> n` sees the voltage
`dp = v_p - v_n` (V) and carries the current `q` (A) that enters pin `p` and leaves pin `n`.
A source that holds node `a` at `+V` relative to ground is the edge `a -> ground`; a source
delivering power carries a negative current. Every parameter is per edge (or, for two-port
laws, per device: one value per group of two consecutive edges) and a scalar broadcasts, so
one law instance carries all devices of its kind.

Phasors are peak-amplitude, sine-referenced phasors: the complex value `X` stands for the time
signal `x(t) = |X| sin(omega t + arg X)`, so the complex power of a device is
`S = (1/2) dp conj(q)` (W). A source has one phase, its waveform's: in phasor mode a `Sine`
source is its steady-state phasor and a `Constant` source is its amplitude (phase 0, the
analysis magnitude). Each device docstring names the MSL model it transcribes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.electrical.waveforms import Constant, Signal, Sine, _Waveform
from noodl.elements import Element, ImplicitLaw, LawContext

Tensor = torch.Tensor


def _driver(drivers: Mapping[str, Tensor] | None, key: str, kind: str) -> Tensor:
    """`drivers[key]`, or a `KeyError` naming the law kind and the missing key."""
    if drivers is None or key not in drivers:
        raise KeyError(f"law kind {kind!r} reads the driver {key!r}, which is missing from "
                       f"drivers")
    return torch.as_tensor(drivers[key])


# ------------------------------------------------------------------------------- sources
def _waveform_list(waveforms, kind: str):
    """`waveforms` as given (one waveform for every edge, or one per edge), checked."""
    if waveforms is None:
        return Constant()
    items = [waveforms] if isinstance(waveforms, _Waveform) else list(waveforms)
    for w in items:
        if not isinstance(w, _Waveform):
            raise TypeError(
                f"source kind {kind!r}: {w!r} is not a waveform (Constant, Step, Ramp, Sine, "
                f"Pulse or Signal from noodl.apps.electrical)"
            )
    return waveforms if isinstance(waveforms, _Waveform) else tuple(items)


def _items(waveforms) -> tuple:
    return (waveforms,) if isinstance(waveforms, _Waveform) else waveforms


def _per_edge(values: list[Tensor], single: bool) -> Tensor:
    """Per-waveform values as a tensor whose last dimension broadcasts against the edges."""
    if single:
        return values[0].unsqueeze(-1)
    return torch.stack(torch.broadcast_tensors(*values), dim=-1)


def _phasor_factor(waveforms, ctx: LawContext, kind: str, dtype, device) -> Tensor:
    """`exp(j (phase - omega start_time))` per edge for `Sine`, 1 for `Constant`."""
    phases, starts = [], []
    for w in _items(waveforms):
        if isinstance(w, Sine):
            phases.append(w.phase)
            starts.append(w.start_time)
        elif isinstance(w, Constant):
            phases.append(0.0)
            starts.append(0.0)
        else:
            raise ValueError(
                f"source kind {kind!r}: a {type(w).__name__} waveform has no phasor; phasor "
                f"sources are Sine (its steady-state phasor) or Constant (the amplitude)"
            )
    angle = torch.tensor(phases, dtype=dtype, device=device)
    if any(t0 != 0.0 for t0 in starts):
        omega = torch.as_tensor(ctx.omega, dtype=dtype, device=device)
        angle = angle - omega * torch.tensor(starts, dtype=dtype, device=device)
    return torch.exp(1j * angle)


def _source_value(amplitude: Tensor, offset: Tensor, waveforms,
                  drivers: Mapping[str, Tensor] | None, ctx: LawContext | None,
                  kind: str, dtype: torch.dtype) -> Tensor:
    """The source value per edge: `offset + amplitude * unit(t)` at the evaluation time
    `ctx.t`, with the waveform values computed in `dtype` (the real dtype of the layer);
    in phasor mode the peak-amplitude phasor `amplitude * exp(j (phase - omega start_time))`
    of a `Sine` waveform, or `amplitude` for `Constant` (the offset is dropped: a constant
    has no phasor)."""
    device = amplitude.device
    if ctx is not None and ctx.mode == "phasor":
        return amplitude * _phasor_factor(waveforms, ctx, kind, dtype, device)
    t = None if ctx is None else ctx.t
    items = _items(waveforms)
    if t is None and any(w.time_dependent for w in items):
        raise ValueError(
            f"source kind {kind!r} has time-dependent waveforms; pass t (the evaluation time)"
        )
    values = []
    for w in items:
        if isinstance(w, Signal):
            signal = _driver(drivers, w.key, kind)[..., w.index]
            values.append(signal.to(dtype=dtype, device=device))
        else:  # with t None only a time-independent waveform (Constant) gets here
            values.append(torch.tensor(w.unit(0.0 if t is None else float(t)), dtype=dtype,
                                       device=device))
    return offset + amplitude * _per_edge(values, isinstance(waveforms, _Waveform))


class _Source:
    """The parameters and value of an independent source (shared by both source laws)."""

    def _setup(self, amplitude, waveforms, offset, learnable) -> None:
        self.amplitude = self._param(amplitude, learnable)
        self.offset = self._param(offset, False)
        self.waveforms = _waveform_list(waveforms, self.kind)

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        """Every waveform's breakpoints in `(t0, t1]`, sorted, without repeats."""
        return sorted({t for w in _items(self.waveforms) for t in w.breakpoints(t0, t1)})

    def value(self, dp: Tensor, drivers, ctx) -> Tensor:
        if not isinstance(self.waveforms, _Waveform) and len(self.waveforms) != dp.shape[-1]:
            raise ValueError(
                f"source kind {self.kind!r}: {len(self.waveforms)} waveforms for "
                f"{dp.shape[-1]} edges; give one waveform per edge or a single one for all"
            )
        dtype = dp.real.dtype if dp.is_complex() else dp.dtype
        return _source_value(self.amplitude, self.offset, self.waveforms, drivers, ctx,
                             self.kind, dtype)


class PrescribedVoltage(_Source, ImplicitLaw):
    """An ideal voltage source: `dp = offset + amplitude * unit(t)` (V), with `unit` the
    waveform (`Constant` by default; one for all edges or one per edge) evaluated at the
    evaluation time `t` (s). In phasor mode a `Sine(frequency, phase, start_time)` source is
    `dp = amplitude * exp(j (phase - omega start_time))` (peak-amplitude, sine-referenced; at
    the layer's frequency) and a `Constant` source is `dp = amplitude`; other waveforms are
    refused there. MSL `Sources.ConstantVoltage`, `StepVoltage`, `SineVoltage`,
    `PulseVoltage`, `SignalVoltage`. `learnable` applies to `amplitude`.
    """

    phasor_capable = True

    def __init__(self, amplitude, *, waveforms=None, offset=0.0,
                 kind: str = "voltage_source", learnable: bool = False) -> None:
        super().__init__(kind)
        self._setup(amplitude, waveforms, offset, learnable)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.value(dp, drivers, ctx)


class PrescribedCurrent(_Source, Element):
    """An ideal current source: `q = offset + amplitude * unit(t)` (A) entering pin `p`
    whatever the voltage, with the waveform and its phasor as for `PrescribedVoltage`. MSL
    `Sources.ConstantCurrent`, `StepCurrent`, `SineCurrent`, `PulseCurrent`,
    `SignalCurrent`. `learnable` applies to `amplitude`.
    """

    dp_independent = True
    contextual = True
    phasor_capable = True

    def __init__(self, amplitude, *, waveforms=None, offset=0.0,
                 kind: str = "current_source", learnable: bool = False) -> None:
        super().__init__(kind)
        self._setup(amplitude, waveforms, offset, learnable)

    def flow(self, dp, drivers=None, ctx=None):
        return self.value(dp, drivers, ctx) + 0 * dp


# -------------------------------------------------------------------- one-port devices
class Resistance(Element):
    """A linear resistor: `q = dp / R_actual` with `R_actual = R (1 + alpha (T - T_ref))`
    (`R` in ohm at `T_ref` in K, `alpha` in 1/K); `T` (K) is read per edge from
    `drivers[temperature]` when `temperature` names a driver, else `T = T_ref`. MSL
    `Basic.Resistor`.
    """

    phasor_capable = True

    def __init__(self, R, *, alpha=0.0, T_ref=300.15, temperature: str | None = None,
                 kind: str = "resistor", learnable: bool = False) -> None:
        super().__init__(kind)
        self.R = self._param(R, learnable)
        self.alpha = self._param(alpha, False)
        self.T_ref = self._param(T_ref, False)
        self.temperature = temperature

    def resistance(self, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """`R_actual` (ohm) per edge at the driver temperature."""
        if self.temperature is None:
            return self.R * torch.ones_like(self.alpha)
        T = _driver(drivers, self.temperature, self.kind)
        return self.R * (1.0 + self.alpha * (T - self.T_ref))

    def flow(self, dp, drivers=None):
        return dp / self.resistance(drivers)

    def dflow(self, dp, drivers=None):
        return torch.ones_like(dp) / self.resistance(drivers)

    def linear_init(self, drivers=None):
        g = 1.0 / self.resistance(drivers)
        return torch.zeros_like(g), g


class Capacitance(ImplicitLaw):
    """A capacitor `q = C d(dp)/dt` (`C` in F, initial voltage `v0` in V), as companion
    forms: dc `q = 0`; initial `dp = v0`; phasor `q = j omega C dp`; transient backward Euler
    `q = C (dp - dp_prev) / dt` or trapezoidal `q + q_prev = 2 C / dt (dp - dp_prev)`. MSL
    `Basic.Capacitor`.
    """

    dynamic = True
    phasor_capable = True

    def __init__(self, C, *, v0=0.0, kind: str = "capacitor", learnable: bool = False) -> None:
        super().__init__(kind)
        self.C = self._param(C, learnable)
        self.v0 = self._param(v0, False)

    def residual(self, dp, q, drivers, ctx):
        if ctx.mode == "dc":
            return q
        if ctx.mode == "initial":
            return dp - self.v0
        if ctx.mode == "phasor":
            return q - 1j * ctx.omega * self.C * dp
        if ctx.integration == "backward_euler":
            return q - self.C * (dp - ctx.dp_prev) / ctx.dt
        return q + ctx.q_prev - 2.0 * self.C / ctx.dt * (dp - ctx.dp_prev)


class Inductance(ImplicitLaw):
    """An inductor `dp = L dq/dt` (`L` in H, initial current `i0` in A), as companion forms:
    dc `dp = 0`; initial `q = i0`; phasor `dp = j omega L q`; transient backward Euler
    `dp = L (q - q_prev) / dt` or trapezoidal `dp + dp_prev = 2 L / dt (q - q_prev)`. MSL
    `Basic.Inductor`.
    """

    dynamic = True
    phasor_capable = True

    def __init__(self, L, *, i0=0.0, kind: str = "inductor", learnable: bool = False) -> None:
        super().__init__(kind)
        self.L = self._param(L, learnable)
        self.i0 = self._param(i0, False)

    def residual(self, dp, q, drivers, ctx):
        if ctx.mode == "dc":
            return dp
        if ctx.mode == "initial":
            return q - self.i0
        if ctx.mode == "phasor":
            return dp - 1j * ctx.omega * self.L * q
        if ctx.integration == "backward_euler":
            return dp - self.L * (q - ctx.q_prev) / ctx.dt
        return dp + ctx.dp_prev - 2.0 * self.L / ctx.dt * (q - ctx.q_prev)


class JunctionDiode(Element):
    """A junction diode with a parallel leak: `q = Ids (exlin(dp / Vt) - 1) + dp / R`, where
    `exlin(x) = exp(x)` up to `Maxexp` and continues linearly above it,
    `exp(Maxexp) (1 + x - Maxexp)` (`Ids` saturation current in A, `Vt` voltage equivalent of
    temperature in V, `R` parallel resistance in ohm). MSL `Semiconductors.Diode` without
    temperature dependency. Nonlinear, so not usable in a phasor layer.
    """

    def __init__(self, Ids=1e-6, Vt=0.04, Maxexp=15.0, R=1e8, *, kind: str = "diode") -> None:
        super().__init__(kind)
        self.Ids = self._param(Ids, False)
        self.Vt = self._param(Vt, False)
        self.Maxexp = self._param(Maxexp, False)
        self.R = self._param(R, False)

    def flow(self, dp, drivers=None):
        x = dp / self.Vt
        # the clamp keeps the unused exponential finite at large forward voltage
        z = torch.where(x > self.Maxexp,
                        torch.exp(self.Maxexp) * (1.0 + x - self.Maxexp),
                        torch.exp(torch.clamp(x, max=self.Maxexp)))
        return self.Ids * (z - 1.0) + dp / self.R


def _driver_entries(spec, drivers: Mapping[str, Tensor] | None, kind: str) -> Tensor:
    """The per-edge values a control specification reads (see `SwitchedConductance`)."""
    if isinstance(spec, str):
        return _driver(drivers, spec, kind)
    parts = []
    for entry in spec:
        key, index = (entry, 0) if isinstance(entry, str) else entry
        parts.append(_driver(drivers, key, kind)[..., index])
    return torch.stack(torch.broadcast_tensors(*parts), dim=-1)


def _is_entry(control) -> bool:
    """True for a lone `(key, index)` pair."""
    return (isinstance(control, tuple) and len(control) == 2 and isinstance(control[0], str)
            and isinstance(control[1], int))


class SwitchedConductance(Element):
    """An ideal closing switch: closed `q = dp / Ron`, open `q = Goff dp` (`Ron` in ohm,
    `Goff` in S); closed where the control value exceeds 0.5 (or is True). `control` is a
    driver key whose last dimension holds one value per switch edge (or one value for all),
    a single `(key, index)` pair read by every switch, or a sequence with one entry per edge,
    each a key (read at index 0) or a `(key, index)` pair. MSL `Ideal.IdealClosingSwitch`.
    """

    phasor_capable = True

    def __init__(self, control, *, Ron=1e-5, Goff=1e-5, kind: str = "switch") -> None:
        super().__init__(kind)
        if isinstance(control, str):
            self.control = control
        elif _is_entry(control):
            self.control = (control,)
        else:
            self.control = tuple(control)
        self.Ron = self._param(Ron, False)
        self.Goff = self._param(Goff, False)

    def closed(self, drivers: Mapping[str, Tensor] | None, n_edges: int | None = None
               ) -> Tensor:
        """True per edge where the switch conducts; with `n_edges`, the control must hold
        one value per edge or one for all."""
        state = _driver_entries(self.control, drivers, self.kind)
        if n_edges is not None and state.dim() > 0 and state.shape[-1] not in (1, n_edges):
            raise ValueError(
                f"switch kind {self.kind!r}: the control holds {state.shape[-1]} values per "
                f"instance for {n_edges} switch edges; give one per edge or one for all"
            )
        return state if state.dtype == torch.bool else state > 0.5

    def flow(self, dp, drivers=None):
        return torch.where(self.closed(drivers, dp.shape[-1]), dp / self.Ron, self.Goff * dp)


# -------------------------------------------------------------------- two-port devices
def _ports(x: Tensor) -> tuple[Tensor, Tensor]:
    """The (port 1, port 2) values of consecutive edge pairs."""
    g = x.unflatten(-1, (-1, 2))
    return g[..., 0], g[..., 1]


def _pairs(a: Tensor, b: Tensor) -> Tensor:
    """Interleave per-device port values back into layer edge order."""
    a, b = torch.broadcast_tensors(a, b)
    return torch.stack([a, b], dim=-1).flatten(-2)


class TransformerRatio(ImplicitLaw):
    """An ideal transformer without magnetisation: `dp1 = n dp2`, `q2 = -n q1` (turns ratio
    `n`, primary to secondary voltage). Edges `[primary p1 -> n1, secondary p2 -> n2]` per
    device. MSL `Ideal.IdealTransformer` with `considerMagnetization = false`.
    """

    ports = 2
    phasor_capable = True

    def __init__(self, n, *, kind: str = "ideal_transformer", learnable: bool = False) -> None:
        super().__init__(kind)
        self.n = self._param(n, learnable)

    def residual(self, dp, q, drivers, ctx):
        v1, v2 = _ports(dp)
        i1, i2 = _ports(q)
        return _pairs(v1 - self.n * v2, i2 + self.n * i1)


class MutualInductance(ImplicitLaw):
    """Two coupled windings: `dp1 = L1 dq1/dt + M dq2/dt`, `dp2 = M dq1/dt + L2 dq2/dt`
    (`L1`, `L2`, `M` in H; initial currents `i0 = (i1, i2)` in A, per device or shared).
    Companion forms as `Inductance` with the 2x2 inductance matrix. Edges
    `[primary, secondary]` per device. MSL `Basic.Transformer`.
    """

    ports = 2
    dynamic = True
    phasor_capable = True

    def __init__(self, L1, L2, M, *, i0=(0.0, 0.0), kind: str = "transformer",
                 learnable: bool = False) -> None:
        super().__init__(kind)
        self.L1 = self._param(L1, learnable)
        self.L2 = self._param(L2, learnable)
        self.M = self._param(M, learnable)
        self.i0 = self._param(i0, False)

    def _flux(self, q: Tensor) -> Tensor:
        """The flux linkage `Lmat q` of every device, in edge order."""
        i1, i2 = _ports(q)
        return _pairs(self.L1 * i1 + self.M * i2, self.M * i1 + self.L2 * i2)

    def residual(self, dp, q, drivers, ctx):
        if ctx.mode == "dc":
            return dp
        if ctx.mode == "initial":
            return (q.unflatten(-1, (-1, 2)) - self.i0).flatten(-2)
        if ctx.mode == "phasor":
            return dp - 1j * ctx.omega * self._flux(q)
        if ctx.integration == "backward_euler":
            return dp - self._flux(q - ctx.q_prev) / ctx.dt
        return dp + ctx.dp_prev - 2.0 / ctx.dt * self._flux(q - ctx.q_prev)


_DEPENDENT = ("voltage", "current")


class DependentSource(ImplicitLaw):
    """A linear controlled source with edges `[control, output]` per device: the control
    port is an open circuit (`q1 = 0`, voltage control) or a short (`dp1 = 0`, current
    control), and the output port imposes `dp2` (voltage output) or `q2` (current output)
    as `gain` times the control voltage `dp1` or current `q1`. Default kinds `"vcvs"`,
    `"vccs"`, `"ccvs"`, `"cccs"`. MSL `Basic.VCV`, `VCC`, `CCV`, `CCC` (gain,
    transconductance in S, transresistance in ohm, current gain).
    """

    ports = 2
    phasor_capable = True

    def __init__(self, gain, *, control: str, output: str, kind: str | None = None,
                 learnable: bool = False) -> None:
        if control not in _DEPENDENT:
            raise ValueError(f"DependentSource: control={control!r}; expected one of "
                             f"{_DEPENDENT}")
        if output not in _DEPENDENT:
            raise ValueError(f"DependentSource: output={output!r}; expected one of "
                             f"{_DEPENDENT}")
        super().__init__(kind or f"{control[0]}c{output[0]}s")
        self.control = control
        self.output = output
        self.gain = self._param(gain, learnable)

    def residual(self, dp, q, drivers, ctx):
        v1, v2 = _ports(dp)
        i1, i2 = _ports(q)
        controlled = self.gain * (v1 if self.control == "voltage" else i1)
        r1 = i1 if self.control == "voltage" else v1
        r2 = (v2 if self.output == "voltage" else i2) - controlled
        return _pairs(r1, r2)


__all__ = [
    "Capacitance", "DependentSource", "Inductance", "JunctionDiode", "MutualInductance",
    "PrescribedCurrent", "PrescribedVoltage", "Resistance", "SwitchedConductance",
    "TransformerRatio",
]
