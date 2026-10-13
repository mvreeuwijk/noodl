"""Test-only circuit laws shared by the Modified Nodal Analysis tests: ideal potential
sources and the capacitance and inductance companion forms (all implicit laws)."""

from noodl.elements import ImplicitLaw


class Source(ImplicitLaw):
    """dp = V: an ideal potential source (V per edge)."""

    phasor_capable = True

    def __init__(self, V, kind="source"):
        super().__init__(kind)
        self.V = self._param(V, False)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.V


class DrivenSource(ImplicitLaw):
    """dp = drivers["V"]: an ideal potential source set by a driver."""

    phasor_capable = True

    def __init__(self, kind="source"):
        super().__init__(kind)

    def residual(self, dp, q, drivers, ctx):
        return dp - drivers["V"]


class Cap(ImplicitLaw):
    """Capacitance C (initial voltage v0) in every mode."""

    dynamic = True
    phasor_capable = True

    def __init__(self, C, v0=0.0, kind="cap", learnable=False):
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


class Ind(ImplicitLaw):
    """Inductance L (initial current i0) in every mode."""

    dynamic = True
    phasor_capable = True

    def __init__(self, L, i0=0.0, kind="ind", learnable=False):
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
