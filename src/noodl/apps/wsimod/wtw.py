"""WSIMOD 0.8.1's treatment works (``wsimod/nodes/wtw.py``): ``WTW``, ``WWTW`` and ``FWTW``,
batched.

``treat_current_input`` (``wtw.py:163-214``) splits the day's input into treated water, liquor
and solids. The volume's constant is ``c_v = 1 - percent_solids - liquor_multiplier["volume"]``
(``:92``, ``:104-110``); each additive pollutant ``k`` has its own ``process_parameters[k] =
{constant, exponent}`` and ``liquor_multiplier[k]`` (``:72-85``, ``:121-128``, ``:130-153``), and

    treated[k] = influent[k] * constant[k] * exponent[k] ** (20 - T)      (``:188-199``)
    liquor[k]  = influent[k] * liquor_multiplier[k]                        (``:201``)
    solids[k]  = influent[k] - treated[k] - liquor[k]                      (``:207-208``)

with ``T`` the influent's temperature and no temperature factor for the volume (``:195``). The
power is :func:`~noodl.apps.wsimod.vqip.pow_python`, evaluated per instance by Python, and the
products run left to right, so every value is bitwise WSIMOD's (``replay_bound = 0``). The
factor rises as the temperature falls when ``exponent > 1``, and nothing caps ``constant x
factor + liquor_multiplier``: where it exceeds 1 an additive solids column is negative (the
works creates mass; WSIMOD's lists still close, the solids being the remainder) and noodl
records ``"wtw_negative_solids"`` (X22). Drivers (X10, X34), one row per node of the class:
``wsimod.<Class>.process_parameters.constant`` and ``.exponent`` ``(..., n, K_add)`` (absent in a
quantity-only run) and ``wsimod.<Class>.liquor_multiplier`` ``(..., n, 1 + K_add)``, columns
``volume`` then the additive pollutants (``(..., n, 1)`` in a quantity-only run). The volume
constant is derived from the day's ``percent_solids`` and ``liquor_multiplier["volume"]`` (D5).

The non-additive liquor blend divides by ``V_liquor + V_in x m_v`` (``:174-182``), which is 0 on
a day with no input and no previous liquor, and on every day with
``liquor_multiplier["volume"] = 0``. WSIMOD raises ``ZeroDivisionError`` there; noodl keeps the
previous liquor value (a double ``where`` over a safe denominator, so the untaken branch has a
finite gradient) and records ``"wtw_liquor_zero_denominator"`` (spec, defect table).

State (spec, State): ``liquor`` (every ``WWTW`` and ``FWTW``: neither ``end_timestep`` resets
it), ``treated`` (a ``WWTW``'s: only ``make_discharge`` empties it) and ``previous_pulled`` (an
``FWTW``'s) are carried from day to day. The other stores (``current_input``, ``solids``,
``total_pulled``, ``total_deficit``, ``unpushed_sludge``, an ``FWTW``'s ``treated``) live
within a day; they are stores so that a replay can load them at every call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod import constants
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.tanks import Tank
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    pow_python,
    safe_den,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = ["FWTW", "WTW", "WWTW", "wtw_derived"]

# wtw.py:608: where an FWTW's liquor and solids go
SLUDGE_TYPES = "Sewer"


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def _volume_column(width: int) -> Tensor:
    """``[True, False, ...]``: the volume column of a VQIP of ``width`` columns."""
    return torch.arange(width) == 0


def _join(first: Tensor, rest: Tensor) -> Tensor:
    """``first`` and ``rest`` side by side along the last axis, their batch shapes broadcast."""
    shape = torch.broadcast_shapes(first.shape[:-1], rest.shape[:-1])
    return torch.cat(
        [first.expand(*shape, first.shape[-1]), rest.expand(*shape, rest.shape[-1])], dim=-1
    )


def _liquor_multiplier(values: Mapping) -> Mapping:
    """``liquor_multiplier`` as ``WTW.__init__`` sets it: WSIMOD's default when it is empty
    (``wtw.py:81-85``; the reader fills it already, this covers a node built directly)."""
    liquor = values.get("liquor_multiplier") or {}
    return liquor if len(liquor) > 0 else {"volume": 0.03}


def wtw_derived(spec: NodeSpec) -> dict[str, float]:
    """The volume constant from ``spec``'s merged parameters (D5):
    ``process_parameters["volume"]["constant"] = 1 - percent_solids -
    liquor_multiplier["volume"]`` (``wtw.py:92``, ``:104-110``; the setters recompute it after
    every override, ``:116-128``)."""
    values = spec.parameters
    percent_solids = values.get("percent_solids", 0.0002)
    liquor = _liquor_multiplier(values)
    if "volume" not in liquor:
        raise ValueError(
            f"node {spec.name!r}: liquor_multiplier has no 'volume' (wtw.py:110 reads it)"
        )
    return {"volume_constant": 1 - percent_solids - liquor["volume"]}


class WTW(Node):
    """``WTW`` (``wtw.py:14-220``): the treatment processes. A bare ``WTW`` denies pushes
    (``:95-96``) and has no orchestration method, so it is accepted and inert.

    Parameters (:meth:`bind`): ``treatment_throughput_capacity``, ``percent_solids`` and
    ``liquor_multiplier`` (its volume entry, D6).
    """

    wsimod_type: ClassVar[str] = "WTW"
    store_paths: ClassVar[tuple[str, ...]] = ("current_input", "treated", "liquor", "solids")
    distributes: ClassVar[tuple[str, ...]] = ()
    parameter_defaults: ClassVar[dict[str, float]] = {
        "treatment_throughput_capacity": 10,
        "percent_solids": 0.0002,
    }
    # the parameters the tanks take: parameter -> (tank attribute, tank field)
    tank_parameters: ClassVar[dict[str, tuple[str, str]]] = {}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        self.site = f"node:{self.name}"
        self._ctx: Ctx | None = None
        values = spec.parameters
        self.defaults: dict[str, Tensor] = {
            name: _f64(values.get(name, default))
            for name, default in self.parameter_defaults.items()
        }
        # X10, X34: the volume entry, then one per additive pollutant (none in a quantity-only run)
        liquor = _liquor_multiplier(values)
        additive = pollutants.additive
        self.defaults["liquor_multiplier"] = _f64(
            [liquor.get("volume", 0.03)] + [liquor.get(name, 0.7) for name in additive]
        )
        if additive:  # wtw.py:73-85: WSIMOD's defaults where a node is built without any
            process = values.get("process_parameters") or {}
            for field, default in (("constant", 0.01), ("exponent", 1.001)):
                self.defaults[f"process_parameters.{field}"] = _f64(
                    [process.get(name, {}).get(field, default) for name in additive]
                )
        # VQIP column -> column of the (volume, additive...) vectors; non-additive columns take
        # the volume's, which the treatment never reads
        position = {name: i + 1 for i, name in enumerate(additive)}
        self._vector_columns = torch.tensor([0] + [position.get(n, 0) for n in pollutants.names])
        # wtw.py:95-96
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_check_deny
        # wtw.py:99-102
        self.current_input = empty(pollutants, ())
        self.treated = empty(pollutants, ())
        self.liquor = empty(pollutants, ())
        self.solids = empty(pollutants, ())

    # --- the day's context and parameters ---------------------------------------------------
    @property
    def ctx(self) -> Ctx:
        if self._ctx is None:
            raise RuntimeError(f"node {self.name!r}: no day context")
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx
        for tank in self._tanks():
            tank.ctx = ctx

    def _tanks(self) -> list[Tank]:
        return []

    def _param(self, name: str) -> Tensor:
        return self.params.get(name, self.defaults[name])

    def driver_columns(self) -> dict[str, tuple[str, ...]]:
        """``liquor_multiplier``: ``volume`` then the additive pollutants; the process
        constants and exponents: the additive pollutants (X10, X34)."""
        columns: dict[str, tuple[str, ...]] = {"liquor_multiplier": ("volume", *self.p.additive)}
        if self.p.additive:
            columns["process_parameters.constant"] = self.p.additive
            columns["process_parameters.exponent"] = self.p.additive
        return columns

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameters by name; the tank parameters reach the tank. An ``exponent``
        of 0 or below is refused (``0 ** negative`` raises where ``T > 20``, a negative base
        gives a complex number)."""
        super().bind(parameters)
        exponent = self.params.get("process_parameters.exponent")
        if exponent is not None and not bool((exponent.detach() > 0).all()):
            raise ValueError(
                f"node {self.name!r}: process_parameters exponent must be positive (wtw.py:191-193 "
                f"raises 0 or a negative base to a power); got {exponent.detach().tolist()}"
            )
        for name, (attribute, field) in self.tank_parameters.items():
            if name in self.params:
                setattr(getattr(self, attribute), field, self.params[name])

    def volume_constant(self) -> Tensor:
        """``wtw.py:104-110``: ``1 - percent_solids - liquor_multiplier["volume"]``."""
        return 1 - self._param("percent_solids") - self._param("liquor_multiplier")[..., 0]

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        out = {path: getattr(self, path) for path in self.store_paths if "." not in path}
        for tank in self._tanks():
            for path, value in tank.stores().items():
                out[f"{tank.path}.{path}"] = value
        return out

    def end_timestep(self) -> None:
        """``wtw.py:216-220``."""
        self.current_input = empty(self.p, ())
        self.treated = empty(self.p, ())

    # --- the processes ----------------------------------------------------------------------
    def get_excess_throughput(self, *, active: Tensor) -> Tensor:
        """``wtw.py:155-161``: ``max(capacity - current_input V, 0)``."""
        return dmax(
            self.ctx,
            f"{self.site}/get_excess_throughput/headroom",
            self._param("treatment_throughput_capacity") - self.current_input[..., 0],
            0.0,
            active=active,
            call=self.ctx.call(self.site, "get_excess_throughput"),
        )

    def treat_current_input(self, *, active: Tensor) -> None:
        """``wtw.py:163-214`` (module text).

        The non-additive columns of the treated water are the influent's; those of the liquor
        blend the old liquor with ``V_in x m_v`` of influent (``:174-182``, before the liquor
        volume changes), keeping the old value where that denominator is 0. The volume and each
        additive column ``k`` are independent (``:188-201``): treated ``x c_k f_k`` with ``f_k =
        e_k ** (20 - T)`` (1 for the volume), liquor ``x m_k``. Solids volume ``V x
        percent_solids`` and additive solids ``influent - treated - liquor`` (``:204-208``);
        the solids' non-additive columns are never written. ``treated`` accumulates (``:211``).
        """
        ctx, p = self.ctx, self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "treat_current_input")
        influent = self.current_input.clone()  # copy_vqip (wtw.py:167)
        multiplier = self._param("liquor_multiplier")  # (..., 1 + K_add): volume, additive
        m_v = multiplier[..., 0]
        c_v = self.volume_constant()
        percent_solids = self._param("percent_solids")
        v_in = influent[..., :1]
        old = self.liquor
        v_liq = old[..., :1]
        # wtw.py:174-182: num / den, kept as the old value where den == 0 (double where). The
        # loop runs over the non-additive pollutants only, so without any (a quantity-only
        # run among them) WSIMOD never divides and nothing is decided or recorded.
        blended = old
        if p.non_additive:
            num = old * v_liq + influent * v_in * m_v[..., None]
            den = v_liq + v_in * m_v[..., None]
            zero = ctx.decide(
                f"{self.site}/treat_current_input/liquor_zero_denominator",
                den[..., 0],
                "==",
                0.0,
                active=active,
                call=call,
            )
            ctx.event("wtw_liquor_zero_denominator", self.name, active & zero)
            blended = torch.where(zero[..., None], old, num / safe_den(den))
        # wtw.py:188-201: volume and additive columns, each its own constant, factor and
        # multiplier; the vectors are (volume, additive...) and ``columns`` spreads them over
        # the VQIP's columns (the non-additive ones are replaced below)
        columns = self._vector_columns
        constant = c_v[..., None]
        if p.additive:
            constant = _join(constant, self._param("process_parameters.constant"))
            exponent = self._param("process_parameters.exponent")
            temperature = influent[..., p.column("temperature") : p.column("temperature") + 1]
            # wtw.py:191-193: exponent ** (20 - T), Python's power per instance
            factor = pow_python(exponent, constants.DECAY_REFERENCE_TEMPERATURE - temperature)
            factor = _join(torch.ones_like(factor[..., :1]), factor)
            treated_add = influent * constant[..., columns] * factor[..., columns]
        else:  # only the volume: x * c * 1 is x * c exactly (wtw.py:195)
            treated_add = influent * constant[..., columns]
        liquor_add = influent * multiplier[..., columns]
        solids_add = influent - treated_add - liquor_add
        solids_volume = v_in * percent_solids[..., None]
        non_add = p.non_additive_mask
        discharge = torch.where(non_add, influent, treated_add)
        liquor = torch.where(non_add, blended, liquor_add)
        solids = torch.where(p.additive_mask, solids_add, self.solids)
        solids = torch.where(_volume_column(1 + p.K), solids_volume, solids)
        self.liquor = write(active, liquor, self.liquor)
        self.solids = write(active, solids, self.solids)
        # wtw.py:211
        self.treated = write(active, sum_vqip(p, self.treated, discharge), self.treated)
        seen = ctx.treated_input.get(self.name, ctx.empty())
        ctx.treated_input[self.name] = write(active, influent, seen)
        # X22: the solids are the unclamped remainder, so they can be negative
        negative = ((solids_add.detach() < 0) & p.additive_mask).any(dim=-1)
        ctx.event("wtw_negative_solids", self.name, active & negative)
        more = (self.treated[..., 0] > self.current_input[..., 0]).detach()
        ctx.event("more_treated_than_input", self.name, active & more)

    # --- validation -------------------------------------------------------------------------
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """``liquor_multiplier`` needs a ``volume`` entry (``wtw.py:110``). A configured volume
        process constant must be the derived one (WSIMOD replaces a configured one, ``:92``, but
        an override's ``update`` would replace the derived one, ``:146-148``). With pollutants
        on, additive pollutants need ``temperature`` among the pollutants (``:191-193`` reads
        it), and the entries WSIMOD never reads are refused (D20): one for a pollutant that is
        not additive (``:188``), a volume ``exponent`` (``:195``), and an ``exponent`` of 0 or
        below (``0 ** negative`` raises where ``T > 20``, a negative base gives a complex
        number). A configuration lacking an additive pollutant is refused by the reader. A
        quantity-only run ignores the pollutant entries of ``process_parameters`` and
        ``liquor_multiplier``."""
        super().validate(spec, network, pollutants=pollutants)
        where = f"node {spec.name!r}"
        values = spec.parameters
        c_v = wtw_derived(spec)["volume_constant"]
        liquor = _liquor_multiplier(values)
        process = values.get("process_parameters") or {}
        volume = process.get("volume")
        if volume is not None and "constant" in volume and volume["constant"] != c_v:
            raise ValueError(
                f"{where}: process_parameters['volume'] constant {volume['constant']!r} is not "
                f"1 - percent_solids - liquor_multiplier['volume'] = {c_v!r} (wtw.py:92, "
                ":104-110); WSIMOD uses one or the other depending on where it is given"
            )
        if not pollutants:
            ignored = []
            if any(k != "volume" for k in process):
                ignored.append("process_parameters")
            if any(k != "volume" for k in liquor):
                ignored.append("liquor_multiplier")
            return ignored
        additive = network.pollutants.additive
        for table, entries in (("process_parameters", process), ("liquor_multiplier", liquor)):
            for name in entries:
                if name != "volume" and name not in additive:
                    raise ValueError(
                        f"{where}: {table} has an entry for {name!r}, which is not an additive "
                        "pollutant of the model (wtw.py:188 reads only the additive ones)"
                    )
        if volume is not None and "exponent" in volume:
            raise ValueError(
                f"{where}: process_parameters['volume'] has an 'exponent' (wtw.py:195 never "
                "reads it)"
            )
        for name in additive:
            exponent = process[name]["exponent"]
            if not exponent > 0:
                raise ValueError(
                    f"{where}: process_parameters[{name!r}] exponent {exponent!r} must be "
                    "positive (wtw.py:191-193 raises 0 or a negative base to a power)"
                )
        if additive and "temperature" not in network.pollutants.names:
            raise ValueError(
                f"{where}: additive pollutants without a 'temperature' pollutant "
                "(treat_current_input reads the influent's temperature, wtw.py:191-193)"
            )
        return []

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """A bare ``WTW`` keeps ``Node``'s default pull check, ``pull_check_basic``, which asks
        its in-arcs (``wtw.py:95-96`` replace only the push handlers); a ``WWTW``'s
        ``pull_check_reuse`` (``wtw.py:274``) and an ``FWTW``'s reservoir check terminate."""
        return cls is WTW and direction == "pull" and tag == "default"


class WWTW(WTW):
    """``WWTW`` (``wtw.py:223-453``): wastewater treatment with a stormwater tank
    (``stormwater_tank``). Pushes go to the spare throughput first, then to the tank;
    ``calculate_discharge`` drains the tank into the spare throughput, adds yesterday's liquor
    and treats; ``make_discharge`` pushes ``treated`` on and empties it. ``treated`` is carried
    (``end_timestep_`` does not reset it, ``:447-453``), so it accumulates when a custom
    orchestration leaves ``make_discharge`` out.
    """

    wsimod_type: ClassVar[str] = "WWTW"
    store_paths: ClassVar[tuple[str, ...]] = (
        "stormwater_tank.storage",
        "current_input",
        "treated",
        "liquor",
        "solids",
    )
    # wtw.py:447-453 keeps liquor and treated; current_input and solids are emptied
    carried_paths: ClassVar[tuple[str, ...]] = ("stormwater_tank.storage", "treated", "liquor")
    step_methods: ClassVar[tuple[str, ...]] = ("calculate_discharge", "make_discharge")
    distributes: ClassVar[tuple[str, ...]] = ("out",)  # make_discharge's push_distributed
    # wtw.py:293: solids go to landfill; :294-297 the storage changes are the engine's (D11)
    mass_balance: ClassVar[dict[str, str]] = {
        "WWTW.solids": "out",
        # W69: water pushed in after the last treatment sits in current_input and vanishes at
        # end_timestep_ (wtw.py:447-453); the balance counts it as an outflow
        "WWTW.unprocessed_input": "out",
    }
    parameter_defaults: ClassVar[dict[str, float]] = {
        **WTW.parameter_defaults,
        "stormwater_storage_capacity": 10,
        "stormwater_storage_area": 1,
        "stormwater_storage_elevation": 10,
    }
    tank_parameters: ClassVar[dict[str, tuple[str, str]]] = {
        "stormwater_storage_capacity": ("stormwater_tank", "capacity"),
        "stormwater_storage_area": ("stormwater_tank", "area"),
        "stormwater_storage_elevation": ("stormwater_tank", "datum"),
    }

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # wtw.py:273-278
        self.pull_set_handler["default"] = self.pull_set_reuse
        self.pull_check_handler["default"] = self.pull_check_reuse
        self.push_set_handler["Sewer"] = self.push_set_sewer
        self.push_check_handler["Sewer"] = self.push_check_sewer
        self.push_check_handler["default"] = self.push_check_sewer
        self.push_set_handler["default"] = self.push_set_sewer
        # wtw.py:281-285
        self.stormwater_tank = Tank(
            pollutants,
            capacity=self.defaults["stormwater_storage_capacity"],
            area=self.defaults["stormwater_storage_area"],
            datum=self.defaults["stormwater_storage_elevation"],
            initial_storage=_f64(0.0),
            owner=self.name,
            path="stormwater_tank",
        )

    def _tanks(self) -> list[Tank]:
        return [self.stormwater_tank] if hasattr(self, "stormwater_tank") else []

    def end_timestep(self) -> None:
        """``end_timestep_`` (``wtw.py:447-453``): ``current_input`` and ``solids`` are emptied
        and the tank ends its day; ``liquor`` and ``treated`` are kept (``liquor_`` and
        ``previous_input`` serve WSIMOD's mass balance only and are not ported)."""
        self.current_input = empty(self.p, ())
        self.solids = empty(self.p, ())
        self.stormwater_tank.end_timestep()

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        ctx.term("WWTW.solids", self.solids, every)
        treated = ctx.treated_input.get(self.name, ctx.empty())
        late = extract_vqip(self.p, self.current_input, treated)
        ctx.term("WWTW.unprocessed_input", late, every)

    # --- orchestration ----------------------------------------------------------------------
    def calculate_discharge(self, *, active: Tensor) -> None:
        """``wtw.py:323-341``: when the tank holds more than ``FLOAT_ACCURACY`` and the spare
        throughput is above it (both tested, ``&``), pull ``min(spare, tank V)`` of the tank's
        water into ``current_input``; then add the liquor and treat."""
        ctx, p, tank = self.ctx, self.p, self.stormwater_tank
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "calculate_discharge")
        excess = self.get_excess_throughput(active=active)
        held = ctx.decide(
            f"{self.site}/calculate_discharge/stormwater",
            tank.get_avail(active=active)[..., 0],
            ">",
            ctx.fa,
            active=active,
            call=call,
        )
        spare = ctx.decide(
            f"{self.site}/calculate_discharge/spare", excess, ">", ctx.fa, active=active, call=call
        )
        run = active & held & spare
        to_pull = dmin(
            ctx,
            f"{self.site}/calculate_discharge/clip",
            excess,
            tank.get_avail(active=run)[..., 0],
            active=run,
            call=call,
        )
        to_pull = v_change_vqip(p, tank.storage, to_pull)
        cleared = tank.pull_storage(to_pull, active=run)
        self.current_input = write(
            run, sum_vqip(p, self.current_input, cleared), self.current_input
        )
        self.current_input = write(
            active, sum_vqip(p, self.current_input, self.liquor), self.current_input
        )
        self.treat_current_input(active=active)

    def make_discharge(self, *, active: Tensor) -> None:
        """``wtw.py:343-349``: push ``treated`` over every out-arc and empty it. A rejection above
        ``FLOAT_ACCURACY`` is forced into the stormwater tank (``"wwtw_couldnt_push"``); one at
        or below it vanishes, recorded as dropped water (W25)."""
        ctx = self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "make_discharge")
        reply = self.push_distributed(self.treated, active=active)
        self.treated = write(active, torch.zeros_like(self.treated), self.treated)
        back = ctx.decide(
            f"{self.site}/make_discharge/rejected",
            reply[..., 0],
            ">",
            ctx.fa,
            active=active,
            call=call,
        )
        self.stormwater_tank.push_storage(reply, active=active & back, force=True)
        ctx.event("wwtw_couldnt_push", self.name, active & back)
        ctx.drop(reply, active & ~back)

    # --- the handlers -----------------------------------------------------------------------
    def push_check_sewer(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``wtw.py:351-372``: the tank's excess plus the spare throughput, cut to the request
        when one is given, at the request's concentration (an empty VQIP without one)."""
        ctx = self.ctx
        excess_throughput = self.get_excess_throughput(active=active)
        excess_tank = self.stormwater_tank.get_excess(active=active)
        volume = excess_tank[..., 0] + excess_throughput
        if vqip is None:
            vqip = ctx.empty()
        else:
            volume = dmin(
                ctx,
                f"{self.site}/push_check_sewer/clip",
                volume,
                vqip[..., 0],
                active=active,
                call=ctx.call(self.site, "push_check_sewer"),
            )
        return v_change_vqip(self.p, vqip, volume)

    def push_set_sewer(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``wtw.py:374-407``: up to the spare throughput into ``current_input``; when that was
        all of it (an exact comparison, ``:394``) nothing is rejected. The rest goes to the
        stormwater tank without force, and the tank's rejection is returned only when it is at
        least ``FLOAT_ACCURACY``; a smaller one vanishes, recorded as dropped water."""
        ctx, p = self.ctx, self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "push_set_sewer")
        vqip = vqip.clone()  # copy_vqip (wtw.py:384)
        sent_direct = self.get_excess_throughput(active=active)
        sent_direct = dmin(
            ctx,
            f"{self.site}/push_set_sewer/direct",
            sent_direct,
            vqip[..., 0],
            active=active,
            call=call,
        )
        sent_direct = v_change_vqip(p, vqip, sent_direct)
        self.current_input = write(
            active, sum_vqip(p, self.current_input, sent_direct), self.current_input
        )
        done = ctx.decide(
            f"{self.site}/push_set_sewer/all_direct",
            sent_direct[..., 0],
            "==",
            vqip[..., 0],
            active=active,
            call=call,
        )
        run = active & ~done
        rest = v_change_vqip(p, vqip, vqip[..., 0] - sent_direct[..., 0])
        rejected = self.stormwater_tank.push_storage(rest, active=run)
        small = ctx.decide(
            f"{self.site}/push_set_sewer/below_fa",
            rejected[..., 0],
            "<",
            ctx.fa,
            active=run,
            call=call,
        )
        ctx.drop(rejected, run & small)
        return torch.where((done | small)[..., None], ctx.empty(), rejected)

    def pull_set_reuse(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``wtw.py:409-428``: ``min(request, treated V)`` of the treated water."""
        p = self.p
        reply_vol = dmin(
            self.ctx,
            f"{self.site}/pull_set_reuse/clip",
            vqip[..., 0],
            self.treated[..., 0],
            active=active,
            call=self.ctx.call(self.site, "pull_set_reuse"),
        )
        reply = v_change_vqip(p, self.treated, reply_vol)
        left = v_change_vqip(p, self.treated, self.treated[..., 0] - reply_vol)
        self.treated = write(active, left, self.treated)
        return reply

    def pull_check_reuse(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``wtw.py:430-445``: the treated water, whatever the request."""
        return self.treated.clone()


class FWTW(WTW):
    """``FWTW`` (``wtw.py:456-657``): freshwater treatment into a service reservoir
    (``service_reservoir_tank``). ``treat_water`` pulls what would fill the reservoir, creates
    any shortfall as water at yesterday's pulled quality (a mass-balance inflow,
    ``"FWTW.total_deficit"``), treats, sends liquor and solids to sewers (what they reject is
    unpushed sludge, which leaves the system) and puts the treated water in the reservoir.
    Pulls come from the reservoir.
    """

    wsimod_type: ClassVar[str] = "FWTW"
    store_paths: ClassVar[tuple[str, ...]] = (
        "service_reservoir_tank.storage",
        "current_input",
        "treated",
        "liquor",
        "solids",
        "total_deficit",
        "total_pulled",
        "previous_pulled",
        "unpushed_sludge",
    )
    # wtw.py:646-653 keeps liquor and sets previous_pulled; the day's totals and treated are
    # emptied. current_input and solids are not, but treat_water overwrites current_input
    # before anything reads it and the solids' non-additive columns are never written, so
    # neither carries anything (spec, State).
    carried_paths: ClassVar[tuple[str, ...]] = (
        "service_reservoir_tank.storage",
        "liquor",
        "previous_pulled",
    )
    step_methods: ClassVar[tuple[str, ...]] = ("treat_water",)
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")
    # wtw.py:542-544
    mass_balance: ClassVar[dict[str, str]] = {
        "FWTW.total_deficit": "in",
        "FWTW.unpushed_sludge": "out",
    }
    parameter_defaults: ClassVar[dict[str, float]] = {
        **WTW.parameter_defaults,
        "service_reservoir_storage_capacity": 10,
        "service_reservoir_storage_area": 1,
        "service_reservoir_storage_elevation": 10,
    }
    tank_parameters: ClassVar[dict[str, tuple[str, str]]] = {
        "service_reservoir_storage_capacity": ("service_reservoir_tank", "capacity"),
        "service_reservoir_storage_area": ("service_reservoir_tank", "area"),
        "service_reservoir_storage_elevation": ("service_reservoir_tank", "datum"),
    }

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # wtw.py:517-521
        self.pull_set_handler["default"] = self.pull_set_fwtw
        self.pull_check_handler["default"] = self.pull_check_fwtw
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_check_deny
        # wtw.py:524-527
        self.total_deficit = empty(pollutants, ())
        self.total_pulled = empty(pollutants, ())
        self.previous_pulled = empty(pollutants, ())
        self.unpushed_sludge = empty(pollutants, ())
        # wtw.py:530-535
        initial = spec.parameters.get("service_reservoir_initial_storage", 0)
        if isinstance(initial, Mapping):
            known = {k: v for k, v in initial.items() if k == "volume" or k in pollutants.names}
            initial = pollutants.to_vqip(known)
        self.service_reservoir_tank = Tank(
            pollutants,
            capacity=self.defaults["service_reservoir_storage_capacity"],
            area=self.defaults["service_reservoir_storage_area"],
            datum=self.defaults["service_reservoir_storage_elevation"],
            initial_storage=_f64(initial),
            owner=self.name,
            path="service_reservoir_tank",
        )

    def _tanks(self) -> list[Tank]:
        return [self.service_reservoir_tank] if hasattr(self, "service_reservoir_tank") else []

    def end_timestep(self) -> None:
        """``end_timestep_`` (``wtw.py:646-653``): ``previous_pulled`` becomes the day's
        ``total_pulled``; the day's totals and ``treated`` are emptied; ``liquor`` is kept."""
        self.service_reservoir_tank.end_timestep()
        self.total_deficit = empty(self.p, ())
        self.previous_pulled = self.total_pulled.clone()
        self.total_pulled = empty(self.p, ())
        self.treated = empty(self.p, ())
        self.unpushed_sludge = empty(self.p, ())

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        ctx.term("FWTW.total_deficit", self.total_deficit, every)
        ctx.term("FWTW.unpushed_sludge", self.unpushed_sludge, every)

    # --- orchestration ----------------------------------------------------------------------
    def treat_water(self, *, active: Tensor) -> None:
        """``wtw.py:573-617``:

        1. ``target = min(reservoir excess, throughput capacity)``, pulled with
           ``pull_distributed`` (no ``of_type``);
        2. the deficit ``max(target - pulled, 0)`` is created at ``previous_pulled``'s quality
           and joins the pulled water as ``current_input`` (``"service_reservoirs_not_filled"``
           above ``FLOAT_ACCURACY``);
        3. treat; push liquor + solids to ``Sewer`` nodes, and the rejection is unpushed sludge
           (``"sludge_nowhere"`` above ``FLOAT_ACCURACY``);
        4. the treated water into the reservoir without force, the excess then forced in too
           (``"excess_treated_water"`` when positive).
        """
        ctx, p, tank = self.ctx, self.p, self.service_reservoir_tank
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "treat_water")
        target = dmin(
            ctx,
            f"{self.site}/treat_water/target",
            tank.get_excess(active=active)[..., 0],
            self._param("treatment_throughput_capacity"),
            active=active,
            call=call,
        )
        throughput = self.pull_distributed(self._volume_vqip(target), active=active)
        deficit = dmax(
            ctx,
            f"{self.site}/treat_water/deficit",
            target - throughput[..., 0],
            0.0,
            active=active,
            call=call,
        )
        deficit = v_change_vqip(p, self.previous_pulled, deficit)
        self.current_input = write(active, sum_vqip(p, throughput, deficit), self.current_input)
        self.total_deficit = write(
            active, sum_vqip(p, self.total_deficit, deficit), self.total_deficit
        )
        ctx.event(
            "service_reservoirs_not_filled",
            self.name,
            active & (self.total_deficit[..., 0] > ctx.fa).detach(),
        )
        self.treat_current_input(active=active)
        push_back = sum_vqip(p, self.liquor, self.solids)
        rejected = self.push_distributed(push_back, of_type=SLUDGE_TYPES, active=active)
        self.unpushed_sludge = write(
            active, sum_vqip(p, self.unpushed_sludge, rejected), self.unpushed_sludge
        )
        ctx.event("sludge_nowhere", self.name, active & (rejected[..., 0] > ctx.fa).detach())
        excess = tank.push_storage(self.treated, active=active)
        tank.push_storage(excess, active=active, force=True)
        ctx.event("excess_treated_water", self.name, active & (excess[..., 0] > 0).detach())

    # --- the handlers -----------------------------------------------------------------------
    def pull_check_fwtw(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``wtw.py:619-629``: the reservoir's ``get_avail``."""
        return self.service_reservoir_tank.get_avail(vqip, active=active)

    def pull_set_fwtw(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``wtw.py:631-644``: from the reservoir, added to the day's ``total_pulled``."""
        pulled = self.service_reservoir_tank.pull_storage(vqip, active=active)
        self.total_pulled = write(
            active, sum_vqip(self.p, self.total_pulled, pulled), self.total_pulled
        )
        return pulled

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """A quantity-only run also ignores a VQIP ``service_reservoir_initial_storage``'s
        pollutants."""
        ignored = super().validate(spec, network, pollutants=pollutants)
        initial = spec.parameters.get("service_reservoir_initial_storage", 0)
        if not pollutants and isinstance(initial, Mapping) and any(k != "volume" for k in initial):
            ignored.append("service_reservoir_initial_storage")
        return ignored


NODE_CLASSES.update({"WTW": WTW, "WWTW": WWTW, "FWTW": FWTW})
