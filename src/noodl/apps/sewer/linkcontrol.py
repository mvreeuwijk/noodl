"""The link settings of a dynamic-wave run: pump startup and shutoff depths, control rules and
the orifice opening rate, as SWMM applies them at the start of every routing step.

`LinkControl.apply` is ``routing.c::evaluateControlRules`` (routing.c:271-310) on the state at
the START of a step, in SWMM's order:

1. ``link_setTargetSetting`` (link.c:604-624): every pump's target is its setting, switched OFF
   when its inlet depth is below its shutoff depth and ON when it is above its startup depth;
2. ``controls_evaluate`` (`controls.ControlEngine`) when the rule step has come (every routing
   step with RULE_STEP 0): the fired actions set the targets of the links they name;
3. every link whose target differs from its setting is changed (``link_setSetting``,
   link.c:628-640): a pump, weir or outlet takes its target at once, an orifice moves
   toward it at its opening rate (``orifice_setSetting``, link.c:1729-1760: the setting changes
   by ``step / orate`` per step, ``orate`` being CloseTime in seconds, and lands on the target
   when within that and 0.001), and a link opened or closed (target or setting 0) has its
   time-last-set updated for the TIMEOPEN and TIMECLOSED premises.

The settings, targets and times are the carried state keys of `dynwave.CONTROL_KEYS`, so they
are discrete state between steps: nothing differentiates through them (the rules compare
values and pick actions). A closing orifice keeps giving its end nodes the surface area it last
had, as SWMM does (it is no longer evaluated, link.c:550): the areas are frozen into
``"sewer.orifice_hold"`` as it closes.

Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.sewer.controls import ControlEngine, start_serial, swmm_datetime
from noodl.apps.sewer.xsect_tables import LCF, QCF

Tensor = torch.Tensor
F64 = torch.float64
#: SI area per SWMM ft2, ``dynwave._AREA_SI`` (kept in step with it)
_AREA_SI = QCF / LCF


class _Context:
    """The simulation as `controls.ControlEngine` reads it, in the file's units, at the start
    of a step: node values from the carried state, link flows from ``"water.q"`` and settings
    from the (pump-thresholded) targets."""

    def __init__(self, hyd, state: Mapping, target: Tensor, lateral: Tensor, controls) -> None:
        self.hyd = hyd
        self.lf = controls.length_factor
        self.qf = controls.flow_factor
        invert = hyd.node_invert.tolist()
        phi = state[f"{hyd.layer}.phi"].tolist()
        self.depth = [max(h - z, 0.0) for h, z in zip(phi, invert, strict=True)]    # m
        self.invert = invert
        self.q = state[f"{hyd.layer}.q"].tolist()
        self.setting = state["sewer.setting"].tolist()
        self.target = target.tolist()
        self.last = state["sewer.time_last_set"].tolist()
        self.lateral = lateral.tolist()
        self.state = state
        self.volume = None
        self.default_last = float(int(start_serial(controls.start)))

    def node_value(self, name: str, attribute: str) -> float:
        i = self.hyd.node_index[name]
        if attribute == "depth":
            return self.depth[i] / self.lf
        if attribute == "head":
            return (self.depth[i] + self.invert[i]) / self.lf
        if attribute == "inflow":
            return self.lateral[i] / self.qf
        if self.volume is None:                      # volume
            heads = self.state[f"{self.hyd.layer}.phi"]
            self.volume = self.hyd._node_volume_ft3(heads).tolist()
        j = int(self.hyd.junction_pos[i])
        if j < 0:
            return 0.0
        return self.volume[j] * (1.0 if self.lf != 1.0 else QCF)    # ft3 -> the file volume unit

    def link_flow(self, name: str) -> float:
        column = self.hyd.link_column[name]
        return self.q[column] * self.hyd.link_direction[column] / self.qf

    def link_setting(self, name: str) -> tuple[str, float, float, float]:
        kind = self.hyd.link_kind[name]
        if kind == "conduit":
            return kind, 1.0, 1.0, self.default_last
        i = self.hyd.setting_index[name]
        return kind, self.setting[i], self.target[i], self.last[i]


class LinkControl:
    """Applies pump depth limits, control rules and the orifice opening rate to a run's state
    (module docstring). Make one per run: it holds the rule engine's memory (the last
    comparison, each PID's two previous errors) and the next rule time."""

    def __init__(self, hyd) -> None:
        self.hyd = hyd
        self.engine = ControlEngine(hyd.controls) if hyd.controls is not None else None
        self.next_rule_ms = 0.0                                  # routing.c:127, NewRuleTime

    def rule_step_limit(self, now_ms: float, step: float) -> float:
        """The routing `step` (s) from `now_ms` shortened to end at the next rule time when it
        would reach it (routing.c:189-195)."""
        controls = self.hyd.controls
        if controls is None or controls.rule_step <= 0.0:
            return step
        next_rule = self.next_rule_ms + 1000.0 * controls.rule_step
        if now_ms + 1000.0 * step >= next_rule:
            return (next_rule - now_ms) / 1000.0
        return step

    def apply(self, state: Mapping, now_ms: float, step: float, lateral: Tensor | None
              ) -> dict[str, Tensor]:
        """The state at the start of the step that begins `now_ms` ms into the run and lasts
        `step` s, with its settings, targets, times last set and orifice holds updated.
        `lateral` is the full-node lateral inflow (m3/s) of the PREVIOUS step (None at the
        first), which is what a rule's NODE INFLOW reads (SWMM evaluates the rules before
        the new inflows are added, routing.c:271-276)."""
        hyd = self.hyd
        setting = state["sewer.setting"]
        target = setting.clone()                  # link_setTargetSetting: target = setting
        pumps = hyd.structures.get("pump")
        if pumps is not None:
            sl = slice(pumps.first, pumps.first + len(pumps.names))
            now = setting[..., sl]
            depth = state["sewer.junction_depth"][..., hyd.pump_inlet] / LCF        # ft
            off, on = hyd.pump_off_ft, hyd.pump_on_ft
            t = torch.where((off > 0.0) & (now > 0.0) & (depth < off),
                            torch.zeros_like(now), now)                              # :617-619
            t = torch.where((on > 0.0) & (now == 0.0) & (depth > on),
                            torch.ones_like(now), t)                                 # :620-622
            target[..., sl] = t
        controls = hyd.controls
        elapsed_ms = now_ms
        current = swmm_datetime(controls.start, elapsed_ms) if controls is not None else (
            elapsed_ms / 86400000.0)
        if self.engine is not None and (controls.rule_step == 0.0
                                        or abs(now_ms - self.next_rule_ms) < 1.0):
            if setting.dim() != 1:
                raise ValueError("simulate_dynamic_wave: control rules need an unbatched "
                                 "state (each batch member would need its own settings)")
            zero = torch.zeros(hyd.node_invert.shape, dtype=F64)
            ctx = _Context(hyd, state, target, zero if lateral is None else lateral, controls)
            changed = self.engine.evaluate(
                ctx, current, current - start_serial(controls.start), step / 86400.0)
            for link, value in changed.items():
                target[hyd.setting_index[link]] = value
        out = dict(state)
        differs = target != setting
        if bool(differs.any()):
            toggled = differs & (target * setting == 0.0)                           # :290-291
            new = target.clone()
            orifices = hyd.structures.get("orifice")
            hold = state["sewer.orifice_hold"]
            if orifices is not None:
                new, hold = self._orifices(state, orifices, setting, target, new, hold, step)
            out.update({
                "sewer.setting": new,
                "sewer.target_setting": target,
                "sewer.time_last_set": torch.where(
                    toggled, torch.full_like(setting, current), state["sewer.time_last_set"]),
                "sewer.orifice_hold": hold,
            })
        else:
            out["sewer.target_setting"] = target
        if controls is not None and controls.rule_step > 0.0:
            next_rule = self.next_rule_ms + 1000.0 * controls.rule_step
            if abs(now_ms + 1000.0 * step - next_rule) < 1.0:                        # :305-306
                self.next_rule_ms = next_rule
        return out

    def _orifices(self, state, orifices, setting, target, new, hold, step):
        """`orifice_setSetting`'s moves of the orifices' settings toward their targets
        (link.c:1742-1752) and the end areas of the ones that close."""
        hyd = self.hyd
        sl = slice(orifices.first, orifices.first + len(orifices.names))
        rate = hyd.orifice_rate
        cur, tgt = setting[..., sl], target[..., sl]
        delta = tgt - cur
        move = (rate > 0.0) & (step / torch.where(rate > 0.0, rate, torch.ones_like(rate))
                               + 0.001 < delta.abs())
        stepped = cur + torch.sign(delta) * step / torch.where(rate > 0.0, rate,
                                                               torch.ones_like(rate))
        moved = torch.where(move, stepped, tgt)
        new = new.clone()
        new[..., sl] = moved
        closing = (cur > 0.0) & (moved <= 0.0)
        if bool(closing.any()):
            end1, end2 = hyd.orifice_end_areas(state)
            n = len(orifices.names)
            hold = hold.clone()
            hold[:n] = torch.where(closing, end1 * _AREA_SI, hold[:n])
            hold[n:] = torch.where(closing, end2 * _AREA_SI, hold[n:])
        return new, hold
