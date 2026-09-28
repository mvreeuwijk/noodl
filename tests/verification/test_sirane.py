"""SIRANE as the independent reference: code-to-code verification of the street application
against SIRANE v2.1 rev 128 on its own South Kensington case (46 streets, 36 junctions).

The fixture (`tests/data/street/sirane_south_kensington`, see its NOTICE.md) is the deck
plus the archived results of one run, hours 00 and 01 of 7 January 2014. That run was NOT
the deck as shipped: its `Resul_Meteo.dat` records a wind from 315 degrees (the deck: 135),
and every street's `Sigma_wH` is SIRANE's default floor of 0.30 m/s (the deck sets the floor
to 0). noodl physics is therefore driven here with the RESULTS' own meteorology --
`read_results(...).meteo`: u*, the boundary-layer height, the Obukhov length, the direction
and the direction spread -- never with the deck's.

Five comparisons, each by case:

- the roof exchange velocity: noodl physics' `exchange_velocity(form="turbulent_velocity")` at
  SIRANE's own `Sigma_wH` against SIRANE's printed `u_d`, street by street and hour by hour;
- the in-canyon wind: noodl physics' Soulhac canyon velocity under SIRANE's u* and
  direction against SIRANE's `U_moy`, with the one-sided streets reported separately;
- the in-canyon concentration with SIRANE's above-roof field imposed. The archived results
  turn out to come from a run with a different emission field from the deck's (a mass
  balance of SIRANE's own output, independent of noodl physics, shows it); that finding is
  what is asserted, and noodl physics' residual under the deck's emission is recorded;
- the above-roof plume model on its own: noodl physics' kernel applied to SIRANE's own roof
  fluxes, `u_d W L (Cint - Cext)`, against SIRANE's printed `Cext`, with the inputs and
  options the archive can discriminate;
- the same kernel against SIRANE's own concentration grids, centrelines and receptors in
  ten cases of one isolated street (`tests/data/street/sirane_kernel_probe`, see its
  NOTICE.md) across, along and oblique to the wind, neutral and stable.

Tolerances are set from the measured values and each comment states the measured number and
why the tolerance is what it is. SIRANE prints every velocity to two decimals, which is the
floor on how well any of these comparisons can agree; nothing here is loosened silently.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.street_aq import build_model, drivers_at, plume, read_case, read_results
from noodl.apps.street_aq.above_roof import roof_fluxes, roof_wiring
from noodl.apps.street_aq.canyon import SIRANE_EXCHANGE, exchange_velocity
from noodl.apps.street_aq.network import Street, StreetNetwork
from noodl.apps.street_aq.routing import StreetFlows

DT = torch.float64
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "street" / "sirane_south_kensington"
PRINTED_HALF_STEP = 0.005          # m/s: SIRANE prints U_moy, Sigma_wH, u_d, Ustar to 0.01
SIGMA_W_FLOOR = 0.30               # m/s: SIRANE's default sigma_w floor, active in this run
SPECIES = "O3"


@functools.lru_cache(maxsize=1)
def _case():
    return read_case(FIXTURE / "Donnees_SouthKensington.dat")


@functools.lru_cache(maxsize=1)
def _results():
    return read_results(FIXTURE / "RESULT_SOUTHKENSINGTON", case=_case())


def _driven_case():
    """The deck's case with the RESULTS' meteorology and SIRANE's `Cext` as the per-street
    background -- everything else (network, emissions, species) from the deck."""
    case, res = _case(), _results()
    meteo = {key: value for key, value in res.meteo.items() if key != "sigma_theta"}
    background = np.zeros_like(case.background)
    for s, name in enumerate(case.species):
        background[:, :, s] = res.c_above[name]
    return replace(case, meteo=meteo, meteo_junction={}, background=background)


def _model(**extra):
    """The model of the case's own options (`model_options()`), `extra` overriding them."""
    case = _case()
    model, state, _ = build_model(
        case.network, species=(SPECIES,), meteo="per_street", background="per_street",
        **dict(case.model_options(), **extra),
    )
    return model, state


def _drivers(model, k):
    """`drivers_at` on the driven case at hour `k`, plus SIRANE's own direction spread
    (`Resul_Meteo.dat`'s `SigmaTheta`, one station for the whole case) on every junction."""
    res = _results()
    drivers = drivers_at(_driven_case(), model, k, species=[SPECIES])
    spread = np.unique(res.meteo["sigma_theta"][k])
    assert spread.size == 1                        # one meteorological station
    drivers["sigma_theta"] = torch.full((len(_case().network.junctions),),
                                        float(spread[0]), dtype=DT)
    return drivers


def test_the_exchange_velocity_of_every_street_and_hour_is_sirane_own(record_property):
    """noodl physics' `exchange_velocity(form="turbulent_velocity")`, evaluated at SIRANE's printed
    `Sigma_wH` with each street's own height and width, reproduces SIRANE's printed `u_d`
    on all 46 streets in both hours, to the printed two decimals -- and the reading
    `sigma_w / sqrt(2 pi)` does not (Soulhac et al. 2011, Eq. 5)."""
    case, res = _case(), _results()
    height = torch.tensor([s.height for s in case.network.streets], dtype=DT)
    width = torch.tensor([s.width for s in case.network.streets], dtype=DT)
    worst = 0.0
    for k in range(len(res.times)):
        sigma_w = torch.as_tensor(res.sigma_w_roof[k], dtype=DT)
        ours = exchange_velocity(sigma_w, height, width, form="turbulent_velocity").numpy()
        miss = np.abs(ours - res.u_exchange[k])
        worst = max(worst, float(miss.max()))
        # The rejected reading misses by far more than the printing can explain: measured
        # 0.1197 - 0.07 = 0.0497 m/s on every street, ten printed half-steps; 5 is the bound.
        wrong = res.sigma_w_roof[k] / np.sqrt(2.0 * np.pi)
        assert bool((np.abs(wrong - res.u_exchange[k]) > 5 * PRINTED_HALF_STEP).all())
    # Measured |0.0675 - 0.07| = 0.0025 m/s at sigma_w = 0.30 (u_d = 0.0675, printed 0.07):
    # inside the printed half-step, which is the tightest bound two-decimal output allows.
    assert worst <= PRINTED_HALF_STEP
    record_property("check", "Roof exchange velocity u_d vs SIRANE, 46 streets x 2 hours")
    record_property("tolerance", "abs 0.005 m/s (SIRANE prints u_d to 0.01)")
    record_property("measured_abs_max", worst)
    record_property("constant", SIRANE_EXCHANGE)


def test_the_sigma_w_floor_is_the_exchange_velocity_floor_on_the_case():
    """SIRANE's sigma_w floor as `sigma_w_min` gives the same flows, bit for bit, as the
    same floor applied to the exchange velocity, `u_d_min = 0.30 / (sqrt(2) pi)`: `u_d`
    is linear in sigma_w, and the floor binds on every street of this run."""
    floored, state = _model(sigma_w_min=SIGMA_W_FLOOR)
    on_u_d, _ = _model(sigma_w_min=0.0, u_d_min=SIGMA_W_FLOOR * SIRANE_EXCHANGE)
    bare, _ = _model()
    for k in range(len(_results().times)):
        drivers = _drivers(floored, k)
        q = floored.current_flows("street", state, drivers)
        assert torch.equal(q, on_u_d.current_flows("street", state, drivers))
        assert not torch.equal(q, bare.current_flows("street", state, drivers))


def test_the_canyon_wind_of_every_street_against_sirane(record_property, capsys):
    """noodl physics' Soulhac canyon velocity `|u_canyon|` (Soulhac et al. 2008), driven
    per street with SIRANE's own u* (0.14 m/s) and direction (315 degrees FROM), against
    SIRANE's `U_moy` on all 46 streets; the one-sided streets (one building row, height
    `mean(HG, HD)`) are reported separately.

    The comparison is limited by SIRANE's printing: `U_moy` is 0.01-0.13 m/s printed to two
    decimals (a 0.01 m/s street carries +-50 %), and u* is printed 0.14 (+-3.6 %), which
    noodl physics' velocity scales with linearly. So the per-street check is against that
    rounding envelope, `|ours - U_moy| <= 0.005 + 0.036 |ours|`, and the relative errors
    are recorded for the record."""
    case, res = _case(), _results()
    model, state = _model()
    flows = next(c for c in model.closures if isinstance(c, StreetFlows))
    one_sided = np.array([sid in case.native["one_sided"] for sid in case.street_ids])
    u_star_rel = PRINTED_HALF_STEP / float(np.unique(res.meteo["u_star"])[0])
    per_hour = []
    for k in range(len(res.times)):
        ours = flows(state, _drivers(model, k))["street.u_canyon"].abs().numpy()
        theirs = res.u_canyon[k]
        rel = np.abs(ours / theirs - 1.0)
        envelope = PRINTED_HALF_STEP + u_star_rel * ours
        outside = np.abs(ours - theirs) > envelope
        worst = int(np.argmax(rel))
        with capsys.disabled():
            print(f"\nSIRANE canyon wind, hour {k:02d}: median |rel| {np.median(rel):.3f}, "
                  f"p90 {np.percentile(rel, 90):.3f}, worst street {case.street_ids[worst]} "
                  f"(noodl {ours[worst]:.4f} vs SIRANE {theirs[worst]:.2f} m/s); "
                  f"one-sided {np.round(rel[one_sided], 3).tolist()}; "
                  f"{int(outside.sum())} of 46 outside the rounding envelope")
        # Measured, both hours: median 0.045, p90 0.128, worst street 40 (0.0147 vs 0.01,
        # the smallest printed value, where the printing alone allows 50 %); median ratio
        # 1.03, i.e. u* = 0.14 printed for ~0.136. The one-sided streets 3, 14, 42: 0.030,
        # 0.054, 0.069 -- no worse than the rest, so the D2 height rule mean(HG, HD) does
        # not show here. Tolerances: median 0.08 and p90 0.20 are a regression guard
        # about 1.6x the measured values; every street must sit inside the rounding
        # envelope (measured: all 46 do).
        assert float(np.median(rel)) < 0.08
        assert float(np.percentile(rel, 90)) < 0.20
        assert float(np.median(rel[one_sided])) < 0.08
        assert not bool(outside.any())
        per_hour.append({"hour": k, "median_rel": float(np.median(rel)),
                         "p90_rel": float(np.percentile(rel, 90)),
                         "one_sided_rel": np.round(rel[one_sided], 4).tolist()})
    # Here and below, the scalar measured_* properties are the last hour's (the archive's
    # two hours are identical); measured_per_hour records every hour.
    record_property("check", "In-canyon wind |u_canyon| vs SIRANE U_moy, 46 streets")
    record_property("tolerance", "median rel < 0.08, p90 < 0.20, all inside the "
                                 "two-decimal rounding envelope")
    record_property("measured_median_rel", float(np.median(rel)))
    record_property("measured_p90_rel", float(np.percentile(rel, 90)))
    record_property("measured_one_sided_rel", np.round(rel[one_sided], 4).tolist())
    record_property("measured_per_hour", per_hour)


def test_the_archived_concentrations_come_from_another_emission_field(record_property,
                                                                       capsys):
    """The in-canyon concentration with SIRANE's above-roof field imposed.

    Set-up, as SIRANE's: per-street background = SIRANE's `Cext` (O3), the deck's unit
    emission (1 g/s of O3 on street 4), SIRANE's closures (`model_options()`), per-street
    meteorology and SIRANE's direction spread from `Resul_Meteo.dat`. O3 is passive here:
    NO and NO2 are zero in every `Cint` and `Cext` (asserted below), so Chapman's
    `k3 [NO][O3]` vanishes (and `k1 = 0` at night), even though `chemistry_on` is set.
    The archived results use SIRANE's default sigma_w floor of 0.30 m/s (the deck zeroes it,
    so `model_options()` gives `sigma_w_min=0`); it binds on every street in this run
    (noodl's own sigma_w here is 0.17 m/s), so the model is built with
    `sigma_w_min=0.30`.

    The comparison fails, and not because of noodl physics. SIRANE's own output alone
    fixes how much O3 leaves the canyons through their roofs,
    `sum u_d W L (Cint - Cext)`, plus what deposits (`DepSec_O3 = 0.01 Cint`: 1 cm/s,
    which noodl physics does not model): 21.75 + 3.49 g/s against the deck's 1 g/s. Every
    street's `Cint` exceeds its `Cext`, i.e. every street is a net source, and the highest
    `Cint` is on street 31, not the emitter 4. The archived run emitted about 25 g/s spread
    over the whole network -- NOTICE.md's "the emission field may differ too" -- so the
    deck's emission cannot reproduce it (noodl physics: median |rel| 0.888). A SIRANE
    comparison of the in-canyon concentration needs runs with known emissions (the
    source-receptor responses, or new runs of `write_sweep`'s decks).
    """
    case, res = _case(), _results()
    for other in ("NO", "NO2"):
        for field in (res.c_in, res.c_above):
            assert not bool(np.any(field[other]))
    area = np.array([s.length * s.width for s in case.network.streets])
    model, state = _model(sigma_w_min=SIGMA_W_FLOOR)
    per_hour = []
    for k in range(len(res.times)):
        c_in, c_above = res.c_in[SPECIES][k], res.c_above[SPECIES][k]
        roof_export = float((res.u_exchange[k] * area * (c_in - c_above)).sum())
        deposition = float((0.01 * area * c_in).sum())
        emitted = float(case.emissions[k, :, case.species.index(SPECIES)].sum())
        # SIRANE's roof export alone (measured 21.75 g/s) is more than 20x the deck's
        # emission (1 g/s); 10x is a margin no printing or routing detail could close.
        assert roof_export > 10.0 * emitted
        assert bool((c_in > c_above).all())
        emitter = int(np.argmax(case.emissions[k, :, case.species.index(SPECIES)]))
        assert int(np.argmax(c_in)) != emitter

        ours = model.steady(state, _drivers(model, k))["street.x"].numpy()
        rel = np.abs(ours / c_in - 1.0)
        with capsys.disabled():
            print(f"\nSIRANE in-canyon O3, hour {k:02d}: SIRANE roof export "
                  f"{roof_export * 1e3:.2f} g/s + deposition {deposition * 1e3:.2f} g/s vs "
                  f"deck emission {emitted * 1e3:.2f} g/s; noodl under the deck's "
                  f"emission: median |rel| {np.median(rel):.3f}, "
                  f"p90 {np.percentile(rel, 90):.3f}")
        per_hour.append({"hour": k, "sirane_roof_export_kg_s": roof_export,
                         "sirane_deposition_kg_s": deposition, "deck_emission_kg_s": emitted,
                         "median_rel": float(np.median(rel)),
                         "p90_rel": float(np.percentile(rel, 90))})
    record_property("check", "In-canyon O3 with SIRANE's Cext imposed vs SIRANE Cint")
    record_property("tolerance", "not asserted: the archived run's emission field is not "
                                 "the deck's (SIRANE roof export > 10x deck emission)")
    record_property("sirane_roof_export_kg_s", roof_export)
    record_property("sirane_deposition_kg_s", deposition)
    record_property("deck_emission_kg_s", emitted)
    record_property("measured_median_rel", float(np.median(rel)))
    record_property("measured_p90_rel", float(np.percentile(rel, 90)))
    record_property("measured_per_hour", per_hour)


# The above-roof plume kernel on SIRANE's own fluxes.
#
# SIRANE's archive fixes, per street, the roof flux it computed, `u_d W L (Cint - Cext)`, and
# the `Cext` it got from the upwind plumes. Feeding those fluxes through noodl physics'
# kernel (`plume.street_kernel` + `plume.junction_kernel`) under SIRANE's own meteorology
# and comparing with the printed `Cext` isolates the above-roof model: no emission, no
# canyon solve and no chemistry enters. The background is zero in this deck, so
# `C_ext = K F`.


def _plume_inputs(k):
    """`(F_s, F_up, F_down, theta_w)` at hour `k`: every street's roof flux and every
    junction's vertical fluxes, kg/s, from SIRANE's `Cint`/`Cext` and noodl physics' flows.

    The model is built with SIRANE's sigma_w floor (`sigma_w_min`), so its exchange flow is
    `u_d W L` with SIRANE's own `u_d` (the first test above) and `F_s` is SIRANE's roof
    flux. The junction fluxes are noodl physics' own: its vent flows (junction routing
    under SIRANE's u* and direction) carrying SIRANE's `Cint` of the street each vent
    leaves (`F_up`) and SIRANE's `Cext` of the street it enters (`F_down`), as excess over
    the (zero) background -- SIRANE does not print its intersection fluxes."""
    res = _results()
    model, state = _model(sigma_w_min=SIGMA_W_FLOOR)
    drivers = _drivers(model, k)
    q = model.current_flows("street", state, drivers)
    c_in = torch.as_tensor(res.c_in[SPECIES][k], dtype=DT)[:, None]
    c_ext = torch.as_tensor(res.c_above[SPECIES][k], dtype=DT)[:, None]
    f_s, up, down = roof_fluxes(roof_wiring(model), q, c_in, c_ext, torch.zeros_like(c_in))
    theta = np.unique(drivers["theta_w"].numpy())
    assert theta.size == 1                         # one meteorological station
    return f_s[:, 0], up[:, 0], down[:, 0], float(theta[0])


# The unrounded u* and the theta* of the archived hour. SIRANE prints u* to two decimals
# (0.14 m/s); its own value for this hour is recovered from the kernel-probe case of the same
# meteorology (`G2_M3F`, below), whose near-field plateau fixes U(H_R) = 0.83835 m/s and so
# u* = 0.13803 m/s (`test_the_probe_plateau_fixes_the_wind_at_the_reflection_height`).
# SIRANE's preprocessor prints theta* = 0.072 K for this hour; the kernel reproduces the
# probe cases of this meteorology with 0.060 K (an empirical input,
# `plume.plume_table(theta_star=)`; the printed value doubles the p90 error below).
SK_U_STAR = 0.13803
SK_THETA_STAR = 0.060


_DECK_CELL = object()


def _plume_c_ext(k, *, u_star=SK_U_STAR, theta_star=SK_THETA_STAR, floors=True,
                 junction_source="upward", self_contribution=False, junctions=True,
                 rotate_rad=0.0, meteo_cell_dx=_DECK_CELL, **table_options):
    """noodl physics' `C_ext = K_s F_s + K_j F_j` at hour `k` on SIRANE's fluxes, with
    SIRANE's meteorology (`Resul_Meteo.dat`: the boundary-layer height, the Obukhov
    length, the direction, `SigmaTheta` and the temperature; u* and theta* as above), the
    deck's reflection height `H_R` = 20 m and the dispersion site's `Z0D`, `ZDISPL`.
    `floors=True` keeps SIRANE's default `sigma_v`, `sigma_w` floors, which the archived
    run used. The downwind cut-off follows the deck's meteo-grid cell (750 m x 600 m,
    `read_case(...).native["meteo_grid"]`): 750 m / |cos 135 deg| = 1061 m;
    `meteo_cell_dx=None` falls back to the table's 700 m. Diagnostics only:
    `junctions=False` leaves the junction plumes out and `rotate_rad` turns SIRANE's wind
    direction; `table_options` go to `plume_table`."""
    case, res = _case(), _results()
    f_s, up, down, theta = _plume_inputs(k)
    site = case.native["site_disp"]
    met = {key: float(np.unique(res.meteo[key][k])[0])
           for key in ("h_abl", "lmo", "sigma_theta", "temperature")}
    if not floors:
        table_options.update(sigma_v_min=0.0, sigma_w_min=0.0)
    if meteo_cell_dx is _DECK_CELL:
        meteo_cell_dx = case.native["meteo_grid"]["dx"]
    if meteo_cell_dx is not None:
        table_options.setdefault("x_max", float(plume.downwind_cutoff(
            theta + rotate_rad, meteo_cell_dx)))
    table = plume.plume_table(
        u_star=u_star, h_abl=met["h_abl"], lmo=met["lmo"],
        h_canopy=float(case.native["options"]["H_R"]), z0=site["Z0D"], d=site["ZDISPL"],
        sigma_theta=met["sigma_theta"], theta_star=theta_star,
        temperature=met["temperature"], **table_options)
    k_s = plume.street_kernel(case.network, table, theta_w=theta + rotate_rad,
                              self_contribution=self_contribution,
                              meteo_cell_dx=meteo_cell_dx)
    k_j = plume.junction_kernel(case.network, table, theta_w=theta + rotate_rad,
                                meteo_cell_dx=meteo_cell_dx)
    f_j = up if junction_source == "upward" else up - down
    return (k_s @ f_s + (k_j @ f_j if junctions else 0.0)).numpy()


def _against_sirane(ours, theirs):
    """`(|rel|, ratio, streets)` over the streets where SIRANE's `Cext` is non-zero (45 of
    46: the upwind corner street 1 has exactly 0)."""
    seen = np.flatnonzero(theirs > 0)
    ratio = ours[seen] / theirs[seen]
    return np.abs(ratio - 1.0), ratio, seen


def test_the_plume_kernel_on_sirane_fluxes_reproduces_sirane_cext(record_property, capsys):
    """noodl physics' above-roof kernel, fed SIRANE's own roof fluxes (and junction fluxes
    from noodl physics' vent flows and SIRANE's `Cint`), reproduces SIRANE's printed `Cext`
    street by street: the default kernel (the default time offsets and plume-centre law,
    upward junction source, own street excluded, SIRANE's floors, the downwind cut-off of
    the deck's meteo cell) with this hour's u* and theta*. Both hours (identical
    meteorology and fields in this archive).

    Out of sample in geometry only: the network (45 streets of mixed length, width, height
    and angle, 36 junctions) is not one of the probe cases, but its hour is the M3F
    meteorology on which the default time offsets' `sigma_v` = 0.5 / `sigma_w` = 0.3 knots
    and theta* = 0.060 K were fitted."""
    case, res = _case(), _results()
    per_hour = []
    for k in range(len(res.times)):
        theirs = res.c_above[SPECIES][k]
        rel, ratio, seen = _against_sirane(_plume_c_ext(k), theirs)
        worst = [(case.street_ids[seen[i]], round(float(ratio[i]), 4))
                 for i in np.argsort(-rel)[:3]]
        with capsys.disabled():
            print(f"\nSIRANE Cext from SIRANE's fluxes, hour {k:02d}: median |rel| "
                  f"{np.median(rel):.4f}, p90 {np.percentile(rel, 90):.4f}, max "
                  f"{rel.max():.4f}, median ratio {np.median(ratio):.4f}; worst (street, "
                  f"ours/SIRANE) {worst}")
        # Measured, both hours: median |rel| 0.0032, p90 0.0138, max 0.028, median ratio
        # 1.0031, over all 45 streets (with the 700 m cut-off instead: 0.0031, 0.0106,
        # 0.023, 1.0000; see the next test). Tolerances: median 0.005, p90 0.02 and max
        # 0.04 are a regression guard about 1.5x the measured values; the median ratio
        # within 0.6 % pins the absence of a bias (a frame or sign error shows here first:
        # the reversed wind gives median ratio 1.52).
        assert float(np.median(rel)) < 0.005
        assert float(np.percentile(rel, 90)) < 0.02
        assert float(rel.max()) < 0.04
        assert abs(float(np.median(ratio)) - 1.0) < 0.006
        per_hour.append({"hour": k, "median_rel": float(np.median(rel)),
                         "p90_rel": float(np.percentile(rel, 90)),
                         "max_rel": float(rel.max()),
                         "median_ratio": float(np.median(ratio)), "worst_streets": worst})
    record_property("check", "Above-roof Cext from SIRANE's own roof and junction fluxes "
                             "vs SIRANE Cext, 45 streets")
    record_property("tolerance", "median |rel| < 0.005, p90 < 0.02, max < 0.04, "
                                 "|median ratio - 1| < 0.006")
    record_property("measured_median_rel", float(np.median(rel)))
    record_property("measured_p90_rel", float(np.percentile(rel, 90)))
    record_property("measured_max_rel", float(rel.max()))
    record_property("measured_median_ratio", float(np.median(ratio)))
    record_property("worst_streets", worst)
    record_property("measured_per_hour", per_hour)


def test_the_plume_inputs_sirane_discriminates(record_property, capsys):
    """The inputs and options SIRANE's archive can discriminate, each against the default
    of the test above, plus diagnostics that are recorded, not asserted:

    - theta* from u* and L (`theta_star=None`) instead of 0.060 K: 16x worse;
    - SIRANE's printed theta* 0.072 K doubles the p90, its printed u* 0.14 m/s the
      largest error;
    - SIRANE's floors off (the deck's values): the archived run used the defaults;
    - no junction plumes: the junction plumes are needed;
    - own street included (`self_contribution=True`): SIRANE excludes it;
    - `junction_source="net"`: indistinguishable from `"upward"` here;
    - the wind reversed: fails the bias bound of the test above, so that bound is a working
      check of the wind frame and sign;
    - recorded only: the wind rotated by 90 degrees, the 700 m cut-off (no meteo cell).
    """
    theirs = _results().c_above[SPECIES][0]
    measured = {}
    for label, kwargs in {
        "default": {},
        "theta_star_from_u_star_and_L": {"theta_star": None},
        "theta_star_printed": {"theta_star": 0.072},
        "u_star_printed": {"u_star": 0.14},
        "no_floors": {"floors": False},
        "no_junctions": {"junctions": False},
        "self_contribution": {"self_contribution": True},
        "junction_net": {"junction_source": "net"},
        "wind_reversed": {"rotate_rad": math.pi},
        "wind_rotated_90": {"rotate_rad": math.pi / 2.0},
        "cutoff_700": {"meteo_cell_dx": None},
    }.items():
        rel, ratio, _ = _against_sirane(_plume_c_ext(0, **kwargs), theirs)
        measured[label] = (float(np.median(rel)), float(np.percentile(rel, 90)),
                           float(np.median(ratio)), float(rel.max()))
    with capsys.disabled():
        print("\nSIRANE Cext, input variants (median |rel|, p90, median ratio): "
              + "; ".join(f"{k} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"
                          for k, v in measured.items()))
    default = measured["default"]
    # theta* from u* and L (0.0135 K): measured median 0.052, p90 0.097, ratio 0.948.
    assert measured["theta_star_from_u_star_and_L"][0] > 5 * default[0]
    # Printed theta* 0.072 K: measured p90 0.0307, 2.2x the default's 0.0138. Printed u*
    # 0.14: p90 0.0157, max 0.055 against the default's 0.028. 1.5x is the bound.
    assert measured["theta_star_printed"][1] > 1.5 * default[1]
    assert measured["u_star_printed"][3] > 1.5 * default[3]
    # Floors off: measured median 0.64 (ratio 1.62).
    assert measured["no_floors"][0] > 0.3
    # No junction plumes: measured median 0.008, p90 0.063, 4.5x the default's p90.
    assert measured["no_junctions"][1] > 3 * default[1]
    # Own street included: measured median ratio 2.33.
    assert measured["self_contribution"][2] > 1.5
    # "net": measured median 0.0029 against 0.0032 -- the downward vent flux carries
    # Cext, far less than the Cint going up. 0.005 is a guard on that equivalence.
    assert abs(measured["junction_net"][0] - default[0]) < 0.005
    # Wind reversed: measured median ratio 1.52, far outside the 0.006 bias bound.
    assert abs(measured["wind_reversed"][2] - 1.0) > 0.1
    # Not asserted: wind rotated 90 degrees, measured median 0.92, p90 5.5; the 700 m
    # cut-off 0.0031, 0.0106 -- this hour's archive cannot choose between the two
    # cut-offs (the kernel-probe cases do: `plume.downwind_cutoff`).
    for label, values in measured.items():
        record_property(f"{label}_median_rel_p90_ratio", values)


# The kernel against SIRANE's own grids: the kernel-probe cases.
#
# Each case is one isolated street (200 m, W = H = 20 m unless the name says W10 or H10)
# under imposed meteorology, with SIRANE's concentration grid written out
# (`tests/data/street/sirane_kernel_probe`, see its NOTICE.md). G1 lies across the wind,
# G2 along it (its flux vents at the downwind junction), G3 at 45 degrees and G6 at 20
# degrees; M1 is neutral (u* = 0.6 m/s), M2 stable (u* = 0.3 m/s, L = 100 m), M3F the South
# Kensington hour with SIRANE's floors. The street's and the junctions' fluxes are
# SIRANE's own (its Listing), so the comparison isolates the kernel.
#
# These checks are IN SAMPLE: the default time-offset knots, the plume-centre law and
# theta* = 0.060 K were fitted on these cases and their meteorologies (M1, M2, M3F among
# them). They verify that the implementation reproduces SIRANE where it was calibrated,
# and pin the per-case agreement; the out-of-sample check (in geometry) is the South
# Kensington C_ext above.

PROBES = Path(__file__).resolve().parents[1] / "data" / "street" / "sirane_kernel_probe"
NEUTRAL_LMO = 1.0e5        # m: SIRANE prints L = 1e6 m for a neutral hour


@functools.lru_cache(maxsize=1)
def _probe_manifest():
    return json.loads((PROBES / "manifest.json").read_text())


def _probe_table(run):
    """The run's plume table: SIRANE's printed meteorology, except u* and theta* of the
    M3F hour (`SK_U_STAR`, `SK_THETA_STAR`); the M2 hour's printed theta* (0.063 K) is
    the default `u*^2 T / (kappa g L)`, so none is passed."""
    spec = _probe_manifest()[run]
    met, opt, site = spec["meteo"], spec["options"], spec["site"]
    m3f = run.endswith("_M3F")
    lmo = math.inf if met["lmo"] >= NEUTRAL_LMO else met["lmo"]
    return plume.plume_table(
        u_star=SK_U_STAR if m3f else met["u_star"], h_abl=met["h_abl"], lmo=lmo,
        h_canopy=opt["H_R"], z0=site["Z0D"], d=site["ZDISPL"],
        sigma_theta=math.radians(met["sigma_theta_deg"]),
        theta_star=SK_THETA_STAR if m3f else None,
        temperature=met["temperature_c"] + 273.15,
        sigma_v_min=opt["SIGMA_V_MIN"], sigma_w_min=opt["SIGMA_W_MIN"])


def _probe_net(run):
    spec = _probe_manifest()[run]
    (x0, y0), (x1, y1) = spec["street"]
    return StreetNetwork([Street("probe", "u", "v", math.hypot(x1 - x0, y1 - y0), spec["W"],
                                 spec["H"])], {"u": x0, "v": x1}, {"u": y0, "v": y1})


def _probe_prediction(run, xy):
    """noodl physics' concentration (micrograms/m3) at points `xy` `(n, 2)` from the run's
    street and junction fluxes (g/s, SIRANE's Listing), wind from the west (0 rad)."""
    spec = _probe_manifest()[run]
    net = _probe_net(run)
    table = _probe_table(run)
    rec = torch.as_tensor(np.asarray(xy, dtype=float), dtype=DT)
    src = plume.source_points(net, theta_w=0.0)
    street = plume.plume_kernel(rec, src.xy, table, theta_w=0.0, width=src.width,
                                clamp=src.clamp) @ src.weight
    jxy, width, clamp = plume.junction_sources(net)
    node_flux = torch.tensor([spec["node_flux_g_s"][0 if j == "u" else 1]
                              for j in net.junctions], dtype=DT)
    junctions = plume.plume_kernel(rec, jxy, table, theta_w=0.0, width=width,
                                   clamp=clamp) @ node_flux
    return (1e6 * (street * spec["street_flux_g_s"] + junctions)).numpy()


def _probe_arrays(run):
    with np.load(PROBES / f"{run}.npz") as data:
        return {key: data[key].astype(float) for key in data.files}


# Per run: the (median, p90) of |ours/SIRANE - 1| over the 1200 sampled cells (drawn from
# those above 1 % of the grid's maximum, outside the street), as measured, and the
# tolerances asserted, about 1.5x the measured values rounded up.
PROBE_TOLERANCES = {
    # measured (median, p90): G1_M1 1.54e-3, 4.49e-3; G1_M1_H10 1.54e-3, 4.70e-3;
    # G1_M1_W10 1.20e-3, 7.72e-3; G1_M2 3.80e-3, 8.01e-3; G1_M3F 4.06e-3, 2.92e-2;
    # G2_M1 1.87e-4, 1.19e-3; G2_M3F 1.57e-4, 1.63e-3; G3_M1 1.62e-3, 3.51e-3;
    # G3_M2 5.84e-3, 1.87e-2; G6_M1 1.17e-3, 2.89e-3. The largest are the stable runs'
    # far field (the stable sigma_z law) and the ends of the oblique street in stable air.
    "G1_M1": (2.5e-3, 7e-3), "G1_M1_H10": (2.5e-3, 7.5e-3), "G1_M1_W10": (2e-3, 1.2e-2),
    "G1_M2": (6e-3, 1.2e-2), "G1_M3F": (6.5e-3, 4.5e-2), "G2_M1": (3e-4, 2e-3),
    "G2_M3F": (2.5e-4, 2.5e-3), "G3_M1": (2.5e-3, 5.5e-3), "G3_M2": (9e-3, 2.8e-2),
    "G6_M1": (2e-3, 4.5e-3),
}


@pytest.mark.parametrize("run", sorted(PROBE_TOLERANCES))
def test_the_kernel_reproduces_sirane_probe_grids(run, record_property, capsys):
    """The kernel against SIRANE's grid on the sampled cells."""
    arr = _probe_arrays(run)
    ours = _probe_prediction(run, np.stack([arr["cell_x"], arr["cell_y"]], axis=1))
    rel = np.abs(ours / arr["cell_c"] - 1.0)
    median, p90 = float(np.median(rel)), float(np.percentile(rel, 90))
    with capsys.disabled():
        print(f"\nSIRANE probe {run}: median |rel| {median:.2e}, p90 {p90:.2e}, "
              f"max {rel.max():.2e} over {rel.size} cells")
    tol_median, tol_p90 = PROBE_TOLERANCES[run]
    assert median < tol_median
    assert p90 < tol_p90
    record_property("check", f"Kernel vs SIRANE probe grid {run}, 1200 cells")
    record_property("tolerance", f"median |rel| < {tol_median}, p90 < {tol_p90}")
    record_property("measured_median_rel", median)
    record_property("measured_p90_rel", p90)


@pytest.mark.parametrize("run", sorted(PROBE_TOLERANCES))
def test_receptor_height_is_ignored_and_receptors_agree(run, record_property):
    """SIRANE's point receptors at z = 20, 25, 30 and 40 m above the same (x, y) hold
    exactly the same value in every run: the above-roof field has no height dependence,
    and the kernel takes none. The kernel against the receptors off the street (receptors
    over the street or on its end hold its `Cint` or half of it) above 1 % of the grid's
    maximum: measured median |rel| 6e-4 to 6.3e-3 (G1_M3F); 1.5e-2 is the bound."""
    spec, arr = _probe_manifest()[run], _probe_arrays(run)
    by_point: dict = {}
    for x, y, _z, c in arr["receptors"]:
        by_point.setdefault((x, y), set()).add(c)
    assert all(len(values) == 1 for values in by_point.values())
    xy = np.array(list(by_point))
    theirs = np.array([next(iter(v)) for v in by_point.values()])
    (x0, y0), (x1, y1) = spec["street"]
    a, b = np.array([x0, y0]), np.array([x1, y1])
    along = np.clip(((xy - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    off_street = np.linalg.norm(xy - (a + along[:, None] * (b - a)), axis=1) > spec["W"] / 2
    seen = off_street & (theirs > 0.01 * arr["cell_c"].max())
    rel = np.abs(_probe_prediction(run, xy[seen]) / theirs[seen] - 1.0)
    assert seen.sum() >= 10
    assert float(np.median(rel)) < 1.5e-2
    record_property("measured_median_rel", float(np.median(rel)))


@pytest.mark.parametrize("run", sorted(PROBE_TOLERANCES))
def test_the_downwind_cutoff(run):
    """Along the centreline (y = 0) SIRANE's field ends between 688 and 700 m downwind of
    the most downwind source (measured: 690 m in the stable runs, 698-700 m in the neutral
    ones, on a 2 m grid); the kernel's `x_max` is 700 m, and both are exactly zero beyond
    it."""
    spec, arr = _probe_manifest()[run], _probe_arrays(run)
    xs, theirs = arr["centre_x"], arr["centre_c"]
    last = max(spec["street"][0][0], spec["street"][1][0])
    reach = float(xs[np.flatnonzero(theirs)[-1]]) - last
    assert 686.0 <= reach <= plume.DOWNWIND_CUTOFF_M
    beyond = xs > last + plume.DOWNWIND_CUTOFF_M
    ours = _probe_prediction(run, np.stack([xs, np.zeros_like(xs)], axis=1))
    assert not np.any(theirs[beyond]) and not np.any(ours[beyond])


@pytest.mark.parametrize("run, u_star", [("G2_M1", 0.6), ("G2_M3F", SK_U_STAR)])
def test_the_probe_plateau_fixes_the_wind_at_the_reflection_height(run, u_star):
    """Just downwind of the junction a street along the wind vents through, the plume is a
    flat top of the junction's width `W` capped at `P_z = 1/H`, so SIRANE's centreline
    holds `C = F / (W H U(H_R))` exactly: 14 equal cells for G2_M1, 7 for G2_M3F. That
    plateau gives U(H_R), hence u*: 0.60001 m/s for the imposed 0.6 (G2_M1), and
    0.138031 m/s for the South Kensington hour, printed 0.14 (G2_M3F) -- `SK_U_STAR`."""
    spec, arr = _probe_manifest()[run], _probe_arrays(run)
    xs, theirs = arr["centre_x"], arr["centre_c"]
    node = spec["street"][1][0]
    near = theirs[(xs > node) & (xs < node + 30.0)]
    plateau = near.max()
    assert int(np.sum(np.abs(near - plateau) <= 1e-6 * plateau)) >= 5
    u_hr = spec["node_flux_g_s"][1] * 1e6 / (spec["W"] * spec["H"] * plateau)
    met = spec["meteo"]
    per_u_star = float(plume.wind_speed(
        spec["options"]["H_R"], u_star=1.0,
        lmo=math.inf if met["lmo"] >= NEUTRAL_LMO else met["lmo"],
        z0=spec["site"]["Z0D"], d=spec["site"]["ZDISPL"]))
    # measured 1.7e-5 (G2_M1) and 9e-6 (G2_M3F) relative: the grid is float32
    assert u_hr / per_u_star == pytest.approx(u_star, rel=1e-4)


# Per run: the (median, p90) of |ours/SIRANE - 1| on the grid columns nearest x = 150, 300
# and 650 m over the cells above 1 % of the grid's maximum. Measured: G1_M1 2.61e-3,
# 9.09e-3; G1_M1_H10 2.71e-3, 9.08e-3; G1_M1_W10 2.64e-3, 9.14e-3; G1_M2 4.58e-3, 2.76e-2;
# G1_M3F 5.68e-4, 4.24e-2; G2_M1 8.08e-4, 5.51e-3; G2_M3F 4.14e-3, 4.73e-3; G3_M1
# 1.51e-3, 6.42e-2; G3_M2 8.36e-3, 0.123; G6_M1 1.27e-3, 5.30e-3. The tolerances are 1.5x
# these, rounded up; the largest are the 650 m columns of the stable runs and of the
# oblique street in stable air (the stable sigma_z law).
CROSSWIND_TOLERANCES = {
    "G1_M1": (4e-3, 1.4e-2), "G1_M1_H10": (4.1e-3, 1.4e-2), "G1_M1_W10": (4e-3, 1.4e-2),
    "G1_M2": (7e-3, 4.2e-2), "G1_M3F": (9e-4, 6.4e-2), "G2_M1": (1.3e-3, 8.3e-3),
    "G2_M3F": (6.3e-3, 7.1e-3), "G3_M1": (2.3e-3, 9.7e-2), "G3_M2": (1.3e-2, 0.19),
    "G6_M1": (1.9e-3, 8e-3),
}


@pytest.mark.parametrize("run", sorted(CROSSWIND_TOLERANCES))
def test_the_crosswind_profiles_agree(run, record_property):
    """The kernel against SIRANE's grid columns at x = 150, 300 and 650 m, over the cells
    above 1 % of the grid's maximum (tolerances: `CROSSWIND_TOLERANCES`)."""
    arr = _probe_arrays(run)
    x, y = np.meshgrid(arr["cross_x"], arr["cross_y"], indexing="ij")
    seen = arr["cross_c"] > 0.01 * arr["cell_c"].max()
    ours = _probe_prediction(run, np.stack([x[seen], y[seen]], axis=1))
    rel = np.abs(ours / arr["cross_c"][seen] - 1.0)
    tol_median, tol_p90 = CROSSWIND_TOLERANCES[run]
    assert float(np.median(rel)) < tol_median
    assert float(np.percentile(rel, 90)) < tol_p90
    record_property("measured_median_rel", float(np.median(rel)))
    record_property("measured_p90_rel", float(np.percentile(rel, 90)))
