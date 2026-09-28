"""SIRANE's NO-NO2-O3 chemistry: code-to-code verification of the street application's
`closure="sirane"` photostationary split, and of `solar_elevation`, `j_no2_sirane` and
`k_no_o3_sirane`, against SIRANE v2.1 output (`tests/data/street/sirane_chemistry`, see its
NOTICE.md).

Two comparisons, each to SIRANE's print precision:

- per hour, over 768 hours on eleven sets (every season, clear to overcast, day, night and
  the low-sun threshold): the solar elevation from the printed timestamp and latitude
  against the printed elevation (three decimals, so +-0.0005 deg); k1 from that elevation
  and the printed cloud cover against the printed k1 (three significant figures); k3 at
  SIRANE's printed ground temperature and molar volume against the printed k3 in
  ppb^-1 s^-1 (three significant figures);
- per point, over 176 street and receptor values on six single-street cases (day, day
  without background, night, dawn, stable, NO emitted as NO2-equivalent mass): the
  photostationary split applied to the passive NO and NO2 (the same case run without
  chemistry) plus the background, against SIRANE's NO2, NO and O3.

Every input the split takes from a printed value carries that value's rounding, and the
check allows exactly that and nothing more: the passive NO2 and NO (+-0.005 ug/m3 each,
jointly, as a 3 x 3 grid), the ground temperature in k3 (+-0.05 K) and the molar volume
in the 2 ppb floor (+-0.005 L/mol); the prediction's range over those must meet SIRANE's
printed value within its own +-0.005 ug/m3. k1 is computed, not read. The negative control,
the prefactor 1.325e5 that Soulhac et al. (2011) print, fails on every daytime point and
every k3 row; it is recorded, not a tolerance.
"""

from __future__ import annotations

import csv
import functools
import itertools
import json
import math
from pathlib import Path

import torch

from noodl.apps.street_aq.chemistry import (
    j_no2_sirane,
    k_no_o3_sirane,
    photostationary_for_streets,
    solar_elevation,
)

DT = torch.float64
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "street" / "sirane_chemistry"
KELVIN = 273.15
HALF_UG = 0.005            # ug/m3: SIRANE prints concentrations to 0.01
HALF_T = 0.05              # K: the printed ground temperature, to 0.1
HALF_VM = 0.005            # L/mol: the printed molar volume, to 0.01
HALF_ELEVATION = 0.0005    # deg: the printed solar elevation, to 0.001


@functools.lru_cache(maxsize=1)
def _hours():
    with open(FIXTURE / "hours.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {key: torch.tensor([float(r[key]) for r in rows], dtype=DT)
               for key in rows[0] if key != "set"}
    columns["set"] = [r["set"] for r in rows]
    return columns


@functools.lru_cache(maxsize=1)
def _decks():
    return json.loads((FIXTURE / "equilibrium.json").read_text(encoding="utf-8"))["decks"]


def _half_unit(printed: torch.Tensor) -> torch.Tensor:
    """Half a unit in the third significant figure of each printed value (0 for 0)."""
    safe = torch.where(printed > 0, printed, torch.ones_like(printed))
    half = 0.5 * 10.0 ** (torch.floor(torch.log10(safe)) - 2.0)
    return torch.where(printed > 0, half, torch.zeros_like(printed))


def _elevation(h, latitude):
    return solar_elevation(latitude, h["day_of_year"], h["hour"] + h["minute"] / 60.0)


# ------------------------------------------------------------------------------ per hour

def test_the_solar_elevation_matches_to_print_precision(record_property):
    h = _hours()
    own = _elevation(h, h["latitude_deg"])
    worst = float((own - h["elevation_deg"]).abs().max())
    record_property("hours", len(h["set"]))
    record_property("max_abs_elevation_diff_deg", worst)
    # Measured 0.000497 deg: rounding to three decimals, and nothing else.
    assert len(h["set"]) == 768
    assert worst <= HALF_ELEVATION + 1e-9


def test_k1_matches_every_printed_value(record_property):
    """k1 from noodl physics' own elevation (not the rounded printed one, which near the
    1.458 deg threshold moves k1 by more than its printed precision)."""
    h = _hours()
    k1 = j_no2_sirane(_elevation(h, h["latitude_deg"]), h["cloud_octas"])
    printed = h["k1_per_s"]
    night = printed == 0
    assert torch.equal(k1[night], torch.zeros_like(k1[night]))
    miss = (k1 - printed).abs() > _half_unit(printed) * (1 + 1e-9)
    daytime = int((~night).sum())
    record_property("daytime_hours", daytime)
    record_property("k1_mismatches", int(miss.sum()))
    record_property("max_rel_k1_diff_daytime",
                    float(((k1 - printed) / printed)[~night].abs().max()))
    assert daytime > 300
    assert int(miss.sum()) == 0


def test_k3_matches_every_printed_value(record_property):
    """k3 in ppb^-1 s^-1 = k_no_o3_sirane(T_g) 1e-6 / V_m[L]; the interval spanned by the
    printed T_g and V_m's rounding must meet the printed k3's."""
    h = _hours()
    t, v = h["temperature_ground_C"] + KELVIN, h["molar_volume_L"]
    low = k_no_o3_sirane(t - HALF_T) * 1e-6 / (v + HALF_VM)
    high = k_no_o3_sirane(t + HALF_T) * 1e-6 / (v - HALF_VM)
    printed, half = h["k3_per_ppb_s"], _half_unit(h["k3_per_ppb_s"])
    inside = (low <= printed + half) & (high >= printed - half)
    record_property("k3_consistent_hours", int(inside.sum()))
    assert bool(inside.all())
    # The ground temperature is not the input temperature away from neutral conditions:
    # these hours exercise both.
    assert bool((h["temperature_ground_C"] != h["temperature_in_C"]).any())
    # Negative control: the paper's 1.325e5 prefactor matches no row.
    paper = k_no_o3_sirane(t) * 0.1 * 1e-6 / v
    paper_hits = int(((paper - printed).abs() <= half).sum())
    record_property("k3_rows_matched_with_paper_prefactor", paper_hits)
    assert paper_hits == 0


# ----------------------------------------------------------------------------- per point

def _predict(deck, hour, k3_scale=1.0):
    """SIRANE's chemistry values, via the `closure="sirane"` split, over the rounding
    envelope of the printed inputs: returns the chemistry values printed by SIRANE
    `(n_points, 3)` and the prediction `(n_envelope, n_points, 3)`, NO2, NO, O3 in ug/m3."""
    reaction = photostationary_for_streets(("no2", "no", "o3"), closure="sirane")
    keys = sorted(hour["chemistry"])
    printed = torch.tensor([hour["chemistry"][k] for k in keys], dtype=DT)
    passive = torch.tensor([hour["passive"][k] for k in keys], dtype=DT)
    background = torch.tensor(deck["background_NO2_NO_O3"], dtype=DT)
    k1 = j_no2_sirane(_elevation({key: torch.tensor(float(hour[key]), dtype=DT)
                                  for key in ("day_of_year", "hour", "minute")},
                                 deck["latitude_deg"]), hour["cloud_octas"])
    t_g = hour["temperature_ground_C"] + KELVIN
    v_m = hour["molar_volume_L"]
    predictions = []
    for d_no2, d_no, d_t, d_v in itertools.product((-HALF_UG, 0.0, HALF_UG),
                                                   (-HALF_UG, 0.0, HALF_UG),
                                                   (-HALF_T, 0.0, HALF_T),
                                                   (-HALF_VM, 0.0, HALF_VM)):
        state = torch.zeros(len(keys), 3, dtype=DT)
        state[:, 0] = passive[:, 0] + d_no2
        state[:, 1] = passive[:, 1] + d_no
        state = (state + background) * 1e-9                          # kg/m3
        drivers = {"J_NO2": k1 / k3_scale,
                   "temperature": torch.tensor(t_g + d_t, dtype=DT),
                   "molar_volume": torch.tensor((v_m + d_v) * 1e-3, dtype=DT)}
        predictions.append(reaction.apply(state, None, drivers) * 1e9)
    return printed, torch.stack(predictions)


def _central(predictions):
    return predictions[len(predictions) // 2]        # every offset zero


def test_the_split_reproduces_every_chemistry_value(record_property):
    total = inside = 0
    worst_central = 0.0
    for deck in _decks():
        for hour in deck["hours"]:
            printed, pred = _predict(deck, hour)
            low = pred.min(dim=0).values - HALF_UG
            high = pred.max(dim=0).values + HALF_UG
            ok = ((printed >= low - 1e-9) & (printed <= high + 1e-9)).all(dim=-1)
            total += int(ok.numel())
            inside += int(ok.sum())
            worst_central = max(worst_central,
                                float((_central(pred) - printed).abs().max()))
    record_property("points_reproduced", f"{inside}/{total}")
    record_property("max_abs_diff_central_ug_m3", worst_central)
    assert total == 176
    assert inside == total
    # Measured 0.0113 ug/m3 at the central inputs: the rounding of the printed inputs,
    # carried through the split.
    assert worst_central < 0.012


def test_the_paper_prefactor_fails_every_daytime_point(record_property):
    """Negative control: with the prefactor 1.325e5, K = k1/k3 is ten times larger and no
    daytime point is reproduced (at night the 2 ppb floor sets K and k3's prefactor is
    immaterial). Points holding nothing at all (upwind, no background) are 0 either way
    and are left out."""
    daytime = hits = 0
    for deck in _decks():
        for hour in deck["hours"]:
            if hour["k1_per_s"] < 1e-3:              # floor-dominated hours
                continue
            printed, pred = _predict(deck, hour, k3_scale=0.1)
            low = pred.min(dim=0).values - HALF_UG
            high = pred.max(dim=0).values + HALF_UG
            ok = ((printed >= low - 1e-9) & (printed <= high + 1e-9)).all(dim=-1)
            holds_something = printed.abs().sum(dim=-1) > 0
            daytime += int(holds_something.sum())
            hits += int((ok & holds_something).sum())
    record_property("daytime_points_reproduced_with_paper_prefactor", f"{hits}/{daytime}")
    assert daytime >= 88
    assert hits == 0


def test_the_street_file_prints_the_raw_background_as_cext():
    """SIRANE applies the equilibrium to the street's Cint and to receptors; the street
    file's Cext column is the background as given, not its equilibrium."""
    for deck in _decks():
        for hour in deck["hours"]:
            assert hour["street_Cext"] == deck["background_NO2_NO_O3"]


def test_the_floor_is_what_sets_the_night():
    """At night k1 = 0, so without the floor NO and O3 would titrate to whichever runs out;
    SIRANE's night values need K = 2 ppb."""
    deck = next(d for d in _decks() if d["case"] == "night")
    hour = deck["hours"][0]
    printed, pred = _predict(deck, hour)
    unfloored = photostationary_for_streets(("no2", "no", "o3"), closure="sirane",
                                            floor_ppb=0.0)
    keys = sorted(hour["chemistry"])
    passive = torch.tensor([hour["passive"][k] + [0.0] for k in keys], dtype=DT)
    state = (passive + torch.tensor(deck["background_NO2_NO_O3"], dtype=DT)) * 1e-9
    titrated = unfloored.apply(state, None, {
        "J_NO2": torch.tensor(0.0, dtype=DT),
        "temperature": torch.tensor(hour["temperature_ground_C"] + KELVIN, dtype=DT),
    }) * 1e9
    assert float((_central(pred) - printed).abs().max()) < 0.012
    assert float((titrated - printed).abs().max()) > 1.0
    assert math.isclose(float(titrated[:, 2].min()), 0.0, abs_tol=1e-6)
