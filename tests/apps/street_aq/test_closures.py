"""The closure option names, the two presets and the deprecated spellings."""

from __future__ import annotations

import pytest

from noodl.apps.street_aq import closures
from noodl.apps.street_aq.canyon import KAPPA, KAPPA_MUNICH
from noodl.apps.street_aq.closures import (
    LEGACY_NAMES,
    LEGACY_VALUES,
    OPTIONS,
    PRESETS,
    normalise,
    preset_options,
    resolve,
)


def test_both_presets_set_every_option_to_an_allowed_value():
    for name, preset in PRESETS.items():
        for option, allowed in OPTIONS.items():
            assert preset[option] in allowed, (name, option)
        for key in ("kappa", "canyon_wind_min", "u_d_min", "sigma_w_min", "sigma_v_min"):
            assert isinstance(preset[key], float), (name, key)
        assert set(preset["chemistry"]) == {"no_o3_rate", "floor_ppb"}


def test_the_presets_are_the_two_reference_closure_sets():
    sirane, munich = PRESETS["sirane"], PRESETS["munich"]
    assert sirane["kappa"] == KAPPA and munich["kappa"] == KAPPA_MUNICH
    assert (sirane["sigma_w_min"], sirane["sigma_v_min"]) == (0.30, 0.5)
    assert (munich["sigma_w_min"], munich["sigma_v_min"]) == (0.0, 0.0)
    assert (munich["canyon_wind_min"], munich["u_d_min"]) == (0.1, 0.001)
    assert sirane["chemistry"] == {"no_o3_rate": "soulhac_2011", "floor_ppb": 2.0}
    assert munich["chemistry"] == {"no_o3_rate": "jpl_2003", "floor_ppb": 0.0}
    assert munich["direction_averaging"] == "rectangle_rule"
    assert munich["direction_spread"] == "turbulence_intensity"


def test_preset_options_is_a_copy_and_refuses_an_unknown_preset():
    options = preset_options("sirane")
    options["chemistry"]["floor_ppb"] = 99.0
    options["kappa"] = 1.0
    assert PRESETS["sirane"]["chemistry"]["floor_ppb"] == 2.0
    assert PRESETS["sirane"]["kappa"] == 0.40
    with pytest.raises(ValueError, match=r"preset must be one of \('sirane', 'munich'\), "
                                         r"got 'impaq'"):
        preset_options("impaq")


def test_every_legacy_value_maps_to_an_allowed_value_and_warns():
    for option, mapping in LEGACY_VALUES.items():
        for old, new in mapping.items():
            assert new in OPTIONS[option]
            with pytest.warns(DeprecationWarning,
                              match=rf"where: {option}='{old}' is deprecated; use "
                                    rf"{option}='{new}'"):
                assert normalise(option, old, "where") == new


def test_a_current_value_passes_silently_and_an_unknown_one_is_named():
    assert normalise("canyon_wind", "bessel_profile", "f") == "bessel_profile"
    with pytest.raises(ValueError, match=r"f: stability must be one of \('neutral', "
                                         r"'monin_obukhov'\), got 'impaq'"):
        normalise("stability", "impaq", "f")


def test_every_legacy_keyword_maps_and_warns():
    for old, new in LEGACY_NAMES.items():
        value = OPTIONS[new][1]
        with pytest.warns(DeprecationWarning, match=rf"the keyword '{old}' is deprecated; "
                                                    rf"use '{new}'"):
            options = resolve("sirane", {old: value}, "build_model")
        assert options[new] == value


def test_a_legacy_keyword_together_with_its_current_name_is_refused():
    with pytest.raises(TypeError, match=r"'junction_routing' given twice"), \
            pytest.warns(DeprecationWarning):
        resolve("sirane", {"junction_routing": "perfect_mixing", "routing": "mixing"}, "f")
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'wind'"):
        resolve("sirane", {"wind": "x"}, "f")


def test_resolve_takes_the_preset_and_lets_explicit_values_win():
    assert resolve("sirane", {}, "f") == {**preset_options("sirane")}
    assert resolve("munich", {"canyon_wind_min": None}, "f")["canyon_wind_min"] == 0.1
    options = resolve("munich", {"stability": "neutral", "sigma_w_min": 0.2}, "f")
    assert options["stability"] == "neutral" and options["sigma_w_min"] == 0.2
    assert options["kappa"] == 0.41


def test_the_old_munich_averaging_implies_the_turbulence_intensity_spread():
    with pytest.warns(DeprecationWarning):
        options = resolve("sirane", {"direction_averaging": "munich"}, "f")
    assert options["direction_averaging"] == "rectangle_rule"
    assert options["direction_spread"] == "turbulence_intensity"
    with pytest.warns(DeprecationWarning):
        options = resolve("sirane", {"direction_averaging": "munich",
                                     "direction_spread": "driver"}, "f")
    assert options["direction_spread"] == "driver"


def test_kappa_follows_the_preset_or_the_explicit_mix():
    assert resolve("sirane", {}, "f")["kappa"] == 0.40
    assert resolve("munich", {}, "f")["kappa"] == 0.41
    assert resolve("sirane", {"canyon_wind": "exponential_profile"}, "f")["kappa"] == 0.41
    assert resolve("sirane", {"roof_exchange": "aspect_ratio_scaled"}, "f")["kappa"] == 0.41
    assert resolve("sirane", {"roof_wind": "canopy_log_law"}, "f")["kappa"] == 0.41
    mixed = {"canyon_wind": "bessel_profile", "roof_exchange": "turbulent_velocity"}
    assert resolve("munich", mixed, "f")["kappa"] == 0.40
    assert resolve("munich", dict(mixed, kappa=0.38), "f")["kappa"] == 0.38
    assert resolve("sirane", {"sigma_w_min": 0.0}, "f")["kappa"] == 0.40


def test_a_deprecation_warning_names_the_calling_line():
    with pytest.warns(DeprecationWarning) as record:
        normalise("stability", "munich", "f")
    assert record[0].filename == __file__
    assert closures.__name__ == "noodl.apps.street_aq.closures"
