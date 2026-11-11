"""The floating-capacitor benchmark (`benchmarks/floating_capacitors.py`) runs end to end
(`--quick`) and its two forms agree; it times nothing here."""

from benchmarks.floating_capacitors import main


def test_quick_run_steps_both_forms_to_the_same_potentials():
    results = main(["--quick"])
    assert len(results) == 4
    for r in results:
        assert r["max_dphi"] < 1e-12
        assert set(r["iterations_per_step"]) == {"relation", "flow"}
