"""The MNA stepping benchmark runs end to end (`--quick`); it times nothing here."""

from benchmarks.mna_stepping import main


def test_quick_run_counts_the_parts_of_each_newton_iteration():
    results = main(["--quick"])
    assert [r["case"] for r in results] == ["RCModel", "HeatingRectifier"]
    for r in results:
        calls = r["calls_per_step"]
        assert calls["jacobian"] >= 1 and calls["lu"] >= calls["jacobian"]
        assert calls["residual"] >= calls["jacobian"]
