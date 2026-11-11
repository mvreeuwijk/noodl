# Notes for coding agents

Follow [CONTRIBUTING.md](CONTRIBUTING.md): code, tests and documentation change together, and
`ruff check .` runs before every commit.

## Running the tests

The full suite takes about 45 minutes. The applications (`building_physics`, `sewer`,
`street_aq`, `water`, `electrical`) are large and independent of each other. While developing, run the core
tests plus the application you are changing. Run the full suite, with coverage, once, just
before opening a pull request.

Core (always include):

```bash
tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py
```

Add the application you are changing. CI's `branch` job uses the same table
(`.github/scripts/select_tests.py`): keep the two in step.

| Application | Paths |
| --- | --- |
| building | `tests/apps/building_physics tests/verification/test_contam_parity.py tests/verification/test_modelica_parity.py tests/verification/test_airflow_closed_forms.py tests/verification/test_natural_ventilation.py` |
| sewer | `tests/apps/sewer tests/apps/inp tests/verification/test_sewer_*.py tests/benchmarks/test_swmm_corpus.py` |
| water | `tests/apps/water tests/apps/inp tests/verification/test_water_parity.py tests/verification/test_water_networks.py tests/verification/test_water_quality*.py` |
| wsimod | `tests/apps/wsimod tests/verification/test_allocation_wsimod_*.py tests/verification/test_wsimod_*.py` |
| street air quality | `tests/apps/street_aq tests/verification/test_munich.py tests/verification/test_sirane*.py` |
| electrical | `tests/apps/electrical tests/layers/test_mna*.py tests/test_model_mna.py tests/operators/test_mna_operator.py tests/verification/test_power_flow_parity.py tests/verification/test_modelica_electrical_parity.py` |
| coupling | `tests/verification/test_coupling_*.py tests/verification/test_composed_*.py` |

The wsimod row runs in about 3 minutes with `-n 4` (178 s, 651 tests, measured), well inside
the 10-minute budget, so nothing is cut: the default composition is quickstart in full (1456 days,
both modes, 38-41 s each), oxford for its first 365 days (both modes, 36 s each), the sampled
replays and the cases (the rest, under 2 minutes). Full oxford (1456 days, both modes, about
145 s each) runs under `-m slow`.

For example, for the building application:

```bash
python -m pytest -q -n 4 --no-cov tests/test_*.py tests/layers tests/solvers tests/operators \
  tests/elements tests/components tests/apps/test_*.py tests/apps/building_physics \
  tests/verification/test_contam_parity.py tests/verification/test_modelica_parity.py \
  tests/verification/test_airflow_closed_forms.py tests/verification/test_natural_ventilation.py
```

- Pass `--no-cov` on a subset: the coverage threshold (90 %) applies to the whole suite only.
- Tests marked `slow` (full reference-model runs) are deselected by default; select them with
  `-m slow` when you change what they compare.
- Before the pull request: `ruff check .`, the full `pytest`, and `python -m mkdocs build --strict`.
