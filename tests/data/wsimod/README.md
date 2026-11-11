# WSIMOD demo configurations

`quickstart/` and `oxford/` are WSIMOD 0.8.1's two demo models in WSIMOD's own save format
(`Model.save(dir, "config.yml", compress=True)`): `config.yml`, one `<node>-inputs.csv.gz`
per node with forcing, and `config_neutral.yml`, the same configuration plus an `overrides`
section that makes the water quality neutral (WTW process parameters and liquor
multipliers, surface `pollutant_load` zero, soil-temperature weights `(1, 0, 0)`).

## Forcing and licence

The forcing is a subset of WSIMOD's `timeseries_data.csv` (the WSIMOD demo
data, `docs/demo/data/processed/` in the WSIMOD repository), covering only the
sites the two demos use. WSIMOD is distributed under the BSD-3-Clause licence,
Copyright (c) 2022, barneydobson; the data is redistributed here under that
licence for the parity tests. The files are not edited beyond WSIMOD's own save format.

The forcing totals about 0.5 MB gzipped (the YAML files add about 170 kB uncompressed).

## Two corrections to what `Model.save` writes

- `PerviousSurface` depth: `save` writes the simulated depth (`depth * porosity`) under the
  key `depth` (`model.py:419-420`), so a reload shrinks it by the porosity again
  (0.5 -> 0.2 written -> 0.08 reloaded). The script rewrites it to the physical depth.
- Arc order: a node's `in_arcs`/`out_arcs` fill in arc construction order, which fixes the
  float summation order of every push and pull. `create_oxford_model` constructs its arcs
  in a different order from the list it passes on, which `save` writes, so the script
  re-sorts the saved arcs to reproduce each node's arc lists.

The script then reloads `config.yml` and checks that its run equals the original model's
(arc flows and qualities, tanks, surfaces: bitwise), and exits non-zero otherwise.

## Regenerating

```bash
PYTHONPATH=src python scripts/regenerate_wsimod_fixtures.py
```

This downloads the demo forcing and rewrites these directories (and the allocation
fixtures in `tests/data/allocation/`). Tests never download anything.
