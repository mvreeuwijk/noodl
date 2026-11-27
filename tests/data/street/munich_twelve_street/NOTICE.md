# Source

Three MUNICH runs of the 12-street network (`twelve_street_grid(L=100, W=20, H=20)`), one
directory per `[street]` option set: `default`, `alternate`, `sirane_closures` (the sets are
`VARIANTS` in `tests/verification/test_munich_run.py`).

## The inputs

Every file except `results/` is written by `write_case(format="munich")` from the case
`_case()` in `tests/verification/test_munich_run.py` builds, and the test checks they still
are, byte for byte. Eight hours from 2014-03-16 00:00, one passive species (`NO2`, with
chemistry, photolysis, deposition and scavenging off), `With_stationary_hypothesis: yes`.

## The output

`results/NO2.bin` is MUNICH's own output (float32, row-major `(8 hours, 12 streets)`,
micrograms/m3, saved with `Averaged: no`), produced by MUNICH v2.2 (CEREA, open source,
<https://github.com/cerea-lab/munich>; the model is described in Kim et al. 2022, *Geosci.
Model Dev.* 15, 7371-7396), built from commit `3d0f23b` ("Release of Version 2.2") under
Ubuntu 22.04, and run in each directory as

    munich munich.cfg

MUNICH's `munich.log` is not kept. To reproduce: regenerate a directory with `write_case` (as
the test does), run MUNICH there, and compare `results/NO2.bin`.
