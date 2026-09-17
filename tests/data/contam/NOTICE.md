These CONTAM project, weather and contaminant files are NIST's sample files from the
`contamxpy` 0.0.9 source distribution (`demo_files/`). They are works of the United States
Government, not subject to copyright (17 U.S.C. 105), redistributed unchanged.
`doorway_damper_fan.prj` is written for this repository's reader tests and is not an official
sample.

Two of NIST's one-zone stack projects are present, and only one of them is usable:

- `test_OneZoneWthCtmStack-UseApi.prj` is the ContamX parity fixture
  (`tests/verification/test_contam_parity.py`). It is filter-free on every path
  (`f# = a# = s# = c# = 0`), stack-driven through two openings at 0.0 m and 1.5 m, and the
  reader loads it. The `-UseApi` variant is the one copied: its non-`UseApi` sibling
  `test_OneZoneWthCtmStack.prj` puts zone 1 on schedule 1, which the reader does not support,
  and the `-UseApi` variant needs neither the `.wth` nor the `.ctm` sibling because it names
  `null` for both and takes its ambient conditions from the co-simulation API instead.
- `test_OneZoneSsStack-UseApi.prj` is kept for provenance but is NOT usable as a parity
  fixture and is referenced by no test. It attaches a constant-efficiency filter (`f# = 1`)
  that removes 10% of the sarin crossing path 1, and the reader refuses it by design: a
  filter is invisible to the airflow solution but not to the species layer, so loading the
  project while silently dropping the filter would corrupt every contaminant result it
  produced. That refusal is deliberate and is expected to stay.
