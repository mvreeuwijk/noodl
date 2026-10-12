These CONTAM project, weather and contaminant files are NIST's sample files from the
`contamxpy` 0.0.9 source distribution (`demo_files/`). They are works of the United States
Government, not subject to copyright (17 U.S.C. 105), redistributed unchanged.
`doorway_damper_fan.prj` is written for this repository's reader tests and is not an official
sample.

Two of NIST's one-zone stack projects are present:

- `test_OneZoneWthCtmStack-UseApi.prj` is the ContamX parity fixture
  (`tests/verification/test_contam_parity.py`). It is filter-free on every path
  (`f# = a# = s# = c# = 0`), stack-driven through two openings at 0.0 m and 1.5 m, and the
  reader loads it. The `-UseApi` variant is the one copied: its non-`UseApi` sibling
  `test_OneZoneWthCtmStack.prj` puts zone 1 on schedule 1, which the reader does not support,
  and the `-UseApi` variant needs neither the `.wth` nor the `.ctm` sibling because it names
  `null` for both and takes its ambient conditions from the co-simulation API instead.
- `test_OneZoneSsStack-UseApi.prj` attaches a constant-efficiency filter (`f# = 1`) that
  removes 10% of the sarin crossing path 1, with steady airflow and steady contaminants and
  a 1e-4 kg/s sarin source. It is the parity fixture for a filter on a path
  (`tests/verification/test_contam_parity.py`).
- `test_OneZoneWthCtm.{prj,wth,ctm}`: NIST's one-zone project run with its own weather and
  ambient-concentration files (two 1.5 m openings, wind from the `.wth`), unchanged. The
  weather/ambient-file parity fixture (`test_contam_parity.py`).
- `test_OneZoneWthCtmStack.{prj,wth,ctm}`: the non-`UseApi` sibling of the stack fixture, unchanged:
  zone 1 on temperature schedule 1, `densZP = 1`, `dodMdt = 1`, with its own weather and
  ambient-concentration files. The scheduled-zone-temperature parity fixture. Note that its second
  header line names `test_OneZoneWthCtmStack-UseApi.prj`; ContamX ignores that line.
- `scheduled_three_zone.{prj,wth,ctm}` is written for this repository from NIST's
  `valThreeZonesWthCtm.prj`, not an official sample. It adds five day schedules and four week
  schedules: a constant-mass-flow exhaust (path 5) on day type 4 08:00-18:00, a 2e-6 kg/s sarin
  source in zone 1 09:00-12:00 (half from 11:00), zone 2 on a trapezoidal temperature
  schedule, and path 1 half open 10:00-14:00. Its `.wth` has diurnal Ta/Pb/Ws/Wd and its `.ctm`
  an ambient sarin pulse; `dodMdt = 1`. Verified to load and run in ContamX 3.4.1.7.
- `reaction_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. It adds a second species
  (`product`) and two kinetic reactions: zone one reacts sarin to product (1e-3/s) with the
  product decaying (5e-4/s), zone three decays sarin (2e-4/s), half of it to product. All
  zones start at 1e-3 kg/kg sarin; the run is two hours at 300 s. Verified to load and run
  in ContamX 3.4.1.7.
- `filters_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. It has two species, sarin (a
  gas, ambient 1e-3 kg/kg) and `pm` (2 um particles, ambient 2e-3 kg/kg), and three filters:
  a constant-efficiency filter (sarin 0.3) on path 1, a particle filter (a six-point curve)
  on path 3, which the air crosses against the path's declared direction, and both in
  series on path 2. Two hours at 300 s. Verified to load and run in ContamX 3.4.1.7.
