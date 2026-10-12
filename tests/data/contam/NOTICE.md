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
