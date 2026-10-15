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
- `sinks_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. Zones start at 1e-4 kg/kg sarin
  with clean outside air; zone one has a deposition velocity sink (1e-3 m/s on 10 m2,
  multiplier 2) and a cutoff source, zone two a deposition rate sink (5e-4/s) on a schedule
  from 00:30 and an NRCC power-law source, zone three a constant-coefficient source with a
  removal term and an NRCC peak source. The sarin record carries a decay constant, which
  ContamX does not apply. Two hours at 300 s. Verified to load and run in ContamX 3.4.1.7.
- `pressure_source_stack.prj` is written for this repository from NIST's
  `test_OneZoneWthCtmStack.prj` (it uses that sample's `.wth` and `.ctm`, which it names),
  not an official sample. It adds a pressure-driven sarin source (2e-6 kg/(s Pa^0.65)) to
  the zone. Verified to load and run in ContamX 3.4.1.7.
- `fan_curve_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. Path 5 is an exhaust fan with a
  cubic performance curve (rated at 1.2041 kg/m3, off below speed ratio 0.1, path mult 1.5)
  from zone three, which is at 303.15 K. Its schedule sets the speed ratio to 1, 0.8 from
  00:30, 0.05 (off) from 01:00 and 1.2 from 01:30. Sarin outdoors at 1e-3 kg/kg; two hours
  at 300 s. Verified to load and run in ContamX 3.4.1.7.
- `spline_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. Its four paths carry the four
  cubic-spline airflow elements (mass and volume flow against pressure, pressure against
  mass and volume flow) on a five-point curve; path 1 is declared zone -> ambient, path 3
  is crossed against its declared direction, zone two is at 303.15 K, and a 0.020068 kg/s
  supply (path 5) drives the curves past their first segment. Sarin outdoors at 1e-3 kg/kg;
  two hours at 300 s. Verified to load and run in ContamX 3.4.1.7.
- `duct_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. It adds a duct network: a
  supply from a wind-exposed ambient terminal through a filtered Darcy-Colebrook duct, a
  fan curve on a speed schedule (1, 0.7 from 00:30), a vertical rise with a dynamic loss
  and a split to a terminal in zone one (with a balance coefficient) and a filtered
  terminal in zone two (half closed from 01:00); and an exhaust from zone three through a
  backdraft damper and a constant-volume fan to an ambient terminal. Supply junctions at
  303.15 K. Sarin outdoors at 1e-3 kg/kg; two hours at 300 s. Verified to load and run in
  ContamX 3.4.1.7; an independent model built from the measured duct rules reproduces its
  first-step flows to 1.9e-6.
- `ahs_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. It adds a simple air-handling
  system: a zero-volume return plenum and a 2 m3 supply plenum, supply points to zones one
  (0.1 kg/s, halved from 00:30) and two (0.05 kg/s), a filtered return point from zone
  three (0.12 kg/s), and the implicit recirculation (outdoor-air fraction 0.3, 0.6 from
  01:00, minimum 0.02 kg/s; filtered), outdoor-air (filtered) and exhaust paths. Sarin
  outdoors at 1e-3 kg/kg; two hours at 300 s. Verified to load and run in ContamX 3.4.1.7.
- `test_GetPrjInfo.prj` is NIST's own sample from the `contamxpy` 0.0.9 sdist
  (`demo_files/`), unchanged: two simple AHS with zero-volume plenums and two duct systems.
  As written it is refused (its duct elements leak); the parity test runs a copy with the
  leakage, and zone 1's and segment 4's control references, removed.
- `dcv_three_zone.prj`, `pressure_exhaust_three_zone.prj`, `hysteresis_window_three_zone.prj`:
  written for this repository by editing NIST's `valThreeZonesWthCtm-UseApi.prj` (contamxpy
  0.0.9 sample) with the maker script reproduced in the controls work's plan (Task 5); loaded
  and run to their end in ContamX 3.4.1.7.
- `occupants_three_zone.prj` is written for this repository from NIST's
  `valThreeZonesWthCtm-UseApi.prj`, not an official sample. Cold steady weather (263.15 K)
  and three zone temperatures; zone three left out of the building volume; the sample's
  report-only control network removed. Two occupants: one moves between zones (once in the
  middle of a step), leaves the building for an hour and generates sarin on a schedule, the
  other stays in zone three. Output flags for exposure, age of air and the building air
  change rate are set. Four hours at 300 s. Verified to load and run in ContamX 3.4.1.7.
