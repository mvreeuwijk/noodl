# ruff: noqa: E501
"""Writes the SWMM runoff fixtures beside this file (`runoff_*_dynwave.inp`).

    python tests/data/sewer/make_runoff_fixtures.py

Every fixture is a three-junction tree (J1, J2 -> J3 -> Outfall, dynamic wave, tight
HEAD_TOLERANCE like the other parity fixtures) under a different hydrology; `runoff_horton_dynwave.inp`
is written by hand. They are small on purpose: the tests run SWMM 5.2.4 through pyswmm on each.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

HEAD = """[TITLE]
;;Project Title/Notes
{title} - noodl test fixture

[OPTIONS]
;;Option             Value
FLOW_UNITS           {units}
INFILTRATION         {infiltration}
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO

START_DATE           {start_date}
START_TIME           {start_time}
REPORT_START_DATE    {start_date}
REPORT_START_TIME    {start_time}
END_DATE             {end_date}
END_TIME             {end_time}
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          {report}
WET_STEP             {wet}
DRY_STEP             {dry}
ROUTING_STEP         {routing}

INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
VARIABLE_STEP        0
LENGTHENING_STEP     0
MIN_SURFAREA         0
MAX_TRIALS           500
HEAD_TOLERANCE       1e-12
SYS_FLOW_TOL         5
LAT_FLOW_TOL         5
MINIMUM_STEP         0.5
THREADS              1
{extra_options}
"""

NETWORK_SI = """[JUNCTIONS]
;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded
;;-------------- ---------- ---------- ---------- ---------- ----------
J1               12.0       5          0          0          0
J2               11.0       5          0          0          0
J3               10.0       5          0          0          0

[OUTFALLS]
;;Name           Elevation  Type       Stage Data       Gated    Route To
;;-------------- ---------- ---------- ---------------- -------- ----------------
Outfall          9.0        FIXED      9.6              NO

[CONDUITS]
;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow
;;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------
C1               J1               J3               200        0.013      0          0          0          0
C2               J2               J3               200        0.013      0          0          0          0
C3               J3               Outfall          200        0.013      0          0          0          0

[XSECTIONS]
;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert
;;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------
C1               CIRCULAR     1.20             0          0          0          1
C2               CIRCULAR     1.00             0          0          0          1
C3               CIRCULAR     1.60             0          0          0          1
"""

# the same tree in feet
NETWORK_US = """[JUNCTIONS]
;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded
;;-------------- ---------- ---------- ---------- ---------- ----------
J1               40.0       16         0          0          0
J2               36.0       16         0          0          0
J3               33.0       16         0          0          0

[OUTFALLS]
;;Name           Elevation  Type       Stage Data       Gated    Route To
;;-------------- ---------- ---------- ---------------- -------- ----------------
Outfall          30.0       FIXED      31.6             NO

[CONDUITS]
;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow
;;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------
C1               J1               J3               660        0.013      0          0          0          0
C2               J2               J3               660        0.013      0          0          0          0
C3               J3               Outfall          660        0.013      0          0          0          0

[XSECTIONS]
;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert
;;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------
C1               CIRCULAR     4.0              0          0          0          1
C2               CIRCULAR     3.3              0          0          0          1
C3               CIRCULAR     5.3              0          0          0          1
"""

TAIL = """
[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[TAGS]

[MAP]
DIMENSIONS 0.000 0.000 1000.000 1000.000
Units      None

[COORDINATES]
;;Node           X-Coord            Y-Coord
;;-------------- ------------------ ------------------
J1               0                  200
J2               0                  100
J3               100                150
Outfall          200                150
"""


def build(path, *, title, units, network, hydrology, infiltration="HORTON",
          start=("01/01/2026", "06:00:00"), end=("01/01/2026", "06:20:00"),
          wet="00:01:00", dry="00:05:00", routing=10, report="00:01:00", extra_options=""):
    text = HEAD.format(
        title=title, units=units, infiltration=infiltration, start_date=start[0],
        start_time=start[1], end_date=end[0], end_time=end[1], wet=wet, dry=dry,
        routing=routing, report=report, extra_options=extra_options)
    text += "\n" + hydrology.strip("\n") + "\n\n" + network + TAIL
    (HERE / path).write_text(text)


# --- CURVE_NUMBER default with MODIFIED_HORTON and HORTON lines, LPS, co-gages ----------
CN_LPS = """[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
MONTHLY          2.0 2.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0
DRY_ONLY         NO

[RAINGAGES]
;;Name           Format    Interval SCF      Source
;;-------------- --------- ------ ------ ----------
RG1              INTENSITY 0:05     1.0      TIMESERIES Ri
RG2              INTENSITY 0:05     1.0      TIMESERIES Ri
RG3              VOLUME    0:03     1.0      TIMESERIES Rv

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack
;;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- ----------------
S1               RG1              J1               3.0      45       150      2.0      0
S2               RG2              J2               1.5      70       90       1.2      0
S3               RG3              S2               2.0      20       140      3.0      0
S4               RG1              S4               0.5      50       60       1.0      0

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted
;;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
S1               0.013      0.20       1.6        5.0        25         OUTLET
S2               0.012      0.15       2.0        4.0        10         IMPERVIOUS 70
S3               0.014      0.25       1.0        6.0        40         PERVIOUS   35
S4               0.015      0.10       1.0        2.0        25         OUTLET

[INFILTRATION]
;;Subcatchment   CurveNum   Conduct    DryTime
;;-------------- ---------- ---------- ----------
S1               75         0.5        6
S2               85         0.5        4
S3               60         0.5        9
;; per-subcatchment models
S2               30 10 2 5 25 MODIFIED_HORTON
S4               50 5 4 7 0 HORTON

[TIMESERIES]
;;Name           Date       Time       Value
;;-------------- ---------- ---------- ----------
Ri               03/05/2026 12:03      20.0
Ri                          12:08      45.0
Ri                          12:13      10.0
Ri                          12:18      0.0
Rv               03/05/2026 12:00      0.0
Rv                          12:03      1.5
Rv                          12:06      2.5
Rv                          12:09      0.0
Rv                          12:12      3.0
Rv                          12:15      1.0
Rv                          12:18      0.0
"""

# --- GREEN_AMPT in US units, a cumulative gage, DRY_ONLY evaporation, n = 0 --------------
GA_US = """[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
CONSTANT         0.25
DRY_ONLY         YES

[RAINGAGES]
;;Name           Format    Interval SCF      Source
;;-------------- --------- ------ ------ ----------
RG1              CUMULATIVE 0:05    1.0      TIMESERIES Rc

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack
;;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- ----------------
S1               RG1              J1               6.0      40       500      2.0      0
S2               RG1              J2               3.0      15       300      3.5      0
S3               RG1              J3               2.0      100      200      1.0      0
S4               RG1              J2               1.0      0        150      0.5      0

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted
;;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
S1               0.013      0.20       0.06       0.20       25         OUTLET
S2               0.012      0          0.05       0.15       0          PERVIOUS   50
S3               0.011      0.1        0.04       0.1        100        OUTLET
S4               0.013      0.30       0.06       0.30       25         OUTLET

[INFILTRATION]
;;Subcatchment   Suction    Ksat       IMDmax
;;-------------- ---------- ---------- ----------
S1               3.5        0.5        0.30
S2               4.0        0.25       0.40  MODIFIED_GREEN_AMPT
S3               3.0        0.4        0.2
S4               6.0        0.1        0.35

[TIMESERIES]
;;Name           Date       Time       Value
;;-------------- ---------- ---------- ----------
Rc               01/01/2026 06:00      0.00
Rc                          06:05      0.10
Rc                          06:10      0.35
Rc                          06:15      0.60
Rc                          06:20      0.65
Rc                          06:25      0.65
Rc                          06:30      0.05
Rc                          06:35      0.30
"""

# --- month boundary, adjustments, monthly recovery, a gage faster than WET_STEP ----------
CAL_SI = """[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
MONTHLY          6.0 12.0 5 5 5 5 5 5 5 5 5 5
RECOVERY         Rec
DRY_ONLY         NO

[PATTERNS]
;;Name           Type       Multipliers
Rec              MONTHLY    0.5 2.0 1 1 1 1 1 1 1 1 1 1
Nv               MONTHLY    1.0 1.6 1 1 1 1 1 1 1 1 1 1
Dv               MONTHLY    1.0 0.5 1 1 1 1 1 1 1 1 1 1
Iv               MONTHLY    1.0 0.7 1 1 1 1 1 1 1 1 1 1

[RAINGAGES]
;;Name           Format    Interval SCF      Source
;;-------------- --------- ------ ------ ----------
RG1              VOLUME    0:01:30  1.0      TIMESERIES Rf

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack
;;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- ----------------
S1               RG1              J1               2.0      35       120      1.5      0
S2               RG1              J2               1.0      55       100      2.5      0

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted
;;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
S1               0.013      0.20       1.5        4.0        25         OUTLET
S2               0.013      0.15       1.5        3.0        25         OUTLET

[INFILTRATION]
;;Subcatchment   MaxRate    MinRate    Decay      DryTime    MaxInfil
;;-------------- ---------- ---------- ---------- ---------- ----------
S1               40         6          4          7          0
S2               30         4          3          5          25

;; SWMM reads [ADJUSTMENTS] in file order: the N-PERV, DSTORE and INFIL lines count only
;; when they come after the subcatchment's line in [SUBCATCHMENTS]
[ADJUSTMENTS]
;;Parameter      Subcatchment  Pattern / Monthly values
EVAPORATION      1.0 -1.0 0 0 0 0 0 0 0 0 0 0
RAINFALL         1.0 1.5 1 1 1 1 1 1 1 1 1 1
CONDUCTIVITY     1.0 0.5 1 1 1 1 1 1 1 1 1 1
N-PERV           S1       Nv
DSTORE           S1       Dv
INFIL            S2       Iv

[TIMESERIES]
;;Name           Date       Time       Value
;;-------------- ---------- ---------- ----------
Rf               01/31/2026 23:40:00    0.8
Rf                          23:41:30    1.2
Rf                          23:43:00    0.0
Rf                          23:44:30    0.0
Rf                          23:46:00    1.5
Rf                          23:47:30    2.0
Rf                          23:49:00    1.0
Rf                          23:50:30    2.5
Rf                          23:52:00    3.0
Rf                          23:53:30    0.0
Rf                          23:55:00    0.0
Rf                          23:56:30    1.5
Rf                          23:58:00    2.5
Rf                          23:59:30    1.0
Rf                          02/01/2026 00:01:00   2.0
Rf                          00:02:30   0.5
Rf                          00:04:00   0.0
"""

# --- time-series evaporation (a step function that starts before the run), GA ----------
EVS_SI = """[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
TIMESERIES       Ev
DRY_ONLY         NO

[RAINGAGES]
;;Name           Format    Interval SCF      Source
;;-------------- --------- ------ ------ ----------
RG1              VOLUME    0:04     1.0      TIMESERIES Rg

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack
;;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- ----------------
S1               RG1              J1               2.0      10       120      1.5      0
S2               RG1              J2               1.0      20       100      2.5      0

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted
;;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
S1               0.013      0.20       1.5        4.0        25         OUTLET
S2               0.013      0.15       1.5        3.0        25         OUTLET

[INFILTRATION]
;;Subcatchment   Suction    Ksat       IMDmax
;;-------------- ---------- ---------- ----------
S1               90         8          0.30
S2               60         4          0.40

[TIMESERIES]
;;Name           Date       Time       Value
;;-------------- ---------- ---------- ----------
Ev               01/01/2026 00:00      30.0
Ev                          06:05      200.0
Ev                          06:10      100.0
Ev                          06:20      400.0
Rg               01/01/2026 06:04      4.0
Rg                          06:08      6.0
Rg                          06:12      0.0
Rg                          06:16      2.0
"""


def main() -> None:
    build("runoff_cn_mixed_lps_dynwave.inp",
          title="Curve Number, modified Horton and Horton subcatchments, LPS, co-gages",
          units="LPS", network=NETWORK_SI, hydrology=CN_LPS, infiltration="CURVE_NUMBER",
          start=("03/05/2026", "12:00:00"), end=("03/05/2026", "12:20:00"),
          wet="00:00:45", dry="00:03:00")
    build("runoff_greenampt_us_dynwave.inp",
          title="Green-Ampt subcatchments, CFS, cumulative gage, dry-only evaporation",
          units="CFS", network=NETWORK_US, hydrology=GA_US, infiltration="GREEN_AMPT",
          end=("01/01/2026", "06:40:00"), wet="00:02:00", dry="00:10:00", routing=15)
    build("runoff_calendar_dynwave.inp",
          title="Month boundary, adjustments, soil recovery pattern, fast gage",
          units="CMS", network=NETWORK_SI, hydrology=CAL_SI,
          start=("01/31/2026", "23:40:00"), end=("02/01/2026", "00:08:00"),
          wet="00:05:00", dry="00:10:00", routing=120, report="00:02:00")
    build("runoff_evap_series_dynwave.inp",
          title="Time-series evaporation (mm/day), Green-Ampt, MODIFIED_GREEN_AMPT default",
          units="CMS", network=NETWORK_SI, hydrology=EVS_SI,
          infiltration="MODIFIED_GREEN_AMPT", wet="00:01:00", dry="00:05:00")


if __name__ == "__main__":
    main()
