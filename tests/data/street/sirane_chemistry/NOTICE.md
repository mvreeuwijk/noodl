# Source and licence

SIRANE reference data: compact extracts of SIRANE v2.1 rev 128 output (LMFA / École
Centrale de Lyon, http://air.ec-lyon.fr/SIRANE/; the model is described in Soulhac et al.
2011, *Atmospheric Environment* 45, 7379-7395) with its NO-NO2-O3 chemistry switched on.
They are the reference for the street application's `preset="sirane"` chemistry
(`solar_elevation`, `j_no2_elevation_cloud`, `k_no_o3_soulhac2011` and the photostationary
split) and are used by `tests/verification/test_sirane_chemistry.py`.

The cases are single-street variants of the kernel cases in `../sirane_kernel_probe`,
which derive from the South Kensington deck (`../sirane_south_kensington`), included with
the permission of that study's authors (see `../sirane_south_kensington/NOTICE.md`). The
SIRANE output is used here on the same terms.

## `hours.csv`: one row per hour, 768 hours in eleven sets

The site is at latitude 51.49415 deg; every set is 2014. Each set cycles the input
temperature from -10 to 35 C and the cloud cover through 0 to 8 octas (whole or fractional)
across the hours, so that day, night, the low-sun threshold, clear and overcast skies all
occur.

| set | first hour | hours | meteorology |
|---|---|---|---|
| `feb`, `mar`, `jun`, `aug`, `sep`, `dec` | 10 Feb, 19 Mar, 20 Jun, 5 Aug, 21 Sep, 20 Dec | 72 each | imposed neutral, u* = 0.4 m/s |
| `mar_fractional_cloud` | 22 Mar | 96 | as above, fractional cloud cover |
| `jun_cloud_cycle`, `mar_cloud_cycle` | 20 Jun, 22 Mar | 48, 96 | as above, an eleven-value cloud cycle |
| `jun_preprocessed`, `jun_preprocessed_cloud_cycle` | 20 Jun | 48 each | through SIRANE's meteorological preprocessor (albedo 0.2), so the ground temperature differs from the input |

Columns, as SIRANE printed them in its listing and in `RESULT/METEO/Resul_Meteo.dat`: the
timestamp (`year`, `month`, `day`, `hour`, `minute`) and `day_of_year`; `latitude_deg`;
`cloud_octas` (the cloud cover SIRANE used); `temperature_in_C` (input) and
`temperature_ground_C` (SIRANE's ground-level air temperature, to 0.1 C);
`pressure_ground_Pa`; `molar_volume_L` (L/mol, to 0.01); `elevation_deg` (the solar
elevation, to 0.001 deg); `k1_per_s` (NO2 photolysis, 1/s) and `k3_per_ppb_s` (NO + O3,
ppb^-1 s^-1), both to three significant figures.

## `equilibrium.json`: six single-street cases, 176 points

One street of 200 m across a wind from 270 degrees (W = H = 20 m), emitting 0.02 g/s of
NO2 and 0.06 g/s of NO, wind 3 m/s, boundary-layer height 800 m, and ten receptors
downwind, upwind, beside and above it. Each case exists twice: with chemistry, and
without chemistry and without background (the passive NO and NO2 from the emissions
alone).

| case | hours (2014) | conditions | background NO2, NO, O3 (ug/m3) |
|---|---|---|---|
| `day` | 20 Jun 12:00-13:00 | neutral, u* = 0.4 m/s, 2 octas, 20 C | 40, 15, 50 |
| `day_no_background` | as `day` | as `day` | none |
| `night` | 20 Jun 00:00-01:00 | as `day` | 40, 15, 50 |
| `dawn` | 20 Mar 04:00-09:00 | neutral, 8 octas, 5 C | 40, 15, 50 |
| `stable` | 20 Mar 10:00-11:00 | 1/L = 0.02 1/m, u* = 0.3 m/s, 5 C, zero solar radiation (so 8 octas) | 25, 5, 70 |
| `no_as_no2` | as `day` | as `day`, NO emissions read as NO2-equivalent mass | 40, 15, 50 |

Per case: `latitude_deg`, `background_NO2_NO_O3`, and per hour the timestamp, the
meteorology as in `hours.csv`, SIRANE's printed `k1_per_s` and `k3_per_ppb_s`, the street
file's `Cext` (`street_Cext`, NO2, NO, O3), and per point (`street`, the street's `Cint`,
and each receptor by its id, which gives its x, y, z in metres) the passive `[NO2, NO]`
and the chemistry `[NO2, NO, O3]`, in ug/m3 to 0.01 as SIRANE writes them.
