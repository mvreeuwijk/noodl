# Source and licence

Reference data: compact extracts of SIRANE v2.1 rev 128 output (LMFA / École Centrale de
Lyon, http://air.ec-lyon.fr/SIRANE/; the model is described in Soulhac et al. 2011,
*Atmospheric Environment* 45, 7379-7395) for ten single-street cases. Each case is one
isolated street under imposed meteorology, built from the South Kensington deck
(`../sirane_south_kensington`), so that SIRANE's above-roof street-plume kernel can be
compared with noodl physics' directly. They are the reference for `plume.py` and are used
by `tests/verification/test_sirane.py`.

The decks derive from the South Kensington deck, included with the permission of that
study's authors (see `../sirane_south_kensington/NOTICE.md`), and the SIRANE output is
used here on the same terms.

## The cases

Every case has one street of 200 m (W = 20 m and H = 20 m unless the name says `W10` or
`H10`), both ends dead-end junctions, the wind from 270 degrees (towards +x), `H_R` = 20 m,
the dispersion site's z0 = 1 m and d = 13 m, 1 kg/s of NO2 emitted in the street, no
background, no deposition and no chemistry. The hour read is 07/01/2014 01:00.

| case | street | meteorology |
|---|---|---|
| `G1_*` | (0, -100) to (0, 100): across the wind | |
| `G2_*` | (-100, 0) to (100, 0): along the wind; the flux vents at (100, 0) | |
| `G3_*` | 45 degrees to the wind, centred on the origin | |
| `G6_*` | 20 degrees to the wind, centred on the origin | |
| `*_M1` | | neutral, u* = 0.6 m/s, h = 1000 m, no turbulence floors |
| `*_M2` | | stable, u* = 0.3 m/s, L = 100 m, h = 200 m, no turbulence floors |
| `*_M3F` | | the South Kensington hour through SIRANE's preprocessor (u* printed 0.14 m/s, L = 100 m, h = 114.5 m), with SIRANE's floors `SIGMA_V_MIN` = 0.5 and `SIGMA_W_MIN` = 0.3 m/s |

## The files

- `manifest.json`, per case: the street's end points, width `W` (`WG + WD`) and height `H`
  (the mean of `HG`, `HD`); the street's roof flux and the two junctions' vertical fluxes
  (g/s, from `Listing.txt`, NO2, the last solution block; the junction fluxes in the order
  of the street's end points); the street's `Cint` and the grid's maximum (micrograms/m3);
  the number of grid cells above 1 % of the maximum outside the street; the meteorology
  as SIRANE printed it (`RESULT/METEO/Resul_Meteo.dat`: u*, L, h, `SigmaTheta` in degrees,
  theta*, temperature in Celsius, direction); `H_R` and the floors from the master file;
  z0 and d from the dispersion site file.
- `<case>.npz`, float32 arrays extracted from SIRANE's NO2 grid
  (`RESULT/GRILLE/Conc_NO2_2014010701.nc`, 601 x 601 points 2 m apart over
  x in [-200, 1000], y in [-600, 600]) and its receptors:
  - `cell_x`, `cell_y`, `cell_c`: 1200 grid cells drawn at random (numpy
    `default_rng(0)`) from those above 1 % of the grid's maximum whose value is not the
    street's `Cint` (to 1e-3 relative);
  - `centre_x`, `centre_c`: the grid row y = 0, all 601 points;
  - `cross_x`, `cross_y`, `cross_c`: the grid columns nearest x = 150, 300 and 650 m, over
    |y| <= 200 m;
  - `receptors`: SIRANE's point receptors (`RESULT/RECEPT/Recept_2014010701.dat`) as rows
    `x, y, z, C`, without those that hold the street's `Cint`.

Concentrations are in micrograms/m3 as SIRANE writes them.
