# Source and licence

This directory is the South Kensington input deck for SIRANE v2.1 rev 128 (LMFA / École
Centrale de Lyon, http://air.ec-lyon.fr/SIRANE/; the model is described in Soulhac et al.
2011, *Atmospheric Environment* 45, 7379-7395), with part of its archived results, all
redistributed byte for byte (`-text` in `.gitattributes`). 46 streets, 36 junctions.
Included with the permission of the study's authors.

The network represents South Kensington, London, after:

Grylls, T., Suter, I., Sun, C., Reeuwijk, M. van (2019). *Steady-state large-eddy
simulations of convective and stable urban boundary layers*. Boundary-Layer
Meteorology, 173, 1-24.

## The deck

- `Donnees_SouthKensington.dat` -- the master file, in SIRANE's French labels.
- `RESEAU/SIRANE_FINAL.{shp,shx,dbf,prj}` -- the street network (PolyLine Shapefile).
- `RESEAU/Site_Disp.dat`, `METEO/Site_Meteo.dat` -- the dispersion and meteo site files.
- `METEO/Meteo_change_1h.dat` -- hourly wind speed, direction, temperature, precipitation,
  cloud cover.
- `EMISSIONS/` -- the emission-evolution file, the street emissions
  (`EMIS_LIN/Emis_Rues.dat`: a unit emission of O3 on street 4, nothing else), the empty
  surface-emission file, and one point source placed outside the domain with an all-zero
  series.
- `ESPECES/` -- species (NO2, NO, O3 active) and the empty source-groups file.
- `FOND/Concentration_Fond.dat` -- hourly background concentrations (all zero) for a week.

- `GRILLES/Grille_Meteo_L93.dat` -- the meteo grid (2 x 2 cells of 750 m x 600 m); its
  cell x-size sets SIRANE's downwind plume cut-off (`noodl.apps.street_aq.plume`).

The deck's output and surface-emission grids, receptor, colormap and statistics files are
not included: noodl physics does not read them.

## The archived results

`RESULT_SOUTHKENSINGTON/` holds, of the results archived with the deck, only those of one
run (hours 00 and 01 of 7 January 2014): `RUES_PAR_HEURE/Rues_2014010700.dat` and
`Rues_2014010701.dat`, `METEO/Resul_Meteo.dat`, and the 46 `RUES_PAR_RUE/Rue_<i>_<i>.dat`.
The archive also held hour-12 and hour-13 snapshots from an earlier run with other inputs;
they are left out.

That run is not necessarily the deck as shipped: `Resul_Meteo.dat` records a wind from
315 degrees, where `Meteo_change_1h.dat` gives 135, and every street's `Sigma_wH` is 0.30 m/s
-- SIRANE's default `sigma_w` floor -- where the master file sets that floor to 0. The deck's
master, meteo and street-emission files carry a later modification date than the results.
The emission field may differ too: the highest archived `Cint_O3` is on street 31
(32279 micrograms/m3), not on the emitting street 4 (4676). A comparison against these
results must therefore drive noodl physics with the results' own meteorology
(`Resul_Meteo.dat`), not the deck's. `tests/apps/street_aq/test_sirane_io.py` pins the
direction and `sigma_w` mismatches.
