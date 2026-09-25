# Source and licence

`street.dat` and `intersection.dat` are the same four-street/five-junction excerpt as
`tests/data/street/munich_case_excerpt/` (see that directory's own `NOTICE.md`).

The `.bin` files here are genuine excerpts (3 hours x 4 streets = 12 float32 records each,
well under the 20-record limit) of the REAL per-street MUNICH input arrays for those same
four streets (rows 0, 1, 6, 9 of `street_ARmodif.dat`, i.e. street ids 1, 3, 8, 11), sliced
from:

Kim, Youngseob (2022). *MUNICH input data for GMD paper (Le Perreux-sur-Marne)*. Zenodo.
DOI: [10.5281/zenodo.6167477](https://doi.org/10.5281/zenodo.6167477), file
`munich-data.tar.bz2`, members `meteo/WindDirection.bin`, `meteo/WindSpeed.bin`,
`meteo/PBLH.bin`, `meteo/UST.bin`, `meteo/LMO.bin`, `meteo/SurfaceTemperature.bin`,
`emission/NO2.bin`, `background/NO2.bin`.

Licence: **CC-BY 4.0** (Zenodo record 6167477).

This is the fixture that exercises `read_munich_case` and `drivers_at` against REAL binary
values on all the required fields at once, including `background_concentration` being
per-street (not the domain-wide single value the excerpt-based
`tests/data/street/munich_case_excerpt/` fixture, built before the real archive was fetched,
first assumed).

`munich.cfg`, `munich-data.cfg` and `species.dat` here are authored (not copied from CEREA),
to the same schema as the real run's inputs.
