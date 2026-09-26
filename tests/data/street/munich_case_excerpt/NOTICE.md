# Source and licence

`street.dat` and `intersection.dat` in this directory are excerpts (four streets, five
junctions -- one hub junction `1` and its four immediate neighbours, a real connected
subgraph) of the Le Perreux-sur-Marne (TRAFIPOLLU) street network shipped as the MUNICH
v2.0.1 test-case archive on Zenodo:

Kim, Youngseob (2022). *MUNICH test case for GMD paper*. Zenodo.
DOI: [10.5281/zenodo.6167477](https://doi.org/10.5281/zenodo.6167477),
file `munich-testcase-0.1.tar.bz2`, members `street_ARmodif.dat` and `graph/intersection.dat`.

Licence: **CC-BY 4.0** (Zenodo record 6167477).

`munich.cfg`, `munich-data.cfg`, `species.dat` and `wind_direction.bin` in this directory are
NOT copied from CEREA's distribution -- they are authored here, to the input-file schema of
the MUNICH v2.2 source (github.com/cerea-lab/munich: `InputFiles::Read` and
`StreetNetworkTransport::ReadConfiguration`) and its shipped `processing/photochemistry`
example, so that this excerpt is a small, self-contained case for the reader's unit tests.
It mixes the two ways MUNICH's `InputFiles::Read` takes a field: `WindDirection` names a
float32 binary (`wind_direction.bin`: 2 hours x 4 streets, toward north in the first hour
and toward east in the second), and every other field uses the `is_num` constant shortcut (a
`Filename` that parses as a number is broadcast as a constant instead of naming a file).
