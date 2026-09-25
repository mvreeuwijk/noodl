# Source and licence

`street.dat` and `intersection.dat` in this directory are excerpts (four streets, five
junctions -- one hub junction `1` and its four immediate neighbours, a real connected
subgraph) of the Le Perreux-sur-Marne (TRAFIPOLLU) street network shipped as the MUNICH
v2.0.1 test-case archive on Zenodo:

Kim, Youngseob (2022). *MUNICH test case for GMD paper*. Zenodo.
DOI: [10.5281/zenodo.6167477](https://doi.org/10.5281/zenodo.6167477),
file `munich-testcase-0.1.tar.bz2`, members `street_ARmodif.dat` and `graph/intersection.dat`.

Licence: **CC-BY 4.0** (Zenodo record 6167477).

`munich.cfg`, `munich-data.cfg` and `species.dat` in this directory are NOT copied from
CEREA's distribution -- they are authored here, to the input-file schema discovered by
reading the MUNICH v2.2 source (`cerea-lab/munich`) and the shipped `processing/photochemistry` example, so that this
excerpt is a small, self-contained, runnable-shaped case for the reader's unit tests. They
use the `is_num` constant shortcut MUNICH's own `InputFiles::Read` supports (a `Filename`
that parses as a number is broadcast as a constant instead of naming a binary file), so no
binary fixture is needed here.
