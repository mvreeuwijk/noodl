# A new file reader

How to read a model from a file format noodl does not read yet. A reader is not a third kind of
extension on top of laws and applications: it builds the same kind of network, laws and layers
that [`build_two_zone`](application.md#the-smallest-application) builds by hand, just driven
by a parsed file.

## Where it lives

A reader lives next to the application it feeds, not in a shared location, because each
format's own section names, column layouts and units are specific to it:

| Reader | Module |
|---|---|
| SWMM `.inp` | `apps/sewer/inp.py` |
| EPANET `.inp` | `apps/water/inp.py` |
| CONTAM `.prj` | `apps/building_physics/prj.py` |
| MATPOWER cases | `apps/electrical/power/matpower.py` |

## What it returns

The reader parses the file into the network and domain objects the application needs.
`read_swmm_inp` and `read_epanet_inp` return a network the caller then passes to the
application's own `build_model`. `project_to_model` in `apps/building_physics/prj.py` takes the
extra step of assembling `(model, state, drivers)` itself, from the laws and heads `read_prj`
already built. Either shape is fine; choose the one the file's contents allow. A file that
fixes a complete model (a CONTAM project) can return a model, a file that describes only a
network (an EPANET or SWMM file) should return the network, so that the caller keeps the
options of `build_model`.

## Reuse the line parser

For an `.inp`-style format, reuse `noodl.apps.inpfile` for the shared lexical layer:
`[SECTION]` headers, `;` comments, whitespace-separated fields and blank lines. Do not write a
new line parser. `apps/sewer/inp.py` and `apps/water/inp.py` both build on it.

## Checklist

- [ ] The reader refuses what it does not support, by name, rather than ignoring it; the
  application's limitations page lists what is refused.
- [ ] The format is added to [File formats](../../reference/formats.md) with what the reader
  covers.
- [ ] A fixture file in `tests/data/` and a test that compares the result against the tool that
  wrote the format, to a stated tolerance, as in
  [A new application](application.md#what-a-real-application-must-ship).
