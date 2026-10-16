# Building and electrical applications: measurements behind design choices

## Electrical

The audit of `src/noodl/apps/electrical` (excluding `modelica/` and `thermal.py`) for issue #22
S06 found no benchmark numbers, timings or implementation narratives in the sources. The code
keeps the mathematics (MATPOWER case format and per-unit conventions, the DC power-flow
linearisation, waveform phasor conventions) and the engine-parity facts it reproduces; there is
nothing to move here.
