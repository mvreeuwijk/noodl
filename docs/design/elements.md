# Elements and components: where the evidence lives

The audit of `src/noodl/elements`, `noodl.nodesources` and `src/noodl/components` for issue #22
S06 found no benchmark numbers, timings or implementation narratives in those files. What the
code carries is mathematics, invariants and engine-parity facts, and these stay in the source.

## Engine-parity facts kept in code

These are empirical facts about the reference engines that the elements reproduce. Each is
stated in one line next to the code that depends on it.

| Fact | Where |
| --- | --- |
| ContamX 3.4.1.7 duct, air-viscosity and Colebrook laws (differ from TN 1887's printed Colebrook form by about 0.1 %) | `elements/air_duct.py` module docstring |
| ContamX 3.4.1.7 curve fit ("CSFchk" in its log) | `elements/curve.py` module docstring |
| ContamX 3.4.1.7 fan-law behaviour (stopped fan, speed ratio) | `elements/fan.py` |
| Upstream-density power law: the error of a frozen reference density (figures in the module docstring) | `elements/upstream.py` module docstring |
| MBL medium conventions (`Air.mo`, `PartialOneWayFlowElement.mo`, `TwoWayFlowElement.mo`) | `elements/media.py` |

## How to reproduce

The ContamX figures come from running the corresponding case through ContamX 3.4.1.7 and
comparing against the element; the tests that pin them are
`tests/verification/test_contam_parity.py` and `tests/verification/test_airflow_closed_forms.py`.
