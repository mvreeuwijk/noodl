# Installation

noodl needs **Python 3.11 or newer**. Its only hard dependencies are PyTorch, NetworkX and
NumPy; everything else is an optional extra, described below.

## From PyPI

```bash
pip install noodl
```

## From source

```bash
git clone https://github.com/mvreeuwijk/noodl.git
cd tellegen
pip install -e .
```

On a machine without a CUDA-capable GPU, install the CPU build of PyTorch first — it is a far
smaller download, and noodl runs perfectly well on CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Optional extras

noodl keeps its optional dependencies genuinely optional. Nothing below is needed to build a
network, solve it, or differentiate through it.

```bash
pip install "noodl[sparse]"        # sparse-direct linear solver
pip install "noodl[street]"        # the street application's file I/O and oracle
pip install "noodl[contam]"        # ContamX parity, Windows x86-64 only
pip install "noodl[dev]"           # everything needed to run the test suite
```

| Extra | Adds | Why you might want it |
|---|---|---|
| `sparse` | `scipy>=1.11` | The SciPy SuperLU backend. This is a **runtime** extra, not a developer convenience — see the note below. |
| `street` | `scipy>=1.11` | The [street application](applications/street.md) reads its AQ_DT NetCDF products through `scipy.io.netcdf_file`, and its IMPAQ comparison oracle needs `scipy.optimize` and `scipy.special`. |
| `contam` | `contamxpy>=0.0.9` (Windows x86-64 only) | Runs NIST's ContamX engine directly for the [building application's](applications/building.md) parity tests. The wheel bundles ContamX 3.4.1.7, so no separate CONTAM installation is needed. There is no wheel for any other platform, hence the marker. |
| `dev` | pytest, ruff, hypothesis, pytest-cov, plus the reference engines | The full test suite, including parity against SWMM (`pyswmm`), EPANET (`wntr`) and WSIMOD. |

**The `sparse` extra deserves a moment.** The default linear-solver selection,
`method="auto"`, prefers the SciPy SuperLU backend for a certified-SPD operator at a batch of
at most 32 instances. Without SciPy installed, that path silently never fires and every such
solve falls back to preconditioned conjugate gradients — correct, but measured at roughly 4.6x
slower at ensemble size 1. It remains optional rather than mandatory because the evidence for
that default is one platform at one problem size, and PCG is a complete answer on its own. If
you care about single-instance solve latency, install it.

The reference engines in `dev` are **test-only** and deliberately not runtime extras. `wntr`
in particular pulls in a measured 351 MB of mandatory dependencies, which is not a price to pay
for a package whose water application needs nothing beyond `torch` and `sparse` to run. Every
test that needs one of these engines skips itself when the import fails, so a partial install
never breaks the suite.

## Verifying the install

```python
import noodl
print(noodl.__version__)
```

To run the test suite from a source checkout:

```bash
pip install -e ".[dev]"
pytest -q
```

Slow tests — the performance budgets and scaling gates — are excluded by default and run with
`pytest -m slow`. Tests needing an external engine carry the `external` marker and skip
themselves when that engine is unavailable.

## GPU and dtype

Every network and every solver moves with PyTorch's usual mechanics:

```python
net = net.to(device="cuda", dtype=torch.float64)
```

Use `float64` for anything involving a Newton solve you care about. The default is `float32`,
which is fine for forward evaluation but leaves little headroom for tight convergence
tolerances; every parity result in these docs was measured in double precision.

## Building the documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

The API reference is generated from the package's own docstrings, so noodl must be installed in
the same environment for that section to build.
