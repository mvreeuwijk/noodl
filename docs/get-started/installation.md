# Installation

noodl physics needs **Python 3.11 or newer**. Its only hard dependencies are PyTorch, NetworkX and
NumPy.

## Install

```bash
pip install noodl-physics
```

From a source checkout:

```bash
git clone https://github.com/mvreeuwijk/noodl.git
cd noodl
pip install -e .
```

Without a CUDA-capable GPU, install the much smaller CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Optional extras

Nothing below is needed to build, solve or differentiate a network. Install an extra with
`pip install "noodl-physics[name]"`.

| Extra | Adds | Use it for |
|---|---|---|
| `sparse` | `scipy` | The SuperLU sparse-direct solver. Without it, symmetric positive definite solves fall back to conjugate gradients (correct, but about 4.6 times slower at a batch of one) and circuits with flow unknowns use batched dense LU. Install it if single-instance speed matters. |
| `street_aq` | `scipy` | Writing the [street air quality](../applications/street/index.md) NetCDF product. |
| `contam` | `contamxpy` | Running NIST's ContamX directly, for comparisons with the [building application](../applications/building/index.md). Windows x86-64 only. |
| `test` | pytest, hypothesis, ruff, `scipy` | The core test suite. |
| `reference` | `pyswmm`, `wntr`, `wsimod` | The reference engines the comparisons run against. Large (`wntr` alone adds about 350 MB) and test-only. |
| `dev` | `test` and `reference` | Everything the full test suite needs. |

## Check the install

```bash
python -c "import noodl; print(noodl.__version__)"
```

To run the tests and build the documentation, see [Testing](../project/testing.md).

## GPU and dtype

Move a network and its solvers with the usual PyTorch calls:

```py
net = net.to(device="cuda", dtype=torch.float64)
```

Use `float64` for any Newton solve you care about. The default, `float32`, is fine for forward
evaluation but leaves little headroom for tight convergence tolerances. Every comparison with a
reference tool in these docs was measured in double precision.

Next: [Your first model](first-model.md).
