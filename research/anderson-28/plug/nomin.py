import os

import noodl.solvers.anderson as A

A._ANDERSON_NOISE_MIN = float(os.environ.get("NMIN", "0"))
