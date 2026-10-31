"""MONSTER AI application core.

Importing this package pins numeric libraries to a single thread. On a 1 GB / 1 vCPU server extra BLAS or
OpenMP threads only add memory pressure and context switching, so the caps are applied before numpy,
pandas or LightGBM are imported anywhere in the process.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

__version__ = "1.0.0"
