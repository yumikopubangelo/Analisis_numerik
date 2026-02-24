"""
Linear-system solvers for Ax = b.
"""

from .gaussian_elimination import gaussian_elimination
from .gauss_jordan import gauss_jordan_elimination

__all__ = ["gaussian_elimination", "gauss_jordan_elimination"]
