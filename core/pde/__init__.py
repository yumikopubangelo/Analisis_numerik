"""
PDE Solvers Module

This module provides functionality for solving Partial Differential Equations (PDEs)
using numerical methods. Currently includes a solver for the biharmonic plate equation.

Submodules:
    - biharmonic_solver: Solver for biharmonic plate equation using finite difference methods

Functions:
    - solve_biharmonic_equation: Solve the biharmonic plate equation
    - plot_deflection_3d: Plot 3D visualization of plate deflection
    - plot_deflection_contour: Plot contour visualization of plate deflection
    - plot_convergence: Plot convergence information
    - compute_statistics: Compute statistics of plate deflection
"""

from .biharmonic_solver import (
    solve_biharmonic_equation,
    plot_deflection_3d,
    plot_deflection_contour,
    plot_convergence,
    compute_statistics
)

__all__ = [
    'solve_biharmonic_equation',
    'plot_deflection_3d',
    'plot_deflection_contour',
    'plot_convergence',
    'compute_statistics'
]
