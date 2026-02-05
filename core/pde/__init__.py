"""
PDE Solvers Module

This module provides functionality for solving Partial Differential Equations (PDEs)
using numerical methods. Currently includes:

1. Solver for the biharmonic plate equation (2D)
2. Solver for 1D advection-diffusion equation

Submodules:
    - biharmonic_solver: Solver for biharmonic plate equation using finite difference methods
    - advection_diffusion_1d: Solver for 1D advection-diffusion equation using finite difference methods

Functions:
    - solve_biharmonic_equation: Solve the biharmonic plate equation
    - plot_deflection_3d: Plot 3D visualization of plate deflection
    - plot_deflection_contour: Plot contour visualization of plate deflection
    - plot_convergence: Plot convergence information
    - compute_statistics: Compute statistics of plate deflection
    
    - solve_advection_diffusion_1d: Solve 1D advection-diffusion equation
    - plot_concentration_snapshot: Plot concentration snapshot at specific time
    - plot_concentration_evolution: Plot concentration evolution over time
"""

from .biharmonic_solver import (
    solve_biharmonic_equation,
    plot_deflection_3d,
    plot_deflection_contour,
    plot_convergence,
    compute_statistics
)

from .advection_diffusion_1d import (
    solve_advection_diffusion_1d,
    plot_concentration_snapshot,
    plot_concentration_evolution
)

__all__ = [
    'solve_biharmonic_equation',
    'plot_deflection_3d',
    'plot_deflection_contour',
    'plot_convergence',
    'compute_statistics',
    
    'solve_advection_diffusion_1d',
    'plot_concentration_snapshot',
    'plot_concentration_evolution'
]
