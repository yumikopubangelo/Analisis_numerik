"""
PDE Solvers Module

This module provides functionality for solving Partial Differential Equations (PDEs)
using numerical methods. Currently includes:

1. Solver for the biharmonic plate equation (2D)
2. Solver for 1D advection-diffusion equation
3. Solver for 2D Laplace equation

Submodules:
    - biharmonic_solver: Solver for biharmonic plate equation using finite difference methods
    - advection_diffusion_1d: Solver for 1D advection-diffusion equation using finite difference methods
    - laplace_solver: Solver for 2D Laplace equation using Jacobi, Gauss-Seidel, and SOR methods

Functions:
    - solve_biharmonic_equation: Solve the biharmonic plate equation
    - plot_deflection_3d: Plot 3D visualization of plate deflection
    - plot_deflection_contour: Plot contour visualization of plate deflection
    - plot_convergence: Plot convergence information
    - compute_statistics: Compute statistics of plate deflection
    
    - solve_advection_diffusion_1d: Solve 1D advection-diffusion equation
    - plot_concentration_snapshot: Plot concentration snapshot at specific time
    - plot_concentration_evolution: Plot concentration evolution over time

    - solve_laplace_equation: Solve the 2D Laplace equation
    - solve_laplace_with_function_bc: Solve with function-based BC
    - plot_phi_contour: Plot contour visualization of phi(x,y)
    - plot_phi_wireframe: Plot wireframe 3D visualization
    - plot_phi_surface: Plot surface 3D visualization
    - plot_convergence: Plot convergence information
    - print_solution_summary: Print formatted solution summary
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

from .laplace_solver import (
    solve_laplace_equation,
    solve_laplace_with_function_bc,
    plot_phi_contour,
    plot_phi_wireframe,
    plot_phi_surface,
    plot_convergence,
    plot_convergence_history,
    print_solution_summary,
    example_heat_distribution,
    example_electrostatic,
    create_2d_grid,
    apply_dirichlet_bc_simple,
    apply_neumann_bc_simple,
    apply_mixed_bc_simple,
    compute_laplacian
)

__all__ = [
    # Biharmonic solver
    'solve_biharmonic_equation',
    'plot_deflection_3d',
    'plot_deflection_contour',
    'plot_convergence',
    'compute_statistics',
    
    # Advection-diffusion solver
    'solve_advection_diffusion_1d',
    'plot_concentration_snapshot',
    'plot_concentration_evolution',
    
    # Laplace solver
    'solve_laplace_equation',
    'solve_laplace_with_function_bc',
    'plot_phi_contour',
    'plot_phi_wireframe',
    'plot_phi_surface',
    'plot_convergence',
    'plot_convergence_history',
    'print_solution_summary',
    'example_heat_distribution',
    'example_electrostatic',
    'create_2d_grid',
    'apply_dirichlet_bc_simple',
    'apply_neumann_bc_simple',
    'apply_mixed_bc_simple',
    'compute_laplacian'
]
