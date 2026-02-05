"""
Biharmonic Plate Equation Solver using Finite Difference Methods

This module provides functionality for solving the biharmonic plate equation
using finite difference methods with Jacobi and Gauss-Seidel iterative solvers.

The biharmonic equation for plate deflection is:
∇⁴w = q(x, y) / D

where:
- w: Plate deflection
- q: Transverse load
- D: Flexural rigidity

Boundary conditions:
- Simply supported: w = 0 and ∇²w = 0
- Fixed: w = 0 and ∂w/∂n = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, Optional


def create_2d_grid(x_min: float, x_max: float, y_min: float, y_max: float,
                   nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D grid for finite difference discretization.

    Parameters:
        x_min, x_max: Boundaries in x-direction
        y_min, y_max: Boundaries in y-direction
        nx, ny: Number of grid points in x and y directions

    Returns:
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        X: 2D meshgrid of x-coordinates
        Y: 2D meshgrid of y-coordinates
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def jacobi_solver_biharmonic(u: np.ndarray, f: np.ndarray, dx: float, dy: float,
                            tol: float = 1e-6, max_iter: int = 10000,
                            w_relax: float = 1.0) -> Tuple[np.ndarray, int, float]:
    """
    Jacobi solver for biharmonic equation with correct 13-point stencil.
    
    ∇⁴w = f, where f = q/D
    """
    u_new = u.copy()
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h = dx
    h4 = h**4
    
    while error > tol and iterations < max_iter:
        iterations += 1
        u_old = u_new.copy()
        
        # 13-point stencil for biharmonic equation
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                u_new[i, j] = (
                    8 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1]) -
                    2 * (u_old[i+1, j+1] + u_old[i+1, j-1] + u_old[i-1, j+1] + u_old[i-1, j-1]) -
                    (u_old[i+2, j] + u_old[i-2, j] + u_old[i, j+2] + u_old[i, j-2]) +
                    (h4 * f[i, j])
                ) / 20
        
        # Apply boundary conditions
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        
        # Relaxation
        u_new = w_relax * u_new + (1 - w_relax) * u_old
        
        error = np.max(np.abs(u_new - u_old))
    
    return u_new, iterations, error


def gauss_seidel_solver_biharmonic(u: np.ndarray, f: np.ndarray, dx: float, dy: float,
                                 tol: float = 1e-6, max_iter: int = 10000,
                                 w_relax: float = 1.0) -> Tuple[np.ndarray, int, float]:
    """
    Gauss-Seidel solver for biharmonic equation with correct 13-point stencil.
    """
    u_new = u.copy()
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h = dx
    h4 = h**4
    
    while error > tol and iterations < max_iter:
        iterations += 1
        u_old = u_new.copy()
        
        # 13-point stencil for biharmonic equation
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                u_new[i, j] = (
                    8 * (u_new[i+1, j] + u_new[i-1, j] + u_new[i, j+1] + u_new[i, j-1]) -
                    2 * (u_new[i+1, j+1] + u_new[i+1, j-1] + u_new[i-1, j+1] + u_new[i-1, j-1]) -
                    (u_new[i+2, j] + u_new[i-2, j] + u_new[i, j+2] + u_new[i, j-2]) +
                    (h4 * f[i, j])
                ) / 20
        
        # Apply boundary conditions
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        
        # Relaxation
        u_new = w_relax * u_new + (1 - w_relax) * u_old
        
        error = np.max(np.abs(u_new - u_old))
    
    return u_new, iterations, error


def apply_simply_supported_bc(u, nx, ny):
    """
    Apply simply supported boundary conditions.
    w = 0 and ∇²w = 0 at boundaries.
    """
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u


def analytic_solution_simply_supported(x, y, q, D, Lx=1.0, Ly=1.0, n_terms=5):
    """
    Analytic solution for simply supported rectangular plate with uniform load.
    Navier series solution.
    """
    w = np.zeros_like(x)
    for m in range(1, n_terms*2, 2):  # Odd terms only
        for n in range(1, n_terms*2, 2):
            coeff = 16 * q / (np.pi**6 * D * m * n)
            denom = (m**2/Lx**2 + n**2/Ly**2)**2
            w += coeff/denom * np.sin(m*np.pi*x/Lx) * np.sin(n*np.pi*y/Ly)
    return w


def validate_solution(w_numeric, X, Y, q, D, dx, dy):
    """
    Validate numeric solution against analytic.
    """
    w_analytic = analytic_solution_simply_supported(X, Y, q, D, n_terms=3)
    
    max_error = np.max(np.abs(w_numeric - w_analytic))
    rms_error = np.sqrt(np.mean((w_numeric - w_analytic)**2))
    
    h = dx
    h4 = h**4
    
    i, j = w_numeric.shape[0]//2, w_numeric.shape[1]//2
    biharmonic_numeric = (
        20*w_numeric[i, j] - 
        8*(w_numeric[i+1, j] + w_numeric[i-1, j] + w_numeric[i, j+1] + w_numeric[i, j-1]) +
        2*(w_numeric[i+1, j+1] + w_numeric[i+1, j-1] + w_numeric[i-1, j+1] + w_numeric[i-1, j-1]) +
        (w_numeric[i+2, j] + w_numeric[i-2, j] + w_numeric[i, j+2] + w_numeric[i, j-2])
    ) / h4
    
    q_over_D = q / D
    pde_error = abs(biharmonic_numeric - q_over_D)
    
    return {
        'max_error': max_error,
        'rms_error': rms_error,
        'pde_error': pde_error,
        'biharmonic_numeric': biharmonic_numeric,
        'q_over_D': q_over_D
    }


def solve_biharmonic_equation_corrected(
    x_min: float = 0, x_max: float = 1,
    y_min: float = 0, y_max: float = 1,
    nx: int = 50, ny: int = 50,
    q: float = 10.0,  # kN/m²
    D: float = 10000.0,  # kN·m
    bc_type: str = 'simply-supported',
    solver: str = 'gauss-seidel',
    tol: float = 1e-6,
    max_iter: int = 10000,
    w_relax: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, dict]:
    
    x, y, X, Y = create_2d_grid(x_min, x_max, y_min, y_max, nx, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    if dx != dy:
        print("Warning: Non-uniform grid may affect accuracy")
    
    # Initialize with analytic solution guess
    w = analytic_solution_simply_supported(X, Y, q, D, n_terms=2)
    w = np.maximum(w, 0)  # Ensure positive deflection
    
    f = np.ones((nx, ny)) * (q / D)
    
    if solver == 'jacobi':
        w, iterations, error = jacobi_solver_biharmonic(w, f, dx, dy, tol, max_iter, w_relax)
    elif solver == 'gauss-seidel':
        w, iterations, error = gauss_seidel_solver_biharmonic(w, f, dx, dy, tol, max_iter, w_relax)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    if bc_type == 'simply-supported':
        w = apply_simply_supported_bc(w, nx, ny)
    elif bc_type == 'clamped':
        w[0, :] = 0
        w[-1, :] = 0
        w[:, 0] = 0
        w[:, -1] = 0
    
    validation = validate_solution(w, X, Y, q, D, dx, dy)
    
    return X, Y, w, iterations, error, validation


def solve_biharmonic_equation(x_min: float = 0, x_max: float = 1,
                              y_min: float = 0, y_max: float = 1,
                              nx: int = 50, ny: int = 50,
                              load_func: Optional[Callable[[float, float], float]] = None,
                              solver: str = 'gauss-seidel',
                              tol: float = 1e-4,
                              max_iter: int = 10000,
                              w_relax: float = 1.0,
                              q: float = 10.0,
                              D: float = 10000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Solve the biharmonic plate equation using finite difference methods.
    This is the entry point for Streamlit UI.
    """
    X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
        x_min, x_max, y_min, y_max, nx, ny, q, D,
        bc_type='simply-supported', solver=solver,
        tol=tol, max_iter=max_iter, w_relax=w_relax
    )
    
    return X, Y, w, iterations, error


def plot_deflection_3d(X: np.ndarray, Y: np.ndarray, w: np.ndarray,
                       title: str = "Plate Deflection (3D)") -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, w, cmap=cm.viridis,
                          linewidth=0.5, antialiased=True,
                          shade=True, alpha=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('Deflection w', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = fig.colorbar(surf, shrink=0.8, aspect=10)
    cbar.set_label('Deflection', fontsize=10)

    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()

    return fig


def plot_deflection_contour(X: np.ndarray, Y: np.ndarray, w: np.ndarray,
                          title: str = "Plate Deflection (Contour)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(X, Y, w, cmap=cm.viridis, levels=50)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = fig.colorbar(contour, shrink=0.8, aspect=10)
    cbar.set_label('Deflection', fontsize=10)

    plt.tight_layout()

    return fig


def plot_convergence(iterations: int, error: float, solver_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(['Iterations', 'Final Error'], [iterations, error],
           color=['#667eea', '#764ba2'])

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Convergence - {solver_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def compute_statistics(w: np.ndarray) -> dict:
    return {
        'max_deflection': float(np.max(w)),
        'min_deflection': float(np.min(w)),
        'mean_deflection': float(np.mean(w)),
        'std_deflection': float(np.std(w)),
        'total_deflection': float(np.sum(w))
    }
