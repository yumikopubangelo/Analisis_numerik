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
                            w_relax: float = 0.8) -> Tuple[np.ndarray, int, float]:
    """
    Jacobi solver for biharmonic equation using split formulation.
    
    Splits ∇⁴w = f into two Poisson equations:
    ∇²w = φ  (first Poisson)
    ∇²φ = f  (second Poisson)
    
    Uses Jacobi iteration with under-relaxation for stability.
    """
    u_new = u.copy()
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h = dx
    
    # Initialize auxiliary field phi
    phi = np.zeros((nx, ny))
    
    # Progress reporting interval
    report_interval = max(100, max_iter // 100)
    
    while error > tol and iterations < max_iter:
        iterations += 1
        
        # Solve ∇²φ = f (Poisson for phi) - Jacobi iteration
        phi_old = phi.copy()
        phi[1:-1, 1:-1] = (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1] + 
                          phi_old[1:-1, 2:] + phi_old[1:-1, :-2] + 
                          h**2 * f[1:-1, 1:-1]) / 4
        phi[0, :] = 0
        phi[-1, :] = 0
        phi[:, 0] = 0
        phi[:, -1] = 0
        
        # Solve ∇²w = φ (Poisson for w) - Jacobi iteration
        u_old = u_new.copy()
        u_new[1:-1, 1:-1] = (u_old[2:, 1:-1] + u_old[:-2, 1:-1] + 
                           u_old[1:-1, 2:] + u_old[1:-1, :-2] + 
                           h**2 * phi[1:-1, 1:-1]) / 4
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        
        # Under-relaxation
        u_new = w_relax * u_new + (1 - w_relax) * u_old
        phi = w_relax * phi + (1 - w_relax) * phi_old
        
        # Compute error periodically
        if iterations % report_interval == 0 or error <= tol:
            error = np.max(np.abs(phi[1:-1, 1:-1] - compute_laplacian_5pt(u_new, h)))
            if iterations % (report_interval * 10) == 0:
                print(f'Iteration {iterations}: Error = {error:.2e}, Max w = {np.max(np.abs(u_new)):.6e}')
        
        # Safety check
        if iterations > 100 and np.isnan(np.max(u_new)):
            print(f'Warning: NaN detected at iteration {iterations}')
            break
    
    return u_new, iterations, error


def sor_solver_poisson(u: np.ndarray, f: np.ndarray, dx: float, omega: float = 1.5) -> np.ndarray:
    """
    SOR (Successive Over-Relaxation) solver for Poisson equation.
    Faster convergence than Gauss-Seidel.
    """
    nx, ny = u.shape
    h = dx
    h2 = h**2
    
    # SOR iteration
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = (1 - omega) * u[i, j] + omega * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] + h2 * f[i, j]
            ) / 4
    
    return u


def compute_laplacian_5pt(u, h):
    """Compute 5-point Laplacian stencil."""
    return (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]) / h**2


def gauss_seidel_solver_biharmonic(u: np.ndarray, f: np.ndarray, dx: float, dy: float,
                                 tol: float = 1e-6, max_iter: int = 10000,
                                 w_relax: float = 0.8) -> Tuple[np.ndarray, int, float]:
    """
    Gauss-Seidel solver for biharmonic equation using split formulation with SOR.
    
    Splits ∇⁴w = f into two Poisson equations:
    ∇²w = φ  (first Poisson)
    ∇²φ = f  (second Poisson)
    
    Uses SOR (Successive Over-Relaxation) for much faster convergence.
    """
    u_new = u.copy()
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h = dx
    h2 = h**2
    
    # Optimal SOR parameter for Poisson equation on square grid
    omega = min(1.9, 2.0 / (1.0 + np.sin(np.pi / max(nx, ny))))
    
    # Initialize auxiliary field phi
    phi = np.zeros((nx, ny))
    
    # Progress reporting interval
    report_interval = max(100, max_iter // 100)
    
    # Inner iterations for Poisson solves
    inner_iter = 5
    
    while error > tol and iterations < max_iter:
        iterations += 1
        
        # Solve ∇²φ = f (Poisson for phi) - multiple SOR iterations
        for _ in range(inner_iter):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    phi[i, j] = (1 - omega) * phi[i, j] + omega * (
                        phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] + h2 * f[i, j]
                    ) / 4
            phi[0, :] = 0
            phi[-1, :] = 0
            phi[:, 0] = 0
            phi[:, -1] = 0
        
        # Solve ∇²w = φ (Poisson for w) - multiple SOR iterations
        for _ in range(inner_iter):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    u_new[i, j] = (1 - omega) * u_new[i, j] + omega * (
                        u_new[i+1, j] + u_new[i-1, j] + u_new[i, j+1] + u_new[i, j-1] + h2 * phi[i, j]
                    ) / 4
            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0
        
        # Compute biharmonic residual directly
        if iterations % report_interval == 0 or error <= tol:
            # ∇⁴w using split: ∇²(∇²w) = ∇²φ
            # Compute ∇²w and then ∇² of that
            lap_w = compute_laplacian_5pt(u_new, h)
            # Pad lap_w to match phi dimensions for comparison
            lap_w_padded = np.zeros_like(phi)
            lap_w_padded[1:-1, 1:-1] = lap_w
            biharmonic_residual = np.max(np.abs(lap_w_padded - phi))
            error = biharmonic_residual
            
            if iterations % (report_interval * 10) == 0:
                print(f'Iteration {iterations}: Error = {error:.2e}, Max w = {np.max(np.abs(u_new)):.6e}')
        
        # Safety check
        if iterations > 100 and np.isnan(np.max(u_new)):
            print(f'Warning: NaN detected at iteration {iterations}')
            break
    
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


def compute_biharmonic_residual(u: np.ndarray, f: np.ndarray, h: float) -> float:
    """
    Compute infinity-norm residual of the biharmonic equation:
    (L(u) - f) using the 13-point stencil.
    """
    if u.shape[0] < 5 or u.shape[1] < 5:
        return np.inf

    core = u[2:-2, 2:-2]
    laplace_like = (
        20 * core
        - 8 * (u[3:-1, 2:-2] + u[1:-3, 2:-2] + u[2:-2, 3:-1] + u[2:-2, 1:-3])
        + 2 * (u[3:-1, 3:-1] + u[3:-1, 1:-3] + u[1:-3, 3:-1] + u[1:-3, 1:-3])
        + (u[4:, 2:-2] + u[:-4, 2:-2] + u[2:-2, 4:] + u[2:-2, :-4])
    )
    residual = laplace_like / (h**4) - f[2:-2, 2:-2]
    return float(np.max(np.abs(residual)))


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
    nx: int = 21, ny: int = 21,  # Reasonable default for interactive use
    q: float = 500000.0,  # N/m² (500 kN/m²) - optimal for visible deflection
    D: float = 10000.0,  # N·m (10 kN·m)
    bc_type: str = 'simply-supported',
    solver: str = 'gauss-seidel',
    tol: float = 1e-5,  # Looser tolerance for faster convergence
    max_iter: int = 10000,
    w_relax: float = 0.8  # Under-relaxation is critical for biharmonic convergence
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, dict]:
    
    x, y, X, Y = create_2d_grid(x_min, x_max, y_min, y_max, nx, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    if dx != dy:
        print("Warning: Non-uniform grid may affect accuracy")
    
    # Initialize with zero deflection guess
    w = np.zeros((nx, ny))
    
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
                              nx: int = 21, ny: int = 21,  # Reasonable default for interactive use
                              load_func: Optional[Callable[[float, float], float]] = None,
                              solver: str = 'gauss-seidel',
                              tol: float = 1e-5,  # Looser tolerance for faster convergence
                              max_iter: int = 10000,
                              w_relax: float = 0.8,  # Under-relaxation is critical
                              q: float = 10.0,
                              D: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Solve the biharmonic plate equation using finite difference methods.
    This is the entry point for Streamlit UI.
    
    Note: Input parameters q (kN/m²) and D (kN·m) are converted to SI units (N/m² and N·m)
    internally for calculation.
    """
    # Convert units: kN/m² to N/m² and kN·m to N·m
    q_si = q * 1000  # kN/m² to N/m²
    D_si = D * 1000  # kN·m to N·m
    
    X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
        x_min, x_max, y_min, y_max, nx, ny, q_si, D_si,
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
        'deflection_at_center': float(w[w.shape[0]//2, w.shape[1]//2])
    }
