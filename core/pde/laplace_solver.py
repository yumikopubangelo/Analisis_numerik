"""
Laplace Equation Solver using Finite Difference Methods

This module provides functionality for solving the 2D Laplace equation:
nabla^2 phi = d^2phi/dx^2 + d^2phi/dy^2 = 0

Using iterative methods:
- Jacobi
- Gauss-Seidel
- SOR (Successive Over-Relaxation)

Supports:
- Dirichlet boundary conditions (phi = g on boundary)
- Neumann boundary conditions (dphi/dn = h on boundary)

Applications:
- Steady-state heat distribution on a metal plate
- Electrostatic potential
- Fluid potential flow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, Optional, Dict, Any


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


def jacobi_solver_laplace(u: np.ndarray, f: np.ndarray, h: float,
                          tol: float = 1e-6, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    """
    Jacobi iterative solver for the Laplace/Poisson equation.
    
    For Laplace equation (nabla^2 u = 0):
        u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) / 4
    
    For Poisson equation (nabla^2 u = f):
        u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h^2*f[i,j]) / 4
    
    Parameters:
        u: Initial guess (boundary conditions already applied)
        f: Source term (zeros for Laplace equation)
        h: Grid spacing (assumes uniform grid)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        u: Solution field
        iterations: Number of iterations performed
        error: Final error (max absolute change)
    """
    u_new = u.copy()
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h2 = h ** 2
    
    report_interval = max(100, max_iter // 100)
    
    while error > tol and iterations < max_iter:
        iterations += 1
        u_old = u_new.copy()
        
        # Update interior points (1:-1 to skip boundary and ghost points)
        u_new[1:-1, 1:-1] = (u_old[2:, 1:-1] + u_old[:-2, 1:-1] + 
                            u_old[1:-1, 2:] + u_old[1:-1, :-2] + 
                            h2 * f[1:-1, 1:-1]) / 4
        
        # Compute error (max absolute change in interior)
        error = np.max(np.abs(u_new[1:-1, 1:-1] - u_old[1:-1, 1:-1]))
        
        # Progress reporting
        if iterations % report_interval == 0 and iterations > 0:
            print(f'Jacobi Iteration {iterations}: Error = {error:.2e}')
        
        # Safety check
        if iterations > 100 and np.isnan(error):
            print(f'Warning: NaN detected at iteration {iterations}')
            break
    
    return u_new, iterations, error


def gauss_seidel_solver_laplace(u: np.ndarray, f: np.ndarray, h: float,
                                 tol: float = 1e-6, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    """
    Gauss-Seidel iterative solver for the Laplace/Poisson equation.
    
    Uses updated values immediately (in-place update), typically converges
    faster than Jacobi.
    
    Parameters:
        u: Initial guess (boundary conditions already applied)
        f: Source term (zeros for Laplace equation)
        h: Grid spacing (assumes uniform grid)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        u: Solution field
        iterations: Number of iterations performed
        error: Final error (max absolute change)
    """
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h2 = h ** 2
    
    report_interval = max(100, max_iter // 100)
    
    while error > tol and iterations < max_iter:
        iterations += 1
        u_old = u.copy()
        
        # Update interior points using latest values (in-place)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i, j] = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] + 
                          h2 * f[i, j]) / 4
        
        # Compute error
        error = np.max(np.abs(u[1:-1, 1:-1] - u_old[1:-1, 1:-1]))
        
        # Progress reporting
        if iterations % report_interval == 0 and iterations > 0:
            print(f'Gauss-Seidel Iteration {iterations}: Error = {error:.2e}')
        
        # Safety check
        if iterations > 100 and np.isnan(error):
            print(f'Warning: NaN detected at iteration {iterations}')
            break
    
    return u, iterations, error


def sor_solver_laplace(u: np.ndarray, f: np.ndarray, h: float, 
                       omega: float = 1.5, tol: float = 1e-6, 
                       max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    """
    SOR (Successive Over-Relaxation) iterative solver for the Laplace/Poisson equation.
    
    SOR accelerates convergence by using a weighted average of the previous
    and updated values. Optimal omega is typically between 1 and 2.
    For Poisson on square grid: omega_opt = 2/(1 + sin(pi/n))
    
    Parameters:
        u: Initial guess (boundary conditions already applied)
        f: Source term (zeros for Laplace equation)
        h: Grid spacing (assumes uniform grid)
        omega: Relaxation factor (1 < omega < 2 for over-relaxation)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        u: Solution field
        iterations: Number of iterations performed
        error: Final error (max absolute change)
    """
    error = np.inf
    iterations = 0
    nx, ny = u.shape
    h2 = h ** 2
    
    # Compute optimal omega for square grid if needed
    if omega <= 0:
        omega = min(1.9, 2.0 / (1.0 + np.sin(np.pi / max(nx, ny))))
    
    report_interval = max(100, max_iter // 100)
    
    while error > tol and iterations < max_iter:
        iterations += 1
        u_old = u.copy()
        
        # Update interior points using SOR formula
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i, j] = (1 - omega) * u[i, j] + omega * (
                    u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] + h2 * f[i, j]
                ) / 4
        
        # Compute error
        error = np.max(np.abs(u[1:-1, 1:-1] - u_old[1:-1, 1:-1]))
        
        # Progress reporting
        if iterations % report_interval == 0 and iterations > 0:
            print(f'SOR (omega={omega:.3f}) Iteration {iterations}: Error = {error:.2e}')
        
        # Safety check
        if iterations > 100 and np.isnan(error):
            print(f'Warning: NaN detected at iteration {iterations}')
            break
    
    return u, iterations, error


def apply_dirichlet_bc_simple(u: np.ndarray, bc_values: Dict[str, float]) -> np.ndarray:
    """
    Apply simple Dirichlet boundary conditions with constant values on each edge.
    
    Parameters:
        u: Solution field
        bc_values: Dictionary with keys 'left', 'right', 'bottom', 'top' giving boundary values
        
    Returns:
        u: Solution field with boundary conditions applied
    """
    nx, ny = u.shape
    
    if 'left' in bc_values:
        u[0, :] = bc_values['left']
    if 'right' in bc_values:
        u[-1, :] = bc_values['right']
    if 'bottom' in bc_values:
        u[:, 0] = bc_values['bottom']
    if 'top' in bc_values:
        u[:, -1] = bc_values['top']
    
    return u


def apply_neumann_bc_simple(u: np.ndarray, h: float, 
                           bc_left: Optional[float] = None,
                           bc_right: Optional[float] = None,
                           bc_bottom: Optional[float] = None,
                           bc_top: Optional[float] = None) -> np.ndarray:
    """
    Apply Neumann boundary conditions using ghost point method.
    
    For dphi/dn = value at boundary (using forward/backward differences at boundary):
    - Left boundary: u[0, j] = u[1, j] - h * bc_left
    - Right boundary: u[-1, j] = u[-2, j] + h * bc_right
    - Bottom boundary: u[i, 0] = u[i, 1] - h * bc_bottom
    - Top boundary: u[i, -1] = u[i, -2] + h * bc_top
    
    For homogeneous Neumann (dphi/dn = 0), the boundary value equals the adjacent interior value.
    
    Note: This version uses the boundary value directly in the update,
    rather than ghost points, which avoids array bounds issues.
    
    Parameters:
        u: Solution field (ghost points will be set)
        h: Grid spacing
        bc_left: dphi/dx at x = x_min
        bc_right: dphi/dx at x = x_max
        bc_bottom: dphi/dy at y = y_min
        bc_top: dphi/dy at y = y_max
        
    Returns:
        u: Solution field with Neumann BC applied at boundaries
    """
    nx, ny = u.shape
    
    if bc_left is not None:
        # Left boundary: u[0,:] = u[1,:] - h * bc_left
        u[0, :] = u[1, :] - h * bc_left
        
    if bc_right is not None:
        # Right boundary: u[-1,:] = u[-2,:] + h * bc_right
        u[-1, :] = u[-2, :] + h * bc_right
        
    if bc_bottom is not None:
        # Bottom boundary: u[:,0] = u[:,1] - h * bc_bottom
        u[:, 0] = u[:, 1] - h * bc_bottom
        
    if bc_top is not None:
        # Top boundary: u[:,-1] = u[:,-2] + h * bc_top
        u[:, -1] = u[:, -2] + h * bc_top
        
    return u


def apply_mixed_bc_simple(u: np.ndarray, h: float,
                          bc_values: Dict[str, float],
                          bc_left: Optional[float] = None,
                          bc_right: Optional[float] = None,
                          bc_bottom: Optional[float] = None,
                          bc_top: Optional[float] = None) -> np.ndarray:
    """
    Apply mixed Dirichlet/Neumann boundary conditions.
    
    Parameters:
        u: Solution field
        h: Grid spacing
        bc_values: Dictionary with Dirichlet BC values
        bc_left, bc_right, bc_bottom, bc_top: Neumann BC values
    """
    # First apply Dirichlet BC everywhere
    u = apply_dirichlet_bc_simple(u, bc_values)
    
    # Then override with Neumann where specified
    if bc_left is not None or bc_right is not None or bc_bottom is not None or bc_top is not None:
        u = apply_neumann_bc_simple(u, h, bc_left, bc_right, bc_bottom, bc_top)
    
    return u


def compute_laplacian(u: np.ndarray, h: float) -> np.ndarray:
    """
    Compute the Laplacian nabla^2 u using 5-point finite difference stencil.
    
    Returns:
        Laplacian of u (same size as input, interior points computed)
    """
    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + 
                           u[1:-1, 2:] + u[1:-1, :-2] - 
                           4 * u[1:-1, 1:-1]) / h ** 2
    return laplacian


def solve_laplace_equation(
    x_min: float = 0, x_max: float = 1,
    y_min: float = 0, y_max: float = 1,
    nx: int = 51, ny: int = 51,
    bc_type: str = 'dirichlet',
    bc_left: Optional[float] = None,
    bc_right: Optional[float] = None,
    bc_bottom: Optional[float] = None,
    bc_top: Optional[float] = None,
    solver: str = 'sor',
    omega: float = 1.5,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Dict[str, Any]:
    """
    Solve the 2D Laplace equation using finite difference methods.
    
    Equation: nabla^2 phi = d^2phi/dx^2 + d^2phi/dy^2 = 0
    
    Parameters:
        x_min, x_max: Domain boundaries in x-direction
        y_min, y_max: Domain boundaries in y-direction
        nx, ny: Number of grid points
        bc_type: 'dirichlet' or 'mixed'
        bc_left: Dirichlet value at x = x_min OR Neumann value dphi/dx at x = x_min
        bc_right: Dirichlet value at x = x_max OR Neumann value dphi/dx at x = x_max
        bc_bottom: Dirichlet value at y = y_min OR Neumann value dphi/dy at y = y_min
        bc_top: Dirichlet value at y = y_max OR Neumann value dphi/dy at y = y_max
        solver: 'jacobi', 'gauss-seidel', or 'sor'
        omega: Relaxation factor for SOR (1 < omega < 2 for over-relaxation)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        Dictionary containing:
        - 'X', 'Y': Meshgrid coordinates
        - 'phi': Solution field phi(x,y)
        - 'iterations': Number of iterations performed
        - 'error': Final error
        - 'statistics': Dict with min, max, mean of phi
        - 'converged': Boolean indicating if converged
    """
    # Create grid
    x, y, X, Y = create_2d_grid(x_min, x_max, y_min, y_max, nx, ny)
    h = x[1] - x[0]
    
    # Initialize solution field
    phi = np.zeros((nx, ny))
    
    # Source term (zeros for Laplace equation)
    f = np.zeros((nx, ny))
    
    # Prepare BC values dictionary for Dirichlet
    bc_values = {}
    if bc_left is not None:
        bc_values['left'] = bc_left
    if bc_right is not None:
        bc_values['right'] = bc_right
    if bc_bottom is not None:
        bc_values['bottom'] = bc_bottom
    if bc_top is not None:
        bc_values['top'] = bc_top
    
    # Apply boundary conditions
    if bc_type == 'dirichlet':
        phi = apply_dirichlet_bc_simple(phi, bc_values)
    elif bc_type == 'mixed':
        # For mixed, use 0 as default Dirichlet, then apply Neumann where specified
        default_bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        # Determine which BCs are Neumann (need to be detected differently)
        # For now, treat all as Dirichlet for mixed
        phi = apply_dirichlet_bc_simple(phi, bc_values)
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    # Solve using specified method
    if solver == 'jacobi':
        phi, iterations, error = jacobi_solver_laplace(phi, f, h, tol, max_iter)
    elif solver == 'gauss-seidel':
        phi, iterations, error = gauss_seidel_solver_laplace(phi, f, h, tol, max_iter)
    elif solver == 'sor':
        phi, iterations, error = sor_solver_laplace(phi, f, h, omega, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Re-apply boundary conditions after iteration
    if bc_type == 'dirichlet':
        phi = apply_dirichlet_bc_simple(phi, bc_values)
    elif bc_type == 'mixed':
        phi = apply_dirichlet_bc_simple(phi, bc_values)
    
    # Compute statistics (interior points only)
    interior_phi = phi[1:-1, 1:-1]
    statistics = {
        'min_phi': float(np.min(interior_phi)) if interior_phi.size > 0 else 0.0,
        'max_phi': float(np.max(interior_phi)) if interior_phi.size > 0 else 0.0,
        'mean_phi': float(np.mean(interior_phi)) if interior_phi.size > 0 else 0.0,
        'std_phi': float(np.std(interior_phi)) if interior_phi.size > 0 else 0.0,
        'phi_at_center': float(phi[nx//2, ny//2]),
        'phi_at_corner_bl': float(phi[0, 0]),
        'phi_at_corner_br': float(phi[-1, 0]),
        'phi_at_corner_tl': float(phi[0, -1]),
        'phi_at_corner_tr': float(phi[-1, -1])
    }
    
    # Return results
    result = {
        'X': X,
        'Y': Y,
        'phi': phi,
        'iterations': iterations,
        'error': error,
        'statistics': statistics,
        'grid_info': {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'nx': nx, 'ny': ny, 'h': h
        },
        'converged': error <= tol
    }
    
    return result


def solve_laplace_with_function_bc(
    x_min: float = 0, x_max: float = 1,
    y_min: float = 0, y_max: float = 1,
    nx: int = 51, ny: int = 51,
    bc_func: Optional[Callable[[float, float], float]] = None,
    solver: str = 'sor',
    omega: float = 1.5,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Dict[str, Any]:
    """
    Solve the 2D Laplace equation with function-based boundary conditions.
    
    Parameters:
        x_min, x_max: Domain boundaries
        y_min, y_max: Domain boundaries
        nx, ny: Number of grid points
        bc_func: Function that returns phi value at (x, y) for boundary points
        solver: 'jacobi', 'gauss-seidel', or 'sor'
        omega: Relaxation factor for SOR
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Solution dictionary
    """
    # Create grid
    x, y, X, Y = create_2d_grid(x_min, x_max, y_min, y_max, nx, ny)
    h = x[1] - x[0]
    
    # Initialize solution field
    phi = np.zeros((nx, ny))
    
    # Apply boundary conditions using function
    if bc_func is not None:
        # Bottom boundary (j=0)
        phi[0, :] = bc_func(x, np.zeros_like(x))
        # Top boundary (j=ny-1)
        phi[-1, :] = bc_func(x, np.ones_like(x) * y_max)
        # Left boundary (i=0)
        phi[:, 0] = bc_func(np.zeros_like(y) * x_min, y)
        # Right boundary (i=nx-1)
        phi[:, -1] = bc_func(np.ones_like(y) * x_max, y)
    
    # Source term
    f = np.zeros((nx, ny))
    
    # Solve
    if solver == 'jacobi':
        phi, iterations, error = jacobi_solver_laplace(phi, f, h, tol, max_iter)
    elif solver == 'gauss-seidel':
        phi, iterations, error = gauss_seidel_solver_laplace(phi, f, h, tol, max_iter)
    elif solver == 'sor':
        phi, iterations, error = sor_solver_laplace(phi, f, h, omega, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Re-apply BC
    if bc_func is not None:
        phi[0, :] = bc_func(x, np.zeros_like(x))
        phi[-1, :] = bc_func(x, np.ones_like(x) * y_max)
        phi[:, 0] = bc_func(np.zeros_like(y) * x_min, y)
        phi[:, -1] = bc_func(np.ones_like(y) * x_max, y)
    
    # Statistics
    interior_phi = phi[1:-1, 1:-1]
    statistics = {
        'min_phi': float(np.min(interior_phi)) if interior_phi.size > 0 else 0.0,
        'max_phi': float(np.max(interior_phi)) if interior_phi.size > 0 else 0.0,
        'mean_phi': float(np.mean(interior_phi)) if interior_phi.size > 0 else 0.0,
        'std_phi': float(np.std(interior_phi)) if interior_phi.size > 0 else 0.0,
        'phi_at_center': float(phi[nx//2, ny//2])
    }
    
    return {
        'X': X,
        'Y': Y,
        'phi': phi,
        'iterations': iterations,
        'error': error,
        'statistics': statistics,
        'grid_info': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'nx': nx, 'ny': ny, 'h': h},
        'converged': error <= tol
    }


def plot_phi_contour(X: np.ndarray, Y: np.ndarray, phi: np.ndarray,
                      title: str = "phi(x,y) Distribution - Contour",
                      levels: int = 30) -> plt.Figure:
    """
    Plot contourf visualization of phi(x,y).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(X, Y, phi, levels=levels, cmap=cm.viridis)
    contour_lines = ax.contour(X, Y, phi, levels=levels, colors='white', 
                               linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(contour, shrink=0.8, aspect=10)
    cbar.set_label('phi(x,y)', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_phi_wireframe(X: np.ndarray, Y: np.ndarray, phi: np.ndarray,
                       title: str = "phi(x,y) Distribution - Wireframe") -> plt.Figure:
    """
    Plot wireframe 3D visualization of phi(x,y).
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    step = max(1, X.shape[0] // 30)
    X_ds = X[::step, ::step]
    Y_ds = Y[::step, ::step]
    phi_ds = phi[::step, ::step]
    
    ax.plot_wireframe(X_ds, Y_ds, phi_ds, color='steelblue', 
                      linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('phi(x,y)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()
    return fig


def plot_phi_surface(X: np.ndarray, Y: np.ndarray, phi: np.ndarray,
                     title: str = "phi(x,y) Distribution - 3D Surface") -> plt.Figure:
    """
    Plot 3D surface visualization of phi(x,y).
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, phi, cmap=cm.viridis,
                           linewidth=0.3, antialiased=True,
                           alpha=0.9, shade=True)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('phi(x,y)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('phi(x,y)', fontsize=11)
    
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()
    return fig


def plot_convergence(iterations: int, error: float, 
                     solver_name: str = "SOR") -> plt.Figure:
    """
    Plot convergence information as a bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    log_error = -np.log10(error) if error > 0 else 20
    
    bars = ax.bar(['Iterations', f'Final Error\n(10^{{-{log_error:.1f}}})'], 
                  [iterations, error], 
                  color=['#667eea', '#764ba2'])
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Convergence - {solver_name}', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, [iterations, error]):
        height = bar.get_height()
        if isinstance(val, float):
            label = f'{val:.2e}'
        else:
            label = str(val)
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_convergence_history(errors: list, solver_name: str = "SOR") -> plt.Figure:
    """
    Plot convergence history (error vs iteration).
    
    Parameters:
        errors: List of errors at each iteration
        solver_name: Name of the solver used
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(errors) + 1)
    ax.semilogy(iterations, errors, 'b-', linewidth=1.5, label='Error')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Error (max norm)', fontsize=12)
    ax.set_title(f'Convergence History - {solver_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def print_solution_summary(result: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the solution.
    """
    print("\n" + "="*60)
    print("LAPLACE EQUATION SOLVER - SOLUTION SUMMARY")
    print("="*60)
    
    grid_info = result['grid_info']
    print(f"\nGrid Information:")
    print(f"  Domain: [{grid_info['x_min']}, {grid_info['x_max']}] x "
          f"[{grid_info['y_min']}, {grid_info['y_max']}]")
    print(f"  Grid size: {grid_info['nx']} x {grid_info['ny']}")
    print(f"  Grid spacing (h): {grid_info['h']:.6f}")
    
    print(f"\nConvergence:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final error: {result['error']:.6e}")
    print(f"  Converged: {'Yes' if result['converged'] else 'No'}")
    
    stats = result['statistics']
    print(f"\nStatistics (phi values):")
    print(f"  Minimum:  {stats['min_phi']:.6f}")
    print(f"  Maximum:  {stats['max_phi']:.6f}")
    print(f"  Mean:     {stats['mean_phi']:.6f}")
    print(f"  Std Dev:  {stats['std_phi']:.6f}")
    
    print(f"\nSample Points:")
    print(f"  Center:           phi = {stats['phi_at_center']:.6f}")
    print(f"  Bottom-Left:      phi = {stats['phi_at_corner_bl']:.6f}")
    print(f"  Bottom-Right:     phi = {stats['phi_at_corner_br']:.6f}")
    print(f"  Top-Left:         phi = {stats['phi_at_corner_tl']:.6f}")
    print(f"  Top-Right:        phi = {stats['phi_at_corner_tr']:.6f}")
    
    print("="*60 + "\n")


# Example Applications
def example_heat_distribution():
    """
    Example: Steady-state heat distribution on a metal plate.
    
    Left edge: T = 100 C
    Right edge: T = 0 C
    Top and bottom edges: insulated (dT/dn = 0)
    """
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_type='dirichlet',
        bc_left=100.0,
        bc_right=0.0,
        bc_bottom=0.0,
        bc_top=0.0,
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    return result


def example_electrostatic():
    """
    Example: Electrostatic potential with sinusoidal top boundary.
    """
    def bc_func(x, y_val):
        # Top boundary: phi = sin(pi*x)
        if abs(y_val - 1.0) < 0.01:
            return np.sin(np.pi * x)
        return 0.0
    
    result = solve_laplace_with_function_bc(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_func=bc_func,
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    return result


# Export functions
__all__ = [
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
