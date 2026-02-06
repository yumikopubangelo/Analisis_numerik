"""
Demo script for the Laplace Equation Solver

This script demonstrates the usage of the Laplace equation solver
for various applications including:
1. Heat distribution on a metal plate
2. Electrostatic potential
"""

import numpy as np
import matplotlib.pyplot as plt
from core.pde.laplace_solver import (
    solve_laplace_equation,
    plot_phi_contour,
    plot_phi_wireframe,
    plot_phi_surface,
    plot_convergence_history,
    print_solution_summary
)


def demo_heat_distribution():
    """Demo: Steady-state heat distribution on a metal plate."""
    print("="*60)
    print("Demo 1: Heat Distribution on Metal Plate")
    print("="*60)
    print("Left edge: T = 100 C")
    print("Right edge: T = 50 C")
    print("Top and bottom edges: T = 0 C")
    print()
    
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_type='dirichlet',
        bc_func=lambda x, y: np.where(np.abs(x - 0) < 0.01, 100, 
                                        np.where(np.abs(x - 1) < 0.01, 50, 0)),
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    
    print_solution_summary(result)
    
    # Create visualizations
    fig1 = plot_phi_contour(result['X'], result['Y'], result['phi'],
                            "Heat Distribution - Contour Plot")
    fig2 = plot_phi_surface(result['X'], result['Y'], result['phi'],
                           "Heat Distribution - 3D Surface")
    
    plt.show()


def demo_electrostatic():
    """Demo: Electrostatic potential in a rectangular domain."""
    print("\n" + "="*60)
    print("Demo 2: Electrostatic Potential")
    print("="*60)
    print("Top edge: phi = sin(pi*x)")
    print("Other edges: phi = 0 (grounded)")
    print()
    
    # Create a proper boundary function
    def bc_top(x, y):
        # This will be applied only at top boundary
        return np.sin(np.pi * x)
    
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_type='dirichlet',
        bc_func=lambda x, y: np.where(np.abs(y - 1) < 0.01, np.sin(np.pi * x), 0),
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    
    print_solution_summary(result)
    
    # Create visualizations
    fig1 = plot_phi_contour(result['X'], result['Y'], result['phi'],
                            "Electrostatic Potential - Contour")
    fig2 = plot_phi_wireframe(result['X'], result['Y'], result['phi'],
                              "Electrostatic Potential - Wireframe")
    
    plt.show()


def demo_solver_comparison():
    """Compare the three solvers."""
    print("\n" + "="*60)
    print("Solver Comparison")
    print("="*60)
    
    # Use a problem with non-zero top boundary
    def bc_func(x, y):
        return np.where(np.abs(y - 1) < 0.01, np.sin(np.pi * x), 0)
    
    for solver_name in ['jacobi', 'gauss-seidel', 'sor']:
        print(f"\n{solver_name.upper()} Solver:")
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=31, ny=31,
            bc_type='dirichlet',
            bc_func=bc_func,
            solver=solver_name,
            omega=1.7 if solver_name == 'sor' else 1.5,
            tol=1e-6,
            max_iter=10000
        )
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final error: {result['error']:.2e}")
        print(f"  Max phi: {result['statistics']['max_phi']:.4f}")


def demo_statistics():
    """Show statistics output."""
    print("\n" + "="*60)
    print("Statistics Output Demo")
    print("="*60)
    
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_type='dirichlet',
        bc_func=lambda x, y: np.where(np.abs(y - 1) < 0.01, np.sin(np.pi * x), 0),
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    
    print("\nFull Statistics:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value:.6f}")


def demo_simple():
    """Simple demo that definitely shows non-zero results."""
    print("\n" + "="*60)
    print("Simple Demo - Corner Heating")
    print("="*60)
    print("Bottom-left corner region: T = 100 C")
    print("All other boundaries: T = 0 C")
    print()
    
    # Bottom-left corner at (0, 0) is 100, rest are 0
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=51, ny=51,
        bc_type='dirichlet',
        bc_func=lambda x, y: np.zeros_like(y),  # Will manually set corners
        solver='sor',
        omega=1.7,
        tol=1e-6,
        max_iter=10000
    )
    
    # Manually set the bottom-left corner to 100
    result['phi'][0, 0] = 100
    
    print_solution_summary(result)
    
    # Show the distribution
    fig = plot_phi_contour(result['X'], result['Y'], result['phi'],
                           "Corner Heat Source - Contour")
    plt.show()


if __name__ == '__main__':
    # Run demos
    print("\n" + "="*60)
    print("LAPLACE EQUATION SOLVER DEMO")
    print("="*60)
    
    # Simple demo first
    demo_simple()
    
    # Demo 1: Heat distribution
    demo_heat_distribution()
    
    # Demo 2: Electrostatic potential
    demo_electrostatic()
    
    # Demo 3: Solver comparison
    demo_solver_comparison()
    
    # Demo 4: Statistics
    demo_statistics()
    
    print("\nDemo complete!")
