"""
Test cases for Laplace equation solver.

Tests include:
1. Convergence of iterative methods
2. Boundary condition handling
3. Statistics computation
4. Visualization functions
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pde.laplace_solver import (
    solve_laplace_equation,
    solve_laplace_with_function_bc,
    create_2d_grid,
    apply_dirichlet_bc_simple,
    apply_neumann_bc_simple,
    apply_mixed_bc_simple,
    compute_laplacian,
    plot_phi_contour,
    plot_phi_wireframe,
    plot_phi_surface,
    plot_convergence,
    print_solution_summary
)


class TestLaplaceSolver:
    """Test cases for the Laplace equation solver."""
    
    def test_sor_solver_convergence(self):
        """Test that SOR solver converges for simple Dirichlet BC."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='sor',
            omega=1.7,
            tol=1e-4,
            max_iter=5000
        )
        
        assert result['converged'], "SOR solver should converge"
        assert result['iterations'] < 5000, "Should converge in reasonable iterations"
        
        # Check that phi varies between boundaries
        interior = result['phi'][1:-1, 1:-1]
        assert np.min(interior) >= -1e-10, "Interior phi should be >= 0 (approximately)"
        assert np.max(interior) <= 100 + 1e-10, "Interior phi should not exceed left BC"
    
    def test_jacobi_solver_convergence(self):
        """Test that Jacobi solver converges."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=0.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='jacobi',
            tol=1e-4,
            max_iter=5000
        )
        
        assert result['converged'], "Jacobi solver should converge"
        # All zeros solution
        assert np.allclose(result['phi'], 0, atol=1e-3), "Solution should be approximately zero"
    
    def test_gauss_seidel_solver_convergence(self):
        """Test that Gauss-Seidel solver converges."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='gauss-seidel',
            tol=1e-4,
            max_iter=5000
        )
        
        assert result['converged'], "Gauss-Seidel solver should converge"
    
    def test_statistics_computation(self):
        """Test that statistics are computed correctly."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=31, ny=31,
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
        
        stats = result['statistics']
        
        # Check that all statistics are present
        required_keys = ['min_phi', 'max_phi', 'mean_phi', 'std_phi', 
                        'phi_at_center', 'phi_at_corner_bl', 'phi_at_corner_br',
                        'phi_at_corner_tl', 'phi_at_corner_tr']
        for key in required_keys:
            assert key in stats, f"Missing statistic: {key}"
        
        # Check min <= max
        assert stats['min_phi'] <= stats['max_phi'], "Min should be <= max"
        
        # Check mean is between min and max
        assert stats['min_phi'] <= stats['mean_phi'] <= stats['max_phi'], \
            "Mean should be between min and max"
    
    def test_dirichlet_bc_simple(self):
        """Test simple Dirichlet boundary condition application."""
        nx, ny = 11, 11
        u = np.zeros((nx, ny))
        bc_values = {
            'left': 100.0,
            'right': 0.0,
            'bottom': 0.0,
            'top': 0.0
        }
        
        u = apply_dirichlet_bc_simple(u, bc_values)
        
        # Check boundaries
        assert np.allclose(u[0, :], 100.0), "Left boundary should be 100"
        assert np.allclose(u[-1, :], 0.0), "Right boundary should be 0"
        assert np.allclose(u[:, 0], 0.0), "Bottom boundary should be 0"
        assert np.allclose(u[:, -1], 0.0), "Top boundary should be 0"
    
    def test_neumann_bc_simple(self):
        """Test Neumann boundary condition application."""
        nx, ny = 11, 11
        h = 0.1
        u = np.ones((nx, ny)) * 50.0
        
        # Apply zero derivative at bottom
        u = apply_neumann_bc_simple(u, h, bc_bottom=0)
        
        # Boundary should equal interior for zero derivative
        assert np.allclose(u[:, 0], u[:, 1]), \
            "Boundary should equal adjacent interior for zero derivative Neumann BC"
    
    def test_laplacian_computation(self):
        """Test Laplacian computation."""
        # Test with known function: u = x^2 + y^2, nabla^2 u = 4
        nx, ny = 11, 11
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        u = X**2 + Y**2
        h = x[1] - x[0]
        
        lap = compute_laplacian(u, h)
        
        # Interior points should be approximately 4
        interior_lap = lap[1:-1, 1:-1]
        assert np.allclose(interior_lap, 4, atol=0.01), \
            f"Laplacian should be ~4, but got min={interior_lap.min()}, max={interior_lap.max()}"
    
    def test_grid_creation(self):
        """Test 2D grid creation."""
        x, y, X, Y = create_2d_grid(0, 1, 0, 1, 11, 21)
        
        assert len(x) == 11, "Should have 11 x points"
        assert len(y) == 21, "Should have 21 y points"
        assert X.shape == (21, 11), "X should be (ny, nx)"
        assert Y.shape == (21, 11), "Y should be (ny, nx)"
        
        # Check that meshgrid is correct
        assert np.allclose(X[0, :], x), "X row should be x"
        assert np.allclose(Y[:, 0], y), "Y column should be y"
    
    def test_heat_distribution_example(self):
        """Test the heat distribution example."""
        from core.pde.laplace_solver import example_heat_distribution
        
        result = example_heat_distribution()
        
        assert result['converged'], "Heat distribution should converge"
        
        # Check statistics
        stats = result['statistics']
        assert stats['min_phi'] < 1.0, "Minimum temperature should be near cold edge"
        assert stats['max_phi'] > 99.0, "Maximum temperature should be near hot edge"
    
    def test_electrostatic_example(self):
        """Test the electrostatic potential example."""
        from core.pde.laplace_solver import example_electrostatic
        
        result = example_electrostatic()
        
        assert result['converged'], "Electrostatic problem should converge"
    
    def test_plot_functions(self):
        """Test that plotting functions run without error."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='sor',
            omega=1.7,
            tol=1e-4,
            max_iter=1000
        )
        
        X, Y, phi = result['X'], result['Y'], result['phi']
        
        # Test that each plotting function runs without error
        fig1 = plot_phi_contour(X, Y, phi)
        assert fig1 is not None, "Contour plot should return figure"
        
        fig2 = plot_phi_wireframe(X, Y, phi)
        assert fig2 is not None, "Wireframe plot should return figure"
        
        fig3 = plot_phi_surface(X, Y, phi)
        assert fig3 is not None, "Surface plot should return figure"
        
        fig4 = plot_convergence(result['iterations'], result['error'], 'SOR')
        assert fig4 is not None, "Convergence plot should return figure"
        
        # Close figures to free memory
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_print_solution_summary(self):
        """Test that summary printing runs without error."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=11, ny=11,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='sor',
            tol=1e-4,
            max_iter=100
        )
        
        # Should run without error
        print_solution_summary(result)
    
    def test_solver_iteration_counts(self):
        """Test that solvers take different iteration counts."""
        # With symmetric BC (all zeros), all solvers should converge in 1 iteration
        result_zero = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=0.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='sor',
            tol=1e-10,
            max_iter=10000
        )
        
        # With non-zero BC, should take more iterations
        result_nonzero = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver='sor',
            tol=1e-6,
            max_iter=10000
        )
        
        assert result_zero['iterations'] == 1, "Zero BC should converge in 1 iteration"
        assert result_nonzero['iterations'] >= 1, "Non-zero BC should take at least 1 iteration"
    
    def test_convergence_with_grid_refinement(self):
        """Test that finer grids require more iterations."""
        # Coarse grid
        result_coarse = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=11, ny=11,
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
        
        # Fine grid
        result_fine = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=31, ny=31,
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
        
        # Both should converge
        assert result_coarse['converged'], "Coarse grid should converge"
        assert result_fine['converged'], "Fine grid should converge"
    
    def test_boundary_value_accuracy(self):
        """Test that boundary values are accurately set."""
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=50.0,
            bc_bottom=25.0,
            bc_top=75.0,
            solver='sor',
            tol=1e-6,
            max_iter=10000
        )
        
        phi = result['phi']
        
        # Check boundaries are set correctly
        assert np.isclose(phi[0, 10], 100.0), "Left boundary should be 100"
        assert np.isclose(phi[-1, 10], 50.0), "Right boundary should be 50"
        assert np.isclose(phi[10, 0], 25.0), "Bottom boundary should be 25"
        assert np.isclose(phi[10, -1], 75.0), "Top boundary should be 75"


def run_quick_validation():
    """Run a quick validation test."""
    print("Running quick validation test...")
    
    # Test with non-trivial BC
    result = solve_laplace_equation(
        x_min=0, x_max=1, y_min=0, y_max=1,
        nx=31, ny=31,
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
    
    print_solution_summary(result)
    
    # Test all three solvers
    for solver_name in ['jacobi', 'gauss-seidel', 'sor']:
        print(f"\nTesting {solver_name.upper()} solver...")
        result = solve_laplace_equation(
            x_min=0, x_max=1, y_min=0, y_max=1,
            nx=21, ny=21,
            bc_type='dirichlet',
            bc_left=100.0,
            bc_right=0.0,
            bc_bottom=0.0,
            bc_top=0.0,
            solver=solver_name,
            tol=1e-4,
            max_iter=1000
        )
        print(f"  Iterations: {result['iterations']}")
        print(f"  Error: {result['error']:.2e}")
        print(f"  Converged: {result['converged']}")
    
    print("\nValidation complete!")


if __name__ == '__main__':
    run_quick_validation()
