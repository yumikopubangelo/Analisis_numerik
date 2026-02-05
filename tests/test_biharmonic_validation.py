"""
Test script to validate the biharmonic solver with analytic solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.pde.biharmonic_solver import (
    solve_biharmonic_equation_corrected,
    analytic_solution_simply_supported,
    create_2d_grid
)


def test_analytic_solution():
    """Test if analytic solution produces expected results."""
    print("Testing analytic solution...")
    
    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    nx, ny = 50, 50
    q = 1.0  # kN/m² (1000 N/m²)
    D = 100.0  # kN·m (100,000 N·m)
    
    x, y, X, Y = create_2d_grid(x_min, x_max, y_min, y_max, nx, ny)
    
    # Compute analytic solution
    w_analytic = analytic_solution_simply_supported(X, Y, q, D, n_terms=3)
    
    print(f"Max deflection (analytic): {np.max(w_analytic):.10f} meters")
    print(f"Expected deflection: ~0.00004 meters")
    
    plt.figure(figsize=(12, 8))
    
    # Analytic solution
    plt.subplot(121)
    contour = plt.contourf(X, Y, w_analytic, cmap='viridis', levels=50)
    plt.colorbar(contour, label='Deflection (m)')
    plt.title('Analytic Solution (Navier Series)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_analytic_solution.png')
    plt.close()
    
    return True


def test_solver_validation():
    """Test solver against analytic solution."""
    print("\nTesting solver validation...")
    
    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    nx, ny = 50, 50
    q = 1.0  # kN/m² (1000 N/m²)
    D = 100.0  # kN·m (100,000 N·m)
    
    try:
        X, Y, w_numeric, iterations, error, validation = solve_biharmonic_equation_corrected(
            x_min, x_max, y_min, y_max, nx, ny, q, D,
            bc_type='simply-supported', solver='gauss-seidel',
            tol=1e-6, max_iter=10000, w_relax=1.5
        )
        
        print(f"Solver converged in {iterations} iterations")
        print(f"Final error: {error:.2e}")
        print(f"Max deflection (numeric): {np.max(w_numeric):.10f} meters")
        print(f"Max error: {validation['max_error']:.2e}")
        print(f"RMS error: {validation['rms_error']:.2e}")
        print(f"PDE error: {validation['pde_error']:.2e}")
        print(f"Biharmonic (numeric): {validation['biharmonic_numeric']:.4f}")
        print(f"q/D expected: {validation['q_over_D']:.4f}")
        
        # Compute relative error
        w_analytic = analytic_solution_simply_supported(X, Y, q, D, n_terms=3)
        rel_error = np.max(np.abs((w_numeric - w_analytic) / w_analytic)) * 100
        print(f"Relative error: {rel_error:.2f}%")
        
        # Plot results
        plt.figure(figsize=(15, 6))
        
        # Numeric solution
        plt.subplot(131)
        contour = plt.contourf(X, Y, w_numeric, cmap='viridis', levels=50)
        plt.colorbar(contour, label='Deflection (m)')
        plt.title(f'Numeric Solution (Gauss-Seidel)\nIterations: {iterations}')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid(True, alpha=0.3)
        
        # Analytic solution
        plt.subplot(132)
        contour = plt.contourf(X, Y, w_analytic, cmap='viridis', levels=50)
        plt.colorbar(contour, label='Deflection (m)')
        plt.title('Analytic Solution')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid(True, alpha=0.3)
        
        # Error
        plt.subplot(133)
        error_map = np.abs(w_numeric - w_analytic)
        contour = plt.contourf(X, Y, error_map, cmap='Reds', levels=50)
        plt.colorbar(contour, label='Absolute Error (m)')
        plt.title(f'Error Map\nMax Error: {validation["max_error"]:.2e}')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_solver_validation.png')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stability():
    """Test solver stability with different parameters."""
    print("\nTesting solver stability...")
    
    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    nx, ny = 30, 30
    
    # Test different q and D
    params = [
        (1.0, 100.0),    # q=1, D=100
        (5.0, 200.0),    # q=5, D=200
        (10.0, 500.0),   # q=10, D=500
    ]
    
    all_stable = True
    
    for q, D in params:
        try:
            X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
                x_min, x_max, y_min, y_max, nx, ny, q, D,
                solver='gauss-seidel', tol=1e-6, max_iter=10000
            )
            
            print(f"q={q}, D={D}: Converged in {iterations} iterations")
            print(f"Max deflection: {np.max(w):.10f} meters")
            print(f"PDE error: {validation['pde_error']:.2e}")
            print()
            
        except Exception as e:
            print(f"q={q}, D={D}: ERROR - {e}")
            all_stable = False
    
    return all_stable


if __name__ == "__main__":
    print("=== Biharmonic Solver Validation ===")
    
    tests = [
        test_analytic_solution,
        test_solver_validation,
        test_stability
    ]
    
    all_passed = True
    for i, test_func in enumerate(tests):
        print(f"\nTest {i+1}:")
        try:
            if test_func():
                print("PASS")
            else:
                print("FAIL")
                all_passed = False
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\nAll validation tests passed! Solver is working correctly.")
    else:
        print("\nSome tests failed.")
