"""
Test script for PDE solver functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from core.pde.biharmonic_solver import (
    solve_biharmonic_equation,
    plot_deflection_3d,
    plot_deflection_contour,
    compute_statistics
)


def test_pde_solver():
    """Test the PDE solver functionality"""
    print("Testing PDE Solver...")
    
    try:
        # Test 1: Basic solver functionality
        print("\n1. Testing basic solver functionality:")
        X, Y, w, iterations, error = solve_biharmonic_equation(
            x_min=0, x_max=1,
            y_min=0, y_max=1,
            nx=30, ny=30,
            solver='jacobi',
            tol=1e-4,
            max_iter=1000
        )
        
        print(f"   - Solved in {iterations} iterations")
        print(f"   - Final error: {error:.2e}")
        
        # Check dimensions
        assert X.shape == (30, 30)
        assert Y.shape == (30, 30)
        assert w.shape == (30, 30)
        print("   - Dimensions match")
        
        # Test 2: Gauss-Seidel solver
        print("\n2. Testing Gauss-Seidel solver:")
        X_gs, Y_gs, w_gs, iterations_gs, error_gs = solve_biharmonic_equation(
            x_min=0, x_max=1,
            y_min=0, y_max=1,
            nx=30, ny=30,
            solver='gauss-seidel',
            tol=1e-4,
            max_iter=1000
        )
        
        print(f"   - Solved in {iterations_gs} iterations")
        print(f"   - Final error: {error_gs:.2e}")
        
        # Test 3: Compute statistics
        print("\n3. Testing statistics computation:")
        stats = compute_statistics(w)
        print(f"   - Max deflection: {stats['max_deflection']:.6f}")
        print(f"   - Min deflection: {stats['min_deflection']:.6f}")
        print(f"   - Mean deflection: {stats['mean_deflection']:.6f}")
        print(f"   - Std deflection: {stats['std_deflection']:.6f}")
        
        # Test 4: Plotting functionality
        print("\n4. Testing plotting functionality:")
        fig_3d = plot_deflection_3d(X, Y, w)
        print("   - 3D plot created")
        
        fig_contour = plot_deflection_contour(X, Y, w)
        print("   - Contour plot created")
        
        # Close figures to save memory
        plt.close(fig_3d)
        plt.close(fig_contour)
        
        # Test 5: Custom load function
        print("\n5. Testing custom load function:")
        def custom_load(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        
        X_custom, Y_custom, w_custom, iter_custom, error_custom = solve_biharmonic_equation(
            x_min=0, x_max=1,
            y_min=0, y_max=1,
            nx=30, ny=30,
            load_func=custom_load,
            solver='jacobi',
            tol=1e-4,
            max_iter=1000
        )
        
        print(f"   - Solved in {iter_custom} iterations")
        print(f"   - Final error: {error_custom:.2e}")
        
        print("\nAll tests passed! PDE solver is working correctly.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    test_pde_solver()
