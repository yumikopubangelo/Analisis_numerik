#!/usr/bin/env python3
"""Test to verify biharmonic solver iterations"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.pde.biharmonic_solver import solve_biharmonic_equation_corrected
from core.pde.biharmonic_solver import gauss_seidel_solver_biharmonic, jacobi_solver_biharmonic


def test_iteration_count():
    print("=== Testing Solver Iteration Count ===\n")
    
    # Test Gauss-Seidel
    print("1. Gauss-Seidel Solver:")
    X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
        nx=20, ny=20,
        q=500000, D=5000,
        bc_type='simply-supported', solver='gauss-seidel',
        tol=1e-8, max_iter=10000
    )
    
    print(f"   Iterations: {iterations}")
    print(f"   Error: {error:.2e}")
    print(f"   Max Deflection: {w.max():.4f}")
    print()
    
    # Test Jacobi
    print("2. Jacobi Solver:")
    X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
        nx=20, ny=20,
        q=500000, D=5000,
        bc_type='simply-supported', solver='jacobi',
        tol=1e-8, max_iter=10000
    )
    
    print(f"   Iterations: {iterations}")
    print(f"   Error: {error:.2e}")
    print(f"   Max Deflection: {w.max():.4f}")
    print()
    
    # Test with different tolerances
    print("3. Tolerance Effect:")
    print("   Testing with varying tolerance values:")
    
    for tol in [1e-3, 1e-5, 1e-7]:
        X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
            nx=20, ny=20,
            q=500000, D=5000,
            bc_type='simply-supported', solver='gauss-seidel',
            tol=tol, max_iter=10000
        )
        print(f"   Tol: {tol:.0e} | Iterations: {iterations:4d} | Error: {error:.2e}")
    
    print()
    
    # Check if solutions are reasonable
    print("4. Solution Validation:")
    X, Y, w, iterations, error, validation = solve_biharmonic_equation_corrected(
        nx=40, ny=40,
        q=500000, D=5000,
        bc_type='simply-supported', solver='gauss-seidel',
        tol=1e-8, max_iter=10000
    )
    
    print(f"   Max Deflection: {w.max():.4f}")
    print(f"   Validation Error: {validation['max_error']:.2e}")
    print(f"   PDE Satisfaction: {abs(validation['pde_error']) < 1e-2}")
    print()


if __name__ == "__main__":
    test_iteration_count()
