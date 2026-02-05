"""
Validation test for biharmonic solver
"""
import numpy as np
from core.pde.biharmonic_solver import (
    solve_biharmonic_equation,
    solve_biharmonic_equation_corrected,
    analytic_solution_simply_supported
)

print("=" * 60)
print("VALIDATION TEST - BIHARMONIC PLATE SOLVER")
print("=" * 60)

# Test 1: Using solve_biharmonic_equation (UI entry point)
print("\n### Test 1: solve_biharmonic_equation (UI entry point)")
X, Y, w1, iterations1, error1 = solve_biharmonic_equation(
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    nx=50, ny=50,
    q=10.0,  # kN/m²
    D=100.0,  # kN·m
    solver='gauss-seidel',
    tol=1e-10,
    max_iter=100000
)

print(f"  q = 10 kN/m², D = 100 kN·m")
print(f"  -> Converted: q = 10000 N/m², D = 100000 N·m")
print(f"  Max deflection: {np.max(w1)*1000:.4f} mm")
print(f"  Iterations: {iterations1}")
print(f"  Error: {error1:.2e}")

# Test 2: Using solve_biharmonic_equation_corrected directly
print("\n### Test 2: solve_biharmonic_equation_corrected (direct)")
X, Y, w2, iterations2, error2, validation = solve_biharmonic_equation_corrected(
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    nx=50, ny=50,
    q=10000,  # N/m²
    D=100000,  # N·m
    solver='gauss-seidel',
    tol=1e-10,
    max_iter=100000
)

print(f"  q = 10000 N/m², D = 100000 N·m")
print(f"  Max deflection: {np.max(w2)*1000:.4f} mm")
print(f"  Iterations: {iterations2}")
print(f"  Error: {error2:.2e}")
print(f"  PDE residual: {validation['pde_error']:.2e}")

# Test 3: Analytic solution
print("\n### Test 3: Analytic solution")
w_analytic = analytic_solution_simply_supported(X, Y, 10000, 100000, n_terms=10)
print(f"  Max deflection: {np.max(w_analytic)*1000:.4f} mm")

# Test 4: Compare with different D values
print("\n### Test 4: Checking deflection scaling with D")
D_values = [10000, 50000, 100000, 500000]
for D_test in D_values:
    X, Y, w_test, iters, err, val = solve_biharmonic_equation_corrected(
        q=10000, D=D_test, max_iter=5000
    )
    max_w = np.max(w_test) * 1000
    expected_w = (0.00406 * 10000 / D_test) * 1e3  # mm
    print(f"  D={D_test:6d} N·m: numeric={max_w:.4f}mm, expected={expected_w:.4f}mm, ratio={max_w/expected_w:.2f}")

# Test 5: Check grid convergence
print("\n### Test 5: Grid convergence")
for nx in [21, 41, 61]:
    X, Y, w_grid, iters, err, val = solve_biharmonic_equation_corrected(
        nx=nx, ny=nx, q=10000, D=100000, max_iter=20000
    )
    max_w = np.max(w_grid) * 1000
    print(f"  Grid {nx:2d}x{nx:2d}: max deflection = {max_w:.4f} mm, error = {err:.2e}")

print("\n" + "=" * 60)
