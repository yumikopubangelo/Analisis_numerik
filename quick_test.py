"""
Quick validation test - Using correct SI units
"""
import numpy as np
from core.pde.biharmonic_solver import (
    solve_biharmonic_equation,  # UI entry point (converts kN to N)
    analytic_solution_simply_supported
)

print("Testing with UI entry point (converts kN to N automatically)")
print("Input: q=10 kN/m^2, D=100 kN.m")
print("Expected: q=10000 N/m^2, D=100000 N.m")
print()

# Use the UI entry point which handles unit conversion
# Parameters tuned for reasonable speed with accurate results
X, Y, w, iterations, error = solve_biharmonic_equation(
    q=10.0,  # kN/m^2
    D=100.0,  # kN.m
    nx=21, ny=21,  # Coarser grid for faster testing
    max_iter=2000,  # Reasonable number of iterations
    tol=1e-3,  # Looser tolerance since we just need accurate deflection
    w_relax=0.8  # Under-relaxation for stability
)

print(f"\nMax deflection: {np.max(w)*1000:.4f} mm")
print(f"Iterations: {iterations}")
print(f"Convergence error: {error:.2e}")
print()

# Analytic solution for comparison
w_analytic = analytic_solution_simply_supported(X, Y, 10000, 100000, n_terms=5)
print(f"Analytic max deflection: {np.max(w_analytic)*1000:.4f} mm")
print()

# Theoretical deflection
alpha = 0.00406
q = 10000  # N/m^2
D = 100000  # N.m
L = 1.0  # m
w_theoretical = alpha * q * L**4 / D
print(f"Theoretical deflection: {w_theoretical*1000:.4f} mm")

# Validation
rel_error = abs(np.max(w) - w_theoretical) / w_theoretical * 100
print(f"\nRelative error vs theoretical: {rel_error:.2f}%")
if rel_error < 5:
    print("PASSED - Test results are within acceptable error tolerance")
else:
    print("FAILED - Test results exceed acceptable error tolerance")
