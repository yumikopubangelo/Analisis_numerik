#!/usr/bin/env python3
"""
Test script for ODE y' = x/y, y(0) = 1
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.series.taylor_series import taylor_series
import numpy as np


def test_ode_x_over_y():
    """Test ODE y' = x/y, y(0) = 1 at x=1.0"""
    print("=== Testing ODE y' = x/y, y(0) = 1 ===\n")
    
    # Use Taylor series method with more terms for better accuracy
    result = taylor_series(
        func_str='x/y', 
        x0=0, 
        n_terms=10, 
        x_eval=1.0, 
        y0=1, 
        is_ode=True
    )
    
    # Print results
    print("=== Taylor Series Results ===")
    print(f"Series expression: {result['series_expr']}")
    print(f"Number of terms: {len(result['terms'])}")
    
    print("\n=== Terms ===")
    for i, term in enumerate(result['terms']):
        print(f"  Term {i}: Coefficient = {term['coefficient']:.6f}, Term = {term['term']}")
    
    print("\n=== Approximations ===")
    for approx in result['approximations']:
        print(f"  Terms used: {approx['Jumlah Suku']}")
        print(f"  Approximation: {approx['Nilai Aproksimasi']:.6f}")
        
        if approx['Nilai Sebenarnya'] is not None:
            print(f"  Actual: {approx['Nilai Sebenarnya']:.6f}")
            print(f"  Absolute error: {approx['Error Absolut']:.6e}")
            print(f"  Relative error: {approx['Error Relatif']:.4%}")
        else:
            print("  Actual value: Not available (analytical solution not known)")
        print()
    
    # Verify with known analytical solution
    print("=== Verification ===")
    known_solution = np.sqrt(1**2 + 1)
    print(f"Known analytical solution at x=1.0: y = sqrt(x² + 1) = {known_solution:.6f}")
    
    # Get the final approximation
    final_approx = result['approximations'][-1]['Nilai Aproksimasi']
    final_error = abs(final_approx - known_solution)
    print(f"Final approximation: {final_approx:.6f}")
    print(f"Final error: {final_error:.6e}")
    
    return final_error < 1e-4  # Should be very accurate


if __name__ == "__main__":
    print("Testing ODE y' = x/y with Taylor series")
    print("=" * 50)
    
    try:
        success = test_ode_x_over_y()
        
        if success:
            print("\n✅ Test PASSED")
            print("Taylor series implementation for this ODE is correct.")
        else:
            print("\n❌ Test FAILED")
            print("Taylor series approximation is not accurate enough.")
            
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
