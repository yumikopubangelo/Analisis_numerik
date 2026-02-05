#!/usr/bin/env python3
"""
Test script to verify the Taylor series implementation for cos(x)
Focusing on:
1. Correctness of signs (+/-) in terms
2. Proper factorial calculations
3. Ensuring cos(x) ≤ 1 for all x
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.series.taylor_series import taylor_series, taylor_cos_specific


def test_signs_and_factorials():
    """Test if terms have correct signs and factorial calculations."""
    print("=== Testing Signs and Factorials ===")
    
    # Test with cos(x) at x=0 (should have only even terms)
    n_terms = 6
    x = 0.1
    result_main = taylor_series('cos(x)', 0, n_terms, x_eval=x)
    
    print(f"Function: cos(x)")
    print(f"x={x}")
    print(f"Number of terms: {n_terms}")
    print()
    
    print(f"{'Term #':<6} {'Actual Order':<12} {'Coefficient':<15} {'Value':<15}")
    print("-" * 50)
    
    for term in result_main['terms']:
        term_num = term['n']
        actual_order = term['actual_order']
        coeff = term['coefficient']
        term_str = term['term']
        
        # Check expected sign and factorial
        expected_sign = (-1)**(actual_order // 2)
        expected_factorial = np.math.factorial(actual_order)
        expected_coeff = expected_sign / expected_factorial
        
        print(f"{term_num:<6} {actual_order:<12} {coeff:<15.10f} {expected_coeff:<15.10f}")
        
        # Verify correctness
        assert abs(coeff - expected_coeff) < 1e-10, f"Coefficient mismatch for term {term_num}"


def test_cos_values_leq_1():
    """Test if cos(x) approximations are ≤ 1 in magnitude."""
    print("\n=== Testing cos(x) Values <= 1 ===")
    
    x_values = [0, 0.5, 1.0, 1.5, np.pi/2, np.pi]
    n_terms = 10
    
    for x in x_values:
        true_val = np.cos(x)
        result_main = taylor_series('cos(x)', 0, n_terms, x_eval=x)
        approx = result_main['approximations'][-1]['Nilai Aproksimasi']
        
        print(f"x={x:.4f}:")
        print(f"  True value:     {true_val:.10f}")
        print(f"  Approximation:  {approx:.10f}")
        print(f"  Error:          {abs(approx - true_val):.10e}")
        
        # Check if approximation is within [-1, 1]
        assert abs(approx) <= 1 + 1e-8, f"Approximation {approx:.10f} exceeds [-1, 1] for x={x:.4f}"


def test_against_specific_function():
    """Test that both Taylor implementations produce identical results."""
    print("\n=== Testing Consistency Between Implementations ===")
    
    x_values = [0, 0.5, 1.0, np.pi/2, np.pi]
    n_terms = 5
    
    for x in x_values:
        result_main = taylor_series('cos(x)', 0, n_terms, x_eval=x)
        result_specific = taylor_cos_specific(x_eval=x, n_terms=n_terms)
        
        main_approx = result_main['approximations'][-1]['Nilai Aproksimasi']
        specific_approx = result_specific['approximations'][-1]['Nilai Aproksimasi']
        
        print(f"x={x:.4f}:")
        print(f"  Taylor Series:  {main_approx:.10f}")
        print(f"  Specific Func:  {specific_approx:.10f}")
        print(f"  Difference:     {abs(main_approx - specific_approx):.10e}")
        
        assert abs(main_approx - specific_approx) < 1e-10, f"Results mismatch for x={x:.4f}"


if __name__ == "__main__":
    print("Testing Taylor Series Implementation for cos(x)")
    print("=" * 50)
    
    try:
        test_signs_and_factorials()
        test_cos_values_leq_1()
        test_against_specific_function()
        
        print("\n" + "=" * 50)
        print("All tests passed! Taylor series implementation is correct.")
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
