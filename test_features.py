"""
Test file untuk memverifikasi 4 fitur analisis numerik
"""

import sys
import io
import numpy as np
from core.analysis.numerical_features import NumericalAnalysis

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_fitur_1():
    """Test Fitur 1: Nilai sebenarnya f(x)"""
    print("Testing Fitur 1: Nilai sebenarnya f(x)")
    
    na = NumericalAnalysis("sin(x)")
    result = na.true_value(np.pi/2)
    assert abs(result - 1.0) < 1e-10, "sin(π/2) should be 1.0"
    print("✓ Test 1 passed: sin(π/2) = 1.0")
    
    na2 = NumericalAnalysis("x**2")
    result2 = na2.true_value(3.0)
    assert abs(result2 - 9.0) < 1e-10, "3² should be 9.0"
    print("✓ Test 2 passed: 3² = 9.0")
    
    print()


def test_fitur_2():
    """Test Fitur 2: Error absolut / relatif"""
    print("Testing Fitur 2: Error absolut / relatif")
    
    na = NumericalAnalysis("pi")
    
    # Test absolute error
    abs_err = na.absolute_error(3.14, np.pi)
    assert abs_err > 0, "Absolute error should be positive"
    print(f"✓ Test 1 passed: Absolute error = {abs_err:.6e}")
    
    # Test relative error
    rel_err = na.relative_error(3.14, np.pi)
    assert 0 < rel_err < 1, "Relative error should be between 0 and 1"
    print(f"✓ Test 2 passed: Relative error = {rel_err:.6e}")
    
    # Test error analysis
    error_result = na.error_analysis(3.14, 1.0)
    assert 'absolute_error' in error_result, "Should contain absolute_error"
    assert 'relative_error' in error_result, "Should contain relative_error"
    print("✓ Test 3 passed: Error analysis complete")
    
    print()


def test_fitur_3():
    """Test Fitur 3: Toleransi error"""
    print("Testing Fitur 3: Toleransi error")
    
    na = NumericalAnalysis("sqrt(2)")
    true_val = np.sqrt(2)
    
    # Test tolerance check - should converge
    result1 = na.check_tolerance(1.4142, true_val, 0.001, 'absolute')
    assert result1['converged'] == True, "Should converge with tolerance 0.001"
    print("✓ Test 1 passed: Tolerance check (converged)")
    
    # Test tolerance check - should not converge
    result2 = na.check_tolerance(1.4, true_val, 0.001, 'absolute')
    assert result2['converged'] == False, "Should not converge with tolerance 0.001"
    print("✓ Test 2 passed: Tolerance check (not converged)")
    
    # Test iterative tolerance
    result3 = na.iterative_tolerance_check(1.4142, 1.414, 0.001)
    assert 'converged' in result3, "Should contain converged status"
    print("✓ Test 3 passed: Iterative tolerance check")
    
    # Test adaptive tolerance
    result4 = na.adaptive_tolerance(1.414, true_val, 3)
    assert 'achieved_digits' in result4, "Should contain achieved_digits"
    print("✓ Test 4 passed: Adaptive tolerance")
    
    print()


def test_fitur_4():
    """Test Fitur 4: Bentuk polinom Taylor"""
    print("Testing Fitur 4: Bentuk polinom Taylor")
    
    # Test Taylor polynomial
    na = NumericalAnalysis("exp(x)")
    taylor_result = na.taylor_polynomial(0, 5)
    
    assert 'polynomial_str' in taylor_result, "Should contain polynomial_str"
    assert 'coefficients' in taylor_result, "Should contain coefficients"
    assert len(taylor_result['coefficients']) == 5, "Should have 5 coefficients"
    print("✓ Test 1 passed: Taylor polynomial generation")
    
    # Test Taylor approximation
    approx_result = na.taylor_approximation(0, 5, 1.0)
    assert 'approximation' in approx_result, "Should contain approximation"
    assert 'error_analysis' in approx_result, "Should contain error_analysis"
    print("✓ Test 2 passed: Taylor approximation")
    
    # Test Taylor convergence
    convergence = na.taylor_convergence(0, 5, 1.0)
    assert len(convergence) == 5, "Should have 5 convergence results"
    assert convergence[-1]['absolute_error'] < convergence[0]['absolute_error'], \
        "Error should decrease with more terms"
    print("✓ Test 3 passed: Taylor convergence")
    
    # Test Taylor remainder
    remainder = na.taylor_remainder(0, 5, 1.0)
    assert 'actual_remainder' in remainder, "Should contain actual_remainder"
    print("✓ Test 4 passed: Taylor remainder")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("RUNNING TESTS FOR 4 FITUR ANALISIS NUMERIK")
    print("=" * 70)
    print()
    
    try:
        test_fitur_1()
        test_fitur_2()
        test_fitur_3()
        test_fitur_4()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
