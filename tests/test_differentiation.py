"""
Unit tests for numerical differentiation methods.
Tests include:
- Forward Difference
- Backward Difference  
- Central Difference (1st derivative)
- Central Difference (2nd derivative)
"""

import unittest
import numpy as np
from core.differentiation import NumericalDifferentiation
from core.differentiation import (
    forward_difference,
    backward_difference,
    central_difference_first,
    central_difference_second
)


class TestNumericalDifferentiation(unittest.TestCase):
    """Tests for the NumericalDifferentiation class."""
    
    def test_quadratic_function_forward_difference(self):
        """Test forward difference on f(x) = x² at x=2."""
        nd = NumericalDifferentiation("x**2")
        result = nd.forward_difference(2, h=0.1)
        
        self.assertAlmostEqual(result['approximation'], 4.1, places=2)
        self.assertAlmostEqual(result['true_value'], 4.0, places=8)
        self.assertLess(result['absolute_error'], 0.11)
    
    def test_quadratic_function_backward_difference(self):
        """Test backward difference on f(x) = x² at x=2."""
        nd = NumericalDifferentiation("x**2")
        result = nd.backward_difference(2, h=0.1)
        
        self.assertAlmostEqual(result['approximation'], 3.9, places=2)
        self.assertAlmostEqual(result['true_value'], 4.0, places=8)
        self.assertLess(result['absolute_error'], 0.11)
    
    def test_quadratic_function_central_difference_first(self):
        """Test central difference (1st derivative) on f(x) = x² at x=2."""
        nd = NumericalDifferentiation("x**2")
        result = nd.central_difference_first(2, h=0.1)
        
        self.assertAlmostEqual(result['approximation'], 4.0, places=8)
        self.assertAlmostEqual(result['true_value'], 4.0, places=8)
        self.assertLess(result['absolute_error'], 1e-10)
    
    def test_quadratic_function_central_difference_second(self):
        """Test central difference (2nd derivative) on f(x) = x² at x=2."""
        nd = NumericalDifferentiation("x**2")
        result = nd.central_difference_second(2, h=0.1)
        
        self.assertAlmostEqual(result['approximation'], 2.0, places=8)
        self.assertAlmostEqual(result['true_value'], 2.0, places=8)
        self.assertLess(result['absolute_error'], 1e-10)
    
    def test_sine_function_forward_difference(self):
        """Test forward difference on f(x) = sin(x) at x=π/2."""
        nd = NumericalDifferentiation("sin(x)")
        x_val = np.pi / 2
        result = nd.forward_difference(x_val, h=0.01)
        
        true_value = 0.0
        self.assertAlmostEqual(result['true_value'], true_value, places=8)
        self.assertLess(result['absolute_error'], 0.01)
    
    def test_sine_function_backward_difference(self):
        """Test backward difference on f(x) = sin(x) at x=π/2."""
        nd = NumericalDifferentiation("sin(x)")
        x_val = np.pi / 2
        result = nd.backward_difference(x_val, h=0.01)
        
        true_value = 0.0
        self.assertAlmostEqual(result['true_value'], true_value, places=8)
        self.assertLess(result['absolute_error'], 0.01)
    
    def test_sine_function_central_difference_first(self):
        """Test central difference (1st derivative) on f(x) = sin(x) at x=π/2."""
        nd = NumericalDifferentiation("sin(x)")
        x_val = np.pi / 2
        result = nd.central_difference_first(x_val, h=0.01)
        
        true_value = 0.0
        self.assertAlmostEqual(result['true_value'], true_value, places=8)
        self.assertLess(result['absolute_error'], 1e-4)
    
    def test_sine_function_central_difference_second(self):
        """Test central difference (2nd derivative) on f(x) = sin(x) at x=π/2."""
        nd = NumericalDifferentiation("sin(x)")
        x_val = np.pi / 2
        result = nd.central_difference_second(x_val, h=0.01)
        
        true_value = -1.0
        self.assertAlmostEqual(result['true_value'], true_value, places=8)
        self.assertLess(result['absolute_error'], 1e-4)
    
    def test_exponential_function(self):
        """Test all methods on f(x) = e^x at x=0 (where f'(x) = e^x = 1)."""
        nd = NumericalDifferentiation("exp(x)")
        x_val = 0.0
        h = 0.001
        
        forward_result = nd.forward_difference(x_val, h)
        backward_result = nd.backward_difference(x_val, h)
        central_result = nd.central_difference_first(x_val, h)
        
        true_value = 1.0
        self.assertAlmostEqual(forward_result['true_value'], true_value, places=8)
        self.assertAlmostEqual(backward_result['true_value'], true_value, places=8)
        self.assertAlmostEqual(central_result['true_value'], true_value, places=8)
        
        # Central difference should be most accurate
        self.assertLess(central_result['absolute_error'], forward_result['absolute_error'])
        self.assertLess(central_result['absolute_error'], backward_result['absolute_error'])
    
    def test_convergence_analysis(self):
        """Test convergence analysis with decreasing h values."""
        nd = NumericalDifferentiation("x**3")
        x_val = 1.0
        h_values = [0.1, 0.01, 0.001, 0.0001]
        
        results = nd.convergence_analysis(x_val, h_values, derivative_order=1)
        
        # As h decreases, error should decrease for all methods
        for i in range(1, len(results)):
            self.assertLess(results[i]['forward_error'], results[i-1]['forward_error'])
            self.assertLess(results[i]['backward_error'], results[i-1]['backward_error'])
            self.assertLess(results[i]['central_error'], results[i-1]['central_error'])
    
    def test_compare_methods(self):
        """Test method comparison functionality."""
        nd = NumericalDifferentiation("sin(x)")
        x_val = np.pi / 4
        h = 0.01
        
        results = nd.compare_first_derivative_methods(x_val, h)
        
        self.assertEqual(len(results), 3)
        method_names = [result['method'] for result in results]
        self.assertIn('Forward Difference', method_names)
        self.assertIn('Backward Difference', method_names)
        self.assertIn('Central Difference (1st Derivative)', method_names)


class TestDifferentiationUtils(unittest.TestCase):
    """Tests for standalone differentiation utility functions."""
    
    def test_forward_difference_function(self):
        """Test standalone forward difference function."""
        def f(x):
            return x ** 2
        
        result = forward_difference(f, 2, 0.1)
        self.assertAlmostEqual(result, 4.1, places=2)
    
    def test_backward_difference_function(self):
        """Test standalone backward difference function."""
        def f(x):
            return x ** 2
        
        result = backward_difference(f, 2, 0.1)
        self.assertAlmostEqual(result, 3.9, places=2)
    
    def test_central_difference_first_function(self):
        """Test standalone central difference (1st derivative) function."""
        def f(x):
            return x ** 2
        
        result = central_difference_first(f, 2, 0.1)
        self.assertAlmostEqual(result, 4.0, places=8)
    
    def test_central_difference_second_function(self):
        """Test standalone central difference (2nd derivative) function."""
        def f(x):
            return x ** 2
        
        result = central_difference_second(f, 2, 0.1)
        self.assertAlmostEqual(result, 2.0, places=8)


if __name__ == '__main__':
    print("Running numerical differentiation tests...")
    unittest.main()
