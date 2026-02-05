import sys
import os

# Add current directory to Python path
sys.path.insert(0, 'c:/Users/vanguard/OneDrive/Documents/Analisis_numerik')

print("Testing imports...")

try:
    from core.differentiation import NumericalDifferentiation
    print("NumericalDifferentiation imported")
    
    from ui.sidebar import sidebar
    print("sidebar imported")
    
    from ui.input_form import input_form
    print("input_form imported")
    
    from ui.displays import display_differentiation_results
    print("display_differentiation_results imported")
    
    print("\nAll modules imported successfully!")
    
except ImportError as e:
    print(f"Import Error: {e}")

try:
    nd = NumericalDifferentiation('sin(x)')
    import numpy as np
    result = nd.forward_difference(0.785398, 0.01)  # π/4 in decimal
    print("\nFunctionality test successful:")
    print(f"  Forward Difference at 0.7854: {result['approximation']:.6f}")
    print(f"  True Value: {result['true_value']:.6f}")
    print(f"  Error: {result['absolute_error']:.2e}")
    
except Exception as e:
    print(f"\nFunctionality test failed: {e}")
