import sys
import os

sys.path.insert(0, 'c:/Users/vanguard/OneDrive/Documents/Analisis_numerik')

print("Testing differentiation module...")

try:
    from core.differentiation import NumericalDifferentiation
    print("NumericalDifferentiation imported")
    
    nd = NumericalDifferentiation('sin(x)')
    x = 0.785398  # π/4
    
    # Test Forward Difference
    result1 = nd.forward_difference(x, 0.01)
    print(f"\nForward Difference:")
    print(f"  Approximation: {result1['approximation']:.6f}")
    print(f"  True Value: {result1['true_value']:.6f}")
    print(f"  Error: {result1['absolute_error']:.2e}")
    
    # Test Central Difference 1st
    result2 = nd.central_difference_first(x, 0.01)
    print(f"\nCentral Difference (1st):")
    print(f"  Approximation: {result2['approximation']:.6f}")
    print(f"  True Value: {result2['true_value']:.6f}")
    print(f"  Error: {result2['absolute_error']:.2e}")
    
    # Test Convergence Analysis
    print("\nConvergence Analysis:")
    import numpy as np
    h_values = np.logspace(-6, -1, 10)
    conv_result = nd.convergence_analysis(x, h_values, 1)
    print(f"  Tested {len(conv_result)} values of h")
    min_error = min([r['central_error'] for r in conv_result])
    print(f"  Minimum error: {min_error:.2e}")
    
    # Test Optimal h Analysis
    print("\nOptimal h Analysis:")
    optimal_result = nd.optimal_h_analysis(x, 1e-7, 0.1, 20)
    print(f"  Optimal h for Central 1st: {optimal_result['optimal']['central1']['h']:.2e}")
    print(f"  Minimum error: {optimal_result['optimal']['central1']['central1_error']:.2e}")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    print(traceback.format_exc())
