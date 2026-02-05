#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from core.pde import solve_advection_diffusion_1d


def test_advection_diffusion_solver():
    """Test the 1D advection-diffusion solver with simple parameters."""
    print("Testing Advection-Diffusion 1D Solver...")
    
    # Test parameters (from user example)
    params = {
        'U': 1.0,
        'K': 0.1,
        'x_min': 0.0,
        'x_max': 1.0,
        'nx': 50,
        'dt': 0.001,
        'num_steps': 500
    }
    
    try:
        # Solve the equation
        result = solve_advection_diffusion_1d(**params)
        
        print("✅ Solver succeeded!")
        print(f"  Grid points: {result['parameters']['nx']}")
        print(f"  Time steps: {result['parameters']['num_steps']}")
        print(f"  Grid spacing: {result['parameters']['dx']:.4f}")
        print(f"  Time step: {result['parameters']['dt']:.4f}")
        
        # Check concentration array
        assert 'x' in result
        assert 'C' in result
        assert 't' in result
        assert len(result['x']) == params['nx']
        assert result['C'].shape[0] == params['num_steps'] + 1
        assert result['C'].shape[1] == params['nx']
        
        print("✅ All arrays have correct dimensions")
        
        # Check concentration values are reasonable
        max_concentration = np.max(result['C'])
        min_concentration = np.min(result['C'])
        
        print(f"  Max concentration: {max_concentration:.4f}")
        print(f"  Min concentration: {min_concentration:.4f}")
        
        # Initial condition should be a Gaussian pulse around x=0.5
        initial_concentration = result['C'][0]
        max_initial = np.max(initial_concentration)
        max_index = np.argmax(initial_concentration)
        x_at_max = result['x'][max_index]
        
        assert np.abs(x_at_max - 0.5) < 0.1
        assert max_initial > 0.9
        
        print("✅ Initial condition is correctly centered")
        
        # Concentration should be positive everywhere
        assert np.all(result['C'] >= 0)
        
        print("✅ All concentrations are non-negative")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_advection_diffusion_solver()
