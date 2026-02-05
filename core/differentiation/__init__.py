"""
Core differentiation module untuk metode-metode diferensiasi numerik:
1. Selisih Maju (Forward Difference) untuk f'(x)
2. Selisih Mundur (Backward Difference) untuk f'(x)  
3. Selisih Pusat (Central Difference) untuk f'(x) dan f''(x)
"""

from .numerical_differentiation import (
    NumericalDifferentiation,
    forward_difference,
    backward_difference,
    central_difference_first,
    central_difference_second,
    create_differentiation_table
)

__all__ = [
    'NumericalDifferentiation',
    'forward_difference',
    'backward_difference',
    'central_difference_first',
    'central_difference_second',
    'create_differentiation_table'
]
