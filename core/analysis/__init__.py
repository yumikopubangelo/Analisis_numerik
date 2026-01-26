"""
Core analysis module untuk fitur-fitur analisis numerik.
"""

from .numerical_features import (
    NumericalAnalysis,
    create_error_table,
    format_error_output
)

__all__ = [
    'NumericalAnalysis',
    'create_error_table',
    'format_error_output'
]
