"""
Display modules untuk setiap kategori metode numerik.
Memisahkan display logic dari app.py untuk better modularity.
"""

from .root_finding_display import display_root_finding_results
from .integration_display import display_integration_results
from .interpolation_display import display_interpolation_results
from .series_display import display_taylor_results, display_ode_results
from .analysis_display import (
    display_true_value_results,
    display_error_analysis_results,
    display_tolerance_check_results,
    display_taylor_polynomial_results
)
from .differentiation_display import display_differentiation_results
from .pde_display import display_pde_results

__all__ = [
    'display_root_finding_results',
    'display_integration_results',
    'display_interpolation_results',
    'display_taylor_results',
    'display_ode_results',
    'display_true_value_results',
    'display_error_analysis_results',
    'display_tolerance_check_results',
    'display_taylor_polynomial_results',
    'display_differentiation_results',
]
