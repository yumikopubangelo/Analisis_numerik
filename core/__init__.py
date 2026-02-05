"""
Core module untuk analisis numerik yang berisi:
- Integration (integrasi numerik)
- Interpolation (interpolasi)
- Root Findings (pencarian akar)
- Series (deret)
- Analysis (analisis numerik)
- Differentiation (diferensiasi numerik)
"""

from . import integration
from . import interpolation
from . import root_findings
from . import series
from . import analysis
from . import differentiation
from . import errors
from . import utils

__all__ = [
    'integration',
    'interpolation', 
    'root_findings',
    'series',
    'analysis',
    'differentiation',
    'errors',
    'utils'
]
