import numpy as np
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.linear_systems.gaussian_elimination import gaussian_elimination
from core.linear_systems.gauss_jordan import gauss_jordan_elimination


def test_gaussian_elimination_solves_system():
    A = np.array(
        [
            [2.0, -1.0, 1.0],
            [3.0, 3.0, 9.0],
            [3.0, 3.0, 5.0],
        ]
    )
    b = np.array([2.0, -1.0, 4.0])

    result = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)

    assert np.allclose(result["solution"], expected, atol=1e-10)
    assert np.linalg.norm(result["residual"], ord=np.inf) < 1e-10
    assert len(result["steps"]) > 0


def test_gauss_jordan_solves_system():
    A = np.array(
        [
            [2.0, -1.0, 1.0],
            [3.0, 3.0, 9.0],
            [3.0, 3.0, 5.0],
        ]
    )
    b = np.array([2.0, -1.0, 4.0])

    result = gauss_jordan_elimination(A, b)
    expected = np.linalg.solve(A, b)

    assert np.allclose(result["solution"], expected, atol=1e-10)
    assert np.linalg.norm(result["residual"], ord=np.inf) < 1e-10
    assert len(result["steps"]) > 0


@pytest.mark.parametrize("solver", [gaussian_elimination, gauss_jordan_elimination])
def test_linear_system_singular_raises_value_error(solver):
    A = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 5.0, 9.0],
        ]
    )
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        solver(A, b)
