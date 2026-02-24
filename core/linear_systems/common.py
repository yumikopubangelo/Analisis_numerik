import numpy as np


def validate_linear_system(A, b):
    """
    Validate and normalize Ax = b input.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A matrix (n x n) and b vector (n,)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("A must be a square matrix (n x n).")
    if b.shape[0] != rows:
        raise ValueError("Vector b length must match matrix A size.")

    return A, b
