import numpy as np

from .common import validate_linear_system


def gaussian_elimination(A, b, tol=1e-12):
    """
    Solve Ax = b with Gaussian elimination + partial pivoting.

    Parameters
    ----------
    A : array-like
        Coefficient matrix (n x n)
    b : array-like
        Right-hand side vector (n,)
    tol : float
        Near-zero pivot threshold

    Returns
    -------
    dict
        {
            "solution": np.ndarray,
            "steps": list[dict],
            "back_substitution": list[dict],
            "final_augmented": np.ndarray,
            "residual": np.ndarray,
            "method": str
        }
    """
    A, b = validate_linear_system(A, b)
    n = A.shape[0]
    aug = np.hstack((A.copy(), b.reshape(-1, 1)))
    steps = []

    for k in range(n - 1):
        pivot_row = k + int(np.argmax(np.abs(aug[k:, k])))
        pivot_value = aug[pivot_row, k]

        if abs(pivot_value) < tol:
            raise ValueError("Matrix is singular or nearly singular; no unique solution.")

        if pivot_row != k:
            aug[[k, pivot_row]] = aug[[pivot_row, k]]
            steps.append(
                {
                    "type": "swap",
                    "description": f"Swap R{k + 1} <-> R{pivot_row + 1}",
                    "pivot_column": k + 1,
                    "matrix": aug.copy(),
                }
            )

        for i in range(k + 1, n):
            factor = aug[i, k] / aug[k, k]
            aug[i, k:] = aug[i, k:] - factor * aug[k, k:]
            aug[i, k] = 0.0
            steps.append(
                {
                    "type": "eliminate",
                    "description": f"R{i + 1} = R{i + 1} - ({factor:.6g})R{k + 1}",
                    "pivot_column": k + 1,
                    "target_row": i + 1,
                    "factor": float(factor),
                    "matrix": aug.copy(),
                }
            )

    for i in range(n):
        if abs(aug[i, i]) < tol:
            raise ValueError("Matrix is singular or nearly singular; no unique solution.")

    x = np.zeros(n, dtype=float)
    back_substitution = []

    for i in range(n - 1, -1, -1):
        rhs = aug[i, -1] - np.dot(aug[i, i + 1 : n], x[i + 1 : n])
        x[i] = rhs / aug[i, i]
        back_substitution.append(
            {
                "row": i + 1,
                "description": f"x{i + 1} = {x[i]:.10g}",
                "value": float(x[i]),
            }
        )

    residual = A @ x - b

    return {
        "solution": x,
        "steps": steps,
        "back_substitution": list(reversed(back_substitution)),
        "final_augmented": aug.copy(),
        "residual": residual,
        "method": "Gaussian Elimination",
    }
