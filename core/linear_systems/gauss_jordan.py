import numpy as np

from .common import validate_linear_system


def gauss_jordan_elimination(A, b, tol=1e-12):
    """
    Solve Ax = b with Gauss-Jordan elimination + partial pivoting.

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
            "final_augmented": np.ndarray,
            "residual": np.ndarray,
            "method": str
        }
    """
    A, b = validate_linear_system(A, b)
    n = A.shape[0]
    aug = np.hstack((A.copy(), b.reshape(-1, 1)))
    steps = []

    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        pivot_value = aug[pivot_row, col]

        if abs(pivot_value) < tol:
            raise ValueError("Matrix is singular or nearly singular; no unique solution.")

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
            steps.append(
                {
                    "type": "swap",
                    "description": f"Swap R{col + 1} <-> R{pivot_row + 1}",
                    "pivot_column": col + 1,
                    "matrix": aug.copy(),
                }
            )

        pivot = aug[col, col]
        aug[col, :] = aug[col, :] / pivot
        steps.append(
            {
                "type": "normalize",
                "description": f"R{col + 1} = R{col + 1} / {pivot:.6g}",
                "pivot_column": col + 1,
                "matrix": aug.copy(),
            }
        )

        for row in range(n):
            if row == col:
                continue

            factor = aug[row, col]
            if abs(factor) < tol:
                continue

            aug[row, :] = aug[row, :] - factor * aug[col, :]
            aug[row, col] = 0.0
            steps.append(
                {
                    "type": "eliminate",
                    "description": f"R{row + 1} = R{row + 1} - ({factor:.6g})R{col + 1}",
                    "pivot_column": col + 1,
                    "target_row": row + 1,
                    "factor": float(factor),
                    "matrix": aug.copy(),
                }
            )

    x = aug[:, -1].copy()
    residual = A @ x - b

    return {
        "solution": x,
        "steps": steps,
        "final_augmented": aug.copy(),
        "residual": residual,
        "method": "Gauss-Jordan",
    }
