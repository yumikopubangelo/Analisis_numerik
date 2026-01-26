import numpy as np

def newton_divided_difference(x_points, y_points):
    """
    Calculate divided differences for Newton interpolation.
    
    Parameters:
    x_points: array of x coordinates
    y_points: array of y coordinates
    
    Returns:
    divided_diff_table: 2D array of divided differences
    """
    n = len(x_points)
    divided_diff_table = np.zeros((n, n))
    divided_diff_table[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff_table[i][j] = (
                (divided_diff_table[i+1][j-1] - divided_diff_table[i][j-1]) /
                (x_points[i+j] - x_points[i])
            )
    
    return divided_diff_table

def newton_interpolation(x_points, y_points, x_eval):
    """
    Newton polynomial interpolation using divided differences.
    
    Parameters:
    x_points: array of x coordinates
    y_points: array of y coordinates
    x_eval: point(s) to evaluate
    
    Returns:
    y_eval: interpolated value(s)
    divided_diff_table: table of divided differences
    polynomial_str: string representation
    """
    n = len(x_points)
    divided_diff_table = newton_divided_difference(x_points, y_points)
    
    # Build polynomial string
    terms = [f"{divided_diff_table[0][0]:.4f}"]
    for i in range(1, n):
        term_parts = [f"{divided_diff_table[0][i]:.4f}"]
        for j in range(i):
            term_parts.append(f"(x - {x_points[j]:.2f})")
        terms.append(" × ".join(term_parts))
    
    polynomial_str = " + ".join(terms)
    
    # Evaluate polynomial
    if np.isscalar(x_eval):
        y_eval = divided_diff_table[0][0]
        product = 1.0
        for i in range(1, n):
            product *= (x_eval - x_points[i-1])
            y_eval += divided_diff_table[0][i] * product
    else:
        y_eval = np.zeros_like(x_eval, dtype=float)
        for idx, x in enumerate(x_eval):
            result = divided_diff_table[0][0]
            product = 1.0
            for i in range(1, n):
                product *= (x - x_points[i-1])
                result += divided_diff_table[0][i] * product
            y_eval[idx] = result
    
    return y_eval, divided_diff_table, polynomial_str