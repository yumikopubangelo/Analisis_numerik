import numpy as np

def lagrange_interpolation(x_points, y_points, x_eval):
    """
    Lagrange polynomial interpolation.
    
    Parameters:
    x_points: array of x coordinates
    y_points: array of y coordinates (function values)
    x_eval: point(s) to evaluate the interpolating polynomial
    
    Returns:
    y_eval: interpolated value(s)
    steps: calculation steps for understanding
    polynomial_str: string representation of the polynomial
    """
    n = len(x_points)
    steps = []
    
    # Calculate Lagrange basis polynomials
    def L(i, x):
        """Lagrange basis polynomial L_i(x)"""
        result = 1.0
        for j in range(n):
            if j != i:
                result *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return result
    
    # Build polynomial string representation
    polynomial_terms = []
    for i in range(n):
        # Build L_i(x) string
        numerator_terms = []
        denominator_terms = []
        for j in range(n):
            if j != i:
                numerator_terms.append(f"(x - {x_points[j]:.2f})")
                denominator_terms.append(f"({x_points[i]:.2f} - {x_points[j]:.2f})")
        
        L_str = " × ".join(numerator_terms) + " / [" + " × ".join(denominator_terms) + "]"
        polynomial_terms.append(f"{y_points[i]:.4f} × [{L_str}]")
    
    polynomial_str = " + ".join(polynomial_terms)
    
    # Evaluate at x_eval
    if np.isscalar(x_eval):
        y_eval = sum(y_points[i] * L(i, x_eval) for i in range(n))
        
        # Store calculation steps
        for i in range(n):
            L_i = L(i, x_eval)
            steps.append({
                'i': i,
                'x_i': x_points[i],
                'y_i': y_points[i],
                'L_i(x)': L_i,
                'y_i × L_i(x)': y_points[i] * L_i
            })
    else:
        y_eval = np.array([sum(y_points[i] * L(i, x) for i in range(n)) for x in x_eval])
    
    return y_eval, steps, polynomial_str