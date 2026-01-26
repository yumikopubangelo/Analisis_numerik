import numpy as np
from core.errors.error_analysis import absolute_error, relative_error

def secant_method(func, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method for finding root of f(x) = 0.
    
    Parameters:
    func: function to find root of
    x0, x1: two initial guesses
    tol: tolerance for convergence
    max_iter: maximum iterations
    
    Returns:
    root: approximate root
    iterations: list of iteration data
    """
    iterations = []
    
    for i in range(max_iter):
        fx0 = func(x0)
        fx1 = func(x1)
        
        if abs(fx1 - fx0) < 1e-12:
            raise ValueError(f"Division by zero: f(x1) - f(x0) too small at iteration {i+1}")
        
        # Secant formula
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = func(x2)
        
        abs_err = absolute_error(x2, x1)
        rel_err = relative_error(x2, x1) if x1 != 0 else None
        
        iterations.append({
            'iteration': i+1,
            'x0': x0,
            'x1': x1,
            'f(x0)': fx0,
            'f(x1)': fx1,
            'x2': x2,
            'f(x2)': fx2,
            'abs_error': abs_err,
            'rel_error': rel_err
        })
        
        if abs(fx2) < tol or abs_err < tol:
            break
        
        # Update for next iteration
        x0 = x1
        x1 = x2
    
    return x2, iterations