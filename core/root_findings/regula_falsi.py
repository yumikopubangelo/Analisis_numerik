import numpy as np
from core.errors.error_analysis import absolute_error, relative_error

def regula_falsi_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Regula Falsi (False Position) method for finding root of f(x) = 0 in [a, b].
    
    Parameters:
    func: function to find root of
    a, b: interval endpoints, f(a)*f(b) < 0
    tol: tolerance for convergence
    max_iter: maximum iterations
    
    Returns:
    root: approximate root
    iterations: list of iteration data
    """
    if func(a) * func(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    iterations = []
    c_prev = a
    
    for i in range(max_iter):
        fa = func(a)
        fb = func(b)
        
        # Calculate c using the regula falsi formula
        c = b - (fb * (b - a)) / (fb - fa)
        fc = func(c)
        
        abs_err = absolute_error(c, c_prev) if i > 0 else None
        rel_err = relative_error(c, c_prev) if i > 0 and c_prev != 0 else None
        
        iterations.append({
            'iteration': i+1,
            'a': a,
            'b': b,
            'c': c,
            'f(a)': fa,
            'f(b)': fb,
            'f(c)': fc,
            'abs_error': abs_err,
            'rel_error': rel_err
        })
        
        if abs(fc) < tol or (abs_err is not None and abs_err < tol):
            break
        
        if fa * fc < 0:
            b = c
        else:
            a = c
        
        c_prev = c
    
    return c, iterations