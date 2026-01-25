import numpy as np
from core.errors.error_analysis import absolute_error, relative_error
from core.utils.convergence import check_convergence

def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for finding root of f(x) = 0 in [a, b].
    
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
    for i in range(max_iter):
        c = (a + b) / 2
        fc = func(c)
        
        abs_err = absolute_error(c, (a + b)/2) if i > 0 else None
        rel_err = relative_error(c, (a + b)/2) if i > 0 else None
        
        iterations.append({
            'iteration': i+1,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'abs_error': abs_err,
            'rel_error': rel_err
        })
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            break
        
        if func(a) * fc < 0:
            b = c
        else:
            a = c
    
    root = (a + b) / 2
    return root, iterations