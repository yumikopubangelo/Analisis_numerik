import numpy as np
import sympy as sp
from core.errors.error_analysis import absolute_error, relative_error

def newton_raphson_method(func, func_str, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for finding root of f(x) = 0.
    
    Parameters:
    func: function to find root of
    func_str: string representation of function (for derivative calculation)
    x0: initial guess
    tol: tolerance for convergence
    max_iter: maximum iterations
    
    Returns:
    root: approximate root
    iterations: list of iteration data
    """
    # Calculate derivative symbolically
    x = sp.Symbol('x')
    expr = sp.sympify(func_str)
    derivative_expr = sp.diff(expr, x)
    derivative = sp.lambdify(x, derivative_expr, 'numpy')
    
    iterations = []
    x_curr = x0
    
    for i in range(max_iter):
        fx = func(x_curr)
        dfx = derivative(x_curr)
        
        if abs(dfx) < 1e-12:
            raise ValueError(f"Derivative too close to zero at x = {x_curr}. Method fails.")
        
        x_next = x_curr - fx / dfx
        
        abs_err = absolute_error(x_next, x_curr)
        rel_err = relative_error(x_next, x_curr) if x_curr != 0 else None
        
        iterations.append({
            'iteration': i+1,
            'x': x_curr,
            'f(x)': fx,
            "f'(x)": dfx,
            'x_next': x_next,
            'abs_error': abs_err,
            'rel_error': rel_err
        })
        
        if abs(fx) < tol or abs_err < tol:
            break
        
        x_curr = x_next
    
    return x_curr, iterations