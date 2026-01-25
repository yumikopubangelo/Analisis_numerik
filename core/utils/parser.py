import sympy as sp

def parse_function(expr_str):
    """
    Parse a string expression into a callable function.
    e.g., "x**2 - 2" -> lambda x: x**2 - 2
    """
    x = sp.Symbol('x')
    try:
        expr = sp.sympify(expr_str)
        func = sp.lambdify(x, expr, 'numpy')
        return func
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr_str}. Error: {e}")