import sympy as sp

def parse_function(expr_str):
    """
    Parse a string expression into a callable function.
    e.g., "x**2 - 2" -> lambda x: x**2 - 2
    """
    x = sp.Symbol('x')
    locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
    try:
        expr = sp.sympify(expr_str, locals=locals_map)
        unknown = expr.free_symbols - {x}
        if unknown:
            y = sp.Symbol('y')
            if unknown == {y} and x not in expr.free_symbols:
                expr = expr.subs(y, x)
                unknown = expr.free_symbols - {x}
        if unknown:
            unknown_str = ", ".join(sorted(str(s) for s in unknown))
            raise ValueError(f"Fungsi hanya boleh memakai variabel x. Simbol lain: {unknown_str}")
        func = sp.lambdify(x, expr, 'numpy')
        return func
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr_str}. Error: {e}")
