import sympy as sp
import numpy as np

def taylor_series(func_str, x0, n_terms, x_eval=None):
    """
    Calculate Taylor series expansion of a function around point x0.
    
    Parameters:
    func_str: string representation of the function (e.g., "sin(x)", "exp(x)", "x**2")
    x0: point around which to expand
    n_terms: number of terms in the series
    x_eval: optional point(s) to evaluate the series at
    
    Returns:
    series_expr: symbolic Taylor series expression
    terms: list of individual terms with their coefficients
    approximations: if x_eval provided, approximation values at each term
    """
    x = sp.Symbol('x')
    
    try:
        # Parse the function
        func = sp.sympify(func_str)
        
        # Calculate Taylor series
        series = func.series(x, x0, n_terms).removeO()
        
        # Get individual terms
        terms = []
        for i in range(n_terms):
            # Calculate i-th derivative at x0
            derivative = func.diff(x, i)
            derivative_at_x0 = derivative.subs(x, x0)
            
            # Calculate coefficient
            coeff = derivative_at_x0 / sp.factorial(i)
            
            # Term expression
            if i == 0:
                term_expr = coeff
            else:
                term_expr = coeff * (x - x0)**i
            
            terms.append({
                'n': i,
                'derivative': str(derivative),
                'derivative_at_x0': float(derivative_at_x0) if derivative_at_x0.is_number else str(derivative_at_x0),
                'coefficient': float(coeff) if coeff.is_number else str(coeff),
                'term': str(term_expr)
            })
        
        # Evaluate approximations if x_eval is provided
        approximations = None
        if x_eval is not None:
            approximations = []
            partial_sum = 0
            
            for i, term_data in enumerate(terms):
                # Add current term to partial sum
                term_expr = sp.sympify(term_data['term'])
                partial_sum += term_expr
                
                # Evaluate at x_eval
                approx_value = float(partial_sum.subs(x, x_eval))
                
                # Calculate true value
                true_value = float(func.subs(x, x_eval))
                
                # Calculate error
                abs_error = abs(approx_value - true_value)
                rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
                
                approximations.append({
                    'n_terms': i + 1,
                    'approximation': approx_value,
                    'true_value': true_value,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                })
        
        return {
            'series_expr': str(series),
            'terms': terms,
            'approximations': approximations
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating Taylor series: {e}")


def evaluate_taylor_at_points(func_str, x0, n_terms, x_points):
    """
    Evaluate Taylor series at multiple points for plotting.
    
    Parameters:
    func_str: string representation of the function
    x0: expansion point
    n_terms: number of terms
    x_points: array of x values to evaluate at
    
    Returns:
    true_values: true function values
    approx_values: Taylor approximation values
    """
    x = sp.Symbol('x')
    func = sp.sympify(func_str)
    series = func.series(x, x0, n_terms).removeO()
    
    # Convert to numpy-compatible functions
    func_np = sp.lambdify(x, func, 'numpy')
    series_np = sp.lambdify(x, series, 'numpy')
    
    true_values = func_np(x_points)
    approx_values = series_np(x_points)
    
    return true_values, approx_values
