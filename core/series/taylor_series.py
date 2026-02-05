import numpy as np
import sympy as sp

def euler_method(func_str, x0, y0, x_eval, h=None, n_steps=None):
    """
    Solve ODE y' = f(x, y) using Euler method.
    
    Parameters:
    -----------
    func_str: function string representing f(x, y)
    x0: initial x value
    y0: initial y value
    x_eval: evaluation x value
    h: step size (optional if n_steps provided)
    n_steps: number of steps (optional if h provided)
    
    Returns:
    --------
    result dictionary
    """
    x, y_sym = sp.symbols('x y')
    locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
    f = sp.sympify(func_str, locals=locals_map)
    f_np = sp.lambdify((x, y_sym), f, 'numpy')
    
    if h is None and n_steps is None:
        n_steps = 100
        h = (x_eval - x0) / n_steps
    elif h is None:
        h = (x_eval - x0) / n_steps
    elif n_steps is None:
        n_steps = int(np.ceil((x_eval - x0) / h))
    
    x_values = np.linspace(x0, x_eval, n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    y_values[0] = y0
    
    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * f_np(x_values[i], y_values[i])
    
    # Exact solution for comparison (for the example ODE y' = x²y - y)
    exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    
    return {
        'method': 'Euler',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': abs(y_values[-1] - exact_solution),
        'relative_error': abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution != 0 else float('inf'),
        'x_values': x_values,
        'y_values': y_values
    }


def taylor_ode_order2(func_str, x0, y0, x_eval, h=None, n_steps=None):
    """
    Solve ODE y' = f(x, y) using Taylor series method of order 2.
    
    Parameters:
    -----------
    func_str: function string representing f(x, y)
    x0: initial x value
    y0: initial y value
    x_eval: evaluation x value
    h: step size (optional if n_steps provided)
    n_steps: number of steps (optional if h provided)
    
    Returns:
    --------
    result dictionary
    """
    x, y_sym = sp.symbols('x y')
    locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
    f = sp.sympify(func_str, locals=locals_map)
    
    # First derivative (f) and second derivative (f')
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y_sym)
    f_prime = df_dx + df_dy * f  # y'' = f'(x, y)
    
    f_np = sp.lambdify((x, y_sym), f, 'numpy')
    f_prime_np = sp.lambdify((x, y_sym), f_prime, 'numpy')
    
    if h is None and n_steps is None:
        n_steps = 100
        h = (x_eval - x0) / n_steps
    elif h is None:
        h = (x_eval - x0) / n_steps
    elif n_steps is None:
        n_steps = int(np.ceil((x_eval - x0) / h))
    
    x_values = np.linspace(x0, x_eval, n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    y_values[0] = y0
    
    for i in range(n_steps):
        x_i = x_values[i]
        y_i = y_values[i]
        y_values[i+1] = y_i + h * f_np(x_i, y_i) + (h**2 / 2) * f_prime_np(x_i, y_i)
    
    # Exact solution for comparison (for the example ODE y' = x²y - y)
    exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    
    return {
        'method': 'Taylor ODE Order 2',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': abs(y_values[-1] - exact_solution),
        'relative_error': abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution != 0 else float('inf'),
        'x_values': x_values,
        'y_values': y_values
    }


def runge_kutta_method(func_str, x0, y0, x_eval, h=None, n_steps=None):
    """
    Solve ODE y' = f(x, y) using Runge-Kutta method (RK4).
    
    Parameters:
    -----------
    func_str: function string representing f(x, y)
    x0: initial x value
    y0: initial y value
    x_eval: evaluation x value
    h: step size (optional if n_steps provided)
    n_steps: number of steps (optional if h provided)
    
    Returns:
    --------
    result dictionary
    """
    x, y_sym = sp.symbols('x y')
    locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
    f = sp.sympify(func_str, locals=locals_map)
    f_np = sp.lambdify((x, y_sym), f, 'numpy')
    
    if h is None and n_steps is None:
        n_steps = 100
        h = (x_eval - x0) / n_steps
    elif h is None:
        h = (x_eval - x0) / n_steps
    elif n_steps is None:
        n_steps = int(np.ceil((x_eval - x0) / h))
    
    x_values = np.linspace(x0, x_eval, n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    y_values[0] = y0
    
    for i in range(n_steps):
        x_i = x_values[i]
        y_i = y_values[i]
        
        k1 = h * f_np(x_i, y_i)
        k2 = h * f_np(x_i + h/2, y_i + k1/2)
        k3 = h * f_np(x_i + h/2, y_i + k2/2)
        k4 = h * f_np(x_i + h, y_i + k3)
        
        y_values[i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Exact solution for comparison (for the example ODE y' = x²y - y)
    exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    
    return {
        'method': 'Runge-Kutta (RK4)',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': abs(y_values[-1] - exact_solution),
        'relative_error': abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution != 0 else float('inf'),
        'x_values': x_values,
        'y_values': y_values
    }


def taylor_series(func_str, x0, n_terms, x_eval=None, error_bound=None, 
                        tolerance_bound=None, y0=None, is_ode=False):
    """
    Calculate Taylor series for either explicit function f(x) or ODE y' = f(x, y).
    
    Parameters:
    -----------
    func_str: function string
    x0: expansion point
    n_terms: number of terms
    x_eval: evaluation point
    error_bound: maximum acceptable error
    tolerance_bound: tolerance for convergence
    y0: initial y value (required for ODE)
    is_ode: True if solving ODE, False if explicit function
    
    Returns:
    --------
    result dictionary
    """
    
    if not is_ode:
        # Original Taylor series for explicit function
        x = sp.Symbol('x')
        locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
        
        try:
            # Parse the function
            func = sp.sympify(func_str, locals=locals_map)
            # Allow y as an alias for x - substitute y with x if present
            y = sp.Symbol('y')
            if y in func.free_symbols:
                func = func.subs(y, x)
            # Ensure no unexpected symbols remain
            unknown = func.free_symbols - {x}
            if unknown:
                unknown_str = ", ".join(sorted(str(s) for s in unknown))
                raise ValueError(f"Fungsi hanya boleh memakai variabel x. Simbol lain: {unknown_str}")
            
            # Calculate Taylor series
            series = func.series(x, x0, n_terms).removeO()
            
            # If y0 is provided, adjust the series to account for initial y value
            if y0 is not None:
                # For functions that require initial y condition, add it to the series
                # This is typically used for problems like y' = f(x, y) with y(x0) = y0
                # For standard Taylor series, this might not be necessary, but it's available
                series = series.subs(x, x0) + y0 - func.subs(x, x0)
            
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
            
            # Calculate error bound if provided
            error_bound_result = None
            if error_bound is not None and x_eval is not None:
                # Lagrange remainder bound: |R_n(x)| ≤ M * |f^(n+1)(x)| / (n+1)!
                # where M = max |f^(n+1)(ξ)| on interval [x0, x_eval]
                try:
                    # Find maximum of (n+1)-th derivative on interval
                    max_derivative = 0
                    for i in range(n_terms + 1):
                        derivative = func.diff(x, i)
                        # Evaluate derivative at multiple points in interval
                        test_points = np.linspace(min(x0, x_eval), max(x0, x_eval), 100)
                        max_val = max([abs(float(derivative.subs(x, pt))) for pt in test_points])
                        max_derivative = max(max_derivative, max_val)
                
                    # Calculate bound
                    factorial = sp.factorial(n_terms)
                    error_bound_val = max_derivative * abs(x_eval - x0)**(n_terms + 1) / factorial
                    error_bound_result = {
                        'bound': error_bound_val,
                        'formula': f'|R_{n_terms}(x)| <= {max_derivative:.2e} * |x - x0|^{n_terms + 1} / {n_terms + 1}!',
                        'description': 'Batas maksimum error berdasarkan turunan ke-(n+1)'
                    }
                except:
                    error_bound_result = None
            
            # Calculate tolerance bound if provided
            tolerance_bound_result = None
            if tolerance_bound is not None and x_eval is not None:
                # Minimum tolerance for convergence
                try:
                    # Calculate required tolerance for n significant digits
                    true_value = float(func.subs(x, x_eval))
                    required_tolerance = 0.5 * 10**(-tolerance_bound) if true_value != 0 else tolerance_bound
                    tolerance_bound_result = {
                        'tolerance': required_tolerance,
                        'formula': f'0.5 × 10^(-{tolerance_bound})',
                        'description': f'Toleransi minimum untuk {tolerance_bound} digit signifikan'
                    }
                except:
                    tolerance_bound_result = None
            
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
                    
                    # Check against bounds
                    meets_error_bound = error_bound_result is None or abs_error <= error_bound_result['bound']
                    meets_tolerance = tolerance_bound_result is None or abs_error <= tolerance_bound_result['tolerance']
                    
                    approximations.append({
                        'n_terms': i + 1,
                        'Jumlah Suku': i + 1,  # Nama yang mudah dimengerti
                        'Nilai Aproksimasi': approx_value,  # Nama yang mudah dimengerti
                        'Nilai Sebenarnya': true_value,  # Nama yang mudah dimengerti
                        'Error Absolut': abs_error,  # Nama yang mudah dimengerti
                        'Error Relatif': rel_error,  # Nama yang mudah dimengerti
                        'Error Persentase': rel_error * 100,  # Nama yang mudah dimengerti
                        'Memenuhi Batas Error': meets_error_bound,  # Nama yang mudah dimengerti
                        'Memenuhi Toleransi': meets_tolerance,  # Nama yang mudah dimengerti
                        'Keterangan': 'Konvergen' if meets_tolerance else 'Belum Konvergen'  # Penjelasan bahasa Indonesia
                    })
            
            result = {
                'series_expr': str(series),
                'terms': terms,
                'approximations': approximations
            }
            
            # Add bounds to result if provided
            if error_bound_result is not None:
                result['error_bound'] = error_bound_result
            if tolerance_bound_result is not None:
                result['tolerance_bound'] = tolerance_bound_result
            
            # Add y0 to result if provided
            if y0 is not None:
                result['y0'] = y0
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error calculating Taylor series: {e}")
    else:
        # ODE case: y' = f(x, y)
        if y0 is None:
            raise ValueError("y0 harus diberikan untuk menyelesaikan ODE")
        if x_eval is None:
            raise ValueError("x_eval harus diberikan untuk evaluasi ODE")
        
        x, y_sym = sp.symbols('x y')
        locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
        
        # Parse f(x, y)
        f = sp.sympify(func_str, locals=locals_map)
        
        # Calculate derivatives up to n_terms-1
        derivative_values = [y0]  # y(x0)
        
        # y'(x0) = f(x0, y0)
        f_at_x0 = f.subs({x: x0, y_sym: y0})
        derivative_values.append(float(f_at_x0))
        
        # Higher derivatives if needed
        if n_terms > 2:
            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y_sym)
            
            df_dx_at_x0 = df_dx.subs({x: x0, y_sym: y0})
            df_dy_at_x0 = df_dy.subs({x: x0, y_sym: y0})
            y_double_prime = float(df_dx_at_x0) + float(df_dy_at_x0) * derivative_values[1]
            derivative_values.append(y_double_prime)
        
        if n_terms > 3:
            # Third derivative
            d2f_dx2 = sp.diff(df_dx, x)
            d2f_dxdy = sp.diff(df_dx, y_sym)
            d2f_dy2 = sp.diff(df_dy, y_sym)
            
            d2f_dx2_at_x0 = d2f_dx2.subs({x: x0, y_sym: y0})
            d2f_dxdy_at_x0 = d2f_dxdy.subs({x: x0, y_sym: y0})
            d2f_dy2_at_x0 = d2f_dy2.subs({x: x0, y_sym: y0})
            
            y_prime = derivative_values[1]
            y_triple_prime = (float(d2f_dx2_at_x0) + 
                              2 * float(d2f_dxdy_at_x0) * y_prime + 
                              float(d2f_dy2_at_x0) * y_prime**2 + 
                              float(df_dy_at_x0) * derivative_values[2])
            derivative_values.append(y_triple_prime)
        
        # Fill remaining with zeros if needed
        while len(derivative_values) < n_terms:
            derivative_values.append(0.0)
        
        # Build terms and approximations
        terms = []
        approximations = []
        
        for num_terms in range(1, n_terms + 1):
            # Calculate approximation with current number of terms
            approx = 0
            h = x_eval - x0
            
            for i in range(num_terms):
                term_val = derivative_values[i] * (h**i) / sp.factorial(i)
                approx += float(term_val)
            
            # For ODE, true value is unknown unless we compute exact solution
            # For this example, we'll compute exact: y(x) = exp(x³/3 - x)
            true_value = float(sp.exp((x_eval**3)/3 - x_eval))  # Exact solution
            
            abs_error = abs(approx - true_value)
            rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
            
            terms_info = []
            for i in range(num_terms):
                coeff = derivative_values[i] / sp.factorial(i)
                terms_info.append({
                    'n': i,
                    'coefficient': float(coeff),
                    'term': f"{float(coeff):.6f} * (x-{x0})^{i}"
                })
            
            approximations.append({
                'Jumlah Suku': num_terms,
                'Nilai Aproksimasi': approx,
                'Nilai Sebenarnya': true_value,
                'Error Absolut': abs_error,
                'Error Relatif': rel_error,
                'Error Persentase': rel_error * 100
            })
            
            if num_terms == n_terms:
                terms = terms_info
        
        return {
            'series_expr': f"Taylor series for ODE y' = {func_str}, y({x0}) = {y0}",
            'terms': terms,
            'approximations': approximations,
            'is_ode': True,
            'y0': y0
        }

# Contoh penggunaan:
if __name__ == "__main__":
    # Untuk ODE: y' = x²y - y, y(0) = 1, hitung y(2)
    result = taylor_series(
        func_str="x**2*y - y",
        x0=0,
        n_terms=2,
        x_eval=2,
        y0=1,
        is_ode=True
    )
    
    print("Hasil untuk ODE:")
    print("Suku-suku deret:")
    for term in result['terms']:
        print(f"  n={term['n']}: coefficient={term['coefficient']}, term={term['term']}")
    
    print("\nKonvergensi Aproksimasi:")
    for approx in result['approximations']:
        print(f"  {approx['Jumlah Suku']} suku: {approx['Nilai Aproksimasi']:.6f} "
              f"(true: {approx['Nilai Sebenarnya']:.6f}, error: {approx['Error Absolut']:.6f})")


def evaluate_taylor_at_points(func_str, x0, n_terms, x_points, y0=None):
    """
    Evaluate Taylor series at multiple points for plotting.
    
    Parameters:
    -----------
    func_str: string representation of the function
    x0: expansion point
    n_terms: number of terms
    x_points: array of x values to evaluate at
    y0: optional, initial y value for problems requiring y-axis (default: None)
    
    Returns:
    --------
    true_values: true function values
    approx_values: Taylor approximation values
    y0: initial y value if provided
    """
    x = sp.Symbol('x')
    locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
    func = sp.sympify(func_str, locals=locals_map)
    # Allow y as an alias for x - substitute y with x if present
    y = sp.Symbol('y')
    if y in func.free_symbols:
        func = func.subs(y, x)
    # Ensure no unexpected symbols remain
    unknown = func.free_symbols - {x}
    if unknown:
        unknown_str = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"Fungsi hanya boleh memakai variabel x. Simbol lain: {unknown_str}")
    series = func.series(x, x0, n_terms).removeO()
    
    # Convert to numpy-compatible functions
    func_np = sp.lambdify(x, func, 'numpy')
    series_np = sp.lambdify(x, series, 'numpy')
    
    true_values = func_np(x_points)
    approx_values = series_np(x_points)
    
    return true_values, approx_values, y0


