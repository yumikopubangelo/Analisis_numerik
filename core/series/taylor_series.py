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
    
    # Calculate exact solution only for specific ODE cases where we know the analytical solution
    if func_str.strip() == "x**2*y - y" and x0 == 0 and y0 == 1:
        # ODE: y' = x²y - y, y(0) = 1 has analytical solution y = exp(x³/3 - x)
        exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    elif func_str.strip() == "x/y" and x0 == 0 and y0 == 1:
        # ODE: y' = x/y, y(0) = 1 has analytical solution y = sqrt(x² + 1)
        exact_solution = float(np.sqrt(x_eval**2 + 1))
    elif func_str.strip() == "x*sqrt(y)" and x0 == 0 and y0 == 1:
        # ODE: y' = x*sqrt(y), y(0) = 1 has analytical solution y = (1 + x²/4)²
        exact_solution = float((1 + x_eval**2 / 4)**2)
    else:
        # For general ODEs, we don't have the analytical solution
        exact_solution = None
    
    # Calculate errors only if exact solution is available
    absolute_error = abs(y_values[-1] - exact_solution) if exact_solution is not None else None
    relative_error = abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution is not None and exact_solution != 0 else None
    
    return {
        'method': 'Euler',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
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
    
    # Calculate exact solution only for specific ODE cases where we know the analytical solution
    if func_str.strip() == "x**2*y - y" and x0 == 0 and y0 == 1:
        # ODE: y' = x²y - y, y(0) = 1 has analytical solution y = exp(x³/3 - x)
        exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    elif func_str.strip() == "x/y" and x0 == 0 and y0 == 1:
        # ODE: y' = x/y, y(0) = 1 has analytical solution y = sqrt(x² + 1)
        exact_solution = float(np.sqrt(x_eval**2 + 1))
    elif func_str.strip() == "x*sqrt(y)" and x0 == 0 and y0 == 1:
        # ODE: y' = x*sqrt(y), y(0) = 1 has analytical solution y = (1 + x²/4)²
        exact_solution = float((1 + x_eval**2 / 4)**2)
    else:
        # For general ODEs, we don't have the analytical solution
        exact_solution = None
    
    # Calculate errors only if exact solution is available
    absolute_error = abs(y_values[-1] - exact_solution) if exact_solution is not None else None
    relative_error = abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution is not None and exact_solution != 0 else None
    
    return {
        'method': 'Taylor ODE Order 2',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
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
    
    # Calculate exact solution only for specific ODE cases where we know the analytical solution
    if func_str.strip() == "x**2*y - y" and x0 == 0 and y0 == 1:
        # ODE: y' = x²y - y, y(0) = 1 has analytical solution y = exp(x³/3 - x)
        exact_solution = float(sp.exp((x_eval**3)/3 - x_eval))
    elif func_str.strip() == "x/y" and x0 == 0 and y0 == 1:
        # ODE: y' = x/y, y(0) = 1 has analytical solution y = sqrt(x² + 1)
        exact_solution = float(np.sqrt(x_eval**2 + 1))
    elif func_str.strip() == "x*sqrt(y)" and x0 == 0 and y0 == 1:
        # ODE: y' = x*sqrt(y), y(0) = 1 has analytical solution y = (1 + x²/4)²
        exact_solution = float((1 + x_eval**2 / 4)**2)
    else:
        # For general ODEs, we don't have the analytical solution
        exact_solution = None
    
    # Calculate errors only if exact solution is available
    absolute_error = abs(y_values[-1] - exact_solution) if exact_solution is not None else None
    relative_error = abs(y_values[-1] - exact_solution) / abs(exact_solution) if exact_solution is not None and exact_solution != 0 else None
    
    return {
        'method': 'Runge-Kutta (RK4)',
        'x0': x0,
        'y0': y0,
        'x_eval': x_eval,
        'h': h,
        'n_steps': n_steps,
        'approximation': y_values[-1],
        'exact': exact_solution,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'x_values': x_values,
        'y_values': y_values
    }


def taylor_series(func_str, x0, n_terms, x_eval=None, error_bound=None, 
                        tolerance_bound=None, y0=None, is_ode=False):
    """
    Calculate Taylor series for either explicit function f(x) or ODE y' = f(x, y).
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
            
            # Calculate Taylor series (FIXED: gunakan method 'series' dengan benar)
            # Series expansion: func.series(x, x0, n_terms+1).removeO()
            series_expr = sp.series(func, x, x0, n_terms+1).removeO()
            
            # Get individual terms (FIXED LOGIC)
            terms = []
            approximations = [] if x_eval is not None else None
            
            # First, collect all non-zero terms up to n_terms
            collected_terms = 0
            i = 0
            while collected_terms < n_terms and i < n_terms * 2:  # Prevent infinite loop
                # Calculate i-th derivative at x0
                derivative = func.diff(x, i)
                derivative_at_x0 = derivative.subs(x, x0)
                
                # Only add non-zero terms for functions like cos(x)
                if derivative_at_x0 != 0 or i == 0:
                    # Calculate coefficient
                    coeff = derivative_at_x0 / sp.factorial(i)
                    
                    # Term expression
                    if i == 0:
                        term_expr = coeff
                    else:
                        term_expr = coeff * (x - x0)**i
                    
                    terms.append({
                        'n': collected_terms,
                        'actual_order': i,  # Store actual derivative order
                        'derivative': str(derivative),
                        'derivative_at_x0': float(derivative_at_x0) if derivative_at_x0.is_number else str(derivative_at_x0),
                        'coefficient': float(coeff) if coeff.is_number else str(coeff),
                        'term': str(term_expr)
                    })
                    
                    collected_terms += 1
                    
                    # Calculate approximation at x_eval if provided
                    if x_eval is not None:
                        # Direct evaluation for each term count
                        approx_value = 0.0
                        for j in range(collected_terms):
                            term_j = terms[j]
                            d_order = term_j['actual_order']
                            d = func.diff(x, d_order).subs(x, x0)
                            approx_value += float(d) * (x_eval - x0)**d_order / np.math.factorial(d_order)
                        
                        true_value = float(func.subs(x, x_eval))
                        abs_error = abs(approx_value - true_value)
                        rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
                        
                        # Check bounds
                        meets_error_bound = True
                        meets_tolerance = True
                        if error_bound is not None:
                            meets_error_bound = abs_error <= error_bound
                        if tolerance_bound is not None:
                            meets_tolerance = abs_error <= tolerance_bound
                        
                        if approximations is None:
                            approximations = []
                        approximations.append({
                            'Jumlah Suku': collected_terms,
                            'Nilai Aproksimasi': approx_value,
                            'Nilai Sebenarnya': true_value,
                            'Error Absolut': abs_error,
                            'Error Relatif': rel_error,
                            'Error Persentase': rel_error * 100,
                            'Memenuhi Batas Error': meets_error_bound,
                            'Memenuhi Toleransi': meets_tolerance,
                            'Keterangan': 'Konvergen' if meets_tolerance else 'Belum Konvergen'
                        })
                
                i += 1
            
            # Build series expression from individual terms
            if terms:
                series_expr = sp.sympify(0)
                for term in terms:
                    term_expr = sp.sympify(term['term'])
                    series_expr += term_expr
                series_expr = series_expr.removeO()
            
            result = {
                'series_expr': str(series_expr),
                'terms': terms,
                'approximations': approximations
            }
            
            # Add y0 to result if provided (untuk plotting)
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
        
        # Limit maximum terms to 4 (since we only calculate up to 3rd derivative)
        if n_terms > 4:
            n_terms = 4
        
        # Fill remaining with zeros if needed (should not be necessary now)
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
            
            # Calculate exact solution only for specific ODE cases where we know the analytical solution
            if func_str.strip() == "x**2*y - y" and x0 == 0 and y0 == 1:
                # ODE: y' = x²y - y, y(0) = 1 has analytical solution y = exp(x³/3 - x)
                true_value = float(sp.exp((x_eval**3)/3 - x_eval))
            elif func_str.strip() == "x/y" and x0 == 0 and y0 == 1:
                # ODE: y' = x/y, y(0) = 1 has analytical solution y = sqrt(x² + 1)
                true_value = float(np.sqrt(x_eval**2 + 1))
            elif func_str.strip() == "x*sqrt(y)" and x0 == 0 and y0 == 1:
                # ODE: y' = x*sqrt(y), y(0) = 1 has analytical solution y = (1 + x²/4)²
                true_value = float((1 + x_eval**2 / 4)**2)
            else:
                # For general ODEs, we don't have the analytical solution
                true_value = None
            
            # Calculate errors only if exact solution is available
            if true_value is not None:
                abs_error = abs(approx - true_value)
                rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
            else:
                abs_error = None
                rel_error = None
            
            terms_info = []
            for i in range(num_terms):
                coeff = derivative_values[i] / sp.factorial(i)
                terms_info.append({
                    'n': i,
                    'coefficient': float(coeff),
                    'term': f"{float(coeff):.6f} * (x-{x0})^{i}"
                })
            
            # Prepare error values for display
            error_persentase = rel_error * 100 if rel_error is not None else None
            
            approximations.append({
                'Jumlah Suku': num_terms,
                'Nilai Aproksimasi': approx,
                'Nilai Sebenarnya': true_value,
                'Error Absolut': abs_error,
                'Error Relatif': rel_error,
                'Error Persentase': error_persentase
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

def taylor_cos_specific(x_eval=1.0, n_terms=5):
    """
    Khusus untuk soal: cos(x) di x0=0, menghitung seperti pada tabel.
    """
    x0 = 0.0
    true_value = np.cos(x_eval)
    
    terms_info = []
    approximations = []
    
    # Pre-calculate terms
    partial_sum = 0
    for k in range(n_terms):
        exponent = 2 * k
        term = (x_eval**exponent) / np.math.factorial(exponent)
        if k % 2 == 1:
            term *= -1
        
        terms_info.append({
            'n': k,
            'exponent': exponent,
            'coefficient': 1.0 / np.math.factorial(exponent) if k % 2 == 0 else -1.0 / np.math.factorial(exponent),
            'term_value': term
        })
    
    # Calculate approximations for 0 to n_terms-1 suku
    for num_suku in range(n_terms):
        approx = 0
        for k in range(num_suku + 1):
            term = terms_info[k]['term_value']
            approx += term
        
        abs_error = abs(approx - true_value)
        rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
        
        approximations.append({
            'Jumlah Suku': num_suku + 1,
            'Nilai Aproksimasi': approx,
            'Nilai Sebenarnya': true_value,
            'Error Absolut': abs_error,
            'Error Relatif': rel_error,
            'Error Persentase': rel_error * 100,
            'Memenuhi Batas Error': True,  # Karena tidak ada batas yang diberikan
            'Memenuhi Toleransi': True,
            'Keterangan': '✅'
        })
    
    return {
        'function': 'cos(x)',
        'x0': x0,
        'x_eval': x_eval,
        'n_terms': n_terms,
        'true_value': true_value,
        'terms': terms_info,
        'approximations': approximations
    }


# Contoh penggunaan:
if __name__ == "__main__":
    print("=== Debug: Taylor untuk cos(x) ===")
    
    # Gunakan fungsi khusus
    result = taylor_cos_specific(x_eval=1.0, n_terms=5)
    
    print("Suku-suku:")
    for term in result['terms']:
        print(f"  k={term['n']}: exponent={term['exponent']}, coeff={term['coefficient']:.6f}, value={term['term_value']:.6f}")
    
    print("\nTabel Konvergensi:")
    print(f"{'Suku':<6} {'Aproksimasi':<12} {'Sebenarnya':<12} {'Error Abs':<12} {'Error %':<10}")
    print("-" * 60)
    
    for approx in result['approximations']:
        print(f"{approx['Jumlah Suku']:<6} "
              f"{approx['Nilai Aproksimasi']:<12.6f} "
              f"{approx['Nilai Sebenarnya']:<12.6f} "
              f"{approx['Error Absolut']:<12.2e} "
              f"{approx['Error Persentase']:<10.4f}%")
    
    print("\n=== Test dengan fungsi utama ===")
    result_main = taylor_series("cos(x)", 0, 5, x_eval=1.0)
    print(f"Result from main function:")
    print(f"  Series expression: {result_main['series_expr']}")
    print(f"  Approximations at x=1.0:")
    for approx in result_main['approximations']:
        print(f"    {approx['Jumlah Suku']} suku: {approx['Nilai Aproksimasi']:.6f} (error: {approx['Error Persentase']:.4f}%)")


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


