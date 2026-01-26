import numpy as np

def trapezoidal_method(func, a, b, n):
    """
    Trapezoidal rule for numerical integration.
    
    Parameters:
    func: function to integrate
    a, b: integration limits
    n: number of subintervals
    
    Returns:
    integral: approximate value of integral
    steps: list of calculation steps
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    
    steps = []
    
    # Calculate sum of middle terms
    sum_middle = 0
    for i in range(1, n):
        sum_middle += y[i]
        steps.append({
            'i': i,
            'x_i': x[i],
            'f(x_i)': y[i],
            'cumulative_sum': sum_middle
        })
    
    # Trapezoidal formula: h/2 * [f(a) + 2*sum(f(x_i)) + f(b)]
    integral = (h / 2) * (y[0] + 2 * sum_middle + y[-1])
    
    calculation_info = {
        'a': a,
        'b': b,
        'n': n,
        'h': h,
        'f(a)': y[0],
        'f(b)': y[-1],
        'sum_middle': sum_middle,
        'formula': f'({h}/2) × [{y[0]:.6f} + 2×{sum_middle:.6f} + {y[-1]:.6f}]',
        'result': integral
    }
    
    return integral, steps, calculation_info