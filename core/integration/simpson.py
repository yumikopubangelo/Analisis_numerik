import numpy as np

def simpson_method(func, a, b, n):
    """
    Simpson's 1/3 rule for numerical integration.
    
    Parameters:
    func: function to integrate
    a, b: integration limits
    n: number of subintervals (must be even)
    
    Returns:
    integral: approximate value of integral
    steps: list of calculation steps
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's rule")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    
    steps = []
    
    # Calculate odd indices sum
    sum_odd = 0
    for i in range(1, n, 2):
        sum_odd += y[i]
        steps.append({
            'i': i,
            'type': 'odd',
            'x_i': x[i],
            'f(x_i)': y[i],
            'cumulative_sum_odd': sum_odd
        })
    
    # Calculate even indices sum
    sum_even = 0
    for i in range(2, n, 2):
        sum_even += y[i]
        steps.append({
            'i': i,
            'type': 'even',
            'x_i': x[i],
            'f(x_i)': y[i],
            'cumulative_sum_even': sum_even
        })
    
    # Simpson's formula: h/3 * [f(a) + 4*sum(odd) + 2*sum(even) + f(b)]
    integral = (h / 3) * (y[0] + 4 * sum_odd + 2 * sum_even + y[-1])
    
    calculation_info = {
        'a': a,
        'b': b,
        'n': n,
        'h': h,
        'f(a)': y[0],
        'f(b)': y[-1],
        'sum_odd': sum_odd,
        'sum_even': sum_even,
        'formula': f'({h}/3) × [{y[0]:.6f} + 4×{sum_odd:.6f} + 2×{sum_even:.6f} + {y[-1]:.6f}]',
        'result': integral
    }
    
    return integral, steps, calculation_info