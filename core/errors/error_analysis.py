def absolute_error(approx, true):
    """
    Calculate absolute error: |approx - true|
    """
    return abs(approx - true)

def relative_error(approx, true):
    """
    Calculate relative error: |approx - true| / |true| if true != 0
    """
    if true == 0:
        return float('inf') if approx != 0 else 0
    return abs(approx - true) / abs(true)

def iterative_error(current, previous):
    """
    Calculate iterative error: |current - previous|
    """
    return abs(current - previous)