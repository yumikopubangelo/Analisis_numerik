def check_convergence(current, previous, tol):
    """
    Check if the method has converged based on tolerance.
    """
    return abs(current - previous) < tol