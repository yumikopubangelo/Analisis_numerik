import numpy as np
import matplotlib.pyplot as plt


def solve_advection_diffusion_1d(
    U: float,
    K: float,
    x_min: float,
    x_max: float,
    nx: int,
    dt: float,
    num_steps: int
) -> dict:
    """
    Solve 1D advection-diffusion equation using finite difference method.
    
    Equation: ∂C/∂t = -U ∂C/∂x + K ∂²C/∂x²
    
    Boundary conditions: Dirichlet (C=0 at both ends)
    
    Parameters:
    -----------
    U : float
        Velocity of advection
    K : float
        Diffusion coefficient
    x_min : float
        Minimum x value (left boundary)
    x_max : float
        Maximum x value (right boundary)
    nx : int
        Number of grid points
    dt : float
        Time step
    num_steps : int
        Number of time steps
    
    Returns:
    --------
    dict
        Solution dictionary containing:
        - x: grid points
        - C: concentration at each time step
        - t: time points
        - parameters: input parameters
    """
    # Calculate grid spacing
    dx = (x_max - x_min) / (nx - 1)
    
    # Create grid
    x = np.linspace(x_min, x_max, nx)
    
    # Initialize concentration profile - Gaussian pulse
    C = np.exp(-(x - 0.5)**2 / 0.01)
    
    # Store concentration at each time step
    C_history = [C.copy()]
    t_history = [0.0]
    
    # Calculate coefficients for the finite difference scheme
    # Using implicit scheme for stability
    # Equation: C_i^{n+1} = C_i^n + (dt/dx²) * [K(C_{i+1}^n - 2C_i^n + C_{i-1}^n) - (U dx/2)(C_{i+1}^n - C_{i-1}^n)]
    
    # Create tridiagonal matrix
    A = np.zeros((nx, nx))
    b = np.zeros(nx)
    
    # Set boundary conditions (Dirichlet)
    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = 0
    b[-1] = 0
    
    # Fill interior points
    for i in range(1, nx - 1):
        # Coefficients for implicit scheme
        coeff = dt / dx**2
        A[i, i-1] = -coeff * (K + (U * dx)/2)
        A[i, i] = 1 + 2 * coeff * K
        A[i, i+1] = -coeff * (K - (U * dx)/2)
    
    # Time integration
    for n in range(num_steps):
        # Solve linear system for next time step
        C = np.linalg.solve(A, C)
        
        # Store results
        C_history.append(C.copy())
        t_history.append((n + 1) * dt)
    
    # Convert to numpy arrays
    C_history = np.array(C_history)
    t_history = np.array(t_history)
    
    return {
        'x': x,
        'C': C_history,
        't': t_history,
        'parameters': {
            'U': U,
            'K': K,
            'x_min': x_min,
            'x_max': x_max,
            'nx': nx,
            'dx': dx,
            'dt': dt,
            'num_steps': num_steps
        }
    }


def plot_concentration_snapshot(x: np.ndarray, C: np.ndarray, t: float) -> plt.Figure:
    """
    Plot concentration snapshot at a specific time.
    
    Parameters:
    -----------
    x : np.ndarray
        Grid points
    C : np.ndarray
        Concentration values
    t : float
        Time
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, C, 'b-', linewidth=2, label=f't = {t:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('Concentration C(x)')
    ax.set_title('Concentration Profile at Specific Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_concentration_evolution(x: np.ndarray, C_history: np.ndarray, t_history: np.ndarray) -> plt.Figure:
    """
    Plot concentration evolution over time (heatmap).
    
    Parameters:
    -----------
    x : np.ndarray
        Grid points
    C_history : np.ndarray
        Concentration history
    t_history : np.ndarray
        Time points
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(C_history, aspect='auto', extent=[x[0], x[-1], t_history[0], t_history[-1]], origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('Time')
    ax.set_title('Concentration Evolution')
    plt.colorbar(im, label='Concentration C(x, t)')
    return fig
