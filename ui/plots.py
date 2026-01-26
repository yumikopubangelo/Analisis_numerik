import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def plot_function(func, a, b, root=None):
    """Plot function with optional root marker"""
    x = np.linspace(a, b, 1000)
    try:
        y = func(x)
    except:
        y = np.array([func(xi) for xi in x])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    if root is not None:
        ax.axvline(root, color='red', linestyle='--', linewidth=2, label=f'Root ≈ {root:.6f}')
        ax.plot(root, func(root), 'ro', markersize=10, label=f'f({root:.4f}) = {func(root):.2e}')
    
    ax.legend(fontsize=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Plot Fungsi', fontsize=14, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()

def plot_convergence(iterations):
    """Plot convergence of error over iterations"""
    if not iterations:
        return
    
    iters = [it['iteration'] for it in iterations]
    errors = [it.get('abs_error') for it in iterations if it.get('abs_error') is not None]
    
    if not errors:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters[1:len(errors)+1], errors, 'o-', color='#667eea', linewidth=2, markersize=8)
    ax.set_xlabel('Iterasi', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Konvergensi Error', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

def plot_integration(func, a, b, n, method):
    """Plot integration visualization"""
    x = np.linspace(a, b, 1000)
    try:
        y = func(x)
    except:
        y = np.array([func(xi) for xi in x])
    
    # Points for integration
    x_points = np.linspace(a, b, n+1)
    y_points = np.array([func(xi) for xi in x_points])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot function
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    
    # Fill area under curve
    ax.fill_between(x, 0, y, alpha=0.2, color='blue', label='Area to integrate')
    
    # Draw approximation
    if method == "Trapezoidal":
        for i in range(n):
            xs = [x_points[i], x_points[i+1]]
            ys = [y_points[i], y_points[i+1]]
            ax.fill_between(xs, 0, ys, alpha=0.4, color='orange', edgecolor='red', linewidth=1.5)
    
    elif method == "Simpson":
        # For Simpson, show parabolic segments (simplified visualization)
        for i in range(0, n, 2):
            if i+2 <= n:
                xs = np.linspace(x_points[i], x_points[i+2], 50)
                # Fit parabola through 3 points
                coeffs = np.polyfit(x_points[i:i+3], y_points[i:i+3], 2)
                ys = np.polyval(coeffs, xs)
                ax.fill_between(xs, 0, ys, alpha=0.4, color='orange', edgecolor='red', linewidth=1.5)
    
    # Plot points
    ax.plot(x_points, y_points, 'ro', markersize=8, label='Evaluation points')
    
    # Vertical lines
    for xp in x_points:
        ax.axvline(xp, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'Visualisasi {method} Rule (n={n})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

def plot_interpolation(x_points, y_points, x_eval, y_eval, method):
    """Plot interpolation"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Generate smooth curve for interpolated polynomial
    x_min, x_max = min(x_points), max(x_points)
    x_range = x_max - x_min
    x_smooth = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 500)
    
    # Calculate interpolated values
    if method == "Lagrange":
        from core.interpolation.lagrange import lagrange_interpolation
        y_smooth, _, _ = lagrange_interpolation(x_points, y_points, x_smooth)
    else:  # Newton
        from core.interpolation.newton import newton_interpolation
        y_smooth, _, _ = newton_interpolation(x_points, y_points, x_smooth)
    
    # Plot interpolating polynomial
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'{method} Polynomial')
    
    # Plot data points
    ax.plot(x_points, y_points, 'ro', markersize=12, label='Data Points', zorder=5)
    
    # Plot evaluation point
    ax.plot(x_eval, y_eval, 'g*', markersize=20, label=f'Interpolated: ({x_eval:.2f}, {y_eval:.4f})', zorder=6)
    
    # Vertical line at evaluation point
    ax.axvline(x_eval, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y_eval, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Annotations for data points
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        ax.annotate(f'({x:.2f}, {y:.2f})', 
                   xy=(x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.legend(fontsize=11, loc='best')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Interpolasi {method}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()