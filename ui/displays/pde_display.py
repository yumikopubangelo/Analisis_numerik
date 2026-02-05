"""
PDE Solver Display Module

This module provides functions to display the results of the PDE solver
for biharmonic plate equations in the Streamlit interface.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from core.pde.biharmonic_solver import (
    plot_deflection_3d,
    plot_deflection_contour,
    plot_convergence,
    compute_statistics
)


def display_pde_results(X: np.ndarray, Y: np.ndarray, w: np.ndarray,
                       iterations: int, error: float, solver_name: str,
                       q: float, D: float):
    """
    Display PDE solver results with visualizations.

    Parameters:
        X, Y: Meshgrid coordinates
        w: Plate deflection
        iterations: Number of iterations performed
        error: Final error
        solver_name: Name of solver used
        q: Applied load
        D: Flexural rigidity
    """
    # Display numerical discretization explanation
    st.markdown("""
    ### Diskritisasi Numerik untuk Persamaan Biharmonik

    **Persamaan Asli:**
    $$\n    \\frac{\\partial^4 w}{\\partial x^4} + 2\\frac{\\partial^4 w}{\\partial x^2 \\partial y^2} + \\frac{\\partial^4 w}{\\partial y^4} = \\frac{q}{D}\n    $$

    **Stencil 13-titik Finite Difference:**
    $$20w_{i,j} - 8(w_{i+1,j} + w_{i-1,j} + w_{i,j+1} + w_{i,j-1}) + 2(w_{i+1,j+1} + w_{i+1,j-1} + w_{i-1,j+1} + w_{i-1,j-1}) - (w_{i+2,j} + w_{i-2,j} + w_{i,j+2} + w_{i,j-2}) = \\frac{q}{D} h^4$$

    dengan $h = \\Delta x = \\Delta y$
    """)

    st.markdown("---")

    # Display statistics
    stats = compute_statistics(w)

    st.markdown("### Parameter dan Statistik Defleksi Plat")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Beban q (kN/m²)", f"{q:.2f}")
        st.metric("Kekakuan D (kN·m)", f"{D:.2e}")
        st.metric("Jumlah Iterasi", iterations)
    with col2:
        st.metric("Defleksi Maksimum (mm)", f"{stats['max_deflection']*1000:.3f}")
        st.metric("Rata-rata Defleksi (mm)", f"{stats['mean_deflection']*1000:.3f}")
        st.metric("Error Final", f"{error:.2e}")
    with col3:
        st.metric("Defleksi Minimum (mm)", f"{stats['min_deflection']*1000:.3f}")
        st.metric("Standar Deviasi (mm)", f"{stats['std_deflection']*1000:.3f}")
        st.metric("Defleksi di Pusat (mm)", f"{stats['deflection_at_center']*1000:.3f}")

    st.markdown("---")

    # Display convergence information
    st.markdown("### Informasi Konvergensi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Iterasi", iterations)
    with col2:
        st.metric("Error Final", f"{error:.2e}")

    # Convergence plot
    convergence_fig = plot_convergence(iterations, error, solver_name)
    st.pyplot(convergence_fig)
    plt.close()

    st.markdown("---")

    # 3D Visualization
    st.markdown("### Visualisasi 3D Defleksi Plat")
    fig_3d = plot_deflection_3d(X, Y, w, title=f"Plate Deflection - {solver_name}")
    st.pyplot(fig_3d)
    plt.close()

    st.markdown("---")

    # Contour Visualization
    st.markdown("### Kontur Defleksi Plat")
    fig_contour = plot_deflection_contour(X, Y, w, title=f"Plate Deflection Contours - {solver_name}")
    st.pyplot(fig_contour)
    plt.close()

    # Additional information
    st.markdown("### Informasi Tambahan")
    st.info("""
    **Catatan:**
    - Defleksi plat dinyatakan dalam satuan relatif
    - Solusi menggunakan metode finite difference 2D orde tinggi
    - Batasan: Plat persegi dengan kondisi batas simply supported
    """)
