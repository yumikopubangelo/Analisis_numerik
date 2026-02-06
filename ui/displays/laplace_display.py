"""
PDE Solver Display Module

This module provides functions to display the results of the PDE solver
for biharmonic plate equations and Laplace equation in the Streamlit interface.
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
from core.pde.laplace_solver import (
    solve_laplace_equation,
    solve_laplace_with_function_bc,
    plot_phi_contour,
    plot_phi_wireframe,
    plot_phi_surface,
    plot_convergence as plot_laplace_convergence,
    print_solution_summary
)


def display_pde_results(X: np.ndarray, Y: np.ndarray, w: np.ndarray,
                       iterations: int, error: float, solver_name: str,
                       q: float, D: float):
    """
    Display PDE solver results with visualizations.
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


# ============================================================================
# LAPLACE SOLVER UI FUNCTIONS
# ============================================================================

def display_laplace_solver():
    """
    Display the Laplace equation solver interface.
    """
    st.markdown("""
    ## Solver Persamaan Laplace 2D
    
    Persamaan Laplace menggambarkan distribusi steady-state dari:
    - **Suhu** pada plat logam (konduksi panas)
    - **Potensial elektrostatik**
    - **Aliran fluida potensial**
    
    **Persamaan:**
    $$\\nabla^2 \\phi = \\frac{\\partial^2 \\phi}{\\partial x^2} + \\frac{\\partial^2 \\phi}{\\partial y^2} = 0$$
    """)
    
    st.markdown("---")
    
    # Input parameters in main area (not sidebar to avoid conflicts)
    st.markdown("### Parameter Domain")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_min = st.number_input("x_min", value=0.0, format="%.2f")
    with col2:
        x_max = st.number_input("x_max", value=1.0, format="%.2f")
    with col3:
        y_min = st.number_input("y_min", value=0.0, format="%.2f")
    with col4:
        y_max = st.number_input("y_max", value=1.0, format="%.2f")
    
    st.markdown("### Grid Resolution")
    col1, col2 = st.columns(2)
    with col1:
        nx = st.slider("Jumlah titik x (nx)", 11, 101, 51, step=10)
    with col2:
        ny = st.slider("Jumlah titik y (ny)", 11, 101, 51, step=10)
    
    st.markdown("### Kondisi Batas (Dirichlet)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bc_left = st.number_input("Nilai Kiri (x=x_min)", value=100.0, format="%.2f")
    with col2:
        bc_right = st.number_input("Nilai Kanan (x=x_max)", value=0.0, format="%.2f")
    with col3:
        bc_bottom = st.number_input("Nilai Bawah (y=y_min)", value=0.0, format="%.2f")
    with col4:
        bc_top = st.number_input("Nilai Atas (y=y_max)", value=0.0, format="%.2f")
    
    st.markdown("### Parameter Solver")
    col1, col2, col3 = st.columns(3)
    with col1:
        solver = st.selectbox("Metode Solver", ["sor", "gauss-seidel", "jacobi"])
    with col2:
        if solver == "sor":
            omega = st.slider("Omega (ω) untuk SOR", 1.0, 1.99, 1.7, step=0.01)
        else:
            omega = 1.5
    with col3:
        tol = st.number_input("Toleransi Konvergensi", value=1e-6, format="%.2e")
    
    max_iter = st.number_input("Maksimum Iterasi", value=10000, step=1000)
    
    # Solve button
    if st.button("Hitung Solusi Persamaan Laplace", type="primary"):
        with st.spinner("Menghitung solusi..."):
            # Solve the Laplace equation
            result = solve_laplace_equation(
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                nx=nx, ny=ny,
                bc_type='dirichlet',
                bc_left=bc_left,
                bc_right=bc_right,
                bc_bottom=bc_bottom,
                bc_top=bc_top,
                solver=solver,
                omega=omega,
                tol=tol,
                max_iter=max_iter
            )
            
            # Store result in session state
            st.session_state['laplace_result'] = result
            st.session_state['laplace_params'] = {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'nx': nx, 'ny': ny,
                'bc_left': bc_left, 'bc_right': bc_right,
                'bc_bottom': bc_bottom, 'bc_top': bc_top,
                'solver': solver, 'omega': omega
            }
    
    # Display results if available
    if 'laplace_result' in st.session_state:
        display_laplace_results(st.session_state['laplace_result'], 
                               st.session_state['laplace_params'])


def display_laplace_results(result, params):
    """
    Display Laplace equation solver results.
    """
    st.markdown("---")
    
    # Statistics
    stats = result['statistics']
    
    st.markdown("### Hasil Solusi Persamaan Laplace")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("φ Minimum", f"{stats['min_phi']:.4f}")
    with col2:
        st.metric("φ Maksimum", f"{stats['max_phi']:.4f}")
    with col3:
        st.metric("Rata-rata φ", f"{stats['mean_phi']:.4f}")
    with col4:
        st.metric("Std Deviasi", f"{stats['std_phi']:.4f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iterasi", result['iterations'])
    with col2:
        st.metric("Error Final", f"{result['error']:.2e}")
    with col3:
        st.metric("Konvergen", "Ya" if result['converged'] else "Tidak")
    
    st.markdown("---")
    
    # Visualization options
    viz_type = st.selectbox("Tipe Visualisasi", ["Kontur 2D", "Permukaan 3D", "Wireframe 3D"])
    
    if viz_type == "Kontur 2D":
        st.markdown("#### Kontur φ(x,y)")
        fig = plot_phi_contour(result['X'], result['Y'], result['phi'],
                              f"Distribusi phi - {params['solver'].upper()}")
        st.pyplot(fig)
        plt.close()
        
    elif viz_type == "Permukaan 3D":
        st.markdown("#### Permukaan 3D φ(x,y)")
        fig = plot_phi_surface(result['X'], result['Y'], result['phi'],
                              f"Permukaan phi - {params['solver'].upper()}")
        st.pyplot(fig)
        plt.close()
        
    elif viz_type == "Wireframe 3D":
        st.markdown("#### Wireframe 3D φ(x,y)")
        fig = plot_phi_wireframe(result['X'], result['Y'], result['phi'],
                                 f"Wireframe phi - {params['solver'].upper()}")
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Convergence plot
    st.markdown("#### Informasi Konvergensi")
    fig_conv = plot_laplace_convergence(result['iterations'], result['error'], 
                                         params['solver'].upper())
    st.pyplot(fig_conv)
    plt.close()
    
    # Print summary
    st.markdown("#### Ringkasan Solusi")
    print_solution_summary(result)
    
    # Export options
    st.markdown("### Ekspor Hasil")
    
    # Option to download phi values
    phi_data = result['phi']
    phi_csv = "\n".join([",".join(map(str, row)) for row in phi_data])
    st.download_button(
        "Download Data φ (CSV)",
        phi_data.tobytes(),
        file_name="laplace_phi_solution.csv",
        mime="text/csv"
    )
    
    # Theory explanation
    st.markdown("---")
    st.markdown("""
    ### Teori Metode Iteratif
    
    **Metode Jacobi:**
    $$u_{i,j}^{k+1} = \\frac{1}{4}(u_{i+1,j}^k + u_{i-1,j}^k + u_{i,j+1}^k + u_{i,j-1}^k)$$
    
    **Metode Gauss-Seidel:**
    $$u_{i,j}^{k+1} = \\frac{1}{4}(u_{i+1,j}^{k+1} + u_{i-1,j}^{k+1} + u_{i,j+1}^{k+1} + u_{i,j-1}^{k+1})$$
    
    **Metode SOR (Successive Over-Relaxation):**
    $$u_{i,j}^{k+1} = (1-\\omega)u_{i,j}^k + \\frac{\\omega}{4}(u_{i+1,j}^{k+1} + u_{i-1,j}^{k+1} + u_{i,j+1}^{k+1} + u_{i,j-1}^{k+1})$$
    
    dengan $\\omega$ adalah faktor relaksasi ($1 < \\omega < 2$ untuk over-relaksasi).
    """)
