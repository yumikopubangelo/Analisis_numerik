import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from core.pde import solve_advection_diffusion_1d, plot_concentration_snapshot, plot_concentration_evolution


def display_advection_diffusion_results(params):
    """
    Display results for 1D Advection-Diffusion equation solver.
    
    Parameters:
    -----------
    params : dict
        Input parameters including:
        - U: velocity
        - K: diffusion coefficient
        - x_min: minimum x
        - x_max: maximum x
        - nx: number of grid points
        - dt: time step
        - num_steps: number of time steps
    """
    st.success("Advection-Diffusion 1D berhasil diselesaikan")
    
    # Solve the PDE
    result = solve_advection_diffusion_1d(
        U=params['U'],
        K=params['K'],
        x_min=params['x_min'],
        x_max=params['x_max'],
        nx=params['nx'],
        dt=params['dt'],
        num_steps=params['num_steps']
    )
    
    # Show parameters
    st.markdown("### Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kecepatan Aliran (U)", f"{result['parameters']['U']:.2f} m/s")
        st.metric("Koefisien Difusi (K)", f"{result['parameters']['K']:.3f} m²/s")
        st.metric("Jumlah Grid X", result['parameters']['nx'])
    with col2:
        st.metric("Langkah Waktu (Δt)", f"{result['parameters']['dt']:.4f} s")
        st.metric("Jumlah Iterasi Waktu", result['parameters']['num_steps'])
        st.metric("Grid Spacing (Δx)", f"{result['parameters']['dx']:.4f} m")
    
    # Display discretized PDE form
    st.markdown("### Bentuk Diskritisasi PDE")
    
    st.markdown("""
    Persamaan Adveksi-Difusi 1D asli:
    $$
    \\frac{\\partial C}{\\partial t} = -U \\frac{\\partial C}{\\partial x} + K \\frac{\\partial^2 C}{\\partial x^2}
    $$
    """)
    
    st.markdown("""
    **Diskritisasi Finite Difference (Skema Implisit):**
    
    Persamaan diferensial parsial diubah menjadi sistem persamaan linear dengan diskritisasi:
    - Turunan waktu: Metode Euler mundur
    - Turunan spasial pertama: Selisih maju-mundur (central difference)
    - Turunan spasial kedua: Selisih tengah (second-order central difference)
    
    Persamaan diskrit untuk titik interior $i$ pada waktu $n+1$:
    $$
    C_i^{n+1} = C_i^n + \\frac{\\Delta t}{\\Delta x^2} \\left[ K(C_{i+1}^n - 2C_i^n + C_{i-1}^n) - \\frac{U \\Delta x}{2}(C_{i+1}^n - C_{i-1}^n) \\right]
    $$
    
    **Sistem Persamaan Linear:**
    
    Persamaan di atas dapat disusun menjadi sistem tridiagonal:
    $$
    A \\mathbf{C}^{n+1} = \\mathbf{C}^n
    $$
    
    Dimana matriks tridiagonal $A$ memiliki struktur:
    - Elemen diagonal: $1 + 2 \\frac{\\Delta t K}{\\Delta x^2}$
    - Elemen sub-diagonal: $-\\frac{\\Delta t}{\\Delta x^2} \\left( K + \\frac{U \\Delta x}{2} \\right)$
    - Elemen super-diagonal: $-\\frac{\\Delta t}{\\Delta x^2} \\left( K - \\frac{U \\Delta x}{2} \\right)$
    
    **Kondisi Batas:**
    - Dirichlet: $C_0^{n} = 0$ dan $C_{N-1}^{n} = 0$ untuk semua waktu $n$
    """)
    
    # Plot concentration evolution
    st.markdown("### Visualisasi Evolusi Konsentrasi")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Pilih Waktu")
        selected_time = st.slider(
            "Waktu (detik)",
            min_value=0.0,
            max_value=result['t'][-1],
            value=0.0,
            step=result['t'][-1]/10,
            format="%.3f"
        )
        
        # Find closest time index
        time_idx = np.argmin(np.abs(result['t'] - selected_time))
    
    with col2:
        # Plot concentration at selected time
        st.markdown(f"#### Profil Konsentrasi pada t = {result['t'][time_idx]:.3f}")
        fig = plot_concentration_snapshot(
            result['x'],
            result['C'][time_idx],
            result['t'][time_idx]
        )
        st.pyplot(fig)
    
    # Plot concentration evolution
    st.markdown("#### Heatmap Evolusi Konsentrasi")
    fig = plot_concentration_evolution(
        result['x'],
        result['C'],
        result['t']
    )
    st.pyplot(fig)
    
    # Display statistics
    st.markdown("### Statistik Konsentrasi")
    
    max_concentration = np.max(result['C'])
    min_concentration = np.min(result['C'])
    avg_concentration = np.mean(result['C'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Konsentrasi Maksimum", f"{max_concentration:.4f}")
    with col2:
        st.metric("Konsentrasi Minimum", f"{min_concentration:.4f}")
    with col3:
        st.metric("Konsentrasi Rata-rata", f"{avg_concentration:.4f}")
    
    # Save data if requested
    if st.button("Unduh Data Konsentrasi", use_container_width=True):
        import pandas as pd
        
        # Create DataFrame
        data = pd.DataFrame()
        data['x'] = result['x']
        
        # Add concentration at each time step
        for i, t in enumerate(result['t']):
            data[f'C(t={t:.3f})'] = result['C'][i]
        
        # Convert to CSV
        csv = data.to_csv(index=False)
        
        st.download_button(
            "Download CSV",
            csv,
            "advection_diffusion_1d_data.csv",
            "text/csv",
            key='download-csv'
        )
