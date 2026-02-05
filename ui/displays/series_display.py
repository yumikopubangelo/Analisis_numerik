"""
Display module untuk Series methods (Taylor) dan ODE solving methods (Euler, Taylor ODE 2, Runge-Kutta).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.series.taylor_series import evaluate_taylor_at_points
from ui.explanation import explanation


def display_ode_results(result, params):
    """
    Display results for ODE solving methods (Euler, Taylor ODE 2, Runge-Kutta).
    
    Parameters:
    -----------
    result : dict
        Results from ODE solver
    params : dict
        Input parameters
    """
    # Success message
    st.success(f"ODE berhasil diselesaikan dengan metode {result['method']}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Fungsi", params['func_str'])
    with col2:
        st.metric("x₀", f"{params['x0']:.4f}")
    with col3:
        st.metric("y₀", f"{params['y0']:.4f}")
    with col4:
        st.metric("x Evaluasi", f"{params['x_eval']:.4f}")
    with col5:
        st.metric("Jumlah Langkah", params['n_steps'])
    
    st.markdown("---")
    
    # Results table
    st.markdown("#### Hasil Perhitungan")
    
    # Handle case when exact solution is None
    exact_value = result['exact']
    exact_str = f"{exact_value:.6f}" if exact_value is not None else "-"
    
    abs_error = result['absolute_error']
    abs_error_str = f"{abs_error:.6e}" if abs_error is not None else "-"
    
    rel_error = result['relative_error']
    rel_error_str = f"{rel_error:.6e}" if rel_error is not None else "-"
    
    df_results = pd.DataFrame({
        'Metode': [result['method']],
        'Aproksimasi y(x)': [f"{result['approximation']:.6f}"],
        'Nilai Sebenarnya': [exact_str],
        'Error Absolut': [abs_error_str],
        'Error Relatif': [rel_error_str]
    })
    st.dataframe(df_results, use_container_width=True)
    
    st.markdown("---")
    
    # Plot comparison
    st.markdown("#### Plot Solusi")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result['x_values'], result['y_values'], 'b-', label=f"{result['method']} Approksimasi", linewidth=2)
    ax.axvline(params['x0'], color='gray', linestyle=':', label=f'x₀ = {params["x0"]}')
    ax.axvline(params['x_eval'], color='green', linestyle=':', label=f'x = {params["x_eval"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.set_title(f'Solusi ODE: {params["func_str"]}')
    st.pyplot(fig)
    
    # Explanation
    explanation("ODE")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.series.taylor_series import evaluate_taylor_at_points
from ui.explanation import explanation


def display_taylor_results(result, params):
    """
    Display results untuk deret Taylor.
    
    Parameters:
    -----------
    result : dict
        Hasil perhitungan Taylor series
    params : dict
        Parameter input
    """
    # Success message
    st.success(f"Deret Taylor berhasil dihitung dengan {params['n_terms']} suku")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Fungsi", params['func_str'])
    with col2:
        st.metric("Titik Ekspansi (x₀)", f"{params['x0']:.4f}")
    with col3:
        st.metric("Nilai y Awal (y₀)", f"{params['y0']:.4f}")
    with col4:
        st.metric("Jumlah Suku", params['n_terms'])
    with col5:
        if result['approximations']:
            approx = result['approximations'][-1]
            st.metric("Aproksimasi di x", f"{approx['Nilai Aproksimasi']:.6f}")
    
    st.markdown("---")
    
    # Show error bound and tolerance bound if available
    if 'error_bound' in result and result['error_bound'] is not None:
        st.info(f"📊 **Batas Error (Error Bound)**: {result['error_bound']['bound']:.6e}")
        st.caption(f"Rumus: {result['error_bound']['formula']}")
        st.caption(f"Keterangan: {result['error_bound']['description']}")
    
    if 'tolerance_bound' in result and result['tolerance_bound'] is not None:
        st.info(f"📊 **Batas Toleransi (Tolerance Bound)**: {result['tolerance_bound']['tolerance']:.6e}")
        st.caption(f"Rumus: {result['tolerance_bound']['formula']}")
        st.caption(f"Keterangan: {result['tolerance_bound']['description']}")
    
    st.markdown("---")
    
    # Show series expression
    with st.expander("Ekspansi Deret Taylor", expanded=True):
        st.latex(result['series_expr'])
    
    # Two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Plot comparison
        if result['approximations']:
            x_range = np.linspace(params['x0'] - 2, params['x0'] + 2, 100)
            true_vals, approx_vals, y0 = evaluate_taylor_at_points(
                params['func_str'], params['x0'], params['n_terms'], x_range, params['y0']
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x_range, true_vals, 'b-', label='Fungsi Asli', linewidth=2)
            ax.plot(x_range, approx_vals, 'r--', label=f'Taylor ({params["n_terms"]} suku)', linewidth=2)
            ax.axvline(params['x0'], color='gray', linestyle=':', label=f'x₀ = {params["x0"]}')
            ax.axvline(params['x_eval'], color='green', linestyle=':', label=f'x = {params["x_eval"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Perbandingan Fungsi Asli vs Aproksimasi Taylor')
            st.pyplot(fig)
    
    with col2:
        # Show terms table
        st.markdown("#### Suku-suku Deret")
        df_terms = pd.DataFrame(result['terms'])
        st.dataframe(df_terms[['n', 'coefficient', 'term']], use_container_width=True)
    
    # Show approximation table if available
    if result['approximations']:
        st.markdown("#### Konvergensi Aproksimasi")
        
        # Create DataFrame with Indonesian column names
        df_approx = pd.DataFrame(result['approximations'])
        
        # Select columns to display (Indonesian names)
        display_columns = [
            'Jumlah Suku', 'Nilai Aproksimasi', 'Nilai Sebenarnya', 
            'Error Absolut', 'Error Relatif', 'Error Persentase', 
            'Memenuhi Batas Error', 'Memenuhi Toleransi', 'Keterangan'
        ]
        
        # Filter columns that exist
        available_columns = [col for col in display_columns if col in df_approx.columns]
        
        # Format the DataFrame for better display
        df_display = df_approx[available_columns].copy()
        
        # Format numeric columns
        if 'Nilai Aproksimasi' in df_display.columns:
            df_display['Nilai Aproksimasi'] = df_display['Nilai Aproksimasi'].apply(lambda x: f"{x:.6f}" if x is not None else "-")
        if 'Nilai Sebenarnya' in df_display.columns:
            df_display['Nilai Sebenarnya'] = df_display['Nilai Sebenarnya'].apply(lambda x: f"{x:.6f}" if x is not None else "-")
        if 'Error Absolut' in df_display.columns:
            df_display['Error Absolut'] = df_display['Error Absolut'].apply(lambda x: f"{x:.6e}" if x is not None else "-")
        if 'Error Relatif' in df_display.columns:
            df_display['Error Relatif'] = df_display['Error Relatif'].apply(lambda x: f"{x:.6e}" if x is not None else "-")
        if 'Error Persentase' in df_display.columns:
            df_display['Error Persentase'] = df_display['Error Persentase'].apply(lambda x: f"{x:.4f}%" if x is not None else "-")
        
        # Display the table
        st.dataframe(df_display, use_container_width=True)
        
        # Add explanation in Indonesian
        st.markdown("""
        **Penjelasan Tabel:**
        - **Jumlah Suku**: Banyaknya suku yang digunakan dalam aproksimasi
        - **Nilai Aproksimasi**: Nilai yang dihitung menggunakan deret Taylor
        - **Nilai Sebenarnya**: Nilai fungsi asli pada titik tersebut
        - **Error Absolut**: Selisih mutlak antara nilai aproksimasi dan nilai sebenarnya
        - **Error Relatif**: Error absolut dibandingkan dengan nilai sebenarnya
        - **Error Persentase**: Error relatif dalam bentuk persentase
        - **Memenuhi Batas Error**: Apakah error berada dalam batas maksimum yang diizinkan
        - **Memenuhi Toleransi**: Apakah error berada dalam batas toleransi yang ditentukan
        - **Keterangan**: Status konvergensi (Konvergen/Belum Konvergen)
        """)
        
        # Plot error convergence
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(df_approx['Jumlah Suku'], df_approx['Error Absolut'], 'o-', label='Error Absolut')
        ax.set_xlabel('Jumlah Suku')
        ax.set_ylabel('Error (log scale)')
        ax.set_title('Konvergensi Error')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    # Explanation
    explanation("Taylor")
