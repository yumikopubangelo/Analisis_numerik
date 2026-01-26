"""
Display module untuk Series methods (Taylor).
"""

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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fungsi", params['func_str'])
    with col2:
        st.metric("Titik Ekspansi (x₀)", f"{params['x0']:.4f}")
    with col3:
        st.metric("Jumlah Suku", params['n_terms'])
    with col4:
        if result['approximations']:
            approx = result['approximations'][-1]
            st.metric("Aproksimasi di x", f"{approx['approximation']:.6f}")
    
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
            true_vals, approx_vals = evaluate_taylor_at_points(
                params['func_str'], params['x0'], params['n_terms'], x_range
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
        df_approx = pd.DataFrame(result['approximations'])
        st.dataframe(df_approx, use_container_width=True)
        
        # Plot error convergence
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(df_approx['n_terms'], df_approx['abs_error'], 'o-', label='Absolute Error')
        ax.set_xlabel('Jumlah Suku')
        ax.set_ylabel('Error (log scale)')
        ax.set_title('Konvergensi Error')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    # Explanation
    explanation("Taylor")
