"""
Display module untuk Interpolation methods.
"""

import streamlit as st
import pandas as pd
from ui.plots import plot_interpolation
from ui.explanation import explanation


def display_interpolation_results(y_eval, steps, poly_str, params, method, div_diff_table=None):
    """
    Display results untuk metode interpolasi.
    
    Parameters:
    -----------
    y_eval : float
        Hasil interpolasi
    steps : list
        List langkah perhitungan
    poly_str : str
        String representasi polynomial
    params : dict
        Parameter input
    method : str
        Nama metode
    div_diff_table : list, optional
        Tabel divided differences untuk Newton
    """
    # Success message
    st.success(f"Hasil interpolasi di x = {params['x_eval']}: **{y_eval:.8f}**")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("x (input)", f"{params['x_eval']:.4f}")
    with col2:
        st.metric("P(x) (output)", f"{y_eval:.6f}")
    with col3:
        st.metric("Jumlah Titik Data", len(params['x_points']))
    
    st.markdown("---")
    
    # Show polynomial
    with st.expander("Polynomial yang Terbentuk", expanded=False):
        st.text(poly_str)
    
    # Two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        plot_interpolation(
            params['x_points'], params['y_points'],
            params['x_eval'], y_eval, method
        )
    
    with col2:
        if method == "Lagrange" and steps:
            st.markdown("#### Tabel Basis Lagrange")
            df = pd.DataFrame(steps)
            st.dataframe(df, use_container_width=True)
        
        elif method == "Newton" and div_diff_table is not None:
            st.markdown("#### Tabel Divided Differences")
            df = pd.DataFrame(div_diff_table)
            st.dataframe(df, use_container_width=True)
    
    # Explanation
    explanation(method)
