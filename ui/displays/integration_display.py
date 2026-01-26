"""
Display module untuk Integration methods.
"""

import streamlit as st
import pandas as pd
from ui.plots import plot_integration
from ui.explanation import explanation


def display_integration_results(integral, steps, calc_info, params, method):
    """
    Display results untuk metode integrasi numerik.
    
    Parameters:
    -----------
    integral : float
        Hasil integrasi
    steps : list
        List langkah perhitungan
    calc_info : dict
        Informasi perhitungan
    params : dict
        Parameter input
    method : str
        Nama metode
    """
    # Success message
    st.success(f"Hasil integrasi: **{integral:.8f}**")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hasil Integral", f"{integral:.6f}")
    with col2:
        st.metric("Interval [a, b]", f"[{params['a']}, {params['b']}]")
    with col3:
        st.metric("Jumlah Subinterval", params['n'])
    with col4:
        st.metric("Lebar (h)", f"{calc_info['h']:.6f}")
    
    st.markdown("---")
    
    # Show calculation details
    with st.expander("Detail Perhitungan", expanded=True):
        st.markdown(f"**Formula yang digunakan:**")
        st.code(calc_info['formula'], language="text")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**f(a) = f({params['a']})** = {calc_info['f(a)']:.6f}")
            st.write(f"**f(b) = f({params['b']})** = {calc_info['f(b)']:.6f}")
        
        with col2:
            if method == "Simpson":
                st.write(f"**Sum (odd indices)** = {calc_info['sum_odd']:.6f}")
                st.write(f"**Sum (even indices)** = {calc_info['sum_even']:.6f}")
            else:
                st.write(f"**Sum (middle terms)** = {calc_info['sum_middle']:.6f}")
    
    # Visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        plot_integration(params['func'], params['a'], params['b'], params['n'], method)
    
    with col2:
        if steps:
            st.markdown("#### Tabel Evaluasi Fungsi")
            df = pd.DataFrame(steps[:10])  # Show first 10 rows
            st.dataframe(df, use_container_width=True)
    
    # Explanation
    explanation(method)
