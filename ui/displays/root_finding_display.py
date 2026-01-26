"""
Display module untuk Root Finding methods.
"""

import streamlit as st
from ui.output_table import output_table
from ui.plots import plot_function, plot_convergence
from ui.explanation import explanation


def display_root_finding_results(root, iterations, params, method, is_interval=True):
    """
    Display results untuk metode root finding.
    
    Parameters:
    -----------
    root : float
        Akar yang ditemukan
    iterations : list
        List iterasi
    params : dict
        Parameter input
    method : str
        Nama metode
    is_interval : bool
        Apakah metode menggunakan interval
    """
    # Success message
    st.success(f"Akar ditemukan: **{root:.8f}** setelah {len(iterations)} iterasi")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akar (x)", f"{root:.8f}")
    with col2:
        st.metric("f(x)", f"{params['func'](root):.2e}")
    with col3:
        st.metric("Iterasi", len(iterations))
    
    st.markdown("---")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        output_table(iterations)
    
    with col2:
        if is_interval:
            plot_function(params['func'], params['a'], params['b'], root)
        else:
            # For Newton and Secant, create appropriate range
            x_range = max(abs(root) * 2, 5)
            plot_function(params['func'], root - x_range, root + x_range, root)
        
        plot_convergence(iterations)
    
    # Explanation
    explanation(method)
