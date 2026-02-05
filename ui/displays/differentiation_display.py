"""
Display module untuk Numerical Differentiation (diferensiasi numerik).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.differentiation import NumericalDifferentiation


def display_differentiation_results(params):
    """
    Display results untuk numerical differentiation.
    
    Parameters:
    -----------
    params : dict
        Parameter input dari form
    """
    nd = NumericalDifferentiation(params['func_str'])
    
    if params['method'] == "Forward Difference":
        result = nd.forward_difference(params['x_val'], params['h'])
    elif params['method'] == "Backward Difference":
        result = nd.backward_difference(params['x_val'], params['h'])
    elif params['method'] == "Central Difference (1st Derivative)":
        result = nd.central_difference_first(params['x_val'], params['h'])
    elif params['method'] == "Central Difference (2nd Derivative)":
        result = nd.central_difference_second(params['x_val'], params['h'])
    elif params['method'] == "Compare All Methods":
        results = nd.compare_first_derivative_methods(params['x_val'], params['h'])
        display_comparison_results(results, params)
        return
    elif params['method'] == "Convergence Analysis":
        h_values = np.logspace(np.log10(params['h_min']), np.log10(params['h_max']), params['num_points'])
        results = nd.convergence_analysis(params['x_val'], h_values, params['derivative_order'])
        display_convergence_analysis(results, params)
        return
    elif params['method'] == "Optimal h Analysis":
        result = nd.optimal_h_analysis(params['x_val'], params['h_min'], params['h_max'], params['num_points'])
        display_optimal_h_analysis(result, params)
        return
    elif params['method'] == "Multi-point Differentiation":
        results = nd.differentiate_at_points(params['x_points'], params['h'], 
                                           params['derivative_order'], params['diff_method'])
        display_multi_point_results(results, params)
        return
    
    # Display single method results
    display_single_method_result(result, params)


def display_single_method_result(result, params):
    """
    Display results untuk single differentiation method.
    """
    st.success(f"Hasil diferensiasi numerik berhasil dihitung!")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Metode", result['method'])
    with col2:
        st.metric("Titik x", f"{result['x']:.6f}")
    with col3:
        st.metric("Aproksimasi", f"{result['approximation']:.8f}")
    with col4:
        st.metric("Nilai True", f"{result['true_value']:.8f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Detail Perhitungan")
        st.write(f"**Fungsi:** `{params['func_str']}`")
        st.write(f"**Langkah (h):** {result['h']:.2e}")
        st.write(f"**Aproksimasi:** {result['approximation']:.15f}")
        st.write(f"**Nilai sebenarnya:** {result['true_value']:.15f}")
        st.write(f"**Error absolut:** {result['absolute_error']:.8e}")
        st.write(f"**Error relatif:** {result['relative_error']:.8e}")
        st.write(f"**Error persentase:** {result['percentage_error']:.6f}%")
    
    with col2:
        # Plot function and derivative
        plot_differentiation_visualization(params, result)


def display_comparison_results(results, params):
    """
    Display comparison results untuk semua metode diferensiasi.
    """
    st.success(f"Perbandingan {len(results)} metode diferensiasi berhasil!")
    
    # Metrics
    best_result = min(results, key=lambda x: x['absolute_error'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Metode Terbaik", best_result['method'])
    with col2:
        st.metric("Error Terkecil", f"{best_result['absolute_error']:.2e}")
    with col3:
        st.metric("Titik x", f"{best_result['x']:.6f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Tabel Perbandingan")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        methods = [r['method'] for r in results]
        approximations = [r['approximation'] for r in results]
        errors = [r['absolute_error'] for r in results]
        
        # Bar chart for approximations
        ax1.bar(methods, approximations, color=['#667eea', '#764ba2', '#f093fb'])
        ax1.set_ylabel('Aproksimasi')
        ax1.set_title('Perbandingan Aproksimasi')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Bar chart for errors (log scale)
        ax2.bar(methods, errors, color=['#667eea', '#764ba2', '#f093fb'])
        ax2.set_ylabel('Error Absolut (log scale)')
        ax2.set_title('Perbandingan Error')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)


def display_convergence_analysis(results, params):
    """
    Display convergence analysis results.
    """
    st.success(f"Analisis konvergensi untuk {len(results)} nilai h berhasil!")
    
    # Metrics
    min_h_result = min(results, key=lambda x: x['h'])
    max_h_result = max(results, key=lambda x: x['h'])
    best_result = min(results, key=lambda x: x['central_error'] if params['derivative_order'] == 1 else x['central_error'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("h Min", f"{min_h_result['h']:.2e}")
    with col2:
        st.metric("h Max", f"{max_h_result['h']:.2e}")
    with col3:
        st.metric("h Optimal", f"{best_result['h']:.2e}")
    with col4:
        st.metric("Error Minimum", f"{best_result['central_error']:.2e}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Tabel Hasil")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        h_values = [r['h'] for r in results]
        if params['derivative_order'] == 1:
            forward_errors = [r['forward_error'] for r in results]
            backward_errors = [r['backward_error'] for r in results]
            central_errors = [r['central_error'] for r in results]
            
            ax1.loglog(h_values, forward_errors, 'o-', linewidth=2, markersize=6, label='Forward')
            ax1.loglog(h_values, backward_errors, 'o-', linewidth=2, markersize=6, label='Backward')
            ax1.loglog(h_values, central_errors, 'o-', linewidth=2, markersize=6, label='Central')
            ax1.set_ylabel('Error Absolut')
            ax1.set_title('Konvergensi Metode Turunan Pertama')
            ax1.legend()
        else:
            central_errors = [r['central_error'] for r in results]
            ax1.loglog(h_values, central_errors, 'o-', linewidth=2, markersize=6, color='#667eea')
            ax1.set_ylabel('Error Absolut')
            ax1.set_title('Konvergensi Metode Turunan Kedua')
        
        ax1.set_xlabel('h')
        ax1.grid(True, alpha=0.3)
        
        # Plot approximation vs h
        ax2.semilogx(h_values, [r['true_value'] for r in results], 'k--', linewidth=2, label='True Value')
        if params['derivative_order'] == 1:
            ax2.semilogx(h_values, [r['forward_approx'] for r in results], 'o-', linewidth=1, markersize=4, label='Forward')
            ax2.semilogx(h_values, [r['backward_approx'] for r in results], 'o-', linewidth=1, markersize=4, label='Backward')
            ax2.semilogx(h_values, [r['central_approx'] for r in results], 'o-', linewidth=1, markersize=4, label='Central')
            ax2.set_title('Aproksimasi vs h')
            ax2.legend()
        else:
            ax2.semilogx(h_values, [r['central_approx'] for r in results], 'o-', linewidth=1, markersize=4, color='#667eea')
            ax2.set_title('Aproksimasi Turunan Kedua vs h')
        
        ax2.set_xlabel('h')
        ax2.set_ylabel('Aproksimasi')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


def display_optimal_h_analysis(result, params):
    """
    Display optimal h analysis results.
    """
    st.success(f"Analisis h optimal untuk semua metode berhasil!")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("h Optimal Forward", f"{result['optimal']['forward']['h']:.2e}")
    with col2:
        st.metric("h Optimal Backward", f"{result['optimal']['backward']['h']:.2e}")
    with col3:
        st.metric("h Optimal Central 1st", f"{result['optimal']['central1']['h']:.2e}")
    with col4:
        st.metric("h Optimal Central 2nd", f"{result['optimal']['central2']['h']:.2e}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Tabel h Optimal")
        optimal_data = {
            'Method': ['Forward', 'Backward', 'Central (1st)', 'Central (2nd)'],
            'Optimal h': [
                result['optimal']['forward']['h'],
                result['optimal']['backward']['h'],
                result['optimal']['central1']['h'],
                result['optimal']['central2']['h']
            ],
            'Minimum Error': [
                result['optimal']['forward']['forward_error'],
                result['optimal']['backward']['backward_error'],
                result['optimal']['central1']['central1_error'],
                result['optimal']['central2']['central2_error']
            ]
        }
        df = pd.DataFrame(optimal_data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        # Plot error vs h for all methods
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.loglog(result['h_values'], [r['forward_error'] for r in result['results']], 'o-', linewidth=2, markersize=4, label='Forward', color='#667eea')
        ax.loglog(result['h_values'], [r['backward_error'] for r in result['results']], 'o-', linewidth=2, markersize=4, label='Backward', color='#764ba2')
        ax.loglog(result['h_values'], [r['central1_error'] for r in result['results']], 'o-', linewidth=2, markersize=4, label='Central (1st)', color='#f093fb')
        ax.loglog(result['h_values'], [r['central2_error'] for r in result['results']], 'o-', linewidth=2, markersize=4, label='Central (2nd)', color='#4facfe')
        
        ax.set_xlabel('h')
        ax.set_ylabel('Error Absolut (log scale)')
        ax.set_title('Error vs h untuk Semua Metode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)


def display_multi_point_results(results, params):
    """
    Display multi-point differentiation results.
    """
    st.success(f"Diferensiasi untuk {len(results)} titik berhasil!")
    
    # Metrics
    avg_error = np.mean([r['absolute_error'] for r in results])
    max_error = np.max([r['absolute_error'] for r in results])
    min_error = np.min([r['absolute_error'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Titik", len(results))
    with col2:
        st.metric("Error Rata-rata", f"{avg_error:.2e}")
    with col3:
        st.metric("Error Maksimum", f"{max_error:.2e}")
    with col4:
        st.metric("Error Minimum", f"{min_error:.2e}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Tabel Hasil")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        # Plot derivative values and errors
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        x_values = [r['x'] for r in results]
        approximations = [r['approximation'] for r in results]
        true_values = [r['true_value'] for r in results]
        errors = [r['absolute_error'] for r in results]
        
        ax1.plot(x_values, true_values, 'k-', linewidth=2, label='True Value')
        ax1.plot(x_values, approximations, 'ro-', linewidth=2, markersize=6, label='Approximation')
        ax1.set_ylabel('Derivative Value')
        ax1.set_title('Perbandingan Turunan')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(x_values, errors, 'bo-', linewidth=2, markersize=6, color='#667eea')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Error Absolut (log scale)')
        ax2.set_title('Error pada Tiap Titik')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


def plot_differentiation_visualization(params, result):
    """
    Plot visualization untuk differentiation.
    """
    nd = NumericalDifferentiation(params['func_str'])
    
    # Generate x values around x_val
    x_range = np.linspace(params['x_val'] - 2, params['x_val'] + 2, 200)
    y_range = nd.true_value(x_range)
    
    # Calculate true derivative
    if '1st' in result['method'] or result['method'] in ['Forward Difference', 'Backward Difference']:
        dy_range = nd.true_first_derivative(x_range)
    else:
        dy_range = nd.true_second_derivative(x_range)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # Plot function
    ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
    ax1.plot(params['x_val'], nd.true_value(params['x_val']), 'ro', markersize=10, label=f'f({params["x_val"]:.2f})')
    ax1.axvline(params['x_val'], color='gray', linestyle=':', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Grafik Fungsi')
    
    # Plot derivative
    ax2.plot(x_range, dy_range, 'b-', linewidth=2, label='True Derivative')
    ax2.plot(params['x_val'], result['true_value'], 'ro', markersize=8, label='True Value')
    ax2.plot(params['x_val'], result['approximation'], 'go', markersize=8, label='Approximation')
    ax2.axvline(params['x_val'], color='gray', linestyle=':', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title(f"Grafik Turunan ({result['method']})")
    
    plt.tight_layout()
    st.pyplot(fig)
