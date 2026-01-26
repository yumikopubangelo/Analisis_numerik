import streamlit as st
import numpy as np

from ui.sidebar import sidebar
from ui.input_form import input_form
from ui.output_table import output_table
from ui.plots import plot_function, plot_convergence, plot_integration, plot_interpolation
from ui.explanation import explanation

# Root finding methods
from core.root_findings.bisection import bisection_method
from core.root_findings.regula_falsi import regula_falsi_method
from core.root_findings.newton_raphson import newton_raphson_method
from core.root_findings.secant import secant_method

# Integration methods
from core.integration.trapezoidal import trapezoidal_method
from core.integration.simpson import simpson_method

# Interpolation methods
from core.interpolation.lagrange import lagrange_interpolation
from core.interpolation.newton import newton_interpolation

# Series methods
from core.series.taylor_series import taylor_series, evaluate_taylor_at_points

# Analysis Features
from core.analysis.numerical_features import NumericalAnalysis

st.set_page_config(
    page_title="Numerical Analysis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Aplikasi Pembelajaran Analisis Numerik</h1>
        <p>Pelajari metode numerik dengan visualisasi step-by-step yang interaktif!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    category, method = sidebar()
    
    # Main content
    params = input_form(category, method)
    
    if params:
        try:
            # ROOT FINDING METHODS
            if category == "Root Finding":
                if method == "Bisection":
                    root, iterations = bisection_method(
                        params['func'], params['a'], params['b'], 
                        params['tol'], params['max_iter']
                    )
                    display_root_finding_results(root, iterations, params, method)
                
                elif method == "Regula Falsi":
                    root, iterations = regula_falsi_method(
                        params['func'], params['a'], params['b'],
                        params['tol'], params['max_iter']
                    )
                    display_root_finding_results(root, iterations, params, method)
                
                elif method == "Newton-Raphson":
                    root, iterations = newton_raphson_method(
                        params['func'], params['func_str'], params['x0'],
                        params['tol'], params['max_iter']
                    )
                    display_root_finding_results(root, iterations, params, method, is_interval=False)
                
                elif method == "Secant":
                    root, iterations = secant_method(
                        params['func'], params['x0'], params['x1'],
                        params['tol'], params['max_iter']
                    )
                    display_root_finding_results(root, iterations, params, method, is_interval=False)
            
            # INTEGRATION METHODS
            elif category == "Integration":
                if method == "Trapezoidal":
                    integral, steps, calc_info = trapezoidal_method(
                        params['func'], params['a'], params['b'], params['n']
                    )
                    display_integration_results(integral, steps, calc_info, params, method)
                
                elif method == "Simpson":
                    integral, steps, calc_info = simpson_method(
                        params['func'], params['a'], params['b'], params['n']
                    )
                    display_integration_results(integral, steps, calc_info, params, method)
            
            # INTERPOLATION METHODS
            elif category == "Interpolation":
                if method == "Lagrange":
                    y_eval, steps, poly_str = lagrange_interpolation(
                        params['x_points'], params['y_points'], params['x_eval']
                    )
                    display_interpolation_results(y_eval, steps, poly_str, params, method)
                
                elif method == "Newton":
                    y_eval, div_diff_table, poly_str = newton_interpolation(
                        params['x_points'], params['y_points'], params['x_eval']
                    )
                    display_interpolation_results(
                        y_eval, None, poly_str, params, method, div_diff_table
                    )
            
            # SERIES METHODS
            elif category == "Series":
                if method == "Taylor":
                    result = taylor_series(
                        params['func_str'], params['x0'],
                        params['n_terms'], params['x_eval']
                    )
                    display_taylor_results(result, params)
            
            # ANALYSIS FEATURES
            elif category == "Analysis Features":
                if method == "True Value":
                    display_true_value_results(params)
                elif method == "Error Analysis":
                    display_error_analysis_results(params)
                elif method == "Tolerance Check":
                    display_tolerance_check_results(params)
                elif method == "Taylor Polynomial":
                    display_taylor_polynomial_results(params)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def display_root_finding_results(root, iterations, params, method, is_interval=True):
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

def display_integration_results(integral, steps, calc_info, params, method):
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
            import pandas as pd
            df = pd.DataFrame(steps[:10])  # Show first 10 rows
            st.dataframe(df, use_container_width=True)
    
    # Explanation
    explanation(method)

def display_interpolation_results(y_eval, steps, poly_str, params, method, div_diff_table=None):
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
            import pandas as pd
            df = pd.DataFrame(steps)
            st.dataframe(df, use_container_width=True)
        
        elif method == "Newton" and div_diff_table is not None:
            st.markdown("#### Tabel Divided Differences")
            import pandas as pd
            df = pd.DataFrame(div_diff_table)
            st.dataframe(df, use_container_width=True)
    
    # Explanation
    explanation(method)

def display_taylor_results(result, params):
    import pandas as pd
    import matplotlib.pyplot as plt
    
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

def display_true_value_results(params):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    na = NumericalAnalysis(params['func_str'])
    
    if params['input_type'] == 'single':
        # Single point evaluation
        true_val = na.true_value(params['x_val'])
        symbolic_val = na.true_value_symbolic(params['x_val'])
        
        st.success(f"Nilai sebenarnya berhasil dihitung!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fungsi", params['func_str'])
        with col2:
            st.metric("x", f"{params['x_val']:.6f}")
        with col3:
            st.metric("f(x)", f"{true_val:.10f}")
        
        st.markdown("---")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hasil Perhitungan")
            st.write(f"**Fungsi:** `{params['func_str']}`")
            st.write(f"**Nilai x:** {params['x_val']}")
            st.write(f"**Nilai numerik:** {true_val:.15f}")
            st.write(f"**Nilai simbolik:** {symbolic_val}")
        
        with col2:
            # Plot function
            x_range = np.linspace(params['x_val'] - 2, params['x_val'] + 2, 100)
            y_range = na.true_value(x_range)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, y_range, 'b-', linewidth=2, label=f'f(x) = {params["func_str"]}')
            ax.plot(params['x_val'], true_val, 'ro', markersize=10, label=f'f({params["x_val"]:.2f}) = {true_val:.4f}')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Grafik Fungsi')
            st.pyplot(fig)
    
    else:
        # Multiple points evaluation
        result = na.evaluate_at_points(params['x_points'])
        
        st.success(f"Nilai sebenarnya berhasil dihitung untuk {len(params['x_points'])} titik!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fungsi", params['func_str'])
        with col2:
            st.metric("Jumlah Titik", len(params['x_points']))
        with col3:
            st.metric("Range", f"[{min(params['x_points']):.2f}, {max(params['x_points']):.2f}]")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Table
            st.markdown("#### Tabel Nilai")
            df = pd.DataFrame({
                'x': result['x_points'],
                'f(x)': result['y_values']
            })
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(result['x_points'], result['y_values'], 'b-o', linewidth=2, markersize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Grafik f(x) = {params["func_str"]}')
            st.pyplot(fig)

def display_error_analysis_results(params):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    na = NumericalAnalysis(params['func_str'])
    
    if params['analysis_type'] == 'single':
        # Single approximation
        approx = params['approximations'][0]
        error_result = na.error_analysis(approx, params['x_val'])
        
        st.success("Analisis error berhasil!")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nilai True", f"{error_result['true_value']:.8f}")
        with col2:
            st.metric("Aproksimasi", f"{approx:.8f}")
        with col3:
            st.metric("Error Absolut", f"{error_result['absolute_error']:.2e}")
        with col4:
            st.metric("Error Relatif", f"{error_result['relative_error']:.2e}")
        
        st.markdown("---")
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Detail Analisis")
            st.write(f"**Fungsi:** `{params['func_str']}`")
            st.write(f"**Titik evaluasi (x):** {params['x_val']}")
            st.write(f"**Nilai sebenarnya:** {error_result['true_value']:.15f}")
            st.write(f"**Nilai aproksimasi:** {approx:.15f}")
            st.write(f"**Error absolut:** {error_result['absolute_error']:.8e}")
            st.write(f"**Error relatif:** {error_result['relative_error']:.8e}")
            st.write(f"**Error persentase:** {error_result['percentage_error']:.6f}%")
        
        with col2:
            # Visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            
            # Bar chart for errors
            errors = ['Absolut', 'Relatif']
            values = [error_result['absolute_error'], error_result['relative_error']]
            ax1.bar(errors, values, color=['#667eea', '#764ba2'])
            ax1.set_ylabel('Error')
            ax1.set_title('Perbandingan Error')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Comparison
            categories = ['True Value', 'Approximation']
            vals = [error_result['true_value'], approx]
            ax2.bar(categories, vals, color=['#28a745', '#ffc107'])
            ax2.set_ylabel('Nilai')
            ax2.set_title('Perbandingan Nilai')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        # Multiple approximations
        comparison = na.compare_approximations(params['approximations'], params['x_val'])
        
        st.success(f"Analisis error untuk {len(params['approximations'])} aproksimasi berhasil!")
        
        # Metrics
        best_approx = min(comparison, key=lambda x: x['absolute_error'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nilai True", f"{comparison[0]['true_value']:.8f}")
        with col2:
            st.metric("Aproksimasi Terbaik", f"{best_approx['approximation']:.8f}")
        with col3:
            st.metric("Error Terkecil", f"{best_approx['absolute_error']:.2e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Table
            st.markdown("#### Tabel Perbandingan Error")
            df = pd.DataFrame(comparison)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Plot convergence
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            
            iterations = [c['iteration'] for c in comparison]
            abs_errors = [c['absolute_error'] for c in comparison]
            rel_errors = [c['relative_error'] for c in comparison]
            
            ax1.semilogy(iterations, abs_errors, 'o-', linewidth=2, markersize=8, color='#667eea')
            ax1.set_xlabel('Iterasi')
            ax1.set_ylabel('Error Absolut (log scale)')
            ax1.set_title('Konvergensi Error Absolut')
            ax1.grid(True, alpha=0.3)
            
            ax2.semilogy(iterations, rel_errors, 'o-', linewidth=2, markersize=8, color='#764ba2')
            ax2.set_xlabel('Iterasi')
            ax2.set_ylabel('Error Relatif (log scale)')
            ax2.set_title('Konvergensi Error Relatif')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

def display_tolerance_check_results(params):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    na = NumericalAnalysis(params['func_str'])
    
    if params['check_type'] == 'with_true':
        # Check with true value
        true_val = na.true_value(params['x_val'])
        result = na.check_tolerance(params['approx'], true_val, params['tolerance'], params['error_type'])
        
        if result['converged']:
            st.success("✓ Aproksimasi MEMENUHI toleransi!")
        else:
            st.warning("✗ Aproksimasi BELUM memenuhi toleransi")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Aproksimasi", f"{params['approx']:.8f}")
        with col2:
            st.metric("Nilai True", f"{true_val:.8f}")
        with col3:
            st.metric("Error", f"{result['error']:.2e}")
        with col4:
            st.metric("Toleransi", f"{params['tolerance']:.2e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Detail Pengecekan")
            st.write(f"**Fungsi:** `{params['func_str']}`")
            st.write(f"**Tipe error:** {params['error_type']}")
            st.write(f"**Nilai aproksimasi:** {params['approx']:.15f}")
            st.write(f"**Nilai sebenarnya:** {true_val:.15f}")
            st.write(f"**Error:** {result['error']:.8e}")
            st.write(f"**Toleransi:** {params['tolerance']:.8e}")
            st.write(f"**Status:** {'KONVERGEN ✓' if result['converged'] else 'BELUM KONVERGEN ✗'}")
        
        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            
            categories = ['Error', 'Toleransi']
            values = [result['error'], params['tolerance']]
            colors = ['#dc3545' if result['error'] > params['tolerance'] else '#28a745', '#ffc107']
            
            bars = ax.bar(categories, values, color=colors)
            ax.set_ylabel('Nilai')
            ax.set_title('Perbandingan Error vs Toleransi')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2e}',
                       ha='center', va='bottom')
            
            st.pyplot(fig)
    
    elif params['check_type'] == 'iterative':
        # Iterative check
        result = na.iterative_tolerance_check(params['current'], params['previous'], params['tolerance'])
        
        if result['converged']:
            st.success("✓ Iterasi KONVERGEN!")
        else:
            st.warning("✗ Iterasi belum konvergen, lanjutkan iterasi")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nilai Sekarang", f"{params['current']:.8f}")
        with col2:
            st.metric("Error Iteratif", f"{result['iterative_error']:.2e}")
        with col3:
            st.metric("Toleransi", f"{params['tolerance']:.2e}")
        
        st.markdown("---")
        
        st.markdown("#### Detail Pengecekan Iteratif")
        st.write(f"**Nilai iterasi sebelumnya:** {params['previous']:.15f}")
        st.write(f"**Nilai iterasi saat ini:** {params['current']:.15f}")
        st.write(f"**Error iteratif:** {result['iterative_error']:.8e}")
        st.write(f"**Toleransi:** {params['tolerance']:.8e}")
        st.write(f"**Status:** {'KONVERGEN ✓' if result['converged'] else 'LANJUTKAN ITERASI'}")
    
    else:  # adaptive
        # Adaptive tolerance
        true_val = na.true_value(params['x_val'])
        result = na.adaptive_tolerance(params['approx'], true_val, params['target_digits'])
        
        if result['meets_target']:
            st.success(f"✓ Mencapai {params['target_digits']} digit signifikan!")
        else:
            st.warning(f"✗ Belum mencapai {params['target_digits']} digit signifikan")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Target Digit", params['target_digits'])
        with col2:
            achieved = result['achieved_digits'] if result['achieved_digits'] != float('inf') else 'Perfect'
            st.metric("Digit Tercapai", f"{achieved}" if achieved == 'Perfect' else f"{achieved:.2f}")
        with col3:
            st.metric("Error Absolut", f"{result['absolute_error']:.2e}")
        with col4:
            st.metric("Error Relatif", f"{result['relative_error']:.2e}")
        
        st.markdown("---")
        
        st.markdown("#### Detail Toleransi Adaptif")
        st.write(f"**Fungsi:** `{params['func_str']}`")
        st.write(f"**Target digit signifikan:** {params['target_digits']}")
        st.write(f"**Digit tercapai:** {result['achieved_digits']:.4f}")
        st.write(f"**Toleransi yang diperlukan:** {result['required_tolerance']:.8e}")
        st.write(f"**Error absolut:** {result['absolute_error']:.8e}")
        st.write(f"**Error relatif:** {result['relative_error']:.8e}")
        st.write(f"**Status:** {'MEMENUHI TARGET ✓' if result['meets_target'] else 'BELUM MEMENUHI ✗'}")

def display_taylor_polynomial_results(params):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    na = NumericalAnalysis(params['func_str'])
    
    if params['analysis_type'] == "Polynomial Form":
        # Just show polynomial form
        result = na.taylor_polynomial(params['x0'], params['n_terms'])
        
        st.success(f"Polinom Taylor berhasil dibuat dengan {params['n_terms']} suku!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fungsi", params['func_str'])
        with col2:
            st.metric("Titik Ekspansi (x₀)", f"{params['x0']:.4f}")
        with col3:
            st.metric("Jumlah Suku", params['n_terms'])
        
        st.markdown("---")
        
        # Show polynomial
        with st.expander("Polinom Taylor", expanded=True):
            st.latex(result['polynomial_latex'])
            st.code(result['polynomial_str'], language="text")
        
        # Show terms
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Koefisien Suku-suku")
            df_terms = pd.DataFrame(result['terms'])
            st.dataframe(df_terms[['order', 'coefficient', 'derivative_at_x0']], use_container_width=True)
        
        with col2:
            # Plot coefficients
            fig, ax = plt.subplots(figsize=(6, 4))
            orders = [t['order'] for t in result['terms']]
            coeffs = [abs(float(t['coefficient'])) if isinstance(t['coefficient'], (int, float)) else 0 for t in result['terms']]
            
            ax.bar(orders, coeffs, color='#667eea')
            ax.set_xlabel('Order (n)')
            ax.set_ylabel('|Koefisien|')
            ax.set_title('Magnitude Koefisien Taylor')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
    
    elif params['analysis_type'] == "Approximation":
        # Show approximation with error
        result = na.taylor_approximation(params['x0'], params['n_terms'], params['x_eval'])
        
        st.success(f"Aproksimasi Taylor berhasil dihitung!")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Aproksimasi", f"{result['approximation']:.8f}")
        with col2:
            st.metric("Nilai True", f"{result['true_value']:.8f}")
        with col3:
            st.metric("Error Absolut", f"{result['error_analysis']['absolute_error']:.2e}")
        with col4:
            st.metric("Error Relatif", f"{result['error_analysis']['relative_error']:.2e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Detail Aproksimasi")
            st.write(f"**Fungsi:** `{params['func_str']}`")
            st.write(f"**Titik ekspansi (x₀):** {params['x0']}")
            st.write(f"**Jumlah suku:** {params['n_terms']}")
            st.write(f"**Titik evaluasi (x):** {params['x_eval']}")
            st.write(f"**Aproksimasi Taylor:** {result['approximation']:.15f}")
            st.write(f"**Nilai sebenarnya:** {result['true_value']:.15f}")
            st.write(f"**Error absolut:** {result['error_analysis']['absolute_error']:.8e}")
            st.write(f"**Error relatif:** {result['error_analysis']['relative_error']:.8e}")
        
        with col2:
            # Plot comparison
            x_range = np.linspace(params['x0'] - 2, params['x0'] + 2, 100)
            true_vals = na.true_value(x_range)
            
            # Get Taylor approximation for range
            taylor_poly = result['polynomial']['polynomial_symbolic']
            import sympy as sp
            x_sym = sp.Symbol('x')
            taylor_func = sp.lambdify(x_sym, taylor_poly, 'numpy')
            taylor_vals = taylor_func(x_range)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, true_vals, 'b-', linewidth=2, label='Fungsi Asli')
            ax.plot(x_range, taylor_vals, 'r--', linewidth=2, label=f'Taylor ({params["n_terms"]} suku)')
            ax.plot(params['x_eval'], result['approximation'], 'go', markersize=10, label=f'Aproksimasi')
            ax.plot(params['x_eval'], result['true_value'], 'ro', markersize=10, label=f'True Value')
            ax.axvline(params['x0'], color='gray', linestyle=':', alpha=0.5, label=f'x₀ = {params["x0"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Perbandingan Fungsi vs Taylor')
            st.pyplot(fig)
    
    elif params['analysis_type'] == "Convergence Analysis":
        # Show convergence
        convergence = na.taylor_convergence(params['x0'], params['max_terms'], params['x_eval'])
        
        st.success(f"Analisis konvergensi selesai untuk {params['max_terms']} suku!")
        
        # Metrics
        final = convergence[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Maksimum Suku", params['max_terms'])
        with col2:
            st.metric("Aproksimasi Akhir", f"{final['approximation']:.8f}")
        with col3:
            st.metric("Error Akhir", f"{final['absolute_error']:.2e}")
        with col4:
            st.metric("Improvement", f"{(convergence[0]['absolute_error']/final['absolute_error']):.2f}x")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Tabel Konvergensi")
            df = pd.DataFrame(convergence)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Plot convergence
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            
            n_terms = [c['n_terms'] for c in convergence]
            abs_errors = [c['absolute_error'] for c in convergence]
            rel_errors = [c['relative_error'] for c in convergence]
            
            ax1.semilogy(n_terms, abs_errors, 'o-', linewidth=2, markersize=6, color='#667eea')
            ax1.set_xlabel('Jumlah Suku')
            ax1.set_ylabel('Error Absolut (log scale)')
            ax1.set_title('Konvergensi Error Absolut')
            ax1.grid(True, alpha=0.3)
            
            ax2.semilogy(n_terms, rel_errors, 'o-', linewidth=2, markersize=6, color='#764ba2')
            ax2.set_xlabel('Jumlah Suku')
            ax2.set_ylabel('Error Relatif (log scale)')
            ax2.set_title('Konvergensi Error Relatif')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:  # Remainder Analysis
        # Show remainder
        result = na.taylor_remainder(params['x0'], params['n_terms'], params['x_eval'])
        
        st.success("Analisis remainder Taylor selesai!")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Aproksimasi", f"{result['approximation']:.8f}")
        with col2:
            st.metric("Nilai True", f"{result['true_value']:.8f}")
        with col3:
            st.metric("Remainder", f"{result['actual_remainder']:.2e}")
        with col4:
            if result['next_term'] is not None:
                st.metric("Suku Berikutnya", f"{result['next_term']:.2e}")
            else:
                st.metric("Suku Berikutnya", "N/A")
        
        st.markdown("---")
        
        st.markdown("#### Detail Remainder")
        st.write(f"**Fungsi:** `{params['func_str']}`")
        st.write(f"**Titik ekspansi (x₀):** {params['x0']}")
        st.write(f"**Jumlah suku:** {params['n_terms']}")
        st.write(f"**Titik evaluasi (x):** {params['x_eval']}")
        st.write(f"**Aproksimasi Taylor:** {result['approximation']:.15f}")
        st.write(f"**Nilai sebenarnya:** {result['true_value']:.15f}")
        st.write(f"**Remainder aktual:** {result['actual_remainder']:.8e}")
        if result['next_term'] is not None:
            st.write(f"**Suku berikutnya (estimasi):** {result['next_term']:.8e}")
        if result['remainder_ratio'] is not None:
            st.write(f"**Rasio remainder:** {result['remainder_ratio']:.8e}")

if __name__ == "__main__":
    main()