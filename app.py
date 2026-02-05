import streamlit as st
import numpy as np

from ui.sidebar import sidebar
from ui.input_form import input_form
from ui.displays import (
    display_root_finding_results,
    display_integration_results,
    display_interpolation_results,
    display_taylor_results,
    display_ode_results,
    display_true_value_results,
    display_error_analysis_results,
    display_tolerance_check_results,
    display_taylor_polynomial_results,
    display_differentiation_results,
    display_pde_results
)

# PDE Solver methods
from core.pde.biharmonic_solver import solve_biharmonic_equation

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
from core.series.taylor_series import taylor_series, euler_method, taylor_ode_order2, runge_kutta_method

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
    """
    Main application orchestrator.
    Handles routing between different numerical methods and their display functions.
    """
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Aplikasi Pembelajaran Analisis Numerik</h1>
        <p>Pelajari metode numerik dengan visualisasi step-by-step yang interaktif!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar & Input
    category, method = sidebar()
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
                        params['n_terms'], params['x_eval'],
                        params.get('error_bound'),
                        params.get('tolerance_bound'),
                        params.get('y0')
                    )
                    display_taylor_results(result, params)
                elif method == "Euler":
                    result = euler_method(
                        params['func_str'], params['x0'],
                        params['y0'], params['x_eval'],
                        n_steps=params['n_steps']
                    )
                    display_ode_results(result, params)
                elif method == "Taylor ODE 2":
                    result = taylor_ode_order2(
                        params['func_str'], params['x0'],
                        params['y0'], params['x_eval'],
                        n_steps=params['n_steps']
                    )
                    display_ode_results(result, params)
                elif method == "Runge-Kutta":
                    result = runge_kutta_method(
                        params['func_str'], params['x0'],
                        params['y0'], params['x_eval'],
                        n_steps=params['n_steps']
                    )
                    display_ode_results(result, params)
            
            # DIFFERENTIATION
            elif category == "Differentiation":
                display_differentiation_results(params)
            
            # PDE SOLVER
            elif category == "PDE Solver":
                if method == "Biharmonic Plate":
                    X, Y, w, iterations, error = solve_biharmonic_equation(
                        params['x_min'], params['x_max'],
                        params['y_min'], params['y_max'],
                        params['nx'], params['ny'],
                        params['load_func'],
                        params['solver_type'],
                        params['tol'],
                        params['max_iter'],
                        q=params['q'],
                        D=params['D']
                    )
                    display_pde_results(X, Y, w, iterations, error, params['solver_type'].capitalize(), 
                                       params['q'], params['D'])
            
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


if __name__ == "__main__":
    main()
