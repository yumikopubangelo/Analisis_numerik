import streamlit as st
import numpy as np
from core.utils.parser import parse_function

def input_form(category, method):
    st.markdown(f"### {category} - {method}")
    st.markdown("---")
    
    if category == "Root Finding":
        return input_root_finding(method)
    elif category == "Integration":
        return input_integration(method)
    elif category == "Interpolation":
        return input_interpolation(method)
    elif category == "Series":
        return input_series(method)
    elif category == "Analysis Features":
        return input_analysis_features(method)
    else:
        st.error("Kategori tidak dikenal")

def input_root_finding(method):
    st.markdown("#### Masukkan Parameter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_str = st.text_input(
            "Fungsi f(x)", 
            value="x**3 - x - 2",
            help="Contoh: x**2 - 2, sin(x) - x/2, exp(x) - 3*x"
        )
    
    with col2:
        tol = st.number_input(
            "Toleransi", 
            value=1e-6, 
            format="%.2e",
            help="Kriteria konvergensi"
        )
    
    if method in ["Bisection", "Regula Falsi"]:
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Batas bawah (a)", value=1.0)
        with col2:
            b = st.number_input("Batas atas (b)", value=2.0)
        
        max_iter = st.number_input("Maksimum iterasi", value=100, min_value=1, max_value=1000)
        
        if st.button("Hitung", use_container_width=True, type="primary"):
            try:
                func = parse_function(func_str)
                return {
                    'func': func,
                    'func_str': func_str,
                    'a': a,
                    'b': b,
                    'tol': tol,
                    'max_iter': int(max_iter)
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    
    elif method == "Newton-Raphson":
        x0 = st.number_input("Tebakan awal (x₀)", value=1.5)
        max_iter = st.number_input("Maksimum iterasi", value=100, min_value=1, max_value=1000)
        
        if st.button("Hitung", use_container_width=True, type="primary"):
            try:
                func = parse_function(func_str)
                return {
                    'func': func,
                    'func_str': func_str,
                    'x0': x0,
                    'tol': tol,
                    'max_iter': int(max_iter)
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    
    elif method == "Secant":
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("Tebakan pertama (x₀)", value=1.0)
        with col2:
            x1 = st.number_input("Tebakan kedua (x₁)", value=2.0)
        
        max_iter = st.number_input("Maksimum iterasi", value=100, min_value=1, max_value=1000)
        
        if st.button("Hitung", use_container_width=True, type="primary"):
            try:
                func = parse_function(func_str)
                return {
                    'func': func,
                    'func_str': func_str,
                    'x0': x0,
                    'x1': x1,
                    'tol': tol,
                    'max_iter': int(max_iter)
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    
    return None

def input_integration(method):
    st.markdown("#### Masukkan Parameter Integrasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_str = st.text_input(
            "Fungsi f(x)", 
            value="x**2",
            help="Contoh: x**2, sin(x), exp(x)"
        )
    
    with col2:
        n = st.number_input(
            "Jumlah subinterval (n)", 
            value=10, 
            min_value=2,
            max_value=1000,
            step=2 if method == "Simpson" else 1,
            help="Untuk Simpson harus genap"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Batas bawah (a)", value=0.0)
    with col2:
        b = st.number_input("Batas atas (b)", value=1.0)
    
    if st.button("Hitung Integral", use_container_width=True, type="primary"):
        try:
            func = parse_function(func_str)
            return {
                'func': func,
                'func_str': func_str,
                'a': a,
                'b': b,
                'n': int(n)
            }
        except ValueError as e:
            st.error(f"Error: {e}")
            return None
    
    return None

def input_interpolation(method):
    st.markdown("#### Masukkan Data Points")
    
    # Option to input points
    input_method = st.radio(
        "Pilih metode input:",
        ["Manual", "Generate dari Fungsi"],
        horizontal=True
    )
    
    if input_method == "Manual":
        n_points = st.number_input("Jumlah titik data", value=4, min_value=2, max_value=10)
        
        x_points = []
        y_points = []
        
        st.markdown("**Masukkan koordinat (x, y):**")
        cols = st.columns(min(n_points, 4))
        
        for i in range(n_points):
            with cols[i % 4]:
                x = st.number_input(f"x₍{i}₎", value=float(i), key=f"x_{i}")
                y = st.number_input(f"y₍{i}₎", value=float(i**2), key=f"y_{i}")
                x_points.append(x)
                y_points.append(y)
        
    else:  # Generate from function
        func_str = st.text_input("Fungsi f(x)", value="x**2")
        n_points = st.number_input("Jumlah titik", value=5, min_value=2, max_value=10)
        
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("x minimum", value=0.0)
        with col2:
            x_max = st.number_input("x maximum", value=4.0)
        
        x_points = np.linspace(x_min, x_max, n_points).tolist()
        
        try:
            func = parse_function(func_str)
            y_points = [func(x) for x in x_points]
        except:
            y_points = [0] * n_points
    
    x_eval = st.number_input("Nilai x untuk interpolasi", value=2.5)
    
    if st.button("Interpolasi", use_container_width=True, type="primary"):
        return {
            'x_points': np.array(x_points),
            'y_points': np.array(y_points),
            'x_eval': x_eval
        }
    
    return None

def input_series(method):
    st.markdown("#### Masukkan Parameter Deret Taylor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_str = st.text_input(
            "Fungsi f(x)",
            value="exp(x)",
            help="Contoh: exp(x), sin(x), cos(x), x**2"
        )
    
    with col2:
        n_terms = st.number_input(
            "Jumlah suku",
            value=5,
            min_value=1,
            max_value=20,
            help="Jumlah suku dalam ekspansi Taylor"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Titik ekspansi (x₀)", value=0.0, help="Titik di sekitar mana fungsi diekspansi")
    with col2:
        x_eval = st.number_input("Nilai x untuk evaluasi", value=1.0, help="Titik untuk mengevaluasi aproksimasi")
    
    if st.button("Hitung Deret Taylor", use_container_width=True, type="primary"):
        try:
            return {
                'func_str': func_str,
                'x0': x0,
                'n_terms': int(n_terms),
                'x_eval': x_eval
            }
        except ValueError as e:
            st.error(f"Error: {e}")
            return None
    
    return None

def input_analysis_features(method):
    st.markdown("#### Masukkan Parameter")
    
    if method == "True Value":
        # Form untuk menghitung nilai sebenarnya
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Fungsi f(x)",
                value="sin(x)",
                help="Contoh: sin(x), exp(x), x**2, sqrt(x)"
            )
        
        with col2:
            input_type = st.radio(
                "Tipe Input",
                ["Single Point", "Multiple Points"],
                horizontal=True
            )
        
        if input_type == "Single Point":
            x_val = st.number_input("Nilai x", value=1.0)
            
            if st.button("Hitung Nilai Sebenarnya", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'x_val': x_val,
                    'input_type': 'single'
                }
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_min = st.number_input("x minimum", value=0.0)
            with col2:
                x_max = st.number_input("x maximum", value=2.0)
            with col3:
                n_points = st.number_input("Jumlah titik", value=10, min_value=2, max_value=100)
            
            if st.button("Hitung Nilai Sebenarnya", use_container_width=True, type="primary"):
                x_points = np.linspace(x_min, x_max, n_points).tolist()
                return {
                    'func_str': func_str,
                    'x_points': x_points,
                    'input_type': 'multiple'
                }
    
    elif method == "Error Analysis":
        # Form untuk analisis error
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Fungsi f(x)",
                value="sin(x)",
                help="Fungsi untuk menghitung nilai sebenarnya"
            )
        
        with col2:
            x_val = st.number_input("Nilai x untuk evaluasi", value=1.0)
        
        st.markdown("#### Input Aproksimasi")
        
        input_method = st.radio(
            "Metode Input",
            ["Single Approximation", "Multiple Approximations"],
            horizontal=True
        )
        
        if input_method == "Single Approximation":
            approx = st.number_input("Nilai aproksimasi", value=0.84)
            
            if st.button("Analisis Error", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'x_val': x_val,
                    'approximations': [approx],
                    'analysis_type': 'single'
                }
        else:
            n_approx = st.number_input("Jumlah aproksimasi", value=5, min_value=2, max_value=20)
            
            approximations = []
            cols = st.columns(min(n_approx, 5))
            for i in range(n_approx):
                with cols[i % 5]:
                    approx = st.number_input(f"Aproks {i+1}", value=0.8 + i*0.01, key=f"approx_{i}")
                    approximations.append(approx)
            
            if st.button("Analisis Error", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'x_val': x_val,
                    'approximations': approximations,
                    'analysis_type': 'multiple'
                }
    
    elif method == "Tolerance Check":
        # Form untuk pengecekan toleransi
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Fungsi f(x)",
                value="sqrt(2)",
                help="Fungsi untuk nilai sebenarnya"
            )
        
        with col2:
            check_type = st.selectbox(
                "Tipe Pengecekan",
                ["With True Value", "Iterative (No True Value)", "Adaptive Tolerance"]
            )
        
        if check_type == "With True Value":
            col1, col2 = st.columns(2)
            with col1:
                approx = st.number_input("Nilai aproksimasi", value=1.414)
            with col2:
                x_val = st.number_input("Nilai x (untuk true value)", value=1.0)
            
            col1, col2 = st.columns(2)
            with col1:
                tolerance = st.number_input("Toleransi", value=0.001, format="%.6f")
            with col2:
                error_type = st.selectbox("Tipe Error", ["absolute", "relative"])
            
            if st.button("Cek Toleransi", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'check_type': 'with_true',
                    'approx': approx,
                    'x_val': x_val,
                    'tolerance': tolerance,
                    'error_type': error_type
                }
        
        elif check_type == "Iterative (No True Value)":
            col1, col2 = st.columns(2)
            with col1:
                current = st.number_input("Nilai saat ini", value=1.4142)
            with col2:
                previous = st.number_input("Nilai sebelumnya", value=1.414)
            
            tolerance = st.number_input("Toleransi", value=0.001, format="%.6f")
            
            if st.button("Cek Toleransi Iteratif", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'check_type': 'iterative',
                    'current': current,
                    'previous': previous,
                    'tolerance': tolerance
                }
        
        else:  # Adaptive Tolerance
            col1, col2 = st.columns(2)
            with col1:
                approx = st.number_input("Nilai aproksimasi", value=3.14159)
            with col2:
                x_val = st.number_input("Nilai x (untuk true value)", value=1.0)
            
            target_digits = st.number_input("Target digit signifikan", value=4, min_value=1, max_value=15)
            
            if st.button("Cek Toleransi Adaptif", use_container_width=True, type="primary"):
                return {
                    'func_str': func_str,
                    'check_type': 'adaptive',
                    'approx': approx,
                    'x_val': x_val,
                    'target_digits': target_digits
                }
    
    elif method == "Taylor Polynomial":
        # Form untuk polinom Taylor
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Fungsi f(x)",
                value="exp(x)",
                help="Contoh: exp(x), sin(x), cos(x), ln(1+x)"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Tipe Analisis",
                ["Polynomial Form", "Approximation", "Convergence Analysis", "Remainder Analysis"]
            )
        
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("Titik ekspansi (x₀)", value=0.0)
        with col2:
            n_terms = st.number_input("Jumlah suku", value=5, min_value=1, max_value=20)
        
        if analysis_type in ["Approximation", "Convergence Analysis", "Remainder Analysis"]:
            x_eval = st.number_input("Nilai x untuk evaluasi", value=1.0)
        else:
            x_eval = None
        
        if analysis_type == "Convergence Analysis":
            max_terms = st.number_input("Maksimum suku untuk konvergensi", value=10, min_value=2, max_value=20)
        else:
            max_terms = n_terms
        
        if st.button("Hitung Taylor", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x0': x0,
                'n_terms': int(n_terms),
                'x_eval': x_eval,
                'analysis_type': analysis_type,
                'max_terms': int(max_terms)
            }
    
    return None