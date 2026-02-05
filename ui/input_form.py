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
    elif category == "Differentiation":
        return input_differentiation(method)
    elif category == "PDE Solver":
        return input_pde(method)
    else:
        st.error("Kategori tidak dikenal")

def input_pde(method):
    if method == "Biharmonic Plate":
        st.markdown("#### Masukkan Parameter PDE Solver")
        
        col1, col2 = st.columns(2)
        
        with col1:
            solver_type = st.selectbox(
                "Metode Solver",
                ["jacobi", "gauss-seidel"],
                format_func=lambda x: x.capitalize()
            )
            
            nx = st.number_input(
                "Jumlah grid X",
                value=50,
                min_value=10,
                max_value=200,
                step=10,
                help="Jumlah titik grid dalam arah x"
            )
        
        with col2:
            tol = st.number_input(
                "Toleransi",
                value=1e-6,
                format="%.2e",
                help="Kriteria konvergensi"
            )
            
            ny = st.number_input(
                "Jumlah grid Y",
                value=50,
                min_value=10,
                max_value=200,
                step=10,
                help="Jumlah titik grid dalam arah y"
            )
        
        max_iter = st.number_input(
            "Maksimum iterasi",
            value=10000,
            min_value=100,
            max_value=100000,
            step=1000
        )
        
        st.markdown("#### Batas Domain")
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("X minimum", value=0.0)
            x_max = st.number_input("X maximum", value=1.0)
        with col2:
            y_min = st.number_input("Y minimum", value=0.0)
            y_max = st.number_input("Y maximum", value=1.0)
        
        st.markdown("#### Parameter Material dan Beban")
        col1, col2 = st.columns(2)
        with col1:
            q = st.number_input(
                "Beban distribusi q (kN/m²)",
                value=10.0,
                min_value=1.0,
                max_value=1000.0,
                help="Beban transversal yang bekerja pada plat. Nilai tinggi menghasilkan defleksi yang terlihat."
            )
        with col2:
            D = st.number_input(
                "Kekakuan lentur D (kN·m)",
                value=100.0,
                min_value=10.0,
                max_value=100000.0,
                help="Modulus kekakuan lentur plat"
            )
        
        st.markdown("#### Beban Plat")
        load_type = st.radio(
            "Jenis beban",
            ["Uniform Load (Rata-rata)", "Central Load (Pusat)", "Custom Load"],
            horizontal=True
        )
        
        if load_type == "Uniform Load (Rata-rata)":
            load_func = lambda x, y: 1.0
        elif load_type == "Central Load (Pusat)":
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            radius = min(x_max - x_min, y_max - y_min) / 4
            load_func = lambda x, y: np.exp(-((x - center_x)/radius)**2 - ((y - center_y)/radius)**2)
        else:
            load_str = st.text_input(
                "Fungsi beban q(x, y)",
                value="sin(pi*x) * sin(pi*y)",
                help="Contoh: sin(pi*x)*sin(pi*y), exp(-(x-0.5)**2 - (y-0.5)**2)"
            )
            
            # Parse custom load function
            def load_func(x, y):
                try:
                    # Replace variables in string with actual values
                    expr = load_str.replace('x', str(x)).replace('y', str(y)).replace('pi', str(np.pi))
                    return eval(expr)
                except:
                    return 1.0
        
        if st.button("Hitung Defleksi Plat", use_container_width=True, type="primary"):
            try:
                return {
                    'solver_type': solver_type,
                    'nx': int(nx),
                    'ny': int(ny),
                    'tol': tol,
                    'max_iter': int(max_iter),
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'q': q,
                    'D': D,
                    'load_func': load_func,
                    'load_type': load_type
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    
    elif method == "Advection Diffusion 1D":
        st.markdown("#### Masukkan Parameter Advection-Diffusion 1D")
        
        col1, col2 = st.columns(2)
        
        with col1:
            U = st.number_input(
                "Kecepatan Aliran (U)",
                value=1.0,
                min_value=0.0,
                max_value=10.0,
                help="Kecepatan aliran fluida (m/s)"
            )
            
            K = st.number_input(
                "Koefisien Difusi (K)",
                value=0.1,
                min_value=0.001,
                max_value=1.0,
                format="%.3f",
                help="Koefisien difusi (m²/s)"
            )
        
        st.markdown("#### Batas Domain")
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("X minimum", value=0.0)
            x_max = st.number_input("X maximum", value=1.0)
        
        st.markdown("#### Parameter Grid dan Waktu")
        col1, col2 = st.columns(2)
        with col1:
            nx = st.number_input(
                "Jumlah Grid X",
                value=50,
                min_value=10,
                max_value=200,
                step=10,
                help="Jumlah titik grid dalam arah x"
            )
            
            dt = st.number_input(
                "Langkah Waktu (Δt)",
                value=0.001,
                min_value=0.0001,
                max_value=0.01,
                format="%.4f",
                help="Langkah waktu untuk integrasi (detik)"
            )
        
        with col2:
            num_steps = st.number_input(
                "Jumlah Iterasi Waktu",
                value=500,
                min_value=10,
                max_value=2000,
                step=50,
                help="Jumlah langkah waktu total"
            )
        
        if st.button("Hitung Advection-Diffusion", use_container_width=True, type="primary"):
            try:
                return {
                    'U': U,
                    'K': K,
                    'x_min': x_min,
                    'x_max': x_max,
                    'nx': int(nx),
                    'dt': dt,
                    'num_steps': int(num_steps)
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    
    return None

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
    if method == "Taylor":
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.number_input("Titik ekspansi (x₀)", value=0.0, help="Titik di sekitar mana fungsi diekspansi")
        with col2:
            y0 = st.number_input("Nilai y awal (y₀)", value=0.0, help="Nilai awal y untuk fungsi dengan sumbu y", format="%.6f")
        with col3:
            x_eval = st.number_input("Nilai x untuk evaluasi", value=1.0, help="Titik untuk mengevaluasi aproksimasi")
        
        # Error Bound dan Tolerance Bound
        st.markdown("#### Analisis Error dan Toleransi")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_error_bound = st.checkbox("Hitung Batas Error (Error Bound)", value=False, help="Estimasi batas maksimum error")
            if enable_error_bound:
                error_bound = st.number_input(
                    "Batas Error Maksimum",
                    value=0.01,
                    format="%.6f",
                    help="Batas maksimum error yang diizinkan"
                )
            else:
                error_bound = None
        
        with col2:
            enable_tolerance_bound = st.checkbox("Hitung Batas Toleransi (Tolerance Bound)", value=False, help="Tolerance minimum untuk konvergensi")
            if enable_tolerance_bound:
                tolerance_bound = st.number_input(
                    "Digit Signifikan",
                    value=4,
                    min_value=1,
                    max_value=15,
                    help="Jumlah digit signifikan yang diinginkan"
                )
            else:
                tolerance_bound = None
        
        if st.button("Hitung Deret Taylor", use_container_width=True, type="primary"):
            try:
                return {
                    'func_str': func_str,
                    'x0': x0,
                    'y0': y0,
                    'n_terms': int(n_terms),
                    'x_eval': x_eval,
                    'error_bound': error_bound,
                    'tolerance_bound': tolerance_bound
                }
            except ValueError as e:
                st.error(f"Error: {e}")
                return None
    else:  # ODE methods: Euler, Taylor ODE 2, Runge-Kutta
        st.markdown("#### Masukkan Parameter ODE (y' = f(x, y))")
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Fungsi f(x, y)",
                value="x**2*y - y",
                help="Contoh: x**2*y - y, x + y, sin(x)*cos(y)"
            )
        
        with col2:
            y0 = st.number_input(
                "Nilai awal y₀",
                value=1.0,
                format="%.6f",
                help="Nilai y pada x = x₀"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.number_input("Titik awal x₀", value=0.0, help="Titik awal integrasi")
        with col2:
            x_eval = st.number_input("Titik evaluasi x", value=2.0, help="Titik untuk menghitung y(x)")
        with col3:
            n_steps = st.number_input(
                "Jumlah langkah (n)",
                value=100,
                min_value=1,
                max_value=1000,
                help="Jumlah langkah integrasi"
            )
        
        if st.button(f"Hitung dengan Metode {method}", use_container_width=True, type="primary"):
            try:
                return {
                    'func_str': func_str,
                    'x0': x0,
                    'y0': y0,
                    'x_eval': x_eval,
                    'n_steps': int(n_steps)
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

def input_differentiation(method):
    st.markdown("#### Masukkan Parameter Differentiation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_str = st.text_input(
            "Fungsi f(x)",
            value="sin(x)",
            help="Contoh: sin(x), x**2, exp(x), cos(x)"
        )
    
    with col2:
        h = st.number_input(
            "Langkah (h)",
            value=0.01,
            format="%.6f",
            min_value=1e-8,
            max_value=1.0,
            help="Ukuran langkah untuk diferensiasi"
        )
    
    if method in ["Forward Difference", "Backward Difference", "Central Difference (1st Derivative)", "Central Difference (2nd Derivative)"]:
        x_val = st.number_input("Titik x untuk diferensiasi", value=1.0)
        
        if st.button("Hitung Turunan", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x_val': x_val,
                'h': h,
                'method': method
            }
    
    elif method == "Compare All Methods":
        x_val = st.number_input("Titik x untuk diferensiasi", value=1.0)
        
        if st.button("Bandingkan Semua Metode", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x_val': x_val,
                'h': h,
                'method': method
            }
    
    elif method == "Convergence Analysis":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_val = st.number_input("Titik x untuk diferensiasi", value=1.0)
        with col2:
            derivative_order = st.selectbox(
                "Orde Turunan",
                [1, 2],
                help="1 untuk f'(x), 2 untuk f''(x)"
            )
        with col3:
            num_points = st.number_input(
                "Jumlah titik h",
                value=10,
                min_value=3,
                max_value=50,
                help="Jumlah nilai h yang akan diuji"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            h_min = st.number_input(
                "h minimum",
                value=1e-6,
                format="%.2e",
                min_value=1e-8,
                max_value=1e-3,
                help="Nilai h terkecil"
            )
        with col2:
            h_max = st.number_input(
                "h maximum",
                value=0.1,
                format="%.2e",
                min_value=1e-4,
                max_value=1.0,
                help="Nilai h terbesar"
            )
        
        if st.button("Analisis Konvergensi", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x_val': x_val,
                'h_min': h_min,
                'h_max': h_max,
                'num_points': int(num_points),
                'derivative_order': derivative_order,
                'method': method
            }
    
    elif method == "Optimal h Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            x_val = st.number_input("Titik x untuk diferensiasi", value=1.0)
        with col2:
            num_points = st.number_input(
                "Jumlah titik h",
                value=20,
                min_value=5,
                max_value=50,
                help="Jumlah nilai h yang akan diuji"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            h_min = st.number_input(
                "h minimum",
                value=1e-7,
                format="%.2e",
                min_value=1e-8,
                max_value=1e-3,
                help="Nilai h terkecil"
            )
        with col2:
            h_max = st.number_input(
                "h maximum",
                value=0.1,
                format="%.2e",
                min_value=1e-4,
                max_value=1.0,
                help="Nilai h terbesar"
            )
        
        if st.button("Analisis h Optimal", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x_val': x_val,
                'h_min': h_min,
                'h_max': h_max,
                'num_points': int(num_points),
                'method': method
            }
    
    elif method == "Multi-point Differentiation":
        col1, col2 = st.columns(2)
        
        with col1:
            derivative_order = st.selectbox(
                "Orde Turunan",
                [1, 2],
                help="1 untuk f'(x), 2 untuk f''(x)"
            )
        with col2:
            diff_method = st.selectbox(
                "Metode Diferensiasi",
                ["forward", "backward", "central"],
                help="Metode untuk menghitung turunan"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_min = st.number_input("x minimum", value=0.0)
        with col2:
            x_max = st.number_input("x maximum", value=2.0)
        with col3:
            n_points = st.number_input(
                "Jumlah titik",
                value=10,
                min_value=2,
                max_value=50,
                help="Jumlah titik x untuk diferensiasi"
            )
        
        x_points = np.linspace(x_min, x_max, n_points).tolist()
        
        if st.button("Hitung di Beberapa Titik", use_container_width=True, type="primary"):
            return {
                'func_str': func_str,
                'x_points': x_points,
                'h': h,
                'derivative_order': derivative_order,
                'diff_method': diff_method,
                'method': method
            }
    
    return None
