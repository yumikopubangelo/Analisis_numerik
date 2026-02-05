import streamlit as st

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
            <h1 style='color: white; margin: 0;'>Numerical</h1>
            <h3 style='color: white; margin: 0;'>Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category selection
        st.markdown("### Pilih Kategori")
        category = st.selectbox(
            "Kategori Metode",
            ["Root Finding", "Integration", "Interpolation", "Series", "Analysis Features", "Differentiation", "PDE Solver"],
            label_visibility="collapsed"
        )
        
        # Method selection based on category
        st.markdown("### Pilih Metode")
        
        if category == "Root Finding":
            method_options = {
                "Bisection": "Metode Biseksi",
                "Regula Falsi": "Metode Regula Falsi",
                "Newton-Raphson": "Metode Newton-Raphson",
                "Secant": "Metode Secant"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
        
        elif category == "Integration":
            method_options = {
                "Trapezoidal": "Aturan Trapesium",
                "Simpson": "Aturan Simpson"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
        
        elif category == "Interpolation":
            method_options = {
                "Lagrange": "Interpolasi Lagrange",
                "Newton": "Interpolasi Newton"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
        
        elif category == "Series":
            method_options = {
                "Taylor": "Deret Taylor",
                "Euler": "Metode Euler",
                "Taylor ODE 2": "Taylor ODE Orde 2",
                "Runge-Kutta": "Metode Runge-Kutta (RK4)"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
        
        elif category == "Differentiation":
            method_options = {
                "Forward Difference": "Selisih Maju (Forward Difference)",
                "Backward Difference": "Selisih Mundur (Backward Difference)",
                "Central Difference (1st Derivative)": "Selisih Pusat (Turunan Pertama)",
                "Central Difference (2nd Derivative)": "Selisih Pusat (Turunan Kedua)",
                "Compare All Methods": "Bandingkan Semua Metode (Turunan Pertama)",
                "Convergence Analysis": "Analisis Konvergensi dengan Variasi h",
                "Optimal h Analysis": "Analisis h Optimal untuk Error Minimum",
                "Multi-point Differentiation": "Diferensiasi pada Beberapa Titik"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
            
        elif category == "PDE Solver":
            method_options = {
                "Biharmonic Plate": "Persamaan Plat Biharmonik"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Metode",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
            
        else:  # Analysis Features
            method_options = {
                "True Value": "Nilai Sebenarnya f(x)",
                "Error Analysis": "Analisis Error",
                "Tolerance Check": "Pengecekan Toleransi",
                "Taylor Polynomial": "Polinom Taylor"
            }
            method_labels = list(method_options.values())
            method_keys = list(method_options.keys())
            
            selected_label = st.radio(
                "Fitur",
                method_labels,
                label_visibility="collapsed"
            )
            method = method_keys[method_labels.index(selected_label)]
        
        st.markdown("---")
        
        # Info box
        st.markdown("### Informasi")
        
        if category == "Root Finding":
            st.info("""
            **Root Finding** digunakan untuk mencari nilai x dimana f(x) = 0.
            
            Pilih metode yang sesuai dengan kebutuhan Anda!
            """)
        
        elif category == "Integration":
            st.info("""
            **Numerical Integration** menghitung aproksimasi integral definit.
            
            Semakin banyak subinterval, semakin akurat hasilnya.
            """)
        
        elif category == "Interpolation":
            st.info("""
            **Interpolation** membuat fungsi yang melewati titik-titik data yang diberikan.
            
            Berguna untuk estimasi nilai di antara data points.
            """)
        
        elif category == "Series":
            st.info("""
            **Deret Taylor** mengaproksimasi fungsi dengan polinomial di sekitar titik tertentu.
            
            Semakin banyak suku, semakin akurat aproksimasi.
            """)
        
        elif category == "Differentiation":
            st.info("""
            **Numerical Differentiation** menghitung turunan fungsi dengan metode numerik:
            
            - **Selisih Maju (Forward)**: O(h) akurasi, mudah dihitung
            - **Selisih Mundur (Backward)**: O(h) akurasi, mudah dihitung  
            - **Selisih Pusat (Central)**: O(h²) akurasi, lebih presisi
            
            Pilih metode yang sesuai dengan kebutuhan akurasi!
            """)
        
        elif category == "PDE Solver":
            st.info("""
            **PDE Solver** menyelesaikan Persamaan Partial Differential Equation (PDE)
            untuk analisis defleksi plat menggunakan metode finite difference:
            
            - **Persamaan Plat Biharmonik**: Menghitung defleksi plat tebal under beban
            - Solver iteratif: Jacobi dan Gauss-Seidel
            - Visualisasi 3D dan kontur defleksi
            
            Pilih metode solver dan parameter grid untuk perhitungan!
            """)
            
        else:  # Analysis Features
            st.info("""
            **Analysis Features** menyediakan tools untuk:
            - Menghitung nilai sebenarnya fungsi
            - Menganalisis error aproksimasi
            - Memeriksa toleransi konvergensi
            - Membuat polinom Taylor
            """)
        
        st.markdown("---")
        
        # Quick reference
        with st.expander("Quick Reference"):
            st.markdown("""
            **Notasi Matematika:**
            - `**` untuk pangkat (x²  = x**2)
            - `sqrt(x)` untuk akar kuadrat
            - `sin(x), cos(x), tan(x)` untuk trigonometri
            - `exp(x)` untuk eˣ
            - `log(x)` untuk ln(x)
            
            **Contoh:**
            - `x**2 - 2`
            - `sin(x) - x/2`
            - `exp(x) - 3*x`
            """)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; padding: 0.5rem; font-size: 0.8rem; color: #666;'>
            <p>Dibuat untuk pembelajaran<br>Analisis Numerik </p>
        </div>
        """, unsafe_allow_html=True)
    
    return category, method
