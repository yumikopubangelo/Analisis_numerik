import streamlit as st

def explanation(method):
    st.markdown("---")
    st.markdown("### Penjelasan Metode")
    
    if method == "Bisection":
        st.markdown("""
        **Metode Biseksi** adalah metode pencarian akar yang paling sederhana dan reliable. 
        Metode ini bekerja dengan membagi interval menjadi dua bagian secara berulang.
        
        #### Langkah-langkah:
        1. Pastikan f(a) dan f(b) memiliki tanda berlawanan
        2. Hitung titik tengah: c = (a + b)/2
        3. Evaluasi f(c):
           - Jika f(c) ≈ 0, maka c adalah akar
           - Jika f(a)·f(c) < 0, akar di [a, c] → set b = c
           - Jika f(b)·f(c) < 0, akar di [c, b] → set a = c
        4. Ulangi hingga konvergen
        
        #### Keuntungan:
        - **Selalu konvergen** jika f(a)·f(b) < 0
        - Sederhana dan mudah dipahami
        - Tidak perlu turunan fungsi
        
        #### Kekurangan:
        - Konvergensi **lambat** (linear convergence)
        - Hanya bisa mencari satu akar dalam satu interval
        """)
    
    elif method == "Regula Falsi":
        st.markdown("""
        **Metode Regula Falsi** (False Position) mirip dengan biseksi, tetapi menggunakan 
        interpolasi linear untuk menentukan titik pendekatan.
        
        #### Langkah-langkah:
        1. Pastikan f(a) dan f(b) berlawanan tanda
        2. Hitung titik c menggunakan rumus:
           ```
           c = b - f(b)·(b - a)/(f(b) - f(a))
           ```
        3. Evaluasi f(c) dan update interval seperti biseksi
        
        #### Keuntungan:
        - Lebih cepat dari biseksi
        - Selalu konvergen
        
        #### Kekurangan:
        - Kadang salah satu ujung interval tidak berubah
        - Konvergensi bisa lambat dalam kasus tertentu
        """)
    
    elif method == "Newton-Raphson":
        st.markdown("""
        **Metode Newton-Raphson** menggunakan turunan fungsi untuk menemukan akar dengan cepat.
        
        #### Rumus:
        ```
        x₍ₙ₊₁₎ = xₙ - f(xₙ)/f'(xₙ)
        ```
        
        #### Langkah-langkah:
        1. Pilih tebakan awal x₀
        2. Hitung f(xₙ) dan f'(xₙ)
        3. Update: x₍ₙ₊₁₎ = xₙ - f(xₙ)/f'(xₙ)
        4. Ulangi hingga |f(xₙ)| < toleransi
        
        #### Keuntungan:
        - **Sangat cepat** (quadratic convergence)
        - Akurat untuk tebakan awal yang baik
        
        #### Kekurangan:
        - Memerlukan turunan f'(x)
        - Gagal jika f'(x) = 0
        - Sensitif terhadap tebakan awal
        """)
    
    elif method == "Secant":
        st.markdown("""
        **Metode Secant** adalah modifikasi Newton-Raphson yang tidak memerlukan turunan.
        
        #### Rumus:
        ```
        x₍ₙ₊₁₎ = xₙ - f(xₙ)·(xₙ - x₍ₙ₋₁₎)/(f(xₙ) - f(x₍ₙ₋₁₎))
        ```
        
        #### Langkah-langkah:
        1. Pilih dua tebakan awal x₀ dan x₁
        2. Gunakan rumus secant untuk mendapat x₂
        3. Update: x₀ = x₁, x₁ = x₂
        4. Ulangi hingga konvergen
        
        #### Keuntungan:
        - Lebih cepat dari biseksi
        - Tidak perlu turunan
        
        #### Kekurangan:
        - Tidak selalu konvergen
        - Memerlukan dua tebakan awal
        """)
    
    elif method == "Trapezoidal":
        st.markdown("""
        **Aturan Trapesium** mengaproksimasi area di bawah kurva dengan trapesium.
        
        #### Rumus:
        ```
        ∫ₐᵇ f(x)dx ≈ (h/2)·[f(a) + 2·∑f(xᵢ) + f(b)]
        ```
        dimana h = (b-a)/n
        
        #### Konsep:
        - Bagi interval [a,b] menjadi n subinterval
        - Hubungkan titik-titik dengan garis lurus
        - Jumlahkan luas trapesium yang terbentuk
        
        #### Keuntungan:
        - Sederhana dan mudah diimplementasi
        - Akurat untuk fungsi linear
        
        #### Kekurangan:
        - Kurang akurat untuk fungsi nonlinear
        - Perlu banyak subinterval untuk akurasi tinggi
        """)
    
    elif method == "Simpson":
        st.markdown("""
        **Aturan Simpson 1/3** menggunakan parabola untuk aproksimasi yang lebih akurat.
        
        #### Rumus:
        ```
        ∫ₐᵇ f(x)dx ≈ (h/3)·[f(a) + 4·∑f(xᵢ ganjil) + 2·∑f(xᵢ genap) + f(b)]
        ```
        dimana h = (b-a)/n dan **n harus genap**
        
        #### Konsep:
        - Aproksimasi kurva dengan parabola
        - Lebih akurat dari trapesium
        - Cocok untuk fungsi smooth
        
        #### Keuntungan:
        - **Sangat akurat** untuk fungsi polynomial
        - Lebih baik dari trapesium dengan n yang sama
        
        #### Kekurangan:
        - n harus genap
        - Lebih kompleks dari trapesium
        """)
    
    elif method == "Lagrange":
        st.markdown("""
        **Interpolasi Lagrange** membentuk polynomial yang melewati semua titik data.
        
        #### Rumus:
        ```
        P(x) = ∑ yᵢ·Lᵢ(x)
        ```
        dimana Lᵢ(x) adalah basis polynomial Lagrange
        
        #### Konsep:
        - Setiap Lᵢ(x) = 1 pada xᵢ dan 0 pada xⱼ (j≠i)
        - P(x) dijamin melewati semua titik data
        
        #### Keuntungan:
        - Formula eksplisit dan jelas
        - Mudah dipahami secara konseptual
        
        #### Kekurangan:
        - Mahal secara komputasi untuk banyak titik
        - Tidak efisien untuk menambah titik baru
        """)
    
    elif method == "Newton":
        st.markdown("""
        **Interpolasi Newton** menggunakan divided differences untuk efisiensi.
        
        #### Rumus:
        ```
        P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
        ```
        
        #### Konsep:
        - Menggunakan tabel divided differences
        - Efisien untuk menambah titik data baru
        - Menghasilkan polynomial yang sama dengan Lagrange
        
        #### Keuntungan:
        - Efisien untuk data bertambah
        - Mudah dihitung step-by-step
        
        #### Kekurangan:
        - Konsep divided differences agak abstrak
        - Tetap polynomial degree tinggi bisa oscillate
        """)
    
    elif method == "Taylor":
        st.markdown("""
        **Deret Taylor** mengaproksimasi fungsi dengan polynomial di sekitar titik tertentu.
        
        #### Rumus:
        ```
        f(x) ≈ f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + f'''(x₀)(x-x₀)³/3! + ...
        ```
        
        #### Konsep:
        - Ekspansi fungsi menjadi jumlah tak hingga suku polynomial
        - Setiap suku melibatkan turunan fungsi di titik x₀
        - Semakin banyak suku, semakin akurat aproksimasi
        
        #### Keuntungan:
        - Sangat akurat di sekitar titik ekspansi
        - Berguna untuk analisis fungsi kompleks
        - Dasar dari banyak metode numerik
        
        #### Kekurangan:
        - Akurasi menurun jauh dari titik ekspansi
        - Memerlukan perhitungan turunan
        - Tidak semua fungsi memiliki deret Taylor
        """)
    
    else:
        st.info("Penjelasan untuk metode ini sedang dikembangkan.")
