# Dokumentasi Fitur Analisis Numerik

## Daftar Isi
1. [Nilai Sebenarnya f(x)](#1-nilai-sebenarnya-fx)
2. [Error Absolut / Relatif](#2-error-absolut--relatif)
3. [Toleransi Error](#3-toleransi-error)
4. [Bentuk Polinom Taylor](#4-bentuk-polinom-taylor)
5. [Metode Pencarian Akar](#5-metode-pencarian-akar)
6. [Integrasi Numerik](#6-integrasi-numerik)
7. [Diferensiasi Numerik](#7-diferensiasi-numerik)
8. [Interpolasi](#8-interpolasi)
9. [Persamaan Diferensial Parsial (PDE)](#9-persamaan-diferensial-parsial-pde)

---

## 1. Nilai Sebenarnya f(x)

### Deskripsi
Fitur ini menghitung nilai sebenarnya (true value) dari suatu fungsi matematika pada titik tertentu. Nilai ini digunakan sebagai referensi untuk membandingkan dengan nilai aproksimasi.

### Fungsi Utama

#### `true_value(x_val)`
Menghitung nilai numerik fungsi pada titik x.

**Parameter:**
- `x_val`: float atau array - Nilai x untuk evaluasi

**Return:**
- float atau array - Nilai f(x)

**Contoh:**
```python
from core.analysis.numerical_features import NumericalAnalysis

na = NumericalAnalysis("sin(x)")
result = na.true_value(np.pi/2)  # Output: 1.0
```

#### `true_value_symbolic(x_val)`
Menghitung nilai eksak dalam bentuk simbolik.

**Parameter:**
- `x_val`: float - Nilai x untuk evaluasi

**Return:**
- sympy.Expr - Nilai eksak dalam bentuk simbolik

**Contoh:**
```python
na = NumericalAnalysis("sqrt(2)")
result = na.true_value_symbolic(1)  # Output: sqrt(2)
```

### Kasus Penggunaan
- Menghitung nilai referensi untuk perbandingan
- Validasi hasil aproksimasi
- Plotting fungsi asli
- Analisis konvergensi metode numerik

---

## 2. Error Absolut / Relatif

### Deskripsi
Fitur ini menghitung berbagai jenis error (kesalahan) antara nilai aproksimasi dan nilai sebenarnya.

### Jenis-jenis Error

#### Error Absolut
```
Error Absolut = |aproksimasi - nilai_sebenarnya|
```

#### Error Relatif
```
Error Relatif = |aproksimasi - nilai_sebenarnya| / |nilai_sebenarnya|
```

### Fungsi Utama

#### `absolute_error(approx, true)`
Menghitung error absolut.

```python
na = NumericalAnalysis("pi")
error = na.absolute_error(3.14, np.pi)  # Output: 0.001592...
```

#### `relative_error(approx, true)`
Menghitung error relatif.

```python
na = NumericalAnalysis("exp(x)")
error = na.relative_error(2.7, np.e)  # Output: 0.00647...
```

### Kasus Penggunaan
- Mengukur akurasi metode numerik
- Membandingkan berbagai metode aproksimasi
- Validasi konvergensi algoritma

---

## 3. Toleransi Error

### Deskripsi
Fitur ini memeriksa apakah error memenuhi toleransi yang diberikan.

### Fungsi Utama

#### `check_tolerance(approx, true, tolerance, error_type='absolute')`
Memeriksa apakah error memenuhi toleransi.

```python
na = NumericalAnalysis("sqrt(2)")
result = na.check_tolerance(1.414, np.sqrt(2), 0.001, 'absolute')
# Output: {'converged': True, 'error': 0.000213..., ...}
```

### Kriteria Konvergensi
- **Toleransi Absolut**: |x_n - x_true| < ε
- **Toleransi Relatif**: |x_n - x_true| / |x_true| < ε
- **Toleransi Iteratif**: |x_n - x_{n-1}| < ε

---

## 4. Bentuk Polinom Taylor

### Deskripsi
Fitur ini menghitung ekspansi deret Taylor dari suatu fungsi.

### Formula Deret Taylor
```
f(x) ≈ f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + f'''(x₀)(x-x₀)³/3! + ...
```

### Fungsi Utama

#### `taylor_polynomial(x0, n_terms)`
Menghitung bentuk polinom Taylor.

```python
na = NumericalAnalysis("exp(x)")
result = na.taylor_polynomial(0, 5)
# Output: {'polynomial_symbolic': 1 + x + x**2/2 + x**3/6 + x**4/24, ...}
```

#### `taylor_convergence(x0, max_terms, x_eval)`
Analisis konvergensi deret Taylor.

```python
na = NumericalAnalysis("cos(x)")
results = na.taylor_convergence(0, 10, 1.0)
# Menunjukkan bagaimana error berkurang dengan menambah suku
```

### Contoh Ekspansi Taylor Umum

| Fungsi | Ekspansi di x₀ = 0 |
|--------|---------------------|
| exp(x) | 1 + x + x²/2! + x³/3! + x⁴/4! + ... |
| sin(x) | x - x³/3! + x⁵/5! - x⁷/7! + ... |
| cos(x) | 1 - x²/2! + x⁴/4! - x⁶/6! + ... |
| ln(1+x) | x - x²/2 + x³/3 - x⁴/4 + ... |

---

## 5. Metode Pencarian Akar

### Deskripsi
Berbagai metode untuk menemukan akar (root) dari fungsi non-linear.

### Metode yang Tersedia

#### Bisection Method
- **Konvergensi**: Linear
- **Kepercayaan**: Selalu konvergen jika f(a) dan f(b) berlawanan tanda

```python
from core.root_findings.bisection import bisection

# f(x) = x² - 2
root, iterations, error = bisection(f, 1, 2, tol=1e-6)
```

#### Regula Falsi (False Position)
- **Konvergensi**: Linear (lebih cepat dari bisection)

```python
from core.root_findings.regula_falsi import regula_falsi

root, iterations, error = regula_falsi(f, 1, 2, tol=1e-6)
```

#### Newton-Raphson
- **Konvergensi**: Kuadratik
- **Membutuhkan**: Turunan fungsi

```python
from core.root_findings.newton_raphson import newton_raphson

root, iterations, error = newton_raphson(f, df, x0=1.5, tol=1e-6)
```

#### Secant Method
- **Konvergensi**: Superlinear (1.618)
- **Tidak membutuhkan**: Turunan

```python
from core.root_findings.secant import secant

root, iterations, error = secant(f, x0=1, x1=2, tol=1e-6)
```

---

## 6. Integrasi Numerik

### Deskripsi
Metode untuk mengintegralkan fungsi secara numerik.

### Metode yang Tersedia

#### Trapezoidal Rule
```
∫f(x)dx ≈ (h/2) * (f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ))
```

```python
from core.integration.trapezoidal import trapezoidal

result = trapezoidal(f, a=0, b=1, n=100)
```

#### Simpson's Rule
```
∫f(x)dx ≈ (h/3) * (f(x₀) + 4f(x₁) + 2f(x₂) + ... + 4f(xₙ₋₁) + f(xₙ))
```

```python
from core.integration.simpson import simpson

result = simpson(f, a=0, b=1, n=100)
```

---

## 7. Diferensiasi Numerik

### Deskripsi
Metode untuk menghitung turunan fungsi secara numerik.

### Metode yang Tersedia

#### Forward Difference
```
f'(x) ≈ (f(x+h) - f(x)) / h
```

#### Backward Difference
```
f'(x) ≈ (f(x) - f(x-h)) / h
```

#### Central Difference
```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

```python
from core.differentiation.numerical_differentiation import (
    forward_diff, backward_diff, central_diff
)

df = central_diff(f, x=1.0, h=0.001)
```

---

## 8. Interpolasi

### Deskripsi
Metode untuk membangun fungsi yang melewati titik-titik data tertentu.

### Metode yang Tersedia

#### Lagrange Polynomial
```
P(x) = Σ yᵢ * Lᵢ(x)
```

```python
from core.interpolation.lagrange import lagrange_interpolation

# Titik data: (1, 1), (2, 4), (3, 9)
x_points = [1, 2, 3]
y_points = [1, 4, 9]

# Evaluasi di x=1.5
result = lagrange_interpolation(x_points, y_points, 1.5)
```

#### Newton Polynomial
```
P(x) = a₀ + a₁(x-x₀) + a₂(x-x₀)(x-x₁) + ...
```

```python
from core.interpolation.newton import newton_interpolation

result = newton_interpolation(x_points, y_points, x_eval=1.5)
```

---

## 9. Persamaan Diferensial Parsial (PDE)

### Deskripsi
Solver untuk persamaan diferensial parsial dalam teknik.

### Persamaan Biharmonik (Lendutan Plat)

**Persamaan Asli:**
```
∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴ = q/D
```

**Stencil 13-titik Finite Difference:**
```
20wᵢⱼ - 8(wᵢ₊₁,ⱼ + wᵢ₋₁,ⱼ + wᵢ,ⱼ₊₁ + wᵢ,ⱼ₋₁) 
+ 2(wᵢ₊₁,ⱼ₊₁ + wᵢ₊₁,ⱼ₋₁ + wᵢ₋₁,ⱼ₊₁ + wᵢ₋₁,ⱼ₋₁) 
- (wᵢ₊₂,ⱼ + wᵢ₋₂,ⱼ + wᵢ,ⱼ₊₂ + wᵢ,ⱼ₋₂) = (q/D)h⁴
```

dengan h = Δx = Δy

**Boundary Conditions:**
- Simply Supported: w = 0 dan ∇²w = 0 di batas

**Contoh Penggunaan:**
```python
from core.pde.biharmonic_solver import solve_biharmonic_equation

# q dalam kN/m², D dalam kN·m
X, Y, w, iterations, error = solve_biharmonic_equation(
    q=10.0,      # Beban dalam kN/m²
    D=100.0,    # Kekakuan dalam kN·m
    nx=41, ny=41,  # Jumlah grid points
    tol=1e-5
)

# Defleksi maksimum dalam mm
max_deflection = np.max(w) * 1000
```

### Solusi Analitik (Plat Simply Supported)
Solusi Navier untuk plat persegi dengan beban uniform:

```python
from core.pde.biharmonic_solver import analytic_solution_simply_supported

w_analytic = analytic_solution_simply_supported(X, Y, q, D, n_terms=5)
```

### Validasi
```python
from core.pde.biharmonic_solver import validate_solution

validation = validate_solution(w_numeric, X, Y, q, D, dx, dy)
# Output: {'max_error', 'rms_error', 'pde_error', ...}
```

---

## Tips dan Best Practices

### Pemilihan Metode
| Masalah | Metode yang Disarankan |
|---------|----------------------|
| Akar tunggal, turunan tersedia | Newton-Raphson |
| Akar tunggal, turunan tidak tersedia | Secant |
| Konvergensi terjamin | Bisection |
| Integrasi dengan error kecil | Simpson |
| Interpolasi cepat | Lagrange |

### Performa
- Gunakan array numpy untuk evaluasi multiple points
- Cache hasil jika digunakan berulang
- Pilih grid size yang sesuai (tidak terlalu besar)

---

## Referensi

### Buku
1. Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.)
2. Chapra, S. C., & Canale, R. P. (2015). Numerical Methods for Engineers (7th ed.)
3. Timoshenko, S., & Goodier, J. N. (1970). Theory of Elasticity

### Link Terkait
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Versi**: 2.0.0  
**Tanggal**: 2026-02-05  
**Author**: Numerical Analysis Team
