# 4 Fitur Analisis Numerik

## 📋 Ringkasan

Proyek ini menyediakan 4 fitur utama untuk analisis numerik:

1. **Nilai Sebenarnya f(x)** - Menghitung nilai eksak dari fungsi matematika
2. **Error Absolut / Relatif** - Mengukur kesalahan aproksimasi
3. **Toleransi Error** - Memeriksa kriteria konvergensi
4. **Bentuk Polinom Taylor** - Ekspansi deret Taylor untuk aproksimasi fungsi

## 🚀 Quick Start

### Instalasi

```bash
# Clone repository
git clone <repository-url>
cd Analisis_numerik

# Install dependencies
pip install -r requirements.txt
```

### Penggunaan Dasar

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Buat objek analisis untuk fungsi sin(x)
na = NumericalAnalysis("sin(x)")

# Fitur 1: Nilai sebenarnya
true_val = na.true_value(np.pi/4)
print(f"sin(π/4) = {true_val}")

# Fitur 2: Error absolut/relatif
approx = 0.707
error_result = na.error_analysis(approx, np.pi/4)
print(f"Error absolut: {error_result['absolute_error']}")
print(f"Error relatif: {error_result['relative_error']}")

# Fitur 3: Toleransi error
tol_result = na.check_tolerance(approx, true_val, 0.001, 'absolute')
print(f"Konvergen: {tol_result['converged']}")

# Fitur 4: Polinom Taylor
taylor = na.taylor_polynomial(0, 5)
print(f"Taylor: {taylor['polynomial_str']}")
```

## 📁 Struktur File

```
Analisis_numerik/
├── core/
│   └── analysis/
│       ├── __init__.py
│       └── numerical_features.py    # Modul utama dengan 4 fitur
├── docs/
│   └── features_documentation.md    # Dokumentasi lengkap
├── demo_features.py                 # Demo interaktif semua fitur
├── test_features.py                 # Unit tests
└── FEATURES_README.md              # File ini
```

## 🎯 Fitur-Fitur

### 1️⃣ Nilai Sebenarnya f(x)

Menghitung nilai eksak dari fungsi matematika pada titik tertentu.

**Fungsi Utama:**
- `true_value(x_val)` - Nilai numerik
- `true_value_symbolic(x_val)` - Nilai simbolik
- `evaluate_at_points(x_points)` - Evaluasi multiple points

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
result = na.true_value(1.0)  # e^1 = 2.718281828...
```

### 2️⃣ Error Absolut / Relatif

Mengukur kesalahan antara nilai aproksimasi dan nilai sebenarnya.

**Jenis Error:**
- **Error Absolut**: `|aproksimasi - true|`
- **Error Relatif**: `|aproksimasi - true| / |true|`
- **Error Persentase**: `Error Relatif × 100%`

**Fungsi Utama:**
- `absolute_error(approx, true)`
- `relative_error(approx, true)`
- `percentage_error(approx, true)`
- `error_analysis(approx, x_val)` - Analisis lengkap
- `compare_approximations(approximations, x_val)` - Bandingkan multiple

**Contoh:**
```python
na = NumericalAnalysis("pi")
abs_err = na.absolute_error(3.14, np.pi)
rel_err = na.relative_error(3.14, np.pi)
```

### 3️⃣ Toleransi Error

Memeriksa apakah error memenuhi toleransi yang diberikan (kriteria konvergensi).

**Jenis Toleransi:**
- **Absolut**: Error absolut < toleransi
- **Relatif**: Error relatif < toleransi
- **Iteratif**: |x_n - x_{n-1}| < toleransi
- **Adaptif**: Berdasarkan digit signifikan

**Fungsi Utama:**
- `check_tolerance(approx, true, tolerance, error_type)`
- `iterative_tolerance_check(current, previous, tolerance)`
- `adaptive_tolerance(approx, true, target_digits)`

**Contoh:**
```python
na = NumericalAnalysis("sqrt(2)")
result = na.check_tolerance(1.414, np.sqrt(2), 0.001, 'absolute')
if result['converged']:
    print("Konvergen!")
```

### 4️⃣ Bentuk Polinom Taylor

Ekspansi deret Taylor untuk aproksimasi fungsi dengan polinomial.

**Formula:**
```
f(x) ≈ f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + ...
```

**Fungsi Utama:**
- `taylor_polynomial(x0, n_terms)` - Bentuk polinom
- `taylor_approximation(x0, n_terms, x_eval)` - Aproksimasi + error
- `taylor_convergence(x0, max_terms, x_eval)` - Analisis konvergensi
- `taylor_remainder(x0, n_terms, x_eval)` - Estimasi remainder

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
taylor = na.taylor_polynomial(0, 5)
print(taylor['polynomial_str'])
# Output: 1 + x + x**2/2 + x**3/6 + x**4/24
```

## 🧪 Testing

Jalankan unit tests untuk memverifikasi semua fitur:

```bash
python test_features.py
```

Output yang diharapkan:
```
======================================================================
RUNNING TESTS FOR 4 FITUR ANALISIS NUMERIK
======================================================================

Testing Fitur 1: Nilai sebenarnya f(x)
✓ Test 1 passed: sin(π/2) = 1.0
✓ Test 2 passed: 3² = 9.0

Testing Fitur 2: Error absolut / relatif
✓ Test 1 passed: Absolute error = 1.592654e-03
✓ Test 2 passed: Relative error = 5.069574e-04
✓ Test 3 passed: Error analysis complete

Testing Fitur 3: Toleransi error
✓ Test 1 passed: Tolerance check (converged)
✓ Test 2 passed: Tolerance check (not converged)
✓ Test 3 passed: Iterative tolerance check
✓ Test 4 passed: Adaptive tolerance

Testing Fitur 4: Bentuk polinom Taylor
✓ Test 1 passed: Taylor polynomial generation
✓ Test 2 passed: Taylor approximation
✓ Test 3 passed: Taylor convergence
✓ Test 4 passed: Taylor remainder

======================================================================
ALL TESTS PASSED! ✓
======================================================================
```

## 🎨 Demo Interaktif

Jalankan demo untuk melihat semua fitur dalam aksi:

```bash
python demo_features.py
```

Demo ini menampilkan:
- Contoh penggunaan setiap fitur
- Output terformat dengan tabel
- Berbagai kasus penggunaan praktis
- Visualisasi konvergensi

## 📚 Dokumentasi Lengkap

Untuk dokumentasi detail, lihat:
- [`docs/features_documentation.md`](docs/features_documentation.md) - Dokumentasi lengkap dengan contoh
- [`core/analysis/numerical_features.py`](core/analysis/numerical_features.py) - Source code dengan docstrings

## 💡 Contoh Kasus Penggunaan

### Kasus 1: Analisis Metode Newton-Raphson

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Mencari akar √2 dengan Newton-Raphson
na = NumericalAnalysis("x**2 - 2")
iterations = [1.5, 1.4167, 1.4142, 1.41421]
true_value = np.sqrt(2)

for i, approx in enumerate(iterations):
    error = na.error_analysis(approx, true_value)
    tol = na.check_tolerance(approx, true_value, 1e-5, 'absolute')
    
    print(f"Iterasi {i+1}: {approx:.6f}")
    print(f"  Error: {error['absolute_error']:.8e}")
    print(f"  Konvergen: {tol['converged']}")
```

### Kasus 2: Aproksimasi dengan Taylor

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Aproksimasi sin(π/4) dengan Taylor
na = NumericalAnalysis("sin(x)")

for n in [2, 4, 6, 8]:
    result = na.taylor_approximation(0, n, np.pi/4)
    print(f"{n} suku: {result['approximation']:.8f}")
    print(f"  Error: {result['error_analysis']['absolute_error']:.8e}")
```

### Kasus 3: Validasi Konvergensi

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Validasi aproksimasi π
na = NumericalAnalysis("pi")
approximations = [3.0, 3.1, 3.14, 3.141, 3.1415]

for approx in approximations:
    result = na.adaptive_tolerance(approx, np.pi, 4)
    print(f"{approx}: {result['achieved_digits']:.2f} digit")
    print(f"  Memenuhi target: {result['meets_target']}")
```

## 🔧 API Reference

### Class: NumericalAnalysis

```python
NumericalAnalysis(func_str: str)
```

**Parameter:**
- `func_str`: String representasi fungsi (e.g., "sin(x)", "x**2", "exp(x)")

**Methods:**

#### Fitur 1: Nilai Sebenarnya
- `true_value(x_val)` → float
- `true_value_symbolic(x_val)` → sympy.Expr
- `evaluate_at_points(x_points)` → dict

#### Fitur 2: Error
- `absolute_error(approx, true)` → float
- `relative_error(approx, true)` → float
- `percentage_error(approx, true)` → float
- `error_analysis(approx, x_val)` → dict
- `compare_approximations(approximations, x_val)` → list[dict]

#### Fitur 3: Toleransi
- `check_tolerance(approx, true, tolerance, error_type)` → dict
- `iterative_tolerance_check(current, previous, tolerance)` → dict
- `adaptive_tolerance(approx, true, target_digits)` → dict

#### Fitur 4: Taylor
- `taylor_polynomial(x0, n_terms)` → dict
- `taylor_approximation(x0, n_terms, x_eval)` → dict
- `taylor_convergence(x0, max_terms, x_eval)` → list[dict]
- `taylor_remainder(x0, n_terms, x_eval)` → dict

### Utility Functions

```python
create_error_table(approximations, true_values, labels=None)
format_error_output(error_dict, precision=8)
```

## 📊 Dependencies

- Python 3.7+
- NumPy
- SymPy

Install dengan:
```bash
pip install numpy sympy
```

## 🤝 Kontribusi

Untuk berkontribusi:
1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Proyek ini adalah bagian dari pembelajaran Analisis Numerik.

## 👥 Author

Numerical Analysis Team

## 📞 Support

Untuk pertanyaan atau masalah:
- Buka issue di GitHub
- Lihat dokumentasi lengkap di `docs/features_documentation.md`
- Jalankan demo untuk contoh penggunaan

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-26
