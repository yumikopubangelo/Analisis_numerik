# Dokumentasi 4 Fitur Analisis Numerik

## Daftar Isi
1. [Nilai Sebenarnya f(x)](#1-nilai-sebenarnya-fx)
2. [Error Absolut / Relatif](#2-error-absolut--relatif)
3. [Toleransi Error](#3-toleransi-error)
4. [Bentuk Polinom Taylor](#4-bentuk-polinom-taylor)

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

#### `evaluate_at_points(x_points)`
Evaluasi fungsi pada beberapa titik sekaligus.

**Parameter:**
- `x_points`: list - Daftar titik x

**Return:**
- dict - Dictionary berisi x_points, y_values, dan function

**Contoh:**
```python
na = NumericalAnalysis("x**2")
result = na.evaluate_at_points([1, 2, 3, 4])
# Output: {'x_points': [1,2,3,4], 'y_values': [1,4,9,16], 'function': 'x**2'}
```

### Kasus Penggunaan
- Menghitung nilai referensi untuk perbandingan
- Validasi hasil aproksimasi
- Plotting fungsi asli
- Analisis konvergensi metode numerik

---

## 2. Error Absolut / Relatif

### Deskripsi
Fitur ini menghitung berbagai jenis error (kesalahan) antara nilai aproksimasi dan nilai sebenarnya. Error adalah ukuran seberapa jauh hasil aproksimasi dari nilai yang sebenarnya.

### Jenis-jenis Error

#### Error Absolut
Error absolut adalah selisih mutlak antara nilai aproksimasi dan nilai sebenarnya:
```
Error Absolut = |aproksimasi - nilai_sebenarnya|
```

#### Error Relatif
Error relatif adalah rasio error absolut terhadap nilai sebenarnya:
```
Error Relatif = |aproksimasi - nilai_sebenarnya| / |nilai_sebenarnya|
```

#### Error Persentase
Error dalam bentuk persentase:
```
Error Persentase = Error Relatif × 100%
```

### Fungsi Utama

#### `absolute_error(approx, true)`
Menghitung error absolut.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `true`: float - Nilai sebenarnya

**Return:**
- float - Error absolut

**Contoh:**
```python
na = NumericalAnalysis("pi")
error = na.absolute_error(3.14, np.pi)  # Output: 0.001592...
```

#### `relative_error(approx, true)`
Menghitung error relatif.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `true`: float - Nilai sebenarnya

**Return:**
- float - Error relatif (desimal, bukan persen)

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
error = na.relative_error(2.7, np.e)  # Output: 0.00647...
```

#### `percentage_error(approx, true)`
Menghitung error dalam persentase.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `true`: float - Nilai sebenarnya

**Return:**
- float - Error persentase

**Contoh:**
```python
na = NumericalAnalysis("sqrt(2)")
error = na.percentage_error(1.41, np.sqrt(2))  # Output: 0.28%
```

#### `error_analysis(approx, x_val)`
Analisis error lengkap untuk suatu aproksimasi.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `x_val`: float - Titik evaluasi

**Return:**
- dict - Dictionary berisi semua metrik error

**Contoh:**
```python
na = NumericalAnalysis("sin(x)")
result = na.error_analysis(0.707, np.pi/4)
# Output: {
#   'x': 0.785...,
#   'true_value': 0.707106...,
#   'approximation': 0.707,
#   'absolute_error': 0.000106...,
#   'relative_error': 0.00015...,
#   'percentage_error': 0.015...,
#   'function': 'sin(x)'
# }
```

#### `compare_approximations(approximations, x_val)`
Membandingkan beberapa aproksimasi sekaligus.

**Parameter:**
- `approximations`: list - Daftar nilai aproksimasi
- `x_val`: float - Titik evaluasi

**Return:**
- list of dict - Daftar analisis error untuk setiap aproksimasi

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
results = na.compare_approximations([2.5, 2.7, 2.718], 1.0)
```

### Kasus Penggunaan
- Mengukur akurasi metode numerik
- Membandingkan berbagai metode aproksimasi
- Validasi konvergensi algoritma
- Menentukan kapan iterasi harus dihentikan

---

## 3. Toleransi Error

### Deskripsi
Fitur ini memeriksa apakah error memenuhi toleransi yang diberikan. Toleransi adalah batas maksimum error yang dapat diterima. Fitur ini penting untuk kriteria konvergensi dalam metode iteratif.

### Fungsi Utama

#### `check_tolerance(approx, true, tolerance, error_type='absolute')`
Memeriksa apakah error memenuhi toleransi.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `true`: float - Nilai sebenarnya
- `tolerance`: float - Nilai toleransi
- `error_type`: str - 'absolute' atau 'relative'

**Return:**
- dict - Dictionary berisi hasil pengecekan

**Contoh:**
```python
na = NumericalAnalysis("sqrt(2)")
result = na.check_tolerance(1.414, np.sqrt(2), 0.001, 'absolute')
# Output: {
#   'converged': True,
#   'error': 0.000213...,
#   'tolerance': 0.001,
#   'error_type': 'absolute',
#   'approximation': 1.414,
#   'true_value': 1.41421...,
#   'meets_tolerance': True
# }
```

#### `iterative_tolerance_check(current, previous, tolerance)`
Memeriksa toleransi untuk metode iteratif (tanpa nilai true).

**Parameter:**
- `current`: float - Nilai iterasi saat ini
- `previous`: float - Nilai iterasi sebelumnya
- `tolerance`: float - Nilai toleransi

**Return:**
- dict - Dictionary berisi hasil pengecekan

**Contoh:**
```python
na = NumericalAnalysis("x**2 - 2")
result = na.iterative_tolerance_check(1.4142, 1.414, 0.0001)
# Output: {
#   'converged': True,
#   'iterative_error': 0.0002,
#   'tolerance': 0.0001,
#   'current_value': 1.4142,
#   'previous_value': 1.414
# }
```

#### `adaptive_tolerance(approx, true, target_digits)`
Menghitung toleransi adaptif berdasarkan digit signifikan.

**Parameter:**
- `approx`: float - Nilai aproksimasi
- `true`: float - Nilai sebenarnya
- `target_digits`: int - Jumlah digit signifikan yang diinginkan

**Return:**
- dict - Dictionary berisi analisis toleransi adaptif

**Contoh:**
```python
na = NumericalAnalysis("pi")
result = na.adaptive_tolerance(3.14159, np.pi, 5)
# Output: {
#   'target_digits': 5,
#   'achieved_digits': 5.78...,
#   'required_tolerance': 5e-06,
#   'absolute_error': 2.65e-06,
#   'relative_error': 8.43e-07,
#   'meets_target': True
# }
```

### Kriteria Konvergensi

#### Toleransi Absolut
Digunakan ketika kita ingin error tidak melebihi nilai tertentu:
```
|x_n - x_true| < ε
```

**Kapan digunakan:**
- Ketika nilai fungsi dekat dengan nol
- Ketika kita tahu batas error yang dapat diterima

#### Toleransi Relatif
Digunakan ketika kita ingin error proporsional terhadap nilai:
```
|x_n - x_true| / |x_true| < ε
```

**Kapan digunakan:**
- Ketika nilai fungsi sangat besar atau sangat kecil
- Ketika kita ingin akurasi relatif yang konsisten

#### Toleransi Iteratif
Digunakan dalam metode iteratif tanpa nilai true:
```
|x_n - x_{n-1}| < ε
```

**Kapan digunakan:**
- Metode iteratif (Newton-Raphson, Fixed Point, dll)
- Ketika nilai true tidak diketahui

### Kasus Penggunaan
- Kriteria stopping untuk metode iteratif
- Validasi akurasi hasil
- Kontrol kualitas komputasi
- Optimasi jumlah iterasi

---

## 4. Bentuk Polinom Taylor

### Deskripsi
Fitur ini menghitung ekspansi deret Taylor dari suatu fungsi. Deret Taylor adalah representasi fungsi sebagai jumlah tak hingga dari suku-suku yang dihitung dari nilai turunan fungsi pada satu titik.

### Formula Deret Taylor
```
f(x) ≈ f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + f'''(x₀)(x-x₀)³/3! + ...
```

### Fungsi Utama

#### `taylor_polynomial(x0, n_terms)`
Menghitung bentuk polinom Taylor.

**Parameter:**
- `x0`: float - Titik ekspansi
- `n_terms`: int - Jumlah suku dalam ekspansi

**Return:**
- dict - Dictionary berisi berbagai representasi polinom

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
result = na.taylor_polynomial(0, 5)
# Output: {
#   'polynomial_symbolic': 1 + x + x**2/2 + x**3/6 + x**4/24,
#   'polynomial_str': '1 + x + x**2/2 + x**3/6 + x**4/24',
#   'polynomial_latex': '1 + x + \\frac{x^{2}}{2} + ...',
#   'expansion_point': 0,
#   'n_terms': 5,
#   'coefficients': [1.0, 1.0, 0.5, 0.166..., 0.041...],
#   'terms': [...],
#   'function': 'exp(x)'
# }
```

#### `taylor_approximation(x0, n_terms, x_eval)`
Menghitung aproksimasi Taylor dan analisis errornya.

**Parameter:**
- `x0`: float - Titik ekspansi
- `n_terms`: int - Jumlah suku
- `x_eval`: float - Titik evaluasi

**Return:**
- dict - Dictionary berisi aproksimasi dan analisis error

**Contoh:**
```python
na = NumericalAnalysis("sin(x)")
result = na.taylor_approximation(0, 5, np.pi/4)
# Output: {
#   'polynomial': {...},
#   'x_eval': 0.785...,
#   'approximation': 0.707...,
#   'true_value': 0.707106...,
#   'error_analysis': {...}
# }
```

#### `taylor_convergence(x0, max_terms, x_eval)`
Analisis konvergensi deret Taylor.

**Parameter:**
- `x0`: float - Titik ekspansi
- `max_terms`: int - Jumlah suku maksimum
- `x_eval`: float - Titik evaluasi

**Return:**
- list of dict - Daftar hasil untuk setiap jumlah suku

**Contoh:**
```python
na = NumericalAnalysis("cos(x)")
results = na.taylor_convergence(0, 10, 1.0)
# Menunjukkan bagaimana error berkurang dengan menambah suku
```

#### `taylor_remainder(x0, n_terms, x_eval)`
Estimasi remainder (sisa) dari deret Taylor.

**Parameter:**
- `x0`: float - Titik ekspansi
- `n_terms`: int - Jumlah suku yang digunakan
- `x_eval`: float - Titik evaluasi

**Return:**
- dict - Dictionary berisi informasi remainder

**Contoh:**
```python
na = NumericalAnalysis("exp(x)")
result = na.taylor_remainder(0, 5, 1.0)
# Output: {
#   'n_terms': 5,
#   'x_eval': 1.0,
#   'x0': 0,
#   'approximation': 2.708...,
#   'true_value': 2.718...,
#   'actual_remainder': 0.0099...,
#   'next_term': 0.0083...,
#   'remainder_ratio': 0.00365...
# }
```

### Contoh Ekspansi Taylor Umum

#### exp(x) di x₀ = 0
```
exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
```

#### sin(x) di x₀ = 0
```
sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
```

#### cos(x) di x₀ = 0
```
cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
```

#### ln(1+x) di x₀ = 0
```
ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
```

### Kasus Penggunaan
- Aproksimasi fungsi kompleks dengan polinomial
- Analisis perilaku fungsi di sekitar titik tertentu
- Perhitungan numerik fungsi transendental
- Estimasi error dalam aproksimasi
- Pembelajaran konsep kalkulus

---

## Fungsi Utilitas

### `create_error_table(approximations, true_values, labels=None)`
Membuat tabel error untuk beberapa aproksimasi.

**Parameter:**
- `approximations`: list - Daftar nilai aproksimasi
- `true_values`: list - Daftar nilai sebenarnya
- `labels`: list, optional - Label untuk setiap baris

**Return:**
- list of dict - Tabel error

**Contoh:**
```python
from core.analysis.numerical_features import create_error_table

approx = [3.0, 3.1, 3.14, 3.141]
true = [np.pi] * 4
labels = ["Iterasi 1", "Iterasi 2", "Iterasi 3", "Iterasi 4"]

table = create_error_table(approx, true, labels)
```

### `format_error_output(error_dict, precision=8)`
Format output error dalam bentuk string yang mudah dibaca.

**Parameter:**
- `error_dict`: dict - Dictionary hasil analisis error
- `precision`: int - Jumlah digit desimal

**Return:**
- str - String terformat

**Contoh:**
```python
from core.analysis.numerical_features import format_error_output

na = NumericalAnalysis("sin(x)")
error_result = na.error_analysis(0.707, np.pi/4)
print(format_error_output(error_result))
```

---

## Contoh Penggunaan Lengkap

### Contoh 1: Analisis Metode Newton-Raphson

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Fungsi: f(x) = x² - 2, mencari akar √2
na = NumericalAnalysis("x**2 - 2")

# Simulasi iterasi Newton-Raphson
iterations = [1.5, 1.4167, 1.4142, 1.41421, 1.414213]
true_value = np.sqrt(2)
tolerance = 1e-5

print("Iterasi Newton-Raphson untuk √2")
print("-" * 60)

for i, approx in enumerate(iterations):
    # Analisis error
    error_result = na.error_analysis(approx, true_value)
    
    # Cek toleransi
    tol_result = na.check_tolerance(approx, true_value, tolerance, 'absolute')
    
    print(f"Iterasi {i+1}:")
    print(f"  Nilai: {approx:.8f}")
    print(f"  Error absolut: {error_result['absolute_error']:.8e}")
    print(f"  Error relatif: {error_result['relative_error']:.8e}")
    print(f"  Konvergen: {'Ya' if tol_result['converged'] else 'Tidak'}")
    print()
```

### Contoh 2: Perbandingan Aproksimasi Taylor

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Aproksimasi sin(π/4) dengan Taylor
na = NumericalAnalysis("sin(x)")

x_eval = np.pi / 4
n_terms_list = [2, 4, 6, 8, 10]

print("Aproksimasi sin(π/4) dengan Deret Taylor")
print("-" * 70)

for n in n_terms_list:
    result = na.taylor_approximation(0, n, x_eval)
    
    print(f"{n} suku:")
    print(f"  Aproksimasi: {result['approximation']:.10f}")
    print(f"  Error absolut: {result['error_analysis']['absolute_error']:.8e}")
    print(f"  Error relatif: {result['error_analysis']['relative_error']:.8e}")
    print()
```

### Contoh 3: Validasi Konvergensi dengan Toleransi Adaptif

```python
from core.analysis.numerical_features import NumericalAnalysis
import numpy as np

# Aproksimasi π
na = NumericalAnalysis("pi")

approximations = [3.0, 3.1, 3.14, 3.141, 3.1415, 3.14159]
true_value = np.pi
target_digits = 4

print("Validasi Konvergensi Aproksimasi π")
print("-" * 60)

for approx in approximations:
    result = na.adaptive_tolerance(approx, true_value, target_digits)
    
    print(f"Aproksimasi: {approx}")
    print(f"  Digit tercapai: {result['achieved_digits']:.2f}")
    print(f"  Memenuhi target: {'Ya' if result['meets_target'] else 'Tidak'}")
    print()
```

---

## Tips dan Best Practices

### 1. Pemilihan Toleransi
- **Toleransi Absolut**: Gunakan untuk nilai dekat nol atau ketika error mutlak penting
- **Toleransi Relatif**: Gunakan untuk nilai yang sangat besar/kecil atau ketika akurasi relatif penting
- **Toleransi Iteratif**: Gunakan untuk metode iteratif tanpa nilai true

### 2. Jumlah Suku Taylor
- Mulai dengan suku sedikit (3-5) untuk fungsi sederhana
- Tambah suku jika error masih besar
- Perhatikan radius konvergensi
- Untuk x jauh dari x₀, butuh lebih banyak suku

### 3. Analisis Error
- Selalu hitung error absolut DAN relatif
- Gunakan error persentase untuk presentasi
- Monitor konvergensi dengan grafik error vs iterasi

### 4. Performa
- Untuk evaluasi berulang, simpan objek NumericalAnalysis
- Gunakan array numpy untuk evaluasi multiple points
- Cache hasil Taylor polynomial jika digunakan berulang

---

## Referensi

### Buku
1. Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.)
2. Chapra, S. C., & Canale, R. P. (2015). Numerical Methods for Engineers (7th ed.)

### Konsep Matematika
- **Error Absolut**: Ukuran mutlak perbedaan
- **Error Relatif**: Ukuran proporsional perbedaan
- **Deret Taylor**: Representasi fungsi sebagai polinomial
- **Konvergensi**: Proses mendekati nilai sebenarnya

### Link Terkait
- [SymPy Documentation](https://docs.sympy.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Taylor Series - Wikipedia](https://en.wikipedia.org/wiki/Taylor_series)

---

## Troubleshooting

### Error: "String literal is unterminated"
**Solusi**: Pastikan semua string ditutup dengan benar, terutama docstring dengan triple quotes.

### Error: "Division by zero"
**Solusi**: Gunakan pengecekan untuk nilai nol sebelum pembagian, terutama dalam perhitungan error relatif.

### Error: "Series does not converge"
**Solusi**: Kurangi jarak antara x_eval dan x0, atau tambah jumlah suku Taylor.

### Warning: "Overflow in calculation"
**Solusi**: Batasi jumlah suku Taylor (< 20) atau gunakan tipe data dengan presisi lebih tinggi.

---

## Lisensi dan Kontribusi

Modul ini adalah bagian dari proyek Analisis Numerik. Untuk kontribusi atau pertanyaan, silakan hubungi maintainer proyek.

**Versi**: 1.0.0  
**Tanggal**: 2026-01-26  
**Author**: Numerical Analysis Team
