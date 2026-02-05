# Taylor Series Improvements

## Ringkasan

Dokumentasi ini menjelaskan improvements yang telah dilakukan pada fitur Taylor Series di aplikasi Analisis Numerik.

## Improvements yang Dilakukan

### 1. Error Bound (Batas Error)

**Deskripsi:**
Menambahkan fitur untuk menghitung batas maksimum error (Error Bound) berdasarkan teorema sisa Lagrange.

**Implementasi:**
- Menggunakan rumus Lagrange remainder: `|R_n(x)| ≤ M * |f^(n+1)(x)| / (n+1)!`
- M adalah maksimum dari turunan ke-(n+1) pada interval [x0, x_eval]
- Menghitung maksimum turunan dengan mengevaluasi pada 100 titik dalam interval

**Parameter:**
- `error_bound`: Batas maksimum error yang diizinkan (opsional)

**Output:**
- `bound`: Nilai batas error yang dihitung
- `formula`: Rumus yang digunakan
- `description`: Penjelasan dalam bahasa Indonesia

**Contoh Penggunaan:**
```python
result = taylor_series(
    func_str="exp(x)",
    x0=0.0,
    n_terms=5,
    x_eval=1.0,
    error_bound=0.01
)
```

### 2. Tolerance Bound (Batas Toleransi)

**Deskripsi:**
Menambahkan fitur untuk menghitung toleransi minimum yang diperlukan untuk mencapai jumlah digit signifikan tertentu.

**Implementasi:**
- Menggunakan rumus: `0.5 × 10^(-n)` untuk n digit signifikan
- Menghitung toleransi minimum berdasarkan nilai sebenarnya fungsi

**Parameter:**
- `tolerance_bound`: Jumlah digit signifikan yang diinginkan (opsional)

**Output:**
- `tolerance`: Nilai toleransi minimum
- `formula`: Rumus yang digunakan
- `description`: Penjelasan dalam bahasa Indonesia

**Contoh Penggunaan:**
```python
result = taylor_series(
    func_str="sin(x)",
    x0=0.0,
    n_terms=5,
    x_eval=0.5,
    tolerance_bound=4  # 4 digit signifikan
)
```

### 3. Nama Kolom yang Lebih Mudah Dimengerti

**Deskripsi:**
Mengubah nama kolom dalam tabel hasil perhitungan menjadi lebih mudah dimengerti oleh mahasiswa.

**Perubahan Nama Kolom:**

| Nama Lama | Nama Baru |
|-----------|-----------|
| `n_terms` | `Jumlah Suku` |
| `approximation` | `Nilai Aproksimasi` |
| `true_value` | `Nilai Sebenarnya` |
| `abs_error` | `Error Absolut` |
| `rel_error` | `Error Relatif` |
| - | `Error Persentase` (baru) |
| - | `Memenuhi Batas Error` (baru) |
| - | `Memenuhi Toleransi` (baru) |
| - | `Keterangan` (baru) |

**Keterangan:**
- `Konvergen`: Error berada dalam batas toleransi
- `Belum Konvergen`: Error belum berada dalam batas toleransi

### 4. Penjelasan Bahasa Indonesia

**Deskripsi:**
Menambahkan penjelasan dalam bahasa Indonesia untuk membantu mahasiswa memahami hasil perhitungan.

**Penjelasan yang Ditambahkan:**

1. **Penjelasan Tabel:**
   - **Jumlah Suku**: Banyaknya suku yang digunakan dalam aproksimasi
   - **Nilai Aproksimasi**: Nilai yang dihitung menggunakan deret Taylor
   - **Nilai Sebenarnya**: Nilai fungsi asli pada titik tersebut
   - **Error Absolut**: Selisih mutlak antara nilai aproksimasi dan nilai sebenarnya
   - **Error Relatif**: Error absolut dibandingkan dengan nilai sebenarnya
   - **Error Persentase**: Error relatif dalam bentuk persentase
   - **Memenuhi Batas Error**: Apakah error berada dalam batas maksimum yang diizinkan
   - **Memenuhi Toleransi**: Apakah error berada dalam batas toleransi yang ditentukan
   - **Keterangan**: Status konvergensi (Konvergen/Belum Konvergen)

2. **Penjelasan Error Bound:**
   - Rumus: `|R_n(x)| <= M * |x - x0|^(n+1) / (n+1)!`
   - Deskripsi: "Batas maksimum error berdasarkan turunan ke-(n+1)"

3. **Penjelasan Tolerance Bound:**
   - Rumus: `0.5 × 10^(-n)`
   - Deskripsi: "Toleransi minimum untuk n digit signifikan"

## Perubahan pada File

### 1. `core/series/taylor_series.py`

**Perubahan:**
- Menambahkan parameter `error_bound` dan `tolerance_bound` ke fungsi `taylor_series()`
- Menambahkan logika untuk menghitung error bound
- Menambahkan logika untuk menghitung tolerance bound
- Mengubah nama kolom dalam dictionary `approximations`
- Menambahkan kolom baru: `Error Persentase`, `Memenuhi Batas Error`, `Memenuhi Toleransi`, `Keterangan`

### 2. `ui/displays/series_display.py`

**Perubahan:**
- Menambahkan display untuk error bound dan tolerance bound
- Mengubah nama kolom yang ditampilkan dalam tabel
- Menambahkan penjelasan dalam bahasa Indonesia
- Memformat nilai numerik untuk tampilan yang lebih baik

### 3. `ui/input_form.py`

**Perubahan:**
- Menambahkan checkbox untuk mengaktifkan error bound
- Menambahkan checkbox untuk mengaktifkan tolerance bound
- Menambahkan input field untuk batas error maksimum
- Menambahkan input field untuk jumlah digit signifikan

### 4. `app.py`

**Perubahan:**
- Menambahkan parameter `error_bound` dan `tolerance_bound` saat memanggil fungsi `taylor_series()`

## Testing

### Test Script

File `test_taylor_improvements.py` dibuat untuk menguji semua improvements:

1. **Test 1**: Taylor Series dengan Error Bound
2. **Test 2**: Taylor Series dengan Tolerance Bound
3. **Test 3**: Taylor Series dengan Error Bound & Tolerance Bound
4. **Test 4**: Taylor Series tanpa Error Bound & Tolerance Bound

### Hasil Test

```
============================================================
SUMMARY
============================================================
Total Tests: 4
Passed: 4
Failed: 0

[OK] SEMUA TEST BERHASIL!
```

## Contoh Output

### Dengan Error Bound

```
Error Bound: 2.265235e-2
Rumus: |R_5(x)| <= 2.72e+00 * |x - x0|^6 / 6!
Deskripsi: Batas maksimum error berdasarkan turunan ke-(n+1)

Jumlah Suku | Nilai Aproksimasi | Nilai Sebenarnya | Error Absolut | Error Relatif | Error Persentase | Memenuhi Batas Error | Keterangan
-----------|-------------------|------------------|---------------|---------------|------------------|---------------------|------------
5          | 2.708333          | 2.718282         | 9.948495e-03  | 3.659847e-03  | 0.3660%          | True                | Konvergen
```

### Dengan Tolerance Bound

```
Tolerance Bound: 5.000000e-05
Rumus: 0.5 × 10^(-4)
Deskripsi: Toleransi minimum untuk 4 digit signifikan

Jumlah Suku | Nilai Aproksimasi | Nilai Sebenarnya | Error Absolut | Error Relatif | Error Persentase | Memenuhi Toleransi | Keterangan
-----------|-------------------|------------------|---------------|---------------|------------------|-------------------|------------
5          | 0.479167          | 0.479426         | 2.588719e-04  | 5.399628e-04  | 0.0540%          | False             | Belum Konvergen
```

## Cara Menggunakan

### Melalui UI Streamlit

1. Buka aplikasi Streamlit
2. Pilih kategori "Series"
3. Pilih method "Taylor"
4. Masukkan parameter:
   - Fungsi f(x)
   - Jumlah suku
   - Titik ekspansi (x₀)
   - Nilai x untuk evaluasi
5. (Opsional) Centang "Hitung Batas Error (Error Bound)" dan masukkan batas error maksimum
6. (Opsional) Centang "Hitung Batas Toleransi (Tolerance Bound)" dan masukkan jumlah digit signifikan
7. Klik "Hitung Deret Taylor"

### Melalui Python Code

```python
from core.series.taylor_series import taylor_series

# Dengan Error Bound
result = taylor_series(
    func_str="exp(x)",
    x0=0.0,
    n_terms=5,
    x_eval=1.0,
    error_bound=0.01
)

# Dengan Tolerance Bound
result = taylor_series(
    func_str="sin(x)",
    x0=0.0,
    n_terms=5,
    x_eval=0.5,
    tolerance_bound=4
)

# Dengan Keduanya
result = taylor_series(
    func_str="cos(x)",
    x0=0.0,
    n_terms=5,
    x_eval=0.5,
    error_bound=0.01,
    tolerance_bound=4
)

# Tanpa Error Bound dan Tolerance Bound
result = taylor_series(
    func_str="ln(1+x)",
    x0=0.0,
    n_terms=5,
    x_eval=0.5
)
```

## Manfaat untuk Pembelajaran

1. **Pemahaman Error**: Mahasiswa dapat memahami konsep error bound dan bagaimana error berkurang seiring bertambahnya jumlah suku.

2. **Analisis Konvergensi**: Mahasiswa dapat melihat kapan aproksimasi Taylor konvergen ke nilai sebenarnya.

3. **Visualisasi Error**: Tabel dan grafik menunjukkan error absolut, error relatif, dan error persentase untuk setiap jumlah suku.

4. **Bahasa Indonesia**: Penjelasan dalam bahasa Indonesia memudahkan mahasiswa memahami konsep numerik.

5. **Nama Kolom yang Jelas**: Nama kolom yang mudah dimengerti membantu mahasiswa menginterpretasikan hasil perhitungan.

## Kesimpulan

Improvements pada Taylor Series telah berhasil diimplementasikan dan diuji. Fitur baru ini memberikan analisis error yang lebih detail dan penjelasan yang lebih mudah dimengerti, sehingga sangat bermanfaat untuk pembelajaran analisis numerik.
