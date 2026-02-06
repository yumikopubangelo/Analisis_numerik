# Panduan Penggunaan Aplikasi

## Daftar Isi
1. [Instalasi](#1-instalasi)
2. [Menjalankan Aplikasi](#2-menjalankan-aplikasi)
3. [Fitur Utama](#3-fitur-utama)
4. [Contoh Penggunaan](#4-contoh-penggunaan)
5. [Tips dan Trik](#5-tips-dan-trik)

---

## 1. Instalasi

### Persyaratan Sistem
- Python 3.8 atau lebih baru
- pip (package manager Python)

### Langkah Instalasi

```bash
# Clone repository
git clone <repository-url>
cd numerical-analysis-app

# Install dependencies
pip install -r requirements.txt
```

### Dependencies Utama
```
streamlit>=1.0.0
numpy>=1.20.0
matplotlib>=3.4.0
sympy>=1.8.0
pandas>=1.3.0
```

---

## 2. Menjalankan Aplikasi

### Menjalankan Streamlit
```bash
streamlit run app.py
```

### Akses Aplikasi
- URL lokal: http://localhost:8501
- Atau akses melalui browser

---

## 3. Fitur Utama

### 3.1 Pencarian Akar (Root Finding)

#### Metode yang Tersedia
| Metode | Kecepatan | Kepercayaan | Input Diperlukan |
|--------|-----------|-------------|------------------|
| Bisection | Sedang | 100% | a, b (tebakan awal) |
| Regula Falsi | Sedang | Tinggi | a, b |
| Newton-Raphson | Cepat | Moderate | x0, f'(x) |
| Secant | Cepat | Moderate | x0, x1 |

#### Input yang Diperlukan
```
f(x) = fungsi yang ingin dicari akarnya
a, b = interval awal (untuk Bisection/Regula Falsi)
x0    = tebakan awal (untuk Newton-Raphson)
tol   = toleransi error (default: 1e-6)
```

#### Interpretasi Output
```
root       = nilai akar yang ditemukan
iterations = jumlah iterasi yang dilakukan
error      = error akhir
converged  = apakah metode konvergen
```

### 3.2 Integrasi Numerik

#### Metode yang Tersedia
| Metode | Orde Error | Cocok untuk |
|--------|------------|-------------|
| Trapezoidal | O(h²) | Kurva sederhana |
| Simpson | O(h⁴) | Fungsi smooth |

#### Input yang Diperlukan
```
f(x) = fungsi yang ingin diintegralkan
a, b = batas integrasi
n    = jumlah subinterval
```

### 3.3 Diferensiasi Numerik

#### Metode yang Tersedia
| Metode | Orde Error | Kegunaan |
|--------|------------|----------|
| Forward | O(h) | Batas domain |
| Backward | O(h) | Batas domain |
| Central | O(h²) | Interior |

#### Input yang Diperlukan
```
f(x) = fungsi yang ingin diturunkan
x    = titik evaluasi
h    = step size
```

### 3.4 Persamaan Biharmonik (PDE)

#### Input yang Diperlukan
```
q   = beban transversal (kN/m²)
D   = kekakuan lentur (kN·m)
nx  = jumlah grid points arah x
ny  = jumlah grid points arah y
tol = toleransi konvergensi
```

#### Catatan Satuan
- q dalam kN/m² (otomatis dikonversi ke N/m²)
- D dalam kN·m (otomatis dikonversi ke N·m)

#### Output
```
X, Y       = koordinat grid
w          = lendutan plat
iterations = jumlah iterasi
error      = error konvergensi
```

---

## 4. Contoh Penggunaan

### 4.1 Mencari Akar f(x) = x² - 2

```python
from core.root_findings.bisection import bisection
import numpy as np

def f(x):
    return x**2 - 2

root, iterations, error = bisection(f, 1, 2, tol=1e-6)
print(f"Akar: {root}")
print(f"Iterasi: {iterations}")
```

### 4.2 Mengintegralkan f(x) = x²

```python
from core.integration.trapezoidal import trapezoidal

def f(x):
    return x**2

result = trapezoidal(f, 0, 1, n=100)
print(f"Hasil integrasi: {result}")
```

### 4.3 Menghitung Lendutan Plat

```python
from core.pde.biharmonic_solver import solve_biharmonic_equation
import numpy as np

# Input dalam kN
X, Y, w, iterations, error = solve_biharmonic_equation(
    q=10.0,      # 10 kN/m²
    D=100.0,    # 100 kN·m
    nx=41,
    ny=41,
    tol=1e-5
)

# Defleksi dalam mm
max_deflection = np.max(w) * 1000
print(f"Defleksi maksimum: {max_deflection:.4f} mm")
```

---

## 5. Tips dan Trik

### 5.1 Memilih Metode yang Tepat

#### Untuk Pencarian Akar
- **Tidak tahu turunan**: Gunakan Bisection atau Secant
- **Tahu turunan**: Gunakan Newton-Raphson untuk kecepatan
- **Ingin jaminan konvergensi**: Gunakan Bisection

#### Untuk Integrasi
- **Fungsi smooth**: Gunakan Simpson (lebih akurat)
- **Fungsi dengan diskontinuitas**: Gunakan Trapezoidal dengan n besar

#### Untuk PDE
- **Hasil cepat**: Gunakan grid kecil (21×21)
- **Hasil akurat**: Gunakan grid besar (81×81+)

### 5.2 Mengatur Toleransi

#### Terlalu Ketat (tol sangat kecil)
- Waktu komputasi lebih lama
- Akurasi mungkin tidak meningkat signifikan
- Risiko numerical overflow

#### Terlalu Loose (tol besar)
- Hasil kurang akurat
- Konvergensi cepat
- Mungkin tidak memenuhi kebutuhan

#### Rekomendasi
| Aplikasi | Toleransi |
|----------|-----------|
| Root Finding | 1e-6 - 1e-8 |
| Integrasi | 1e-4 - 1e-6 |
| PDE | 1e-3 - 1e-5 |

### 5.3 Visualisasi

#### Plot yang Tersedia
- Grafik fungsi 2D
- Plot error vs iterasi
- Surface plot 3D untuk PDE
- Kontur plot

#### Cara Menampilkan
Semua plot otomatis ditampilkan di UI Streamlit setelah perhitungan selesai.

---

## Troubleshooting

### Error: "No convergence"
**Solusi**: 
- Perbesar toleransi
- Perbesar jumlah iterasi maksimum
- Periksa apakah fungsi memiliki akar di interval

### Error: "Division by zero"
**Solusi**:
- Hindari titik di mana fungsi = 0
- Gunakan toleransi untuk menghindari pembagian sangat kecil

### Plot tidak muncul
**Solusi**:
- Periksa apakah matplotlib.pyplot ditutup setelah penggunaan
- Gunakan `plt.close()` setelah menampilkan plot

---

## Frequently Asked Questions

**Q: Mengapa defleksi plat negatif?**  
A: Konvensi tanda tergantung pada sistem koordinat. Nilai absolut adalah yang penting.

**Q: Berapa grid size yang optimal?**  
A: Untuk aplikasi praktis, 41×41 memberikan keseimbangan antara akurasi dan kecepatan.

**Q: Solver tidak konvergen?**  
A: Coba: (1) turunkan toleransi, (2) tingkatkan iterasi maksimum, (3) gunakan grid lebih halus.

---

**Versi**: 1.0.0  
**Tanggal**: 2026-02-05
