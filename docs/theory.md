# Teori Analisis Numerik

## Daftar Isi
1. [Persamaan Biharmonik - Teori Plat](#1-persamaan-biharmonik---teori-plat)
2. [Metode Finite Difference](#2-metode-finite-difference)
3. [Diskritisasi Stencil 13-Titik](#3-diskritisasi-stencil-13-titik)
4. [Kondisi Batas](#4-kondisi-batas)
5. [Solusi Analitik](#5-solusi-analitik)

---

## 1. Persamaan Biharmonik - Teori Plat

### Deskripsi
Persamaan biharmonik mendeskripsikan lendutan (deflection) plat elastis tipis di bawah beban transversal. Persamaan ini diturunkan dari teori elastisitas dan teori plat.

### Persamaan Dasar
```
∇⁴w = ∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴ = q/D
```

### Parameter
| Simbol | Deskripsi | Satuan |
|--------|-----------|--------|
| w | Lendutan plat | m |
| q | Beban transversal | N/m² |
| D | Kekakuan lentur (flexural rigidity) | N·m |
| ∇⁴ | Operator biharmonik | - |

### Kekakuan Lentur D
```
D = Eh³ / [12(1 - ν²)]
```

| Parameter | Deskripsi |
|-----------|-----------|
| E | Modulus Young (MPa) |
| h | Ketebalan plat (m) |
| ν | Rasio Poisson |

---

## 2. Metode Finite Difference

### Konsep Dasar
Metode finite difference mengaproksimasi turunan dengan perbedaan nilai pada grid points.

### Aproksimasi Turunan

#### Turunan Pertama
```
Forward:  f'(x) ≈ [f(x+h) - f(x)] / h
Backward: f'(x) ≈ [f(x) - f(x-h)] / h
Central:  f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```

#### Turunan Kedua
```
Central: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
```

---

## 3. Diskritisasi Stencil 13-Titik

### Stencil untuk ∇⁴w

Operator biharmonik diskritisasi menggunakan stencil 13-titik:

```
wᵢ₊₂,ⱼ      wᵢ₊₁,ⱼ₊₁    wᵢ₊₁,ⱼ      wᵢ₊₁,ⱼ₋₁    wᵢ₊₂,ⱼ
             ↘      ↙
wᵢ,ⱼ₊₂  ←  wᵢ,ⱼ₊₁  ←  wᵢ,ⱼ  →  wᵢ,ⱼ₋₁  →  wᵢ,ⱼ₋₂
             ↗      ↖
wᵢ₋₂,ⱼ      wᵢ₋₁,ⱼ₋₁    wᵢ₋₁,ⱼ      wᵢ₋₁,ⱼ₊₁    wᵢ₋₂,ⱼ
```

### Formula Diskritisasi
```
20wᵢ,ⱼ 
- 8(wᵢ₊₁,ⱼ + wᵢ₋₁,ⱼ + wᵢ,ⱼ₊₁ + wᵢ,ⱼ₋₁) 
+ 2(wᵢ₊₁,ⱼ₊₁ + wᵢ₊₁,ⱼ₋₁ + wᵢ₋₁,ⱼ₊₁ + wᵢ₋₁,ⱼ₋₁) 
- (wᵢ₊₂,ⱼ + wᵢ₋₂,ⱼ + wᵢ,ⱼ₊₂ + wᵢ,ⱼ₋₂) = (q/D)h⁴
```

### Penjelasan Koefisien
| Koefisien | Jumlah Titik | Total Kontribusi |
|-----------|--------------|------------------|
| 20 | Center (1) | 20 |
| -8 | Tetangga terdekat (4) | -32 |
| +2 | Diagonal (4) | +8 |
| -1 | Tetangga kedua (4) | -4 |
| **Total** | **13** | **-8** |

---

## 4. Kondisi Batas

### Simply Supported (Tumpuan Sederhana)
- w = 0 di batas
- Momen lentur = 0 → ∇²w = 0 di batas

### Clamped (Terjepit)
- w = 0 di batas
- dw/dn = 0 di batas (turunan normal = 0)

---

## 5. Solusi Analitik

### Solusi Navier (Plat Persegi)
Untuk plat persegi dengan kondisi simply supported:

```
w(x,y) = Σ Σ [16q / (π⁶Dm n)] × sin(mπx/Lₓ) sin(nπy/Lᵧ) / (m²/Lₓ² + n²/Lᵧ²)²
```

### Koefisien Defleksi Maksimum

Untuk plat persegi dengan beban uniform:

```
w_max = α × qL⁴ / D
```

| Rasio Aspek (Lᵧ/Lₓ) | α (Simply Supported) | α (Clamped) |
|---------------------|----------------------|-------------|
| 1.0 | 0.00406 | 0.00126 |
| 1.2 | 0.00387 | 0.00124 |
| 1.4 | 0.00359 | 0.00119 |
| 1.6 | 0.00326 | 0.00113 |
| 2.0 | 0.00257 | 0.00101 |

---

## Verifikasi Solver

### Test Case
- Beban: q = 10 kN/m²
- D: D = 100 kN·m
- Dimensi: L = 1 m × 1 m
- Grid: 41 × 41 points

### Hasil yang Diharapkan
| Metode | Defleksi Maksimum |
|--------|------------------|
| Analitik | 0.4062 mm |
| Numerik | ~0.4060 mm |
| Error | < 1% |

---

## Referensi

1. Timoshenko, S., & Goodier, J. N. (1970). Theory of Elasticity (3rd ed.). McGraw-Hill.
2. Reddy, J. N. (2006). An Introduction to the Finite Element Method (3rd ed.). McGraw-Hill.
3. Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.). Brooks/Cole.

---

**Versi**: 1.0.0  
**Tanggal**: 2026-02-05
