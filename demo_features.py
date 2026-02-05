"""
Demo untuk 4 Fitur Analisis Numerik:
1. Nilai sebenarnya f(x)
2. Error absolut / relatif
3. Toleransi error
4. Bentuk polinom Taylor
"""

import numpy as np
from core.analysis.numerical_features import NumericalAnalysis, create_error_table, format_error_output


def demo_fitur_1_nilai_sebenarnya():
    """Demo Fitur 1: Menghitung nilai sebenarnya f(x)"""
    print("=" * 70)
    print("FITUR 1: NILAI SEBENARNYA f(x)")
    print("=" * 70)
    
    # Contoh 1: Fungsi trigonometri
    print("\n1. Fungsi sin(x)")
    print("-" * 70)
    na_sin = NumericalAnalysis("sin(x)")
    
    x_values = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    print(f"{'x':<15} {'sin(x)':<15} {'Nilai Simbolik':<20}")
    print("-" * 70)
    for x in x_values:
        true_val = na_sin.true_value(x)
        symbolic_val = na_sin.true_value_symbolic(x)
        print(f"{x:<15.6f} {true_val:<15.8f} {str(symbolic_val):<20}")
    
    # Contoh 2: Fungsi eksponensial
    print("\n2. Fungsi exp(x)")
    print("-" * 70)
    na_exp = NumericalAnalysis("exp(x)")
    
    x_values = [0, 0.5, 1.0, 1.5, 2.0]
    print(f"{'x':<15} {'exp(x)':<15}")
    print("-" * 70)
    for x in x_values:
        true_val = na_exp.true_value(x)
        print(f"{x:<15.6f} {true_val:<15.8f}")
    
    # Contoh 3: Fungsi polinomial
    print("\n3. Fungsi x**3 - 2*x + 1")
    print("-" * 70)
    na_poly = NumericalAnalysis("x**3 - 2*x + 1")
    
    x_points = [-2, -1, 0, 1, 2]
    result = na_poly.evaluate_at_points(x_points)
    print(f"{'x':<15} {'f(x)':<15}")
    print("-" * 70)
    for x, y in zip(result['x_points'], result['y_values']):
        print(f"{x:<15.6f} {y:<15.8f}")
    
    print("\n")


def demo_fitur_2_error_absolut_relatif():
    """Demo Fitur 2: Menghitung error absolut dan relatif"""
    print("=" * 70)
    print("FITUR 2: ERROR ABSOLUT DAN RELATIF")
    print("=" * 70)
    
    # Contoh 1: Aproksimasi π
    print("\n1. Aproksimasi nilai π")
    print("-" * 70)
    na_const = NumericalAnalysis("pi")
    
    approximations = [3.0, 3.1, 3.14, 3.141, 3.1415, 3.14159]
    true_pi = np.pi
    
    print(f"{'Aproksimasi':<15} {'Error Absolut':<15} {'Error Relatif':<15} {'Error %':<15}")
    print("-" * 70)
    for approx in approximations:
        abs_err = abs(approx - true_pi)
        rel_err = abs_err / abs(true_pi)
        pct_err = rel_err * 100
        print(f"{approx:<15.6f} {abs_err:<15.8e} {rel_err:<15.8e} {pct_err:<15.8f}")
    
    # Contoh 2: Aproksimasi sin(π/4)
    print("\n2. Aproksimasi sin(π/4)")
    print("-" * 70)
    na_sin = NumericalAnalysis("sin(x)")
    
    x_val = np.pi / 4
    approximations = [0.7, 0.707, 0.7071, 0.70710, 0.707106, 0.7071067]
    
    print(f"{'Aproksimasi':<15} {'Nilai True':<15} {'Abs Error':<15} {'Rel Error':<15}")
    print("-" * 70)
    for approx in approximations:
        error_result = na_sin.error_analysis(approx, x_val)
        print(f"{approx:<15.8f} {error_result['true_value']:<15.8f} "
              f"{error_result['absolute_error']:<15.8e} {error_result['relative_error']:<15.8e}")
    
    # Contoh 3: Membandingkan beberapa aproksimasi
    print("\n3. Perbandingan Aproksimasi exp(1)")
    print("-" * 70)
    na_exp = NumericalAnalysis("exp(x)")
    
    approximations = [2.5, 2.7, 2.71, 2.718, 2.7182, 2.71828]
    comparison = na_exp.compare_approximations(approximations, 1.0)
    
    print(f"{'Iterasi':<10} {'Aproksimasi':<15} {'True Value':<15} {'Abs Error':<15} {'Rel Error':<15}")
    print("-" * 70)
    for result in comparison:
        print(f"{result['iteration']:<10} {result['approximation']:<15.8f} "
              f"{result['true_value']:<15.8f} {result['absolute_error']:<15.8e} "
              f"{result['relative_error']:<15.8e}")
    
    print("\n")


def demo_fitur_3_toleransi_error():
    """Demo Fitur 3: Memeriksa toleransi error"""
    print("=" * 70)
    print("FITUR 3: TOLERANSI ERROR")
    print("=" * 70)
    
    # Contoh 1: Pengecekan toleransi absolut
    print("\n1. Pengecekan Toleransi Absolut")
    print("-" * 70)
    na = NumericalAnalysis("sqrt(2)")
    
    true_val = np.sqrt(2)
    approximations = [1.4, 1.41, 1.414, 1.4142, 1.41421]
    tolerance = 0.001
    
    print(f"Toleransi: {tolerance}")
    print(f"{'Aproksimasi':<15} {'Error':<15} {'Status':<20}")
    print("-" * 70)
    for approx in approximations:
        result = na.check_tolerance(approx, true_val, tolerance, 'absolute')
        status = "✓ KONVERGEN" if result['converged'] else "✗ BELUM KONVERGEN"
        print(f"{approx:<15.8f} {result['error']:<15.8e} {status:<20}")
    
    # Contoh 2: Pengecekan toleransi relatif
    print("\n2. Pengecekan Toleransi Relatif")
    print("-" * 70)
    na_exp = NumericalAnalysis("exp(x)")
    
    x_val = 1.0
    true_val = na_exp.true_value(x_val)
    approximations = [2.5, 2.7, 2.71, 2.718, 2.7182]
    tolerance = 0.01  # 1%
    
    print(f"Toleransi relatif: {tolerance} (1%)")
    print(f"{'Aproksimasi':<15} {'Rel Error':<15} {'Status':<20}")
    print("-" * 70)
    for approx in approximations:
        result = na_exp.check_tolerance(approx, true_val, tolerance, 'relative')
        status = "✓ KONVERGEN" if result['converged'] else "✗ BELUM KONVERGEN"
        print(f"{approx:<15.8f} {result['error']:<15.8e} {status:<20}")
    
    # Contoh 3: Toleransi iteratif (tanpa nilai true)
    print("\n3. Toleransi Iteratif (Metode Iteratif)")
    print("-" * 70)
    na_iter = NumericalAnalysis("x**2 - 2")
    
    # Simulasi iterasi metode Newton
    iterations = [1.5, 1.4167, 1.4142, 1.41421, 1.414213]
    tolerance = 1e-5
    
    print(f"Toleransi: {tolerance}")
    print(f"{'Iterasi':<10} {'Nilai':<15} {'Error Iteratif':<15} {'Status':<20}")
    print("-" * 70)
    for i in range(1, len(iterations)):
        result = na_iter.iterative_tolerance_check(iterations[i], iterations[i-1], tolerance)
        status = "✓ KONVERGEN" if result['converged'] else "✗ LANJUT"
        print(f"{i:<10} {iterations[i]:<15.8f} {result['iterative_error']:<15.8e} {status:<20}")
    
    # Contoh 4: Toleransi adaptif berdasarkan digit signifikan
    print("\n4. Toleransi Adaptif (Digit Signifikan)")
    print("-" * 70)
    na_pi = NumericalAnalysis("pi")
    
    true_val = np.pi
    approximations = [3.0, 3.1, 3.14, 3.141, 3.1415, 3.14159, 3.141592]
    target_digits = 4
    
    print(f"Target: {target_digits} digit signifikan")
    print(f"{'Aproksimasi':<15} {'Digit Tercapai':<15} {'Status':<20}")
    print("-" * 70)
    for approx in approximations:
        result = na_pi.adaptive_tolerance(approx, true_val, target_digits)
        status = "✓ MEMENUHI" if result['meets_target'] else "✗ BELUM"
        achieved = result['achieved_digits'] if result['achieved_digits'] != float('inf') else 'inf'
        print(f"{approx:<15.8f} {achieved:<15} {status:<20}")
    
    print("\n")


def demo_fitur_4_polinom_taylor():
    """Demo Fitur 4: Bentuk polinom Taylor"""
    print("=" * 70)
    print("FITUR 4: BENTUK POLINOM TAYLOR")
    print("=" * 70)
    
    # Contoh 1: Ekspansi Taylor exp(x) di x0=0
    print("\n1. Ekspansi Taylor exp(x) di x₀ = 0")
    print("-" * 70)
    na_exp = NumericalAnalysis("exp(x)")
    
    taylor_result = na_exp.taylor_polynomial(0, 6)
    print(f"Fungsi: {taylor_result['function']}")
    print(f"Titik ekspansi: x₀ = {taylor_result['expansion_point']}")
    print(f"Jumlah suku: {taylor_result['n_terms']}")
    print(f"\nPolinom Taylor:")
    print(taylor_result['polynomial_str'])
    
    print(f"\nKoefisien:")
    for term in taylor_result['terms']:
        print(f"  Suku ke-{term['order']}: koefisien = {term['coefficient']}")
    
    # Contoh 2: Ekspansi Taylor sin(x) di x0=0
    print("\n2. Ekspansi Taylor sin(x) di x₀ = 0")
    print("-" * 70)
    na_sin = NumericalAnalysis("sin(x)")
    
    taylor_result = na_sin.taylor_polynomial(0, 7)
    print(f"Polinom Taylor:")
    print(taylor_result['polynomial_str'])
    
    print(f"\nDetail Suku-suku:")
    print(f"{'Order':<8} {'Koefisien':<20} {'Turunan di x₀':<20}")
    print("-" * 70)
    for term in taylor_result['terms']:
        print(f"{term['order']:<8} {str(term['coefficient']):<20} {str(term['derivative_at_x0']):<20}")
    
    # Contoh 3: Aproksimasi dengan Taylor
    print("\n3. Aproksimasi menggunakan Taylor")
    print("-" * 70)
    na_cos = NumericalAnalysis("cos(x)")
    
    x_eval = np.pi / 4
    n_terms_list = [2, 4, 6, 8, 10]
    
    print(f"Aproksimasi cos(π/4) dengan berbagai jumlah suku:")
    print(f"{'n suku':<10} {'Aproksimasi':<15} {'Nilai True':<15} {'Abs Error':<15} {'Rel Error':<15}")
    print("-" * 70)
    
    true_val = na_cos.true_value(x_eval)
    for n in n_terms_list:
        approx_result = na_cos.taylor_approximation(0, n, x_eval)
        print(f"{n:<10} {approx_result['approximation']:<15.10f} "
              f"{approx_result['true_value']:<15.10f} "
              f"{approx_result['error_analysis']['absolute_error']:<15.8e} "
              f"{approx_result['error_analysis']['relative_error']:<15.8e}")
    
    # Contoh 4: Konvergensi Taylor
    print("\n4. Analisis Konvergensi Deret Taylor")
    print("-" * 70)
    na_exp2 = NumericalAnalysis("exp(x)")
    
    convergence = na_exp2.taylor_convergence(0, 10, 1.0)
    
    print(f"Konvergensi exp(1) dengan menambah jumlah suku:")
    print(f"{'n':<5} {'Aproksimasi':<18} {'Error Absolut':<18} {'Error %':<15}")
    print("-" * 70)
    for result in convergence:
        print(f"{result['n_terms']:<5} {result['approximation']:<18.12f} "
              f"{result['absolute_error']:<18.8e} {result['percentage_error']:<15.8f}")
    
    # Contoh 5: Remainder Taylor
    print("\n5. Analisis Remainder (Sisa) Taylor")
    print("-" * 70)
    na_sin2 = NumericalAnalysis("sin(x)")
    
    x_eval = 1.0
    n_terms_list = [3, 5, 7, 9]
    
    print(f"Remainder untuk sin(1) dengan berbagai jumlah suku:")
    print(f"{'n suku':<10} {'Aproksimasi':<15} {'Remainder':<15} {'Suku Berikutnya':<18}")
    print("-" * 70)
    for n in n_terms_list:
        remainder_result = na_sin2.taylor_remainder(0, n, x_eval)
        next_term_str = f"{remainder_result['next_term']:.8e}" if remainder_result['next_term'] is not None else "N/A"
        print(f"{n:<10} {remainder_result['approximation']:<15.10f} "
              f"{remainder_result['actual_remainder']:<15.8e} {next_term_str:<18}")
    
    print("\n")


def demo_format_output():
    """Demo format output yang rapi"""
    print("=" * 70)
    print("BONUS: FORMAT OUTPUT YANG RAPI")
    print("=" * 70)
    
    na = NumericalAnalysis("exp(x)")
    
    # Analisis error
    error_result = na.error_analysis(2.71, 1.0)
    
    print("\n1. Format Error Output:")
    print(format_error_output(error_result))
    
    # Tabel error
    print("\n2. Tabel Error untuk Multiple Approximations:")
    approximations = [2.5, 2.7, 2.71, 2.718]
    true_values = [np.e] * len(approximations)
    labels = [f"Iterasi {i+1}" for i in range(len(approximations))]
    
    error_table = create_error_table(approximations, true_values, labels)
    
    print(f"{'Label':<15} {'Aproksimasi':<15} {'True Value':<15} {'Abs Error':<15} {'Rel Error':<15}")
    print("-" * 75)
    for row in error_table:
        print(f"{row['label']:<15} {row['approximation']:<15.8f} "
              f"{row['true_value']:<15.8f} {row['absolute_error']:<15.8e} "
              f"{row['relative_error']:<15.8e}")
    
    print("\n")


def main():
    """Jalankan semua demo"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  DEMO 4 FITUR ANALISIS NUMERIK".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  1. Nilai sebenarnya f(x)".ljust(68) + "║")
    print("║" + "  2. Error absolut / relatif".ljust(68) + "║")
    print("║" + "  3. Toleransi error".ljust(68) + "║")
    print("║" + "  4. Bentuk polinom Taylor".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Jalankan semua demo
    demo_fitur_1_nilai_sebenarnya()
    demo_fitur_2_error_absolut_relatif()
    demo_fitur_3_toleransi_error()
    demo_fitur_4_polinom_taylor()
    demo_format_output()
    
    print("=" * 70)
    print("DEMO SELESAI!")
    print("=" * 70)
    print("\nUntuk menggunakan fitur-fitur ini dalam kode Anda:")
    print("  from core.analysis.numerical_features import NumericalAnalysis")
    print("  na = NumericalAnalysis('sin(x)')")
    print("  result = na.true_value(1.0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
