"""
Module untuk fitur-fitur analisis numerik:
1. Nilai sebenarnya f(x)
2. Error absolut / relatif
3. Toleransi error
4. Bentuk polinom Taylor
"""

import sympy as sp
import numpy as np
from typing import Union, List, Dict, Tuple, Callable


class NumericalAnalysis:
    """
    Kelas untuk analisis numerik lengkap dengan berbagai fitur perhitungan error
    dan aproksimasi.
    """
    
    def __init__(self, func_str: str):
        """
        Inisialisasi dengan fungsi dalam bentuk string.
        
        Parameters:
        -----------
        func_str : str
            Fungsi dalam bentuk string, contoh: "sin(x)", "x**2 + 2*x", "exp(x)"
        """
        self.func_str = func_str
        self.x = sp.Symbol('x')
        # Map common constants so users can input e, E, pi naturally
        locals_map = {'e': sp.E, 'E': sp.E, 'pi': sp.pi}
        self.func_symbolic = sp.sympify(func_str, locals=locals_map)
        # Allow y as an alias for x - substitute y with x if present
        y = sp.Symbol('y')
        if y in self.func_symbolic.free_symbols:
            self.func_symbolic = self.func_symbolic.subs(y, self.x)
        # Ensure no unexpected symbols remain
        unknown = self.func_symbolic.free_symbols - {self.x}
        if unknown:
            unknown_str = ", ".join(sorted(str(s) for s in unknown))
            raise ValueError(f"Fungsi hanya boleh memakai variabel x. Simbol lain: {unknown_str}")
        self.func_numeric = sp.lambdify(self.x, self.func_symbolic, 'numpy')
    
    # ========== FITUR 1: NILAI SEBENARNYA f(x) ==========
    
    def true_value(self, x_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Menghitung nilai sebenarnya dari fungsi f(x) pada titik x.
        
        Parameters:
        -----------
        x_val : float atau array
            Nilai x atau array nilai x untuk evaluasi
        
        Returns:
        --------
        float atau array
            Nilai f(x) yang sebenarnya
        
        Example:
        --------
        >>> na = NumericalAnalysis("sin(x)")
        >>> na.true_value(np.pi/2)
        1.0
        """
        return self.func_numeric(x_val)
    
    def true_value_symbolic(self, x_val: float) -> sp.Expr:
        """
        Menghitung nilai sebenarnya dalam bentuk simbolik (exact).
        
        Parameters:
        -----------
        x_val : float
            Nilai x untuk evaluasi
        
        Returns:
        --------
        sympy.Expr
            Nilai eksak dalam bentuk simbolik
        """
        return self.func_symbolic.subs(self.x, x_val)
    
    def evaluate_at_points(self, x_points: List[float]) -> Dict:
        """
        Evaluasi fungsi pada beberapa titik sekaligus.
        
        Parameters:
        -----------
        x_points : list
            Daftar titik x untuk evaluasi
        
        Returns:
        --------
        dict
            Dictionary berisi x_points dan nilai fungsi
        """
        y_values = [self.true_value(x) for x in x_points]
        return {
            'x_points': x_points,
            'y_values': y_values,
            'function': self.func_str
        }
    
    # ========== FITUR 2: ERROR ABSOLUT / RELATIF ==========
    
    def absolute_error(self, approx: float, true: float) -> float:
        """
        Menghitung error absolut: |approx - true|
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        true : float
            Nilai sebenarnya
        
        Returns:
        --------
        float
            Error absolut
        
        Example:
        --------
        >>> na = NumericalAnalysis("x**2")
        >>> na.absolute_error(4.1, 4.0)
        0.1
        """
        return abs(approx - true)
    
    def relative_error(self, approx: float, true: float) -> float:
        """
        Menghitung error relatif: |approx - true| / |true|
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        true : float
            Nilai sebenarnya
        
        Returns:
        --------
        float
            Error relatif (dalam bentuk desimal, bukan persen)
        
        Example:
        --------
        >>> na = NumericalAnalysis("x**2")
        >>> na.relative_error(4.1, 4.0)
        0.025
        """
        if true == 0:
            return float('inf') if approx != 0 else 0.0
        return abs(approx - true) / abs(true)
    
    def percentage_error(self, approx: float, true: float) -> float:
        """
        Menghitung error relatif dalam persen.
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        true : float
            Nilai sebenarnya
        
        Returns:
        --------
        float
            Error relatif dalam persen
        """
        return self.relative_error(approx, true) * 100
    
    def error_analysis(self, approx: float, x_val: float) -> Dict:
        """
        Analisis error lengkap untuk suatu aproksimasi.
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        x_val : float
            Titik evaluasi
        
        Returns:
        --------
        dict
            Dictionary berisi berbagai metrik error
        """
        true = self.true_value(x_val)
        abs_err = self.absolute_error(approx, true)
        rel_err = self.relative_error(approx, true)
        
        return {
            'x': x_val,
            'true_value': true,
            'approximation': approx,
            'absolute_error': abs_err,
            'relative_error': rel_err,
            'percentage_error': rel_err * 100,
            'function': self.func_str
        }
    
    def compare_approximations(self, approximations: List[float], x_val: float) -> List[Dict]:
        """
        Membandingkan beberapa aproksimasi sekaligus.
        
        Parameters:
        -----------
        approximations : list
            Daftar nilai aproksimasi
        x_val : float
            Titik evaluasi
        
        Returns:
        --------
        list of dict
            Daftar analisis error untuk setiap aproksimasi
        """
        results = []
        true = self.true_value(x_val)
        
        for i, approx in enumerate(approximations):
            results.append({
                'iteration': i + 1,
                'approximation': approx,
                'true_value': true,
                'absolute_error': self.absolute_error(approx, true),
                'relative_error': self.relative_error(approx, true)
            })
        
        return results
    
    # ========== FITUR 3: TOLERANSI ERROR ==========
    
    def check_tolerance(self, approx: float, true: float, tolerance: float, 
                       error_type: str = 'absolute') -> Dict:
        """
        Memeriksa apakah error memenuhi toleransi yang diberikan.
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        true : float
            Nilai sebenarnya
        tolerance : float
            Nilai toleransi
        error_type : str
            Tipe error: 'absolute' atau 'relative'
        
        Returns:
        --------
        dict
            Dictionary berisi hasil pengecekan toleransi
        
        Example:
        --------
        >>> na = NumericalAnalysis("x**2")
        >>> na.check_tolerance(4.001, 4.0, 0.01, 'absolute')
        {'converged': True, 'error': 0.001, 'tolerance': 0.01, 'error_type': 'absolute'}
        """
        if error_type == 'absolute':
            error = self.absolute_error(approx, true)
        elif error_type == 'relative':
            error = self.relative_error(approx, true)
        else:
            raise ValueError("error_type harus 'absolute' atau 'relative'")
        
        converged = error < tolerance
        
        return {
            'converged': converged,
            'error': error,
            'tolerance': tolerance,
            'error_type': error_type,
            'approximation': approx,
            'true_value': true,
            'meets_tolerance': converged
        }
    
    def iterative_tolerance_check(self, current: float, previous: float, 
                                  tolerance: float) -> Dict:
        """
        Memeriksa toleransi untuk metode iteratif (tanpa nilai true).
        
        Parameters:
        -----------
        current : float
            Nilai iterasi saat ini
        previous : float
            Nilai iterasi sebelumnya
        tolerance : float
            Nilai toleransi
        
        Returns:
        --------
        dict
            Dictionary berisi hasil pengecekan
        """
        error = abs(current - previous)
        converged = error < tolerance
        
        return {
            'converged': converged,
            'iterative_error': error,
            'tolerance': tolerance,
            'current_value': current,
            'previous_value': previous
        }
    
    def adaptive_tolerance(self, approx: float, true: float, 
                          target_digits: int) -> Dict:
        """
        Menghitung toleransi adaptif berdasarkan jumlah digit signifikan yang diinginkan.
        
        Parameters:
        -----------
        approx : float
            Nilai aproksimasi
        true : float
            Nilai sebenarnya
        target_digits : int
            Jumlah digit signifikan yang diinginkan
        
        Returns:
        --------
        dict
            Dictionary berisi analisis toleransi adaptif
        """
        required_tolerance = 0.5 * 10**(-target_digits)
        abs_err = self.absolute_error(approx, true)
        rel_err = self.relative_error(approx, true)
        
        # Hitung digit signifikan yang tercapai
        if abs_err > 0:
            achieved_digits = -np.log10(2 * abs_err)
        else:
            achieved_digits = float('inf')
        
        return {
            'target_digits': target_digits,
            'achieved_digits': achieved_digits,
            'required_tolerance': required_tolerance,
            'absolute_error': abs_err,
            'relative_error': rel_err,
            'meets_target': achieved_digits >= target_digits
        }
    
    # ========== FITUR 4: BENTUK POLINOM TAYLOR ==========
    
    def taylor_polynomial(self, x0: float, n_terms: int) -> Dict:
        """
        Menghitung bentuk polinom Taylor dari fungsi.
        
        Parameters:
        -----------
        x0 : float
            Titik ekspansi
        n_terms : int
            Jumlah suku dalam ekspansi
        
        Returns:
        --------
        dict
            Dictionary berisi berbagai representasi polinom Taylor
        
        Example:
        --------
        >>> na = NumericalAnalysis("exp(x)")
        >>> result = na.taylor_polynomial(0, 4)
        >>> print(result['polynomial_str'])
        """
        # Hitung ekspansi Taylor
        series = self.func_symbolic.series(self.x, x0, n_terms).removeO()
        
        # Dapatkan koefisien dan suku-suku
        terms = []
        coefficients = []
        
        for i in range(n_terms):
            # Hitung turunan ke-i
            derivative = self.func_symbolic.diff(self.x, i)
            derivative_at_x0 = derivative.subs(self.x, x0)
            
            # Hitung koefisien
            coeff = derivative_at_x0 / sp.factorial(i)
            coefficients.append(float(coeff) if coeff.is_number else coeff)
            
            # Bentuk suku
            if i == 0:
                term_expr = coeff
                term_str = f"{coeff}"
            else:
                term_expr = coeff * (self.x - x0)**i
                if x0 == 0:
                    term_str = f"{coeff}*x^{i}"
                else:
                    term_str = f"{coeff}*(x-{x0})^{i}"
            
            terms.append({
                'order': i,
                'coefficient': float(coeff) if coeff.is_number else str(coeff),
                'term_symbolic': term_expr,
                'term_string': term_str,
                'derivative_order': i,
                'derivative_at_x0': float(derivative_at_x0) if derivative_at_x0.is_number else str(derivative_at_x0)
            })
        
        return {
            'polynomial_symbolic': series,
            'polynomial_str': str(series),
            'polynomial_latex': sp.latex(series),
            'expansion_point': x0,
            'n_terms': n_terms,
            'coefficients': coefficients,
            'terms': terms,
            'function': self.func_str
        }
    
    def taylor_approximation(self, x0: float, n_terms: int, x_eval: float) -> Dict:
        """
        Menghitung aproksimasi Taylor dan analisis errornya.
        
        Parameters:
        -----------
        x0 : float
            Titik ekspansi
        n_terms : int
            Jumlah suku
        x_eval : float
            Titik evaluasi
        
        Returns:
        --------
        dict
            Dictionary berisi aproksimasi dan analisis error
        """
        # Dapatkan polinom Taylor
        taylor_result = self.taylor_polynomial(x0, n_terms)
        series = taylor_result['polynomial_symbolic']
        
        # Evaluasi aproksimasi
        approx_value = float(series.subs(self.x, x_eval))
        
        # Nilai sebenarnya
        true_value = self.true_value(x_eval)
        
        # Analisis error
        error_analysis = self.error_analysis(approx_value, x_eval)
        
        return {
            'polynomial': taylor_result,
            'x_eval': x_eval,
            'approximation': approx_value,
            'true_value': true_value,
            'error_analysis': error_analysis
        }
    
    def taylor_convergence(self, x0: float, max_terms: int, x_eval: float) -> List[Dict]:
        """
        Analisis konvergensi deret Taylor dengan menambah jumlah suku.
        
        Parameters:
        -----------
        x0 : float
            Titik ekspansi
        max_terms : int
            Jumlah suku maksimum
        x_eval : float
            Titik evaluasi
        
        Returns:
        --------
        list of dict
            Daftar hasil untuk setiap jumlah suku
        """
        true_value = self.true_value(x_eval)
        results = []
        
        for n in range(1, max_terms + 1):
            series = self.func_symbolic.series(self.x, x0, n).removeO()
            approx = float(series.subs(self.x, x_eval))
            
            abs_err = self.absolute_error(approx, true_value)
            rel_err = self.relative_error(approx, true_value)
            
            results.append({
                'n_terms': n,
                'approximation': approx,
                'true_value': true_value,
                'absolute_error': abs_err,
                'relative_error': rel_err,
                'percentage_error': rel_err * 100
            })
        
        return results
    
    def taylor_remainder(self, x0: float, n_terms: int, x_eval: float) -> Dict:
        """
        Estimasi remainder (sisa) dari deret Taylor.
        
        Parameters:
        -----------
        x0 : float
            Titik ekspansi
        n_terms : int
            Jumlah suku yang digunakan
        x_eval : float
            Titik evaluasi
        
        Returns:
        --------
        dict
            Dictionary berisi informasi remainder
        """
        # Aproksimasi dengan n suku
        series_n = self.func_symbolic.series(self.x, x0, n_terms).removeO()
        approx_n = float(series_n.subs(self.x, x_eval))
        
        # Nilai sebenarnya
        true_value = self.true_value(x_eval)
        
        # Remainder aktual
        actual_remainder = true_value - approx_n
        
        # Estimasi remainder menggunakan suku berikutnya
        if n_terms < 20:  # Batasi untuk menghindari overflow
            series_n_plus_1 = self.func_symbolic.series(self.x, x0, n_terms + 1).removeO()
            approx_n_plus_1 = float(series_n_plus_1.subs(self.x, x_eval))
            next_term = approx_n_plus_1 - approx_n
        else:
            next_term = None
        
        return {
            'n_terms': n_terms,
            'x_eval': x_eval,
            'x0': x0,
            'approximation': approx_n,
            'true_value': true_value,
            'actual_remainder': actual_remainder,
            'next_term': next_term,
            'remainder_ratio': abs(actual_remainder / true_value) if true_value != 0 else None
        }


# ========== FUNGSI UTILITAS ==========

def create_error_table(approximations: List[float], true_values: List[float], 
                      labels: List[str] = None) -> List[Dict]:
    """
    Membuat tabel error untuk beberapa aproksimasi.
    
    Parameters:
    -----------
    approximations : list
        Daftar nilai aproksimasi
    true_values : list
        Daftar nilai sebenarnya
    labels : list, optional
        Label untuk setiap baris
    
    Returns:
    --------
    list of dict
        Tabel error dalam bentuk list of dictionaries
    """
    if labels is None:
        labels = [f"Iterasi {i+1}" for i in range(len(approximations))]
    
    table = []
    for i, (approx, true) in enumerate(zip(approximations, true_values)):
        abs_err = abs(approx - true)
        rel_err = abs_err / abs(true) if true != 0 else float('inf')
        
        table.append({
            'label': labels[i],
            'approximation': approx,
            'true_value': true,
            'absolute_error': abs_err,
            'relative_error': rel_err,
            'percentage_error': rel_err * 100
        })
    
    return table


def format_error_output(error_dict: Dict, precision: int = 8) -> str:
    """
    Format output error dalam bentuk string yang mudah dibaca.
    
    Parameters:
    -----------
    error_dict : dict
        Dictionary hasil analisis error
    precision : int
        Jumlah digit desimal
    
    Returns:
    --------
    str
        String terformat
    """
    output = []
    output.append("=" * 50)
    output.append("ANALISIS ERROR")
    output.append("=" * 50)
    
    if 'x' in error_dict:
        output.append(f"Titik evaluasi (x): {error_dict['x']:.{precision}f}")
    
    if 'true_value' in error_dict:
        output.append(f"Nilai sebenarnya: {error_dict['true_value']:.{precision}f}")
    
    if 'approximation' in error_dict:
        output.append(f"Nilai aproksimasi: {error_dict['approximation']:.{precision}f}")
    
    if 'absolute_error' in error_dict:
        output.append(f"Error absolut: {error_dict['absolute_error']:.{precision}e}")
    
    if 'relative_error' in error_dict:
        output.append(f"Error relatif: {error_dict['relative_error']:.{precision}e}")
    
    if 'percentage_error' in error_dict:
        output.append(f"Error persentase: {error_dict['percentage_error']:.{precision}f}%")
    
    if 'converged' in error_dict:
        status = "KONVERGEN ✓" if error_dict['converged'] else "BELUM KONVERGEN ✗"
        output.append(f"Status: {status}")
    
    output.append("=" * 50)
    
    return "\n".join(output)


# ========== CONTOH PENGGUNAAN ==========

if __name__ == "__main__":
    # Contoh 1: Analisis fungsi sin(x)
    print("CONTOH 1: Analisis sin(x)")
    print("-" * 50)
    
    na = NumericalAnalysis("sin(x)")
    
    # Fitur 1: Nilai sebenarnya
    x_val = np.pi / 4
    true_val = na.true_value(x_val)
    print(f"Nilai sebenarnya sin(π/4) = {true_val:.8f}")
    
    # Fitur 2: Error absolut/relatif
    approx_val = 0.707
    error_result = na.error_analysis(approx_val, x_val)
    print(f"\nAproksimasi: {approx_val}")
    print(f"Error absolut: {error_result['absolute_error']:.8f}")
    print(f"Error relatif: {error_result['relative_error']:.8f}")
    
    # Fitur 3: Toleransi error
    tol_result = na.check_tolerance(approx_val, true_val, 0.001, 'absolute')
    print(f"\nToleransi 0.001: {'Terpenuhi' if tol_result['converged'] else 'Tidak terpenuhi'}")
    
    # Fitur 4: Polinom Taylor
    taylor_result = na.taylor_polynomial(0, 5)
    print(f"\nPolinom Taylor (5 suku):")
    print(taylor_result['polynomial_str'])
    
    print("\n" + "=" * 50)
    print("CONTOH 2: Konvergensi Taylor untuk exp(x)")
    print("-" * 50)
    
    na2 = NumericalAnalysis("exp(x)")
    convergence = na2.taylor_convergence(0, 10, 1.0)
    
    print(f"{'n':<5} {'Aproksimasi':<15} {'Error Absolut':<15} {'Error Relatif':<15}")
    print("-" * 50)
    for result in convergence:
        print(f"{result['n_terms']:<5} {result['approximation']:<15.8f} "
              f"{result['absolute_error']:<15.2e} {result['relative_error']:<15.2e}")
