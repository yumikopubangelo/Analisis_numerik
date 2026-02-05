"""
Module untuk metode diferensiasi numerik:
1. Selisih Maju (Forward Difference) untuk f'(x)
2. Selisih Mundur (Backward Difference) untuk f'(x)  
3. Selisih Pusat (Central Difference) untuk f'(x) dan f''(x)
"""

import sympy as sp
import numpy as np
from typing import Union, List, Dict, Tuple, Callable
from core.analysis.numerical_features import NumericalAnalysis


class NumericalDifferentiation(NumericalAnalysis):
    """
    Kelas untuk diferensiasi numerik dengan berbagai metode.
    Warisan dari NumericalAnalysis untuk akses fitur analisis error.
    """
    
    def __init__(self, func_str: str):
        """
        Inisialisasi dengan fungsi dalam bentuk string.
        
        Parameters:
        -----------
        func_str : str
            Fungsi dalam bentuk string, contoh: "sin(x)", "x**2 + 2*x", "exp(x)"
        """
        super().__init__(func_str)
        # Hitung turunan pertama dan kedua simbolis untuk verifikasi
        self.first_derivative_symbolic = self.func_symbolic.diff(self.x)
        self.second_derivative_symbolic = self.first_derivative_symbolic.diff(self.x)
        self.first_derivative_numeric = sp.lambdify(self.x, self.first_derivative_symbolic, 'numpy')
        self.second_derivative_numeric = sp.lambdify(self.x, self.second_derivative_symbolic, 'numpy')
    
    # ========== FITUR 1: TINJAUAN TEORETIS TURUNAN ==========
    
    def true_first_derivative(self, x_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Menghitung nilai sebenarnya dari turunan pertama f'(x) pada titik x.
        
        Parameters:
        -----------
        x_val : float atau array
            Nilai x atau array nilai x untuk evaluasi
        
        Returns:
        --------
        float atau array
            Nilai f'(x) yang sebenarnya
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("sin(x)")
        >>> nd.true_first_derivative(np.pi/2)
        0.0
        """
        return self.first_derivative_numeric(x_val)
    
    def true_second_derivative(self, x_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Menghitung nilai sebenarnya dari turunan kedua f''(x) pada titik x.
        
        Parameters:
        -----------
        x_val : float atau array
            Nilai x atau array nilai x untuk evaluasi
        
        Returns:
        --------
        float atau array
            Nilai f''(x) yang sebenarnya
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("sin(x)")
        >>> nd.true_second_derivative(np.pi/2)
        -1.0
        """
        return self.second_derivative_numeric(x_val)
    
    # ========== FITUR 2: SELISIH MAJU (FORWARD DIFFERENCE) ==========
    
    def forward_difference(self, x_val: float, h: float = 1e-5) -> Dict:
        """
        Menghitung aproksimasi turunan pertama f'(x) menggunakan metode selisih maju.
        
        Rumus: f'(x) ≈ [f(x + h) - f(x)] / h
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        
        Returns:
        --------
        dict
            Dictionary berisi hasil diferensiasi dan analisis error
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("x**2")
        >>> result = nd.forward_difference(2, 0.1)
        >>> print(result['approximation'])  # Should be approximately 4.1
        """
        f_x = self.true_value(x_val)
        f_x_h = self.true_value(x_val + h)
        approximation = (f_x_h - f_x) / h
        true_value = self.true_first_derivative(x_val)
        
        return {
            'method': 'Forward Difference',
            'x': x_val,
            'h': h,
            'approximation': approximation,
            'true_value': true_value,
            'absolute_error': self.absolute_error(approximation, true_value),
            'relative_error': self.relative_error(approximation, true_value),
            'percentage_error': self.percentage_error(approximation, true_value),
            'function': self.func_str
        }
    
    # ========== FITUR 3: SELISIH MUNDUR (BACKWARD DIFFERENCE) ==========
    
    def backward_difference(self, x_val: float, h: float = 1e-5) -> Dict:
        """
        Menghitung aproksimasi turunan pertama f'(x) menggunakan metode selisih mundur.
        
        Rumus: f'(x) ≈ [f(x) - f(x - h)] / h
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        
        Returns:
        --------
        dict
            Dictionary berisi hasil diferensiasi dan analisis error
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("x**2")
        >>> result = nd.backward_difference(2, 0.1)
        >>> print(result['approximation'])  # Should be approximately 3.9
        """
        f_x = self.true_value(x_val)
        f_x_h = self.true_value(x_val - h)
        approximation = (f_x - f_x_h) / h
        true_value = self.true_first_derivative(x_val)
        
        return {
            'method': 'Backward Difference',
            'x': x_val,
            'h': h,
            'approximation': approximation,
            'true_value': true_value,
            'absolute_error': self.absolute_error(approximation, true_value),
            'relative_error': self.relative_error(approximation, true_value),
            'percentage_error': self.percentage_error(approximation, true_value),
            'function': self.func_str
        }
    
    # ========== FITUR 4: SELISIH PUSAT (CENTRAL DIFFERENCE) - TURUNAN PERTAMA ==========
    
    def central_difference_first(self, x_val: float, h: float = 1e-5) -> Dict:
        """
        Menghitung aproksimasi turunan pertama f'(x) menggunakan metode selisih pusat.
        
        Rumus: f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        
        Returns:
        --------
        dict
            Dictionary berisi hasil diferensiasi dan analisis error
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("x**2")
        >>> result = nd.central_difference_first(2, 0.1)
        >>> print(result['approximation'])  # Should be exactly 4.0
        """
        f_x_plus_h = self.true_value(x_val + h)
        f_x_minus_h = self.true_value(x_val - h)
        approximation = (f_x_plus_h - f_x_minus_h) / (2 * h)
        true_value = self.true_first_derivative(x_val)
        
        return {
            'method': 'Central Difference (1st Derivative)',
            'x': x_val,
            'h': h,
            'approximation': approximation,
            'true_value': true_value,
            'absolute_error': self.absolute_error(approximation, true_value),
            'relative_error': self.relative_error(approximation, true_value),
            'percentage_error': self.percentage_error(approximation, true_value),
            'function': self.func_str
        }
    
    # ========== FITUR 5: SELISIH PUSAT (CENTRAL DIFFERENCE) - TURUNAN KEDUA ==========
    
    def central_difference_second(self, x_val: float, h: float = 1e-5) -> Dict:
        """
        Menghitung aproksimasi turunan kedua f''(x) menggunakan metode selisih pusat.
        
        Rumus: f''(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h²
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        
        Returns:
        --------
        dict
            Dictionary berisi hasil diferensiasi dan analisis error
        
        Example:
        --------
        >>> nd = NumericalDifferentiation("x**2")
        >>> result = nd.central_difference_second(2, 0.1)
        >>> print(result['approximation'])  # Should be exactly 2.0
        """
        f_x_plus_h = self.true_value(x_val + h)
        f_x = self.true_value(x_val)
        f_x_minus_h = self.true_value(x_val - h)
        approximation = (f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2)
        true_value = self.true_second_derivative(x_val)
        
        return {
            'method': 'Central Difference (2nd Derivative)',
            'x': x_val,
            'h': h,
            'approximation': approximation,
            'true_value': true_value,
            'absolute_error': self.absolute_error(approximation, true_value),
            'relative_error': self.relative_error(approximation, true_value),
            'percentage_error': self.percentage_error(approximation, true_value),
            'function': self.func_str
        }
    
    # ========== FITUR 6: BANDINGKAN SEMUA METODE TURUNAN PERTAMA ==========
    
    def compare_first_derivative_methods(self, x_val: float, h: float = 1e-5) -> List[Dict]:
        """
        Membandingkan semua metode diferensiasi untuk turunan pertama.
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        
        Returns:
        --------
        list of dict
            Daftar hasil dari semua metode
        """
        methods = [
            self.forward_difference(x_val, h),
            self.backward_difference(x_val, h),
            self.central_difference_first(x_val, h)
        ]
        
        return methods
    
    # ========== FITUR 7: ANALISIS KONVERGENSI DENGAN VARIASI h ==========
    
    def convergence_analysis(self, x_val: float, h_values: List[float], 
                           derivative_order: int = 1) -> List[Dict]:
        """
        Analisis konvergensi metode diferensiasi dengan berbagai nilai h.
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h_values : list
            Daftar nilai h untuk diuji
        derivative_order : int, optional
            Orde turunan (1 atau 2), default: 1
        
        Returns:
        --------
        list of dict
            Hasil analisis konvergensi untuk setiap h
        """
        results = []
        
        if derivative_order == 1:
            true_value = self.true_first_derivative(x_val)
            for h in h_values:
                forward_result = self.forward_difference(x_val, h)
                backward_result = self.backward_difference(x_val, h)
                central_result = self.central_difference_first(x_val, h)
                
                results.append({
                    'h': h,
                    'forward_approx': forward_result['approximation'],
                    'forward_error': forward_result['absolute_error'],
                    'backward_approx': backward_result['approximation'],
                    'backward_error': backward_result['absolute_error'],
                    'central_approx': central_result['approximation'],
                    'central_error': central_result['absolute_error'],
                    'true_value': true_value
                })
        else:
            true_value = self.true_second_derivative(x_val)
            for h in h_values:
                central_result = self.central_difference_second(x_val, h)
                
                results.append({
                    'h': h,
                    'central_approx': central_result['approximation'],
                    'central_error': central_result['absolute_error'],
                    'true_value': true_value
                })
        
        return results
    
    # ========== FITUR 8: METODE DENGAN OPTIMASI h ==========
    
    def optimal_h_analysis(self, x_val: float, h_min: float = 1e-8, 
                          h_max: float = 1e-1, num_points: int = 20) -> Dict:
        """
        Analisis untuk menemukan nilai h optimal yang memberikan error minimal.
        
        Parameters:
        -----------
        x_val : float
            Titik x yang akan dievaluasi
        h_min : float, optional
            Nilai h minimum, default: 1e-8
        h_max : float, optional
            Nilai h maksimum, default: 1e-1
        num_points : int, optional
            Jumlah titik h untuk diuji, default: 20
        
        Returns:
        --------
        dict
            Hasil analisis optimal h untuk semua metode
        """
        h_values = np.logspace(np.log10(h_min), np.log10(h_max), num_points)
        results = []
        
        for h in h_values:
            forward_result = self.forward_difference(x_val, h)
            backward_result = self.backward_difference(x_val, h)
            central1_result = self.central_difference_first(x_val, h)
            central2_result = self.central_difference_second(x_val, h)
            
            results.append({
                'h': h,
                'forward_error': forward_result['absolute_error'],
                'backward_error': backward_result['absolute_error'],
                'central1_error': central1_result['absolute_error'],
                'central2_error': central2_result['absolute_error']
            })
        
        # Temukan h optimal untuk masing-masing metode
        optimal = {
            'forward': min(results, key=lambda x: x['forward_error']),
            'backward': min(results, key=lambda x: x['backward_error']),
            'central1': min(results, key=lambda x: x['central1_error']),
            'central2': min(results, key=lambda x: x['central2_error'])
        }
        
        return {
            'h_values': h_values,
            'results': results,
            'optimal': optimal,
            'x': x_val
        }
    
    # ========== FITUR 9: DIFERENSIASI PADA DAFTAR TITIK ==========
    
    def differentiate_at_points(self, x_points: List[float], h: float = 1e-5,
                               derivative_order: int = 1, method: str = 'central') -> List[Dict]:
        """
        Melakukan diferensiasi pada beberapa titik x sekaligus.
        
        Parameters:
        -----------
        x_points : list
            Daftar titik x untuk dievaluasi
        h : float, optional
            Langkah (step size), default: 1e-5
        derivative_order : int, optional
            Orde turunan (1 atau 2), default: 1
        method : str, optional
            Metode yang digunakan: 'forward', 'backward', 'central', default: 'central'
        
        Returns:
        --------
        list of dict
            Hasil diferensiasi untuk setiap titik x
        """
        results = []
        
        for x in x_points:
            if derivative_order == 1:
                if method == 'forward':
                    result = self.forward_difference(x, h)
                elif method == 'backward':
                    result = self.backward_difference(x, h)
                else:
                    result = self.central_difference_first(x, h)
            else:
                result = self.central_difference_second(x, h)
            
            results.append(result)
        
        return results


# ========== FUNGSI UTILITAS ==========

def forward_difference(func: Callable, x_val: float, h: float = 1e-5) -> float:
    """
    Fungsi utilitas untuk selisih maju (forward difference).
    
    Parameters:
    -----------
    func : callable
        Fungsi yang akan didiferensiasikan
    x_val : float
        Titik x yang akan dievaluasi
    h : float, optional
        Langkah (step size), default: 1e-5
    
    Returns:
    --------
    float
        Aproksimasi turunan pertama
    """
    return (func(x_val + h) - func(x_val)) / h


def backward_difference(func: Callable, x_val: float, h: float = 1e-5) -> float:
    """
    Fungsi utilitas untuk selisih mundur (backward difference).
    
    Parameters:
    -----------
    func : callable
        Fungsi yang akan didiferensiasikan
    x_val : float
        Titik x yang akan dievaluasi
    h : float, optional
        Langkah (step size), default: 1e-5
    
    Returns:
    --------
    float
        Aproksimasi turunan pertama
    """
    return (func(x_val) - func(x_val - h)) / h


def central_difference_first(func: Callable, x_val: float, h: float = 1e-5) -> float:
    """
    Fungsi utilitas untuk selisih pusat (central difference) turunan pertama.
    
    Parameters:
    -----------
    func : callable
        Fungsi yang akan didiferensiasikan
    x_val : float
        Titik x yang akan dievaluasi
    h : float, optional
        Langkah (step size), default: 1e-5
    
    Returns:
    --------
    float
        Aproksimasi turunan pertama
    """
    return (func(x_val + h) - func(x_val - h)) / (2 * h)


def central_difference_second(func: Callable, x_val: float, h: float = 1e-5) -> float:
    """
    Fungsi utilitas untuk selisih pusat (central difference) turunan kedua.
    
    Parameters:
    -----------
    func : callable
        Fungsi yang akan didiferensiasikan
    x_val : float
        Titik x yang akan dievaluasi
    h : float, optional
        Langkah (step size), default: 1e-5
    
    Returns:
    --------
    float
        Aproksimasi turunan kedua
    """
    return (func(x_val + h) - 2 * func(x_val) + func(x_val - h)) / (h ** 2)


def create_differentiation_table(results: List[Dict], precision: int = 8) -> List[Dict]:
    """
    Membuat tabel perbandingan hasil diferensiasi dari berbagai metode.
    
    Parameters:
    -----------
    results : list of dict
        Hasil diferensiasi dari berbagai metode
    precision : int, optional
        Jumlah digit desimal, default: 8
    
    Returns:
    --------
    list of dict
        Tabel hasil diferensiasi yang diformat
    """
    table = []
    
    for result in results:
        table.append({
            'Method': result['method'],
            'x': result['x'],
            'h': result['h'],
            'Approximation': round(result['approximation'], precision),
            'True Value': round(result['true_value'], precision),
            'Absolute Error': round(result['absolute_error'], precision),
            'Relative Error': round(result['relative_error'], precision),
            'Percentage Error': round(result['percentage_error'], precision)
        })
    
    return table


def format_differentiation_output(result: Dict, precision: int = 8) -> str:
    """
    Format output hasil diferensiasi dalam bentuk string yang mudah dibaca.
    
    Parameters:
    -----------
    result : dict
        Hasil diferensiasi dari satu metode
    precision : int, optional
        Jumlah digit desimal, default: 8
    
    Returns:
    --------
    str
        String terformat
    """
    output = []
    output.append("=" * 70)
    output.append(f"METODE: {result['method']}")
    output.append("=" * 70)
    
    output.append(f"Titik evaluasi (x): {result['x']:.{precision}f}")
    output.append(f"Langkah (h): {result['h']:.{precision}e}")
    output.append(f"Nilai aproksimasi: {result['approximation']:.{precision}f}")
    output.append(f"Nilai sebenarnya: {result['true_value']:.{precision}f}")
    output.append(f"Error absolut: {result['absolute_error']:.{precision}e}")
    output.append(f"Error relatif: {result['relative_error']:.{precision}e}")
    output.append(f"Error persentase: {result['percentage_error']:.{precision}f}%")
    
    output.append("=" * 70)
    
    return "\n".join(output)


# ========== CONTOH PENGGUNAAN ==========

if __name__ == "__main__":
    # Contoh 1: Analisis fungsi sin(x) pada x = π/4
    print("CONTOH 1: Diferensiasi fungsi sin(x) pada x = π/4")
    print("-" * 70)
    
    nd = NumericalDifferentiation("sin(x)")
    x_val = np.pi / 4
    h = 0.01
    
    # Metode selisih maju
    forward_result = nd.forward_difference(x_val, h)
    print(format_differentiation_output(forward_result))
    print()
    
    # Metode selisih mundur
    backward_result = nd.backward_difference(x_val, h)
    print(format_differentiation_output(backward_result))
    print()
    
    # Metode selisih pusat turunan pertama
    central1_result = nd.central_difference_first(x_val, h)
    print(format_differentiation_output(central1_result))
    print()
    
    # Metode selisih pusat turunan kedua
    central2_result = nd.central_difference_second(x_val, h)
    print(format_differentiation_output(central2_result))
    
    print("\n" + "=" * 70)
    print("CONTOH 2: Perbandingan semua metode turunan pertama")
    print("-" * 70)
    
    comparison = nd.compare_first_derivative_methods(x_val, h)
    table = create_differentiation_table(comparison)
    
    print(f"{'Metode':<40} {'Aproksimasi':<15} {'Error Absolut':<15} {'Error Persentase':<15}")
    print("-" * 70)
    for row in table:
        print(f"{row['Method']:<40} {row['Approximation']:<15.8f} "
              f"{row['Absolute Error']:<15.2e} {row['Percentage Error']:<15.2f}%")
