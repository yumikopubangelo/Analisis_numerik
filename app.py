import streamlit as st

from ui.sidebar import sidebar
from ui.input_form import input_form
from ui.output_table import output_table
from ui.plots import plot_function, plot_convergence
from ui.explanation import explanation

from core.root_findings.bisection import bisection_method

st.set_page_config(page_title="Numerical Analysis App", layout="wide")

def main():
    st.title("Aplikasi Pembelajaran Analisis Numerik")
    st.markdown("Selamat datang! Pilih metode dan masukkan parameter untuk melihat proses komputasi langkah demi langkah.")
    
    category, method = sidebar()
    
    params = input_form(category, method)
    
    if params:
        if category == "Root Finding" and method == "Bisection":
            try:
                root, iterations = bisection_method(
                    params['func'], params['a'], params['b'], 
                    params['tol'], params['max_iter']
                )
                
                st.success(f"Akar ditemukan: {root:.6f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    output_table(iterations)
                with col2:
                    plot_function(params['func'], params['a'], params['b'], root)
                    plot_convergence(iterations)
                
                explanation(method)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Metode lain belum diimplementasi.")

if __name__ == "__main__":
    main()