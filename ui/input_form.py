import streamlit as st
from core.utils.parser import parse_function

def input_form(category, method):
    st.header(f"{category} - {method}")
    
    if category == "Root Finding":
        return input_root_finding()
    elif category == "Integration":
        return input_integration()
    elif category == "Interpolation":
        return input_interpolation()
    else:
        st.error("Kategori tidak dikenal")

def input_root_finding():
    st.subheader("Masukkan Parameter")
    
    func_str = st.text_input("Fungsi f(x) (e.g., x**2 - 2)", "x**2 - 2")
    a = st.number_input("Batas bawah a", value=0.0)
    b = st.number_input("Batas atas b", value=2.0)
    tol = st.number_input("Toleransi", value=1e-6, format="%.2e")
    max_iter = st.number_input("Maksimum iterasi", value=100, min_value=1)
    
    if st.button("Hitung"):
        try:
            func = parse_function(func_str)
            return {
                'func': func,
                'func_str': func_str,
                'a': a,
                'b': b,
                'tol': tol,
                'max_iter': max_iter
            }
        except ValueError as e:
            st.error(str(e))
            return None
    return None

def input_integration():
    # Placeholder
    st.subheader("Integration input (belum diimplementasi)")
    return None

def input_interpolation():
    # Placeholder
    st.subheader("Interpolation input (belum diimplementasi)")
    return None