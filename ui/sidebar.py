import streamlit as st

def sidebar():
    st.sidebar.title("Numerical Analysis App")
    st.sidebar.markdown("---")
    
    # Select category
    category = st.sidebar.selectbox(
        "Pilih Kategori",
        ["Root Finding", "Integration", "Interpolation"]
    )
    
    # Select method based on category
    if category == "Root Finding":
        method = st.sidebar.selectbox(
            "Pilih Metode",
            ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"]
        )
    elif category == "Integration":
        method = st.sidebar.selectbox(
            "Pilih Metode",
            ["Trapezoidal", "Simpson"]
        )
    else:
        method = st.sidebar.selectbox(
            "Pilih Metode",
            ["Lagrange", "Newton"]
        )
    
    return category, method