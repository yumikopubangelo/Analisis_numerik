import streamlit as st
import pandas as pd

def output_table(iterations):
    if not iterations:
        st.warning("Tidak ada data iterasi")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(iterations)
    st.subheader("Tabel Iterasi")
    st.dataframe(df)