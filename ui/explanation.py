import streamlit as st

def explanation(method):
    st.subheader("Penjelasan Metode")
    
    if method == "Bisection":
        st.markdown("""
        **Metode Biseksi** adalah metode untuk mencari akar persamaan f(x) = 0 dalam interval [a, b] di mana f(a) dan f(b) memiliki tanda berlawanan.
        
        **Langkah-langkah:**
        1. Hitung titik tengah c = (a + b)/2
        2. Jika f(c) = 0, maka c adalah akar
        3. Jika f(a) * f(c) < 0, maka akar di [a, c], set b = c
        4. Jika f(b) * f(c) < 0, maka akar di [c, b], set a = c
        5. Ulangi sampai konvergen
        
        **Keuntungan:** Selalu konvergen jika kondisi awal terpenuhi.
        **Kekurangan:** Konvergensi lambat.
        """)
    else:
        st.markdown("Penjelasan untuk metode lain belum tersedia.")