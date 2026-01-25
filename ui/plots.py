import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def plot_function(func, a, b, root=None):
    x = np.linspace(a-1, b+1, 1000)
    y = func(x)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.5)
    if root is not None:
        ax.axvline(root, color='red', linestyle='--', label=f'Root ≈ {root:.6f}')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Plot Fungsi')
    
    st.pyplot(fig)

def plot_convergence(iterations):
    if not iterations:
        return
    
    iters = [it['iteration'] for it in iterations]
    errors = [it['abs_error'] for it in iterations if it['abs_error'] is not None]
    
    if errors:
        fig, ax = plt.subplots()
        ax.plot(iters[:len(errors)], errors, marker='o')
        ax.set_xlabel('Iterasi')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Konvergensi Error')
        ax.set_yscale('log')
        st.pyplot(fig)