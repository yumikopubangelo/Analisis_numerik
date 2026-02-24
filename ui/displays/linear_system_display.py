"""
Display module for linear-system methods (Ax = b).
"""

import numpy as np
import pandas as pd
import streamlit as st


def _augmented_to_dataframe(augmented_matrix):
    n = augmented_matrix.shape[1] - 1
    columns = [f"x{i + 1}" for i in range(n)] + ["b"]
    return pd.DataFrame(augmented_matrix, columns=columns)


def _solution_to_dataframe(solution):
    return pd.DataFrame(
        {
            "Variabel": [f"x{i + 1}" for i in range(len(solution))],
            "Nilai": solution,
        }
    )


def display_linear_system_results(result, params, method):
    """
    Display results for Gaussian Elimination / Gauss-Jordan.
    """
    solution = np.array(result["solution"], dtype=float)
    residual = np.array(result["residual"], dtype=float)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))

    st.success("Sistem persamaan linear berhasil diselesaikan.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Metode", method)
    with col2:
        st.metric("Jumlah Variabel", len(solution))
    with col3:
        st.metric("||Ax - b||_inf", f"{residual_norm:.2e}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Solusi")
        st.dataframe(_solution_to_dataframe(solution), use_container_width=True)

    with col2:
        st.markdown("#### Verifikasi")
        Ax = params["A"] @ solution
        verification_df = pd.DataFrame(
            {
                "Ax": Ax,
                "b": params["b"],
                "Residual": residual,
            }
        )
        st.dataframe(verification_df, use_container_width=True)

    st.markdown("#### Matriks Augmented Akhir")
    st.dataframe(_augmented_to_dataframe(result["final_augmented"]), use_container_width=True)

    back_sub = result.get("back_substitution", [])
    if back_sub:
        with st.expander("Detail Substitusi Balik", expanded=False):
            st.dataframe(pd.DataFrame(back_sub), use_container_width=True)

    steps = result.get("steps", [])
    if params.get("show_steps", True) and steps:
        with st.expander("Langkah Eliminasi", expanded=False):
            safe_method = method.lower().replace(" ", "_").replace("-", "_")
            step_state_key = f"linear_step_state_{safe_method}_{len(steps)}"
            slider_key = f"linear_step_slider_{safe_method}_{len(steps)}"
            prev_key = f"linear_step_prev_{safe_method}_{len(steps)}"
            next_key = f"linear_step_next_{safe_method}_{len(steps)}"

            if step_state_key not in st.session_state:
                st.session_state[step_state_key] = 1

            current_step = int(st.session_state[step_state_key])
            current_step = max(1, min(len(steps), current_step))
            st.session_state[step_state_key] = current_step

            col_prev, col_next, col_info = st.columns([1, 1, 2])
            with col_prev:
                if st.button("Prev", key=prev_key, disabled=current_step <= 1):
                    st.session_state[step_state_key] = current_step - 1
            with col_next:
                if st.button("Next", key=next_key, disabled=current_step >= len(steps)):
                    st.session_state[step_state_key] = current_step + 1
            with col_info:
                st.caption(f"Langkah {st.session_state[step_state_key]} dari {len(steps)}")

            selected_step_number = st.slider(
                "Pilih langkah ke-",
                min_value=1,
                max_value=len(steps),
                value=st.session_state[step_state_key],
                key=slider_key,
            )
            st.session_state[step_state_key] = selected_step_number
            selected_step = steps[selected_step_number - 1]
            st.caption(f"Tipe operasi: {selected_step['type']}")
            st.markdown(f"**Operasi:** {selected_step['description']}")
            st.dataframe(
                _augmented_to_dataframe(selected_step["matrix"]),
                use_container_width=True
            )
