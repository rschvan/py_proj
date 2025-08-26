# sttest.py
import streamlit as st
import pandas as pd

st.title("Dataframe Selection Test")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})

event = st.dataframe(
    st.session_state.df,
    use_container_width=True,
    key = "data",
    on_select = "rerun",
    selection_mode=["multi-row"],
)

if event is not None:
    st.write("Selected rows:", event.selection.rows)
    st.write("Selected columns:", event.selection.columns)
