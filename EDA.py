import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

st.write(pd.DataFrame({'name': ['Luoie', 'Matt', 'Peter', 'Luck'], 'scores': [40, 55, 67, 73]}))
