import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

df=pd.read_csv("Dubawa_label_data.csv")

st.write(df)
