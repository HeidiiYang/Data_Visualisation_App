import streamlit as st
import pandas as pd

df=pd.read_csv("Dubawa_label_data.csv")

st.write(len(df))



