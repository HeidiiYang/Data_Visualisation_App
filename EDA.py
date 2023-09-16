import streamlit as st
import pandas as pd

df=pd.read_csv("data/Dubawa_label_data.csv")

st.write(df)



