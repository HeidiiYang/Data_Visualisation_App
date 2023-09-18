import streamlit as st
import pandas as pd

df_pos_tag=pd.read_csv("data/pos_tag.csv")

st.write(df_pos_tag)
st.write("Hello World")



