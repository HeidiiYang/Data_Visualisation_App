import streamlit as st
import pandas as pd
import plotly.express as px

df_pos_tag=pd.read_csv("data/Dubawa_label_data.csv")

st.write(df_pos_tag)
st.write("Hello World")



