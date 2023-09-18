import streamlit as st
import pandas as pd
!pip install plotly.express
import plotly.express as px

df_pos_tag=pd.read_csv("data/pos_tag.csv")

st.write(df_pos_tag)
st.write("Hello World")



