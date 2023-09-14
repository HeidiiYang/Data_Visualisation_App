import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

st.write("DB username:", st.secrets["db_username"])
st.write("DB password:", st.secrets["db_password"])

url='https://drive.google.com/file/d/1ZIxWQ_u5_PkHvPFLMOJjgPRm4IWSB4v_/view?usp=drive_link'
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df_news=pd.read_csv(path)
df_news=df_news.dropna(axis=0)

st.write(df)
