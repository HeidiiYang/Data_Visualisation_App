import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

url='https://drive.google.com/file/d/1ZIxWQ_u5_PkHvPFLMOJjgPRm4IWSB4v_/view?usp=drive_link'
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df=pd.read_csv(path)

st.write(df)
  

