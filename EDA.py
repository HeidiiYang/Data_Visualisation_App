import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

url='https://drive.google.com/file/d/19U3J_1CalIwvIccqtmNUfnfGfqEQ8EMj/view?usp=drive_link'
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df=pd.read_csv(path)

st.write(df)
  

