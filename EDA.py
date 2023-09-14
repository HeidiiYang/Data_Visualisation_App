import streamlit as st
import pandas as pd

st.title("My App")
st.write("My first app")

data=pd.read_csv("F:\Career\A-internship-remote projects\Fake news\data\updated_full_dataset.csv") 
st.write(data)
  

