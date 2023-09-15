import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.title("My App")
st.write("My first app")

conn=st.experimental_connection("gsheets", type=GSheetsConnection)
data=conn.read(worksheet="Dubawa_label_data")
df=st.dataframe(data)

st.write(df)
