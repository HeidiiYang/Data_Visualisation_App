import streamlit as st
import pandas as pd
from pathlib import Path 

streamlit config show

st.title("My App")
st.write("My first app")


p=Path.cwd()
st.write(p)



