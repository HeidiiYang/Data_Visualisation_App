import streamlit as st
import pandas as pd
from pathlib import Path 

st.title("My App")
st.write("My first app")
url = 'https://drive.google.com/file/d/1_cjfajizKAco3t8Sn99xODjcxcbYMshk/view?usp=drive_link'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

p=Path.cwd()
st.write(p)

st.write(df)



