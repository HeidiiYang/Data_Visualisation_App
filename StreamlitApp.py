import streamlit as st
import pandas as pd
import plotly.express as px

df_pos_tag=pd.read_csv("data/pos_tag.csv")
fig=px.histogram(df_pos_tag, x='pos_tag', y='frequency_ratio', title='POS Tagging Frequency in Fake and Real News', barmode='group', color='news_category', color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_xaxes(title='POS Tagging').update_yaxes(title='Frequency')
fig.update_layout(xaxis={'categoryorder':'total descending'}, width=1400, height=500)
fig.show()

st.write(fig)
st.write("Hello World")



