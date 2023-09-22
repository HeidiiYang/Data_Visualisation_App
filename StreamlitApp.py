import streamlit as st
import pandas as pd
import plotly.express as px

#Visualise the news count by year
df_news_calculation=pd.read_csv("data/full_dataset_calculation.csv")
'''fig1=px.histogram(df_news_calculation, x='year', title='News Count by Year', color='Label', color_discrete_sequence=px.colors.qualitative.Vivid)
fig1.update_xaxes(title='Publication Year').update_yaxes(title='News Count')
fig1.update_layout(width=700, height=400, bargap=0.03)
fig1.show()

df_pos_tag=pd.read_csv("data/pos_tag.csv")
fig=px.histogram(df_pos_tag, x='pos_tag', y='frequency_ratio', title='POS Tagging Frequency in Fake and Real News', barmode='group', color='news_category', color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_xaxes(title='POS Tagging').update_yaxes(title='Frequency')
fig.update_layout(xaxis={'categoryorder':'total descending'}, width=1400, height=500)
fig.show()'''

st.write(df_news_calculation)




