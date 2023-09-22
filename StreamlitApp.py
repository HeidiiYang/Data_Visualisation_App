import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px

# Initial page config
st.set_page_config(
     page_title='XXXPageTitle',
     layout="wide",
)

#Set menu on the side 
with st.sidebar:
    selected=option_menu("Fake News Detection", ["Project Description", "Exploratory Data Analysis", "Modelling", "News Detection Tool"], 
                         menu_icon="newspaper", 
                         default_index=1, styles={
                              "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                              "nav-link-selected": {"background-color": "#952e4b"},
                         })
    selected

#Visualise the news count by year
df_news_calculation=pd.read_csv("data/full_dataset_calculation.csv")
fig1=px.histogram(df_news_calculation, x='year', title='News Count by Year', color='Label', color_discrete_sequence=["#84B9EF", "#FF7171"])
fig1.update_xaxes(title='Publication Year').update_yaxes(title='News Count')
fig1.update_layout(width=700, height=400, bargap=0.03)
fig1.show()

df_pos_tag=pd.read_csv("data/pos_tag.csv")
fig=px.histogram(df_pos_tag, x='pos_tag', y='frequency_ratio', title='POS Tagging Frequency in Fake and Real News', barmode='group', color='news_category', color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_xaxes(title='POS Tagging').update_yaxes(title='Frequency')
fig.update_layout(xaxis={'categoryorder':'total descending'}, width=1400, height=500)
fig.show()

if selected=='Project Description':
     st.write("Fake news detection.")
elif selected=='Exploratory Data Analysis':
     st.write(fig1)
elif selected=='Modelling':
     tab1, tab2=st.tabs(["Model Balancing", "Model Evaluation"])
elif selected=="News Detection Tool":
     st.header("News Detection: Fake or Real?")
     user_input=st.text_area("","Please pate news content here.")
     





