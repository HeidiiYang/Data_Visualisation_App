import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import regex as re
import plotly.express as px
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initial page config
st.set_page_config(
     page_title='XXXPageTitle',
     layout="wide",
)

def predict(data):
     logreg=joblib.load('logreg_model.sav')
     proba_result=logreg.predict_proba(data)
     return proba_result

def text_processing(text):
     #remove numbers/digits
     text=re.sub(r'\d+', '', str(text))
     #tokenise
     p=string.punctuation+'’”'
     text=''.join(ch for ch in text if ch not in p)
     text=word_tokenize(text)
     text=[w for w in text if w.lower() not in stopwords.words('english')]
     #converse to lower case
     new_words=[]
     for word in text:
          word=word.lower()
          new_words.append(word)
     #lemmatise
     lemmatizer=WordNetLemmatizer()
     s_lemmatised=[lemmatizer.lemmatize(word) for word in new_words]
     lemmatised_text=' '.join(s_lemmatised)
     return lemmatised_text

def tf_idf(text):
     vectorizer=TfidfVectorizer(stop_words='english')
     response=vectorizer.fit_transform([text, ''])
     tfidf_matrix=pd.DataFrame(response.toarray(),columns=vectorizer.get_feature_names_out())
     text_tfidf=tfidf_matrix.loc[0]
     text_tfidf=text_tfidf.to_frame()
     text_tfidf=text_tfidf.transpose()
     return text_tfidf

def feature_matching(df_text_feature):
     lr_model=joblib.load('logreg_model.sav')
     df_t_fit=pd.DataFrame(columns=lr_model.feature_names)
     for c in df_t_fit.columns:
          if c in df_text_feature.columns:
               df_t_fit[c]=df_text_feature[c]
     df_t_fit=df_t_fit.fillna(0.0)
     return df_t_fit     

#Set menu on the side 
with st.sidebar:
    selected=option_menu("Fake News Detection", ["Project Description", "Exploratory Data Analysis", "Modelling", "News Detection Tool"], 
                         menu_icon="newspaper", 
                         default_index=1, styles={
                              "menue_icon": {"color": "#949cdf"},
                              "icon": {"color": "#949cdf", "font-size": "18px"}, 
                              "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                              "nav-link-selected": {"background-color": "#2b3467"},
                         })
    selected
#Customise the side menu
with open("style.css") as source_style:
          st.markdown(f"<style>{source_style.read()}</style>", unsafe_allow_html=True)
     
#Reading the calculation dataset
df_news_calculation=pd.read_csv("data/dataset_calculation.csv")

#Visualise the news count by year
fig1=px.histogram(df_news_calculation, x='year', title='News Count by Year', color='Label', color_discrete_sequence=["#949CDF", "#EB455F"])
fig1.update_xaxes(title='Publication Year').update_yaxes(title='News Count')
fig1.update_layout(width=700, height=400, bargap=0.03)
fig1.show()

#Visualise fake and real news distribution
fig2=px.histogram(df_news_calculation, x='Label', title='Distribution of fake and factual news', color='Label', color_discrete_sequence=["#949CDF", "#EB455F"])
fig2.update_xaxes(type='category', title='News Label').update_yaxes(title='News Count')
fig2.update_layout(width=700, height=400, bargap=0.03)
fig2.show()

#Visualise content/character lenght count
fig3=px.histogram(df_news_calculation, x='content_length', title='News Text Length Count', barmode='overlay', color='Label', color_discrete_sequence=["#949CDF", "#EB455F"])
fig3.update_xaxes(title='Text length').update_yaxes(title='News Count')
fig3.update_layout(width=700, height=300)
fig3.show()

#Visualise the number of weords
fig4=px.histogram(df_news_calculation, x='word_count', title='Word Count', barmode='overlay', color='Label', color_discrete_sequence=["#949CDF", "#EB455F"])
fig4.update_xaxes(title='The number of words').update_yaxes(title='News Count')
fig4.update_layout(width=700, height=300)
fig4.show()

#Visualise average word length 
fig5=px.histogram(df_news_calculation, x='aver_word_length', title='Average Word Length Count', barmode='overlay', color='Label', color_discrete_sequence=["#949CDF", "#EB455F"])
fig5.update_xaxes(title='Average Word length').update_yaxes(title='News Count')
fig5.update_layout(width=700, height=300)
fig5.show()

#Visualise the pos tag count
df_pos_tag=pd.read_csv("data/pos_tag.csv")
fig6=px.histogram(df_pos_tag[df_pos_tag["frequency_ratio"]>1.3], x='pos_tag', y='frequency_ratio', title='Part-of-Speech Tagging Frequency in Fake and Real News', barmode='group', color='news_category', color_discrete_sequence=["#949CDF", "#EB455F"])
fig6.update_xaxes(title='POS Tagging').update_yaxes(title='Frequency')
fig6.update_layout(xaxis={'categoryorder':'total descending'}, width=700, height=400)
fig6.show()

#Visualise the top 20 proper nouns
df_nnp=pd.read_csv("data/nnp_20.csv")
fig7=px.histogram(df_nnp, x='NNP', y='frequency_ratio', title='Top 20 Proper Nouns', barmode='group', color='news_category', color_discrete_sequence=["#949CDF", "#EB455F"])
fig7.update_xaxes(title='Part of speech').update_yaxes(title='Frequency')
fig7.update_layout(xaxis={'categoryorder':'total ascending'}, width=700, height=800)
fig7.show()

if selected=='Project Description':
     st.write("Fake news detection.")
elif selected=='Exploratory Data Analysis':
     st.write(fig1)
     st.write("Fake news accumulates in 2020, suggesting the potential existence of a significant amount of misinformation concerning COVID-19.")
     st.write(fig3)
     st.write(fig4)
     st.write(fig5)
     st.write("Compared with real news, fake news tends to use simpler words to make their content longer.")
     st.write(fig6)
     st.write("In terms of the part-of-speech differences, fake news contains more nouns, verbs, past participle verbs, and adverbs. In contrast, real news uses more adjectives, proper nouns, gerund verbs and past tense verbs.")
     st.write("The difference lying in the use of proper nouns between fake and real news is noticeable. To illustrate the contrast, the top 20 proper nouns in the two news groups were selected and visualised.")
     st.write(fig7)
     st.write("Apparently, terms associated with health (such as ) occurred more frequently in the fake news dataset than in the real news dataset, which lends support to the assumption of a large volume of fake news related to the covid-19 above. ")
elif selected=='Modelling':
     with open("style.css") as source_style:
          st.markdown(f"<style>{source_style.read()}</style>", unsafe_allow_html=True)
     tab1, tab2, tab3=st.tabs(["Data Balancing", "Model Development", "Model Evaluation"])
     with tab1:
          st.write(fig2)
elif selected=="News Detection Tool":
     with open("style.css") as source_style:
          st.markdown(f"<style>{source_style.read()}</style>", unsafe_allow_html=True)
     st.header("News Detection: Fake or Real?")
     user_input=st.text_area("","Please paste news content here.")
     if st.button("predict"):
          t=text_processing(user_input)
          t=tf_idf(t)
          t=feature_matching(t)
          news_category=predict(t)
          st.text("The probability of being fake is {}".format(round(news_category[0][0], 2)))
          st.text("The probability of being real is {}".format(round(news_category[0][1], 2)))
          #news_category=predict(np.array[[area, bedrooms]])
          #st.text(t)

     





