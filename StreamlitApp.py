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
     page_title='Africa-Fake-News-Detection',
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
fig7.update_layout(xaxis={'categoryorder':'total ascending'}, width=700, height=500)
fig7.show()

if selected=='Project Description':
     st.subheader("Project Background")
     st.write("Fake news leading to misinformation and disrupting confidence in media industry, poses a potential threat to society as a whole. Looking back to 2019, the detriment to various social realms, including health interventions and electoral integrity, cause by fake news, rumours, and misinformation had captured the nation’s attention in Liberia Consequently, actions and strategies were taken to combat the escalating spread of fake news. For example, a high-level regional conference discussing the tools and methodologies to monitor the online space and tackle misinformation and hate speech was held in Liberia in July 2023 joined by other West Africa countries. ([Click here for more information about the conference.](https://chriswizo.substack.com/p/fighting-fake-news-together-liberia))")
     st.write("There is a pressing need to leverage cutting-edge technologies to combat the dissemination of harmful and misleading information. Artificial Intelligence (AI) emerges as a powerful tool for detecting fake news in the digital media landscape. This project aims to apply Natural Language Processing (NLP) to analyse the differences between fake and authentic news in a textual setting enabling the detection of potential fake news.")
     st.write("Additionally, given the detrimental impacts of misinformation on the 2020’s senatorial election and the upcoming national election in October 2023, it is crucial to develop a user-friendly fake news detection tool that will be able to assist in the manually fact-check previously conducted by individuals and promotion of electoral integrity.")
     st.markdown("""<hr style="height:2px;border-radius: 4px 4px 4px 4px;border:none;color:#ff5959;background-color:#676fa3;" /> """, unsafe_allow_html=True)
     st.subheader("Methods and Techniques")
     st.markdown(
          """
          - **Dataset Collection**
               - **Web scrapping:** collected news from various news websites including fact-check websites to construct the news dataset. 
               - **Data integration and data cleansing:** merged the data collected from different sources and maintained data consistency.
          - **Data Analysis**
               - **Exploratory data analysis:** had a comprehensive understanding of the whole dataset and presented initial insights through graphical representations.
          - **Model Development**
               - **Machine Learning Classification:** extracted features to represent news content and selecting and designing models that can distinguish fake news from authentic news while emphasising robustness, ensuring reliable performance across various conditions and datasets.
          """
     )
elif selected=='Exploratory Data Analysis':
     st.subheader("Exploratory Data Analysis")
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
     st.write("Apparently, terms associated with health (such as Health and CDC (Africa Centres for Disease Control and Prevention)) occurred more frequently in the fake news dataset than in the real news dataset, which lends support to the assumption of a large volume of fake news related to the covid-19 above. ")
elif selected=='Modelling':
     with open("style.css") as source_style:
          st.markdown(f"<style>{source_style.read()}</style>", unsafe_allow_html=True)
     tab1, tab2, tab3=st.tabs(["Feature Extraction", "Data Balancing", "Model Development and Evaluation"])
     with tab1:
          st.subheader("Feature Extraction")
          st.write("TF-IDF was employed to represent news text. However, to prevent the matrix from becoming excessively large, words that appeared in less than 10% of the records were excluded. Consequently, 236 features were extracted.")
     with tab2:
          st.subheader("Data Balancing")
          st.write(fig2)
          st.write("The final dataset is imbalanced with 26003 records of real news and 5404 records of fake news resulting in an approximate ratio of 5:1, which would skew the classification model to favour the real news. To deal with this issue, there are mainly two ways: oversampling the minority class and undersampling the majority class. However, a method combing these two ways has been approved to have a better performance in generating a synthetic balanced dataset—[SMOTE](https://www.jair.org/index.php/jair/article/view/10302).")
          st.write("After the resampling the original news dataset, there are 36472 records including 18236 real news records and 18236 fake news records. Then, the resampled data was trained by classification models. 9423 records being split from the original dataset were used for model evaluation.")
     with tab3:
          st.subheader("Model Development and Evaluation")
          st.write("Logistic Regression, Navie Bayes, Random Forest and Gradient Boosting classifiers were employed respectively. A table below shows the accuracy score for each model. ")
          df_model_evaluation=pd.read_csv("data/model_evaluation.csv")
          st.table(df_model_evaluation)
          st.write("Based on model performance, Logistic Regression was finally chosen for predicting fake news in this project.")
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

     





