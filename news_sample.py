import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Reading the dataset
df_news=pd.read_csv('data/news_sample.csv')

#Seperating the target and features
#target->y, features->X
y=data['Label']
X=data.drop(columns='Label', axis=1)

#Splitting into training and test data set
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)

#Fitting the model
logreg=LogisticRegression()
logreg.fit(X_train, y_train)

#Making prediction
y_pred=logreg.predict(X_test)

#Saving the model
import joblib

joblib.dump(logreg, "logreg_model.sav")
