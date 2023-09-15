import streamlit as st
import pandas as pd

# Code to read csv file into Colaboratory:
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

drive_link=input("Enter link to the .csv: ")

# This link can be replaced with any other Google Drive link to a .csv
link=str(drive_link) # The shareable link

fluff, id = link.split('=')
print ("Link ID verification: ", id) # Verify that you have everything after '='

filename_input = input("Enter desired filename (.csv will automatically be appended): ")

filename = str(filename_input) + ".csv"

downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile(filename)  
dataframe = pd.read_csv(filename)
# Dataset is now stored in a Pandas Dataframe

st.write("Data Import Complete.")

p=Path.cwd()
st.write(p)

st.write(df)



