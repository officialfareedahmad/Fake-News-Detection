import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords           #for to stop words like if, we, and, in, with
from nltk.stem.porter import PorterStemmer  # Loved, Loving == love convert the words to base/root words
from sklearn.feature_extraction.text import TfidfVectorizer         #Loved = [0.0] converting to machine language
from sklearn.model_selection import train_test_split        #convert the data into two parts: one is for training and second is for testing
from sklearn.linear_model import LogisticRegression     #classification problems        
from sklearn.metrics import accuracy_score  #accuracy or performance of the model after the test

news_df = pd.read_csv("train.csv")

news_df = news_df.fillna(' ')

news_df.isna().sum()

news_df['content'] = news_df['author']+ " " + news_df['title']

#Stemming- Cleaning of the content

ps = PorterStemmer()

def stemming(content):
    Stemmed_Content = re.sub('[^a-zA-Z]',' ', content)
    Stemmed_Content = Stemmed_Content.lower()
    Stemmed_Content = Stemmed_Content.split()
    Stemmed_Content = [ ps.stem(word) for word in Stemmed_Content if not word in stopwords.words('english')]
    Stemmed_Content = ' '.join(Stemmed_Content)
    return Stemmed_Content

news_df['content'] = news_df['content'].apply(stemming)

x = news_df['content'].values
y = news_df['label'].values

#conversion to machine language

vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)


#splitting content
x_train, x_test, y_train, y_test = train_test_split (x , y, test_size=0.2, stratify=y,random_state= 1)

model = LogisticRegression() # here is your coach   
model.fit(x_train, y_train)  # which is observing and looking for future predictions



#website

st.title ('Fake news Detector | Spam Detection')

input_text = st.text_input("Enter News Article")


def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]
    

if input_text:
        pred = prediction(input_text)
        if pred == 1:
             st.write('The News is Fake')
        else:
             st.write('The News is Real')
