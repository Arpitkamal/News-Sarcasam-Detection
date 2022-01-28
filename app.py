import streamlit as st
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from string import punctuation
import string
import re

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("News Sarcasam Detection")

st.subheader("Enter the text")

# Preprocesssing the text

# removing stopwords and Punctuation
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

# for pulling data from HTML and XML
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

# remove noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

@st.cache
def load_split():
    df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
    # droping article link
    df = df.drop("article_link", axis=1)
    #Apply function on review column
    df['headline']=df['headline'].apply(denoise_text)
    x_train,x_test,y_train,y_test = train_test_split(df.headline,df.is_sarcastic, test_size = 0.3 , random_state = 0)
    return x_train

x_train=load_split()

vocab_size = 20000
embedding_dim = 64
max_length = 200
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
# converting training text to sequence of text 
training_sequences = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


    
def main():
    user_input = st.text_input("Enter the text")
    class_btn = st.button("Classify")
    if user_input is not None:    
        st.write("Ented Text: ",user_input)
    if class_btn:
        if user_input is None:
            st.write("Please enter the text...")
        else:
            with st.spinner('Model working....'):
                predictions = predict(user_input)
                time.sleep(1)
                st.success('Classified')
                st.balloons()
                if predictions == 1:
                    st.write("New headline is Sarcastic")
                else:
                    st.write("New headline is not Sarcastic")
                    
def predict(text): 
    classifier_model = "pre-trained_embedding.h5" 
    model = tf.keras.models.load_model(classifier_model)
    text_headline = denoise_text(text)
    headline_seq=tokenizer.texts_to_sequences([text_headline])
    headline_seq_pad=pad_sequences(headline_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return round(model.predict(headline_seq_pad)[0][0])
    
main()