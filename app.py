"""
Streamlit Web App for SMS Spam Detection
Author: Padmini
"""

import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load model and vectorizer
model = pickle.load(open('models/spam_model.pkl', 'rb'))
cv = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Text Preprocessing
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(word) for word in text 
         if word.isalnum() 
         and word not in stopwords.words('english') 
         and word not in string.punctuation]
    return " ".join(y)

# Streamlit UI
st.title("📩 SMS Spam Detection App")
st.write("Enter your message below to check if it is Spam or Not Spam.")

input_sms = st.text_area("✉️ Enter Message:")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = cv.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 This message is **SPAM!**")
    else:
        st.success("✅ This message is **NOT Spam (Ham)**")

st.caption("Built with ❤️ by Padmini | Streamlit + Machine Learning")
