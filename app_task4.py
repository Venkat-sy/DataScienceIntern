import streamlit as st
import pandas as pd
import urllib.request
import zipfile
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Task 4: Spam Detection", layout="centered")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

@st.cache_resource
def train_spam_model():
    if not os.path.exists("sms_data/SMSSpamCollection"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, "smsspamcollection.zip")
        with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
            zip_ref.extractall("sms_data")
            
    df = pd.read_csv("sms_data/SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
    
    stop_words = set(stopwords.words("english"))
    def preprocess_text(text):
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    df["clean_message"] = df["message"].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df["clean_message"])
    y = df["label"].map({"ham": 0, "spam": 1})
    
    clf = MultinomialNB()
    clf.fit(X_tfidf, y)
    
    return clf, vectorizer, preprocess_text

st.title("✉️ Task 4: Email Spam Detection")
st.write("Enter an email or SMS message below to classify it as Spam or Ham (legitimate).")

with st.spinner("Loading NLP Model..."):
    model, vectorizer, preprocess_fn = train_spam_model()

user_input = st.text_area("Message Content", "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now.")

if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_text = preprocess_fn(user_input)
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**")
        else:
            st.success("✅ This message is classified as **HAM** (Legitimate)")
