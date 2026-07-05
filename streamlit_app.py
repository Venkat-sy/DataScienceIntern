import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK stopwords quietly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Data Science Intern Tasks", layout="wide")

@st.cache_resource
def train_iris_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf, iris.target_names

@st.cache_resource
def train_spam_model():
    # Download dataset if not exists
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

@st.cache_resource
def train_sales_model():
    if not os.path.exists("Advertising.csv"):
        url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
        urllib.request.urlretrieve(url, "Advertising.csv")
        
    df = pd.read_csv("Advertising.csv", index_col=0)
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    
    return rf

st.title("📊 Data Science Intern Tasks Demo")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select a Task", ["Task 1: Iris Classification", "Task 4: Spam Detection", "Task 5: Sales Prediction"])

if app_mode == "Task 1: Iris Classification":
    st.header("🌸 Iris Flower Classification")
    st.write("Adjust the sliders below to predict the species of an Iris flower based on its physical measurements.")
    
    model, target_names = train_iris_model()
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)
        
    if st.button("Predict Species"):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        st.success(f"The predicted species is: **{target_names[prediction[0]].capitalize()}**")

elif app_mode == "Task 4: Spam Detection":
    st.header("✉️ Email Spam Detection")
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

elif app_mode == "Task 5: Sales Prediction":
    st.header("📈 Sales Prediction")
    st.write("Adjust the advertising budget across different media channels to predict total sales.")
    
    with st.spinner("Loading Regression Model..."):
        model = train_sales_model()
        
    col1, col2, col3 = st.columns(3)
    with col1:
        tv = st.number_input("TV Budget ($)", min_value=0.0, value=150.0, step=10.0)
    with col2:
        radio = st.number_input("Radio Budget ($)", min_value=0.0, value=30.0, step=5.0)
    with col3:
        newspaper = st.number_input("Newspaper Budget ($)", min_value=0.0, value=20.0, step=5.0)
        
    if st.button("Predict Sales"):
        prediction = model.predict(pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"]))
        st.success(f"Predicted Sales: **{prediction[0]:.2f}** units")
