import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Task 1: Iris Classification", layout="centered")

@st.cache_resource
def train_iris_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf, iris.target_names

st.title("🌸 Task 1: Iris Flower Classification")
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
