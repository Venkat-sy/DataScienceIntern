import streamlit as st
import pandas as pd
import urllib.request
import os
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Task 5: Sales Prediction", layout="centered")

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

st.title("📈 Task 5: Sales Prediction")
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
