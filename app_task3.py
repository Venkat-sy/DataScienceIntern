import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Task 3: Car Price Prediction", layout="wide")

st.title("🚗 Task 3: Car Price Prediction")
st.write("Please upload the **car_data.csv** (Vehicle dataset from cardekho) to train the model.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("Dataset loaded successfully! Training model...")
    
    # Preprocessing
    df.drop_duplicates(inplace=True)
    df["Car_Age"] = 2024 - df["Year"]
    df["Brand"] = df["Car_Name"].apply(lambda x: str(x).split(" ")[0])
    
    categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission", "Brand"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    df.drop(["Car_Name", "Year"], axis=1, inplace=True)
    
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    st.subheader("Model Trained (Random Forest)")
    st.write(f"R² Score on full data: **{r2_score(y, rf.predict(X)):.4f}**")
    
    st.subheader("Predict Car Price")
    
    col1, col2 = st.columns(2)
    with col1:
        present_price = st.number_input("Present Price (in Lakhs)", value=5.0)
        kms_driven = st.number_input("Kilometers Driven", value=50000)
        owner = st.selectbox("Owner (0 = First, 1 = Second)", [0, 1, 2])
        car_age = st.number_input("Car Age (Years)", value=5)
        
    with col2:
        fuel = st.selectbox("Fuel Type", label_encoders["Fuel_Type"].classes_)
        seller = st.selectbox("Seller Type", label_encoders["Seller_Type"].classes_)
        transmission = st.selectbox("Transmission", label_encoders["Transmission"].classes_)
        brand = st.selectbox("Brand", label_encoders["Brand"].classes_)
        
    if st.button("Predict Selling Price"):
        input_data = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type": label_encoders["Fuel_Type"].transform([fuel])[0],
            "Seller_Type": label_encoders["Seller_Type"].transform([seller])[0],
            "Transmission": label_encoders["Transmission"].transform([transmission])[0],
            "Owner": owner,
            "Car_Age": car_age,
            "Brand": label_encoders["Brand"].transform([brand])[0]
        }
        
        input_df = pd.DataFrame([input_data])[X.columns]
        pred = rf.predict(input_df)[0]
        st.success(f"Estimated Selling Price: **{pred:.2f} Lakhs**")
