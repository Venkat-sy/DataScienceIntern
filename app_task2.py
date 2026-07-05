import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as st_sns
import kagglehub
import os

st.set_page_config(page_title="Task 2: Unemployment Analysis", layout="wide")

st.title("📊 Task 2: Unemployment Analysis (India)")
st.write("Automatically downloading dataset from Kaggle via `kagglehub`...")

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
    csv_path = os.path.join(path, "Unemployment in India.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

with st.spinner("Downloading and loading data..."):
    df = load_data()

st.success("Dataset loaded successfully!")

st.subheader("Data Overview")
st.write(df.head())

st.subheader("1. Region-wise Average Unemployment Rate")
region_stats = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().reset_index()
region_stats = region_stats.sort_values(by="Estimated Unemployment Rate (%)", ascending=False)

fig1, ax1 = plt.subplots(figsize=(12, 6))
st_sns.barplot(data=region_stats, x="Region", y="Estimated Unemployment Rate (%)", ax=ax1)
plt.xticks(rotation=90)
st.pyplot(fig1)

st.subheader("2. Time-Series Line Chart: Major States")
major_states = ["Maharashtra", "Uttar Pradesh", "Andhra Pradesh", "Delhi", "Karnataka"]
subset_df = df[df["Region"].isin(major_states)]

fig2, ax2 = plt.subplots(figsize=(12, 6))
st_sns.lineplot(data=subset_df, x="Date", y="Estimated Unemployment Rate (%)", hue="Region", marker="o", ax=ax2)
st.pyplot(fig2)

st.subheader("3. Pre-COVID vs Post-COVID Comparison")
df["COVID_Era"] = np.where(df["Date"] < "2020-03-01", "Pre-COVID", "Post-COVID")
covid_stats = df.groupby("COVID_Era")["Estimated Unemployment Rate (%)"].mean().reset_index()

col1, col2 = st.columns([1, 2])
with col1:
    st.write(covid_stats)
with col2:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    st_sns.barplot(data=covid_stats, x="COVID_Era", y="Estimated Unemployment Rate (%)", ax=ax3)
    st.pyplot(fig3)
