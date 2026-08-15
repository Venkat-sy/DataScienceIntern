import pandas as pd
import numpy as np

print("==================================================")
print("TASK 1: DATA CLEANING & EXPLORATION WITH PANDAS")
print("==================================================\n")

# 1. CREATE AND LOAD DATASET
data_1 = {
    "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 103, 108],
    "Age": [25, np.nan, 28, 35, 42, 29, np.nan, 31, 26, 38, 28, 31],
    "Department": ["IT", "HR", "IT", "Sales", "HR", "IT", "Sales", "Marketing", "Marketing", "Sales", "IT", "Marketing"],
    "Salary": [60000, 55000, 62000, 75000, 80000, np.nan, 71000, 68000, 65000, 72000, 62000, 68000],
    "Join_Date": ["2020-01-15", "2019-03-22", "2021-07-01", "2018-11-12", "2015-05-30", "2022-02-18", "2020-09-09", "2019-12-01", "2021-04-14", "2017-08-25", "2021-07-01", "2019-12-01"]
}
df = pd.DataFrame(data_1)
df.to_csv("sample_employee_data.csv", index=False)

print("--- 1. INITIAL DATASET ---")
print(df.head(), "\n")
print(f"Duplicates: {df.duplicated().sum()}")
print("Missing Values:\n", df.isnull().sum(), "\n")

# 2. DATA CLEANING
print("... Cleaning Data (Removing duplicates, filling missing values) ...\n")
df = df.drop_duplicates()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df.groupby("Department")["Salary"].transform(lambda x: x.fillna(x.mean()))
df["Join_Date"] = pd.to_datetime(df["Join_Date"])

# 3. RESULTS
print("--- 2. CLEANED DATASET ---")
print(f"Duplicates: {df.duplicated().sum()}")
print("Missing Values:\n", df.isnull().sum(), "\n")
print("--- Basic Statistics ---")
print(df.describe())

df.to_csv("cleaned_employee_data.csv", index=False)
print("\nSUCCESS: Cleaned data saved to 'cleaned_employee_data.csv'")
