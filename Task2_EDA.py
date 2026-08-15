import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("==================================================\n")

# --- 1. GENERATE CLEANED DATASET DIRECTLY ---
data = {
    "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Age": [25.0, 30.0, 28.0, 35.0, 42.0, 29.0, 30.0, 31.0, 26.0, 38.0],
    "Department": ["IT", "HR", "IT", "Sales", "HR", "IT", "Sales", "Marketing", "Marketing", "Sales"],
    "Salary": [60000.0, 55000.0, 62000.0, 75000.0, 80000.0, 61000.0, 71000.0, 68000.0, 65000.0, 72000.0]
}
df = pd.DataFrame(data)

print("--- Dataset Loaded Successfully ---")
print(df.head(), "\n")

print("... Generating Visualizations ...")
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

sns.histplot(df["Age"], bins=5, kde=True, color="skyblue", ax=axs[0, 0])
axs[0, 0].set_title("Distribution of Employee Ages")

sns.countplot(data=df, x="Department", hue="Department", legend=False, palette="viridis", ax=axs[0, 1])
axs[0, 1].set_title("Number of Employees per Department")

sns.barplot(data=df, x="Department", y="Salary", hue="Department", legend=False, errorbar=None, palette="magma", ax=axs[1, 0])
axs[1, 0].set_title("Average Salary by Department")

sns.scatterplot(data=df, x="Age", y="Salary", hue="Department", s=150, ax=axs[1, 1])
axs[1, 1].set_title("Age vs. Salary")

plt.tight_layout()
print("Close the visualization window to complete the script.")
plt.show()
