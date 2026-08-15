import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("==================================================\n")

try:
    df = pd.read_csv("cleaned_employee_data.csv")
    print("--- Dataset Loaded Successfully ---")
    print(df.head(), "\n")
    
    print("... Generating Visualizations ...")
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.histplot(df["Age"], bins=5, kde=True, color="skyblue", ax=axs[0, 0])
    axs[0, 0].set_title("Distribution of Employee Ages")
    
    sns.countplot(data=df, x="Department", palette="viridis", ax=axs[0, 1])
    axs[0, 1].set_title("Number of Employees per Department")
    
    sns.barplot(data=df, x="Department", y="Salary", errorbar=None, palette="magma", ax=axs[1, 0])
    axs[1, 0].set_title("Average Salary by Department")
    
    sns.scatterplot(data=df, x="Age", y="Salary", hue="Department", s=150, ax=axs[1, 1])
    axs[1, 1].set_title("Age vs. Salary")
    
    plt.tight_layout()
    print("Close the visualization window to complete the script.")
    plt.show()
    
except FileNotFoundError:
    print("Error: 'cleaned_employee_data.csv' not found. Please run Task1_DataCleaning.py first!")
