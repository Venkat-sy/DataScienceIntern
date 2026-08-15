import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("==================================================")
print("TASK 3: SIMPLE LINEAR REGRESSION MODEL")
print("==================================================\n")

# 1. CREATE DATASET
np.random.seed(42)
X_data = np.linspace(1, 10, 30)
y_data = 40000 + 9000 * X_data + np.random.normal(0, 5000, 30)
df = pd.DataFrame({"Years_Experience": X_data, "Salary": y_data})

print("--- 1. Dataset Loaded ---")
print(df.head(), "\n")

# 2. SPLIT DATA
X = df[["Years_Experience"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN MODEL
print("... Training Linear Regression Model ...\n")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. EVALUATION
print("--- 2. Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {mean_squared_error(y_test, y_pred)**0.5:.2f}")
print(f"R-squared (R2) Score: {r2_score(y_test, y_pred):.4f}\n")

# 5. VISUALIZATION
print("... Generating Visualization ...")
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.title("Salary vs. Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
print("Close the visualization window to complete the script.")
plt.show()
