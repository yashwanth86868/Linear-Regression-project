"""
Salary Prediction using Simple Linear Regression
Run: python slr.py
Expected CSV: Salary_Data.csv in this project folder
Expected columns: YearsExperience and Salary (or experience in first column, salary in last)
"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"C:\Users\ybali\Downloads\Investment.csv").resolve().parent
DATA_PATH = BASE_DIR / "Salary_Data.csv"
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"
PLOT_PATH = BASE_DIR / "salary_regression_plot.png"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH.name}. Put Salary_Data.csv in this folder."
    )

dataset = pd.read_csv(r"C:\Users\ybali\Downloads\Investment.csv")
if dataset.shape[1] < 2:
    raise ValueError("The CSV must contain at least two columns: experience and salary.")

# Use the first column as the input feature and the last column as the target.
# For the common Salary_Data.csv dataset, these are YearsExperience and Salary.
feature_column = dataset.columns[0]
target_column = dataset.columns[-1]

dataset = dataset[[feature_column, target_column]].dropna()
X = dataset[[feature_column]].to_numpy()
y = dataset[target_column].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
print("\nActual vs Predicted salaries")
print(comparison.to_string(index=False))

print("\nModel parameters")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

print("\nModel performance")
for split_name, actual, predicted in [
    ("Training", y_train, model.predict(X_train)),
    ("Testing", y_test, y_test_pred),
]:
    mse = mean_squared_error(actual, predicted)
    print(f"{split_name} R²:   {r2_score(actual, predicted):.4f}")
    print(f"{split_name} MAE:  {mean_absolute_error(actual, predicted):.2f}")
    print(f"{split_name} MSE:  {mse:.2f}")
    print(f"{split_name} RMSE: {np.sqrt(mse):.2f}")

# Plot test observations and the fitted regression line.
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.scatter(X_test, y_test, color="red", label="Test data")
plt.plot(x_line, model.predict(x_line), color="blue", label="Regression line")
plt.title("Salary vs Years of Experience")
plt.xlabel(feature_column)
plt.ylabel(target_column)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"\nPlot saved to: {PLOT_PATH.name}")

# Save the trained model for the Streamlit app.
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)
print(f"Model saved to: {MODEL_PATH.name}")

# Example predictions; these are model estimates, not guaranteed salaries.
for years in [5, 10, 12, 15, 20]:
    estimate = model.predict([[years]])[0]
    print(f"Estimated salary for {years} years of experience: {estimate:,.2f}")