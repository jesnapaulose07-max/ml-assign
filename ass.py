
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

y = diabetes.target


X_simple = df[["bmi"]]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print(f"\nSimple Linear Regression Training Samples : {len(X_train_s)}")
print(f"Simple Linear Regression Testing Samples  : {len(X_test_s)}")

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

print("\n--- Simple Linear Regression Results ---")
print("Intercept :", round(simple_model.intercept_, 4))
print("Slope     :", round(simple_model.coef_[0], 4))
print(f"Disease Progression = {simple_model.intercept_:.4f} + ({simple_model.coef_[0]:.4f} × BMI)")

y_pred_s = simple_model.predict(X_test_s)

mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)
print("Mean Squared Error (MSE) :", round(mse_s, 4))
print("R² Score                 :", round(r2_s, 4))


plt.figure(figsize=(8, 5))
plt.scatter(
    X_test_s,
    y_test_s,
    color="blue",
    alpha=0.5,
    label="Actual Data"
)
X_sorted = X_test_s.sort_values("bmi")
plt.plot(
    X_sorted,
    simple_model.predict(X_sorted),
    color="red",
    linewidth=2,
    label="Regression Line"
)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression: BMI vs Disease Progression")
plt.legend()
plt.savefig("simple_regression_plot.png", dpi=150)
plt.close()
print("\nRegression plot saved as simple_regression_plot.png")


X_multi = df.drop(columns=["target"])

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)


multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

print("\n--- Multiple Linear Regression Results ---")
print("Intercept :", round(multi_model.intercept_, 4))


y_pred_m = multi_model.predict(X_test_m)


mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)
print("Mean Squared Error (MSE) :", round(mse_m, 4))
print("R² Score                 :", round(r2_m, 4))


print("\n" + "="*50)
print(f"{'Model Performance Comparison':^50}")
print("="*50)
print(f"{'Model Type':<30} | {'MSE':<10} | {'R² Score':<10}")
print("-" * 54)
print(f"{'Simple Linear Regression':<30} | {round(mse_s, 4):<10} | {round(r2_s, 4):<10}")
print(f"{'Multiple Linear Regression':<30} | {round(mse_m, 4):<10} | {round(r2_m, 4):<10}")
print("="*50)
