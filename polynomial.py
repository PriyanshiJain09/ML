import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv('advertising.csv')
X = df[['TV']]  # Single feature for visualization
y = df['Sales']

# 2. Polynomial Features (degree = 2 or 3)
poly = PolynomialFeatures(degree=2)  # Try degree=3 for more curve
X_poly = poly.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Polynomial Degree: 2")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. Visualization (on sorted data)
X_sorted = np.sort(X.values, axis=0)
X_poly_sorted = poly.transform(X_sorted)
y_pred_plot = model.predict(X_poly_sorted)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X_sorted, y_pred_plot, color='red', linewidth=2, label='Polynomial Curve')
plt.title("Polynomial Regression (TV vs Sales)")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.show()

# 8. Cross-validation
cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R² Score (CV): {np.mean(cv_scores):.4f}")
