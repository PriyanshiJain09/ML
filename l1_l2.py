import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

def evaluate(model_name, y_test, y_pred):
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

evaluate("Lasso Regression", y_test, y_pred_lasso)
evaluate("Ridge Regression", y_test, y_pred_ridge)

plt.figure(figsize=(10, 5))
plt.plot(lasso.coef_, label='Lasso Coefficients (L1)')
plt.plot(ridge.coef_, label='Ridge Coefficients (L2)', linestyle='--')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso vs Ridge Coefficients')
plt.legend()
plt.grid(True)
plt.show()
