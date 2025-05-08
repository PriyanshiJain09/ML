
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f}")

