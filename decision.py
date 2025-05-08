import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=columns)

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

X = df.drop('Species', axis=1)  # Features
y = df['Species']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Decision Tree Classifier Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



