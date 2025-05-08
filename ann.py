
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=columns)

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

X = df.drop('Species', axis=1)  # Features
y = df['Species']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))  # Input layer and first hidden layer
model.add(Dense(8, activation='relu'))  # Second hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🔹 ANN Model Accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=-1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

