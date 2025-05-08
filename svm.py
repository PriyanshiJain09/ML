from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))  # Default SVM
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
