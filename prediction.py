from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load standard dataset
iris = load_iris()
X, y = iris.data, iris.target   # features and target

# 2. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the model
rf.fit(X_train, y_train)

# 5. Make predictions on test set
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Predict with a sample input
# Example: first row of test data
sample = X_test[0].reshape(1, -1)
prediction = rf.predict(sample)
predicted_class = iris.target_names[prediction][0]

print("Sample input features:", X_test[0])
print("Predicted class:", predicted_class)

