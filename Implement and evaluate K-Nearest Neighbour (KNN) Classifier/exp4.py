# -----------------------------------------------------
# Implement and Evaluate K-Nearest Neighbour (KNN) Classifier
# Dataset: Iris
# -----------------------------------------------------

# Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Step 1: Load Dataset
# -----------------------------------------------------
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("✅ Iris Dataset Loaded Successfully!")
print("\nFirst 5 Rows of Dataset:\n", df.head())

# -----------------------------------------------------
# Step 2: Split Data into Features and Target
# -----------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

# -----------------------------------------------------
# Step 3: Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n📂 Data Split Complete:")
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# -----------------------------------------------------
# Step 4: Feature Scaling
# -----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✅ Feature Scaling Done (StandardScaler applied).")

# -----------------------------------------------------
# Step 5: Train KNN Classifier
# -----------------------------------------------------
k = 5  # You can experiment with different values
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# -----------------------------------------------------
# Step 6: Evaluate the Model
# -----------------------------------------------------
print(f"\n🔹 KNN Classifier Evaluation (k={k}) 🔹")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"KNN Confusion Matrix (k={k})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------------------------------
# Step 7: Find the Best 'k' using Cross-Validation
# -----------------------------------------------------
k_values = range(1, 16)
cv_scores = []

for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, scaler.fit_transform(X), y, cv=5)
    cv_scores.append(scores.mean())

# Find best k
best_k = k_values[cv_scores.index(max(cv_scores))]
print(f"\n✅ Best k value found using 5-Fold Cross Validation: {best_k}")

# Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o', color='green')
plt.title("KNN Cross-Validation Accuracy for Different k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# Step 8: Retrain with Best k and Evaluate Again
# -----------------------------------------------------
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
best_pred = best_knn.predict(X_test_scaled)

print(f"\n🌟 Final Evaluation with Best k = {best_k}")
print("Final Accuracy:", round(accuracy_score(y_test, best_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, best_pred))

print("\n✅ K-Nearest Neighbour Classifier Implementation Completed Successfully!")
