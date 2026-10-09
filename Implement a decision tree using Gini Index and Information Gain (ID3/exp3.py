# -----------------------------------------------------
# Decision Tree Implementation using Gini Index and Information Gain
# Dataset: Iris
# -----------------------------------------------------

# Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Step 1: Load Dataset
# -----------------------------------------------------
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("✅ Iris Dataset Loaded Successfully!")
print("\nFirst 5 rows:\n", df.head())

# -----------------------------------------------------
# Step 2: Split Dataset into Features and Target
# -----------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📂 Data Split Complete:")
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# -----------------------------------------------------
# Step 3: Decision Tree using Gini Index (CART)
# -----------------------------------------------------
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

print("\n🌳 Decision Tree using Gini Index (CART):")
print("Accuracy:", accuracy_score(y_test, y_pred_gini))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gini))

# Plot the tree
plt.figure(figsize=(12, 6))
plot_tree(dt_gini, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree (Gini Index - CART)")
plt.show()

# -----------------------------------------------------
# Step 4: Decision Tree using Information Gain (ID3)
# -----------------------------------------------------
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

print("\n🧠 Decision Tree using Information Gain (ID3):")
print("Accuracy:", accuracy_score(y_test, y_pred_entropy))
print("\nClassification Report:\n", classification_report(y_test, y_pred_entropy))

# Plot the tree
plt.figure(figsize=(12, 6))
plot_tree(dt_entropy, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree (Information Gain - ID3)")
plt.show()

# -----------------------------------------------------
# Step 5: Confusion Matrix Visualization
# -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_gini), annot=True, fmt="d", cmap="Greens", ax=ax[0])
ax[0].set_title("CART (Gini Index) Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_pred_entropy), annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_title("ID3 (Information Gain) Confusion Matrix")
plt.show()

# -----------------------------------------------------
# Step 6: Compare Both Models
# -----------------------------------------------------
print("\n📊 Model Comparison:")
print("CART (Gini Index) Accuracy:", round(accuracy_score(y_test, y_pred_gini), 3))
print("ID3  (Information Gain) Accuracy:", round(accuracy_score(y_test, y_pred_entropy), 3))

print("\n✅ Decision Tree Implementation Completed Successfully!")
