# -------------------------------
# Data Exploration and Preprocessing in Python
# Using Iris Dataset
# -------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("✅ Dataset Loaded Successfully!\n")

# -------------------------------
# Step 2: Basic Information
# -------------------------------
print("📘 Dataset Information:\n")
print(df.info())

print("\n📊 Dataset Summary Statistics:\n")
print(df.describe())

print("\n🔍 Checking Missing Values:\n")
print(df.isnull().sum())

# -------------------------------
# Step 3: Data Exploration
# -------------------------------
print("\n📈 First 5 Rows of Dataset:\n")
print(df.head())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df, hue='target', palette='husl')
plt.show()

# -------------------------------
# Step 4: Label Encoding (if categorical data exists)
# -------------------------------
# Example for categorical encoding (not needed for Iris)
# df['species'] = LabelEncoder().fit_transform(df['species'])

# -------------------------------
# Step 5: Feature Scaling
# -------------------------------
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Feature Scaling Done (StandardScaler applied).")

# -------------------------------
# Step 6: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📂 Data Split Complete:")
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# -------------------------------
# Step 7: Verify Processed Data
# -------------------------------
print("\n✅ Data Exploration & Preprocessing Completed Successfully!")
