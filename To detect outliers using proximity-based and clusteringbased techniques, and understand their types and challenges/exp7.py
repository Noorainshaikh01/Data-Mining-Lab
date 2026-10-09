# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# -----------------------------------
# 1️⃣ Generate Synthetic Data
# -----------------------------------
# Creating 2D data with some random outliers
X, _ = make_blobs(n_samples=250, centers=1, cluster_std=0.60, random_state=42)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-6, high=6, size=(20, 2))  # Add 20 outliers
X = np.concatenate([X, outliers], axis=0)

# Plot raw data
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], s=30)
plt.title("Generated Data with Outliers")
plt.show()

# -----------------------------------
# 2️⃣ PROXIMITY-BASED OUTLIER DETECTION
# -----------------------------------
# 2.1 Using Z-Score Method
df = pd.DataFrame(X, columns=["X1", "X2"])
df_scaled = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=["X1", "X2"])

# Calculate Z-scores
z_scores = np.abs((df_scaled - df_scaled.mean()) / df_scaled.std())
threshold = 3
z_outliers = (z_scores > threshold).any(axis=1)

# 2.2 Using IQR Method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

# Plot Z-Score Outliers
plt.figure(figsize=(7,5))
plt.scatter(df["X1"], df["X2"], c=~z_outliers, cmap='coolwarm', s=30)
plt.title("Proximity-Based Outlier Detection (Z-Score)")
plt.show()

# Plot IQR Outliers
plt.figure(figsize=(7,5))
plt.scatter(df["X1"], df["X2"], c=~iqr_outliers, cmap='viridis', s=30)
plt.title("Proximity-Based Outlier Detection (IQR Method)")
plt.show()

# -----------------------------------
# 3️⃣ CLUSTERING-BASED OUTLIER DETECTION (DBSCAN)
# -----------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(df_scaled)

# -1 represents noise/outliers in DBSCAN
db_outliers = labels == -1
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN detected {n_clusters} clusters and {db_outliers.sum()} outliers.")

# Plot DBSCAN Clustering with Outliers
plt.figure(figsize=(7,5))
plt.scatter(df_scaled["X1"], df_scaled["X2"], c=labels, cmap='rainbow', s=30)
plt.title("Clustering-Based Outlier Detection (DBSCAN)")
plt.show()

# -----------------------------------
# 4️⃣ ANALYSIS AND DISCUSSION
# -----------------------------------
print("\n🔍 OUTLIER DETECTION ANALYSIS")
print("✅ Proximity-Based Methods (Z-score & IQR):")
print("- Identify points far from the mean or quartile range.")
print("- Simple, effective for numerical continuous data.")
print("- Struggles with non-Gaussian distributions and multi-dimensional data.\n")

print("✅ Clustering-Based Method (DBSCAN):")
print("- Detects outliers as noise points (label = -1).")
print("- Works well for irregularly shaped clusters.")
print("- Sensitive to parameters (eps, min_samples).")
print("- Can handle multi-dimensional and non-linear data.\n")

print("⚠️ Challenges in Outlier Detection:")
print("- Defining what qualifies as an outlier is subjective.")
print("- Sensitive to parameter tuning and data scaling.")
print("- Difficult in high-dimensional spaces (curse of dimensionality).")
print("- Noise vs. Novelty: Some outliers may contain useful information.")
