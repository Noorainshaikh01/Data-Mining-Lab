# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# -----------------------------------
# 1️⃣ Generate synthetic data
# -----------------------------------
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=42)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# 2️⃣ Agglomerative Clustering
# -----------------------------------
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

# Evaluate using Silhouette Score
agg_silhouette = silhouette_score(X_scaled, agg_labels)
print(f"Agglomerative Clustering Silhouette Score: {agg_silhouette:.4f}")

# Plot Agglomerative clusters
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agg_labels, cmap='rainbow', s=30)
plt.title("Agglomerative Clustering")
plt.show()

# -----------------------------------
# 3️⃣ DBSCAN Clustering
# -----------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Count clusters (-1 represents noise)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

# Check if valid silhouette score can be computed
if n_clusters > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
else:
    print("DBSCAN Silhouette Score: Not applicable (only one cluster detected)")

# Plot DBSCAN clusters
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='rainbow', s=30)
plt.title("DBSCAN Clustering (with Noise)")
plt.show()

# -----------------------------------
# 4️⃣ Comparison & Analysis
# -----------------------------------
print("\n🔍 Clustering Comparison:")
print(f"Agglomerative Clustering → {agg_silhouette:.4f} Silhouette Score")
if n_clusters > 1:
    print(f"DBSCAN Clustering → {dbscan_silhouette:.4f} Silhouette Score")

print("\n🧠 Behaviour Analysis:")
print("👉 Agglomerative Clustering:")
print("- Hierarchical method that merges clusters step-by-step.")
print("- Performs well when clusters are well-separated and spherical.")

print("\n👉 DBSCAN:")
print("- Density-based method that can detect clusters of arbitrary shape.")
print("- Automatically identifies noise/outliers (-1 labels).")
print("- Works well when cluster density varies but sensitive to parameters (eps, min_samples).")
