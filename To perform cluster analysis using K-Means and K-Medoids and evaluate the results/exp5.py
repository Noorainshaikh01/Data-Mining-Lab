# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids  # install via: pip install scikit-learn-extra

# Generate synthetic dataset for demonstration
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 1️⃣ K-MEANS CLUSTERING
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluate using Silhouette Score
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

# Plot K-Means Clusters
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='rainbow', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='black', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()

# -------------------------------
# 2️⃣ K-MEDOIDS CLUSTERING
# -------------------------------
kmedoids = KMedoids(n_clusters=4, random_state=42)
kmedoids_labels = kmedoids.fit_predict(X_scaled)

# Evaluate using Silhouette Score
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids_labels)
print(f"K-Medoids Silhouette Score: {kmedoids_silhouette:.4f}")

# Plot K-Medoids Clusters
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmedoids_labels, cmap='rainbow', s=30)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], 
            c='black', marker='X', s=200, label='Medoids')
plt.title("K-Medoids Clustering")
plt.legend()
plt.show()

# -------------------------------
# 3️⃣ COMPARISON OF RESULTS
# -------------------------------
print("\n🔍 Comparison of Clustering Methods:")
print(f"K-Means Silhouette Score : {kmeans_silhouette:.4f}")
print(f"K-Medoids Silhouette Score: {kmedoids_silhouette:.4f}")

if kmeans_silhouette > kmedoids_silhouette:
    print("\n✅ K-Means performed better based on Silhouette Score.")
else:
    print("\n✅ K-Medoids performed better based on Silhouette Score.")
