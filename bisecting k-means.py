# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings

# Load and Preprocess
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method with Bisecting K-Means 
inertia = []
k_range = range(1,11)

for k in k_range:
  bkmeans = BisectingKMeans(n_clusters = k, random_state = 42)
  bkmeans.fit(X_scaled)
  inertia.append(bkmeans.inertia_)


# Silhoutte Score for Bisecting K-Means & Plotting
silhouette_scores = []
k_range_sil = range(2,11)

for k in k_range_sil:
  bkmeans = BisectingKMeans(n_clusters = k, random_state = 42)
  labels = bkmeans.fit_predict(X_scaled)
  silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method (Bisecting K-means)')
plt.xlabel('k')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_range_sil, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score (Bisecting K-means)')
plt.xlabel('k')
plt.ylabel('Score')
plt.tight_layout()
plt.show()


# Apply Bisecting K-means with Optimal K
optimal_k = 5
bkmeans_final = BisectingKMeans(n_clusters = optimal_k, random_state = 42)
labels_bk = bkmeans_final.fit_predict(X_scaled)


# Add to Dataframe
df['ClusterPart3'] = labels_bk


# Visualizing
plt.figure(figsize=(10, 6))
X_original = scaler.inverse_transform(X_scaled)

for i in range(optimal_k):
    plt.scatter(X_original[labels_bk == i, 0],
                X_original[labels_bk == i, 1],
                label=f'Cluster {i}')

if hasattr(bkmeans_final, 'cluster_centers_'):
    plt.scatter(scaler.inverse_transform(bkmeans_final.cluster_centers_)[:, 0],
                scaler.inverse_transform(bkmeans_final.cluster_centers_)[:, 1],
                s=200, c='black', marker='X', label='Centroids')

plt.title(f'Bisecting K-means Clusters (k={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
