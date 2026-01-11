# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load and Preprocess
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method with K-Means++
inertia = []
k_range = range(1,11)

for k in k_range:
  kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)


# Silhouette Score with K-Means++ & Plotting the Results
silhouette_scores = []
k_range_sil = range(2,11)

for k in k_range_sil:
  kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
  kmeans.fit(X_scaled)
  silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method (K-means++)')
plt.xlabel('k')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_range_sil, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score (K-means++)')
plt.xlabel('k')
plt.ylabel('Score')
plt.tight_layout()
plt.show()


# K-Means++ with Optimal K
optimal_k = 5
kmeans_pp = KMeans(n_clusters= k, init = 'k-means++', random_state = 42)
labels_pp = kmeans_pp.fit_predict(X_scaled)


# Add to Dataframe
df['ClusterPart2'] = labels_pp


# Visualizing
plt.figure(figsize=(10, 6))
X_original = scaler.inverse_transform(X_scaled)

for i in range(optimal_k):
    plt.scatter(X_original[labels_pp == i, 0],
                X_original[labels_pp == i, 1],
                label=f'Cluster {i}')

plt.scatter(scaler.inverse_transform(kmeans_pp.cluster_centers_)[:, 0],
            scaler.inverse_transform(kmeans_pp.cluster_centers_)[:, 1],
            s=200, c='black', marker='X', label='Centroids')

plt.title(f'K-means++ Clusters (k={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


