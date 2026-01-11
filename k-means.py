# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mall_Customers.csv')


# Display Dataset Information
df.head() # display first 5 rows
print(f"rows:{df.shape[0]}, columns:{df.shape[1]}") # display dataset shapes
print(df.dtypes) # display data types


# Check For Missing Values and Remove Irrelevant Columns
missing_values = df.isnull().sum()
print(missing_values)

df_clean = df.drop(['CustomerID'], axis = 1)


# Select Meaningful Features for Clustering
X = df_clean[['Annual Income (k$)', 'Spending Score (1-100)']].values


# Normalize/Standardize & Compare
# standard scaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
# min-max scaler
mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)

# comparison using a test k = 5
kmeans_std = KMeans(n_clusters = 5, init = 'random', random_state=42)
labels_std = kmeans_std.fit_predict(X_std)
scores_std = silhouette_score(X_std, labels_std)

kmeans_mm = KMeans(n_clusters = 5, init = 'random', random_state = 42)
labels_mm = kmeans_mm.fit_predict(X_mm)
scores_mm = silhouette_score(X_mm, labels_mm)

X_scaled = X_std


# Find Optimal K Using Elbow Method
inertia = []
k_range = range(1,11)

for k in k_range:
  kmeans = KMeans(n_clusters = k, init = 'random', random_state= 42)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')


# Find Optimal K Using Silhouette Score
silhouette_scores = []
k_range_sil = range(2,11)

for k in k_range_sil:
  kmeans = KMeans(n_clusters = k, init = 'random', random_state = 42)
  labels = kmeans.fit_predict(X_scaled)
  silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.subplot(1, 2, 2)
plt.plot(k_range_sil, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.tight_layout()
plt.show()


# Applying K-Means for Optimal K
optimal_k = 5
kmeans_final = KMeans(n_clusters = optimal_k, init = 'random', random_state = 42)
cluster_labels = kmeans_final.fit_predict(X_scaled)


# Add Labels to Dataset
df['ClusterPart1'] = cluster_labels


# Visualize the Clusters
plt.figure(figsize=(10,6))

X_original = std_scaler.inverse_transform(X_scaled)

for i in range(optimal_k):
  plt.scatter(X_original[cluster_labels == i, 0],
                X_original[cluster_labels == i, 1],
                label=f'Cluster {i}')

plt.scatter(std_scaler.inverse_transform(kmeans_final.cluster_centers_)[:, 0],
            std_scaler.inverse_transform(kmeans_final.cluster_centers_)[:, 1],
            s=200, c='black', marker='X', label='Centroids')

plt.title(f'K-means (Random Init) Clusters (k={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



















































