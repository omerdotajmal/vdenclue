import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KDTree

class VDENCLUE:
    def __init__(self, initial_bandwidth=1.0, epsilon=1e-3, max_iter=100, k=11, min_density=6.0):
        self.initial_bandwidth = initial_bandwidth
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.k = k
        self.min_density = min_density

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.tree = KDTree(X)
        self.densities = np.zeros(self.n_samples)
        self.attractors = np.zeros_like(X)
        self.labels_ = -1 * np.ones(self.n_samples, dtype=int)
        
        self._compute_local_features()
        self._compute_varying_bandwidth()
        self._compute_densities()
        self._find_attractors()
        self._assign_clusters()

    def _compute_local_features(self):
        self.local_features = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            neighbors = self.tree.query_radius(self.X[i].reshape(1, -1), r=self.initial_bandwidth)
            self.local_features[i] = len(neighbors[0])  # Example: local density

    def _compute_varying_bandwidth(self):
        self.bandwidths = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            # Calculate distance to Kth nearest neighbor
            distances, _ = self.tree.query(self.X[i].reshape(1, -1), k=self.k + 1)  # k+1 because the point itself is included
            self.bandwidths[i] = distances[0][-1]  # Kth distance


    def _compute_densities(self):
        for i in range(self.n_samples):
            neighbors = self.tree.query_radius(self.X[i].reshape(1, -1), r=self.bandwidths[i])
            self.densities[i] = np.sum(np.exp(-np.linalg.norm(self.X[neighbors[0]] - self.X[i], axis=1)**2 / (2 * self.bandwidths[i]**2)))

    def _find_attractors(self):
        for i in range(self.n_samples):
            self.attractors[i] = self._find_attractor(self.X[i], self.bandwidths[i])

    def _find_attractor(self, point, bandwidth):
        for _ in range(self.max_iter):
            neighbors = self.tree.query(point.reshape(1, -1), k=self.k, return_distance=False)
            new_point = np.sum(self.X[neighbors[0]] * np.exp(-np.linalg.norm(self.X[neighbors[0]] - point, axis=1)**2 / (2 * bandwidth**2)).reshape(-1, 1), axis=0)
            new_point /= np.sum(np.exp(-np.linalg.norm(self.X[neighbors[0]] - point, axis=1)**2 / (2 * bandwidth**2)))
            
            if np.linalg.norm(new_point - point) < self.epsilon:
                break
            
            point = new_point
        return point

    def _assign_clusters(self):
        cluster_id = 0
        for i in range(self.n_samples):
            if self.labels_[i] == -1 and self.densities[i] > self.min_density:
                self.labels_[i] = cluster_id
                for j in range(self.n_samples):
                    if np.linalg.norm(self.attractors[i] - self.attractors[j]) < self.bandwidths[i]:
                        self.labels_[j] = cluster_id
                cluster_id += 1

    def predict(self, X):
        return self.labels_


# Load the dataset
file_path = ""
data = pd.read_csv(file_path)

# the last column is the ground truth labels
X = data.iloc[:, :-1].values
y_true = data.iloc[:, -1].values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Change the default values as required
bandwidth = 1.0
iter = 500
k_value=12
m_density = 4

vdenclue = VDENCLUE(initial_bandwidth=bandwidth, epsilon=1e-3, max_iter=iter, k=k_value, min_density=m_density)
vdenclue.fit(X_scaled)
y_pred = vdenclue.predict(X_scaled)

# Calculate Clustering quality metrics
ari = adjusted_rand_score(y_true, y_pred)
silhouette = silhouette_score(X_scaled, y_pred)
davies_bouldin = davies_bouldin_score(X_scaled, y_pred)


print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Silhouette Score (SI): {silhouette}")
print(f"Davies-Bouldin Index (DBI): {davies_bouldin}")

# Print the number of clusters and the size of each cluster
unique_labels, counts = np.unique(y_pred, return_counts=True)
print(f"Number of clusters: {len(unique_labels)}")
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")
