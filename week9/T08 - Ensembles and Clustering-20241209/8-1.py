import numpy as np
import pandas as pd

data = np.array([
    [2.5, 3.0], [3.0, 3.0], [2.0, 2.75], [2.5, 2.5], [3.0, 2.5],
    [0.5, 1.0], [1.0, 1.0], [3.0, 1.0], [3.75, 1.0], [0.75, 0.75],
    [1.0, 0.5], [3.5, 0.5]
])

centroids = np.array([
    [0.5, 2.5],  # Centroid A
    [5.0, 2.5],  # Centroid B
    [4.5, 0.5]   # Centroid C
])

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))  # Cluster index (0 for A, 1 for B, 2 for C)
    return np.array(clusters)

def recalculate_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(centroids[i])  # Keep the previous centroid if no points assigned
    return np.array(new_centroids)

k = 3
max_iterations = 10
for iteration in range(max_iterations):
    clusters = assign_clusters(data, centroids)
    new_centroids = recalculate_centroids(data, clusters, k)
    if np.allclose(new_centroids, centroids):  # Check for convergence
        break
    centroids = new_centroids

results = pd.DataFrame(data, columns=["x", "y"])
results["Cluster"] = clusters

print(centroids)
print(results)