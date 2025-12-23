import numpy as np
from scipy.spatial import cKDTree
from collections import deque

class EuclideanCluster:
    def __init__(self, cluster_tolerance, min_cluster_size, max_cluster_size=np.inf):
        self.cluster_tolerance = cluster_tolerance  # This is our epsilon
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster(self, points):
        tree = cKDTree(points)
        clusters = []
        processed = set()

        for i in range(len(points)):
            if i in processed:
                continue

            cluster = self._grow_cluster(i, tree, points, processed)
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(cluster)

        # Create labels array
        labels = np.full(len(points), -1, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            labels[cluster] = cluster_id

        return labels

    def _grow_cluster(self, seed_point_idx, tree, points, processed):
        cluster = []
        queue = deque([seed_point_idx])

        while queue:
            point_idx = queue.popleft()
            if point_idx in processed:
                continue

            processed.add(point_idx)
            cluster.append(point_idx)

            # Here we use the cluster_tolerance (epsilon) to find neighboring points
            neighbors = tree.query_ball_point(points[point_idx], self.cluster_tolerance)
            queue.extend([idx for idx in neighbors if idx not in processed])

        return cluster
    
# Example usage
if __name__ == "__main__":
    # Generate a sample point cloud
    np.random.seed(0)
    points = np.random.rand(10000, 3) * 100

    # Add some clustered points
    points = np.vstack([points, np.random.normal(50, 2, (1000, 3))])
    points = np.vstack([points, np.random.normal(80, 2, (1000, 3))])

    # Cluster the points
    clusterer = EuclideanCluster(cluster_tolerance=0.1, min_cluster_size=10, max_cluster_size=10000)
    labels = clusterer.cluster(points)

    print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"Number of noise points: {np.sum(labels == -1)}")