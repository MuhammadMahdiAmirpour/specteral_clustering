import numpy as np
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score

def chamfer_distance(point_cloud_A, point_cloud_B, batch_size=10):
    """
    Compute the Chamfer distance between two point clouds in a batched manner.

    Parameters:
    - point_cloud_A: np.ndarray, shape (m, n)
        First point cloud.
    - point_cloud_B: np.ndarray, shape (p, n)
        Second point cloud.
    - batch_size: int, optional
        Batch size for processing point clouds in chunks.

    Returns:
    - chamfer_dist: float
        Chamfer distance.
    """
    # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2

    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1

    # TODO: Return Chamfer distance, sum of the average distances in both directions

    distances_A_to_B = np.linalg.norm(point_cloud_A[:, None] - point_cloud_B, axis=2)
    distances_B_to_A = np.linalg.norm(point_cloud_B[:, None] - point_cloud_A, axis=2)
    chamfer_dist = (np.sum(np.min(distances_A_to_B, axis=1)) + np.sum(np.min(distances_B_to_A, axis=1))) / (len(point_cloud_A) + len(point_cloud_B))
    return chamfer_dist

    
def register(point_cloud1, point_cloud2):
    """
    Registers point_cloud1 and point_cloud2 to align them and optimize distance

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - registered_point_cloud1: numpy array, shape (N1, D), representing transformed version of the first point cloud.
    - registered_point_cloud2: numpy array, shape (N2, D), representing transformed version of the second point cloud.
    """

    # TODO: Find a rigid or non-rigid transformation

    # TODO: Transform point clouds by transformation

    # TODO: Return transformed point clouds

    pass


def construct_affinity_matrix(point_clouds, batch_size=10):
    """
    Construct the affinity matrix for a set of point clouds using Chamfer distance in a batched manner.

    Parameters:
    - point_clouds: list of np.ndarray
        List of point clouds, each with shape (m, n).
    - batch_size: int, optional
        Batch size for processing point clouds in chunks.

    Returns:
    - affinity_matrix: np.ndarray, shape (num_point_clouds, num_point_clouds)
        Affinity matrix.
    """
    num_point_clouds = len(point_clouds)
    point_clouds_array = np.array(point_clouds)

    # Calculate the number of batches
    num_batches = (num_point_clouds + batch_size - 1) // batch_size

    affinity_matrix = np.zeros((num_point_clouds, num_point_clouds))

    for batch_start in range(0, num_point_clouds, batch_size):
        batch_end = min(batch_start + batch_size, num_point_clouds)
        batch_point_clouds = point_clouds_array[batch_start:batch_end]

        for i in range(batch_start, min(batch_end, num_point_clouds)):
            for j in range(i + 1, num_point_clouds):
                chamfer_dist_ij = chamfer_distance(batch_point_clouds[i - batch_start], point_clouds_array[j])
                affinity_matrix[i, j] = chamfer_dist_ij
                affinity_matrix[j, i] = chamfer_dist_ij  # Symmetric matrix

    return affinity_matrix


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Plot Ach using its first 3 eigenvectors

