from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np

def laplacian(A: np.ndarray):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """

    m = A.shape[0]
    # TODO: Calculate degree matrix
    D = np.diag(A.sum(axis=0))

    # TODO: Calculate the inverse square root of the symmetric matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # TODO: Return symmetric normalized Laplacian matrix
    return np.eye(A.shape[0]) - (D_inv_sqrt @ A @ D_inv_sqrt)


def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    # TODO: Compute Laplacian matrix
    L = laplacian(affinity)

    # TODO: Compute the first k eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = LA.eig(L)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[:k]]

    # TODO: Apply K-means clustering on the selected eigenvectors
    labels = k_means_clustering(eigenvectors, k)[0]

    # TODO: Return cluster labels
    return labels

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    np.random.seed(1)
    random_points = np.random.randint(0, 100, (100, 2))
    # data = make_blobs(n_samples=100, n_features=2)
    # random_points = data[0]
    pairwise_distances = np.linalg.norm(random_points[:, None] - random_points, axis=2)
    print(spectral_clustering(pairwise_distances, k=3))

